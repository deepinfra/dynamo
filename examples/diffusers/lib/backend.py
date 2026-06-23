# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generic video-pipeline backend.

``GenericVideoBackend`` is model-agnostic glue: it owns the
serialization lock, the optional ``SubprocessPool``, and the Dynamo
``create_video`` endpoint, but delegates model loading to a
caller-supplied factory function (LTX-2's ``ltx2.factory:load_model``
today). The factory's return value must expose a ``generate_video()``
method with the FastVideo keyword-args contract.

Two paths share the lock:

* Legacy in-process: the factory runs in this process at
  ``initialize_model()`` time; ``create_video`` calls
  ``generator.generate_video(...)`` directly via ``asyncio.to_thread``.
* Pool: ``initialize_model()`` creates a ``SubprocessPool`` (does NOT
  load the model in-process); ``create_video`` routes each request to
  a per-shape subprocess that owns its own generator.
"""

import argparse
import asyncio
import base64
import json
import logging
import os
import tempfile
import time
import uuid
from typing import Any, Callable

from dynamo.runtime import dynamo_endpoint

from .models import VideoCreateRequest, VideoCreateResponse, VideoData
from .pool import VIDEO_POOL_MAX_SIZE, VIDEO_POOL_MODE, SubprocessPool

logger = logging.getLogger(__name__)


def _coerce_optional_float(value: object) -> float | None:
    """Best-effort conversion for optional numeric metrics from backend results."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class GenericVideoBackend:
    def __init__(
        self,
        args: argparse.Namespace,
        model_factory_callable: Callable[[str, int, bool], Any],
        model_factory_dotted: str,
        model_label: str,
    ) -> None:
        self.model_name: str = args.model
        self.served_model_name: str = args.served_model_name or args.model
        self.num_gpus: int = args.num_gpus
        self.enable_optimizations: bool = args.enable_optimizations
        self.attention_backend: str = args.attention_backend

        # Model factory: pool mode passes the dotted form to subprocesses
        # via --model-factory; legacy mode calls the callable directly.
        self._model_factory_callable = model_factory_callable
        self._model_factory_dotted = model_factory_dotted
        # Value of the ``model`` Prometheus label attached to every
        # video_pool_* series this pool emits.
        self._model_label = model_label

        # One request at a time — both the legacy in-process generator
        # and the pool path serialize at the pod level via this lock
        # (GPU is single-tenant).
        self._generate_lock = asyncio.Lock()
        self.generator: Any | None = None

        # Phase 2: opt-in subprocess pool. When enabled, customer
        # requests are routed to per-shape persistent subprocesses
        # instead of the in-process generator. See lib/pool.py
        # module-level constants.
        self.pool_mode: bool = VIDEO_POOL_MODE
        self.pool: SubprocessPool | None = None

        os.environ["FASTVIDEO_ATTENTION_BACKEND"] = self.attention_backend
        os.environ["FASTVIDEO_STAGE_LOGGING"] = "1"
        os.environ["FASTVIDEO_ENABLE_RMSNORM_FP4_PREQUANT"] = "0"

    async def initialize_model(self) -> None:
        if self.pool_mode:
            logger.info(
                "pool mode enabled: creating SubprocessPool "
                "(max=%d, model=%s); no in-process generator load",
                VIDEO_POOL_MAX_SIZE,
                self.model_name,
            )
            self.pool = SubprocessPool(
                model_path=self.model_name,
                num_gpus=self.num_gpus,
                enable_optimizations=self.enable_optimizations,
                attention_backend=self.attention_backend,
                model_factory_dotted=self._model_factory_dotted,
                model_label=self._model_label,
            )
            return

        logger.info("Loading generator model=%s", self.model_name)
        loop = asyncio.get_running_loop()
        self.generator = await loop.run_in_executor(
            None,
            self._model_factory_callable,
            self.model_name,
            self.num_gpus,
            self.enable_optimizations,
        )
        logger.info("Generator ready")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _generate_mp4(
        self,
        prompt: str,
        video_id: str,
        width: int,
        height: int,
        num_frames: int,
        fps: int,
        num_inference_steps: int,
        guidance_scale: float,
        seed: int | None,
        negative_prompt: str | None,
        ltx2_images: list[tuple[str, int, float]] | None = None,
        ltx2_image_crf: float = 0.0,
    ) -> bytes:
        """Generate a video clip and return it as MP4 bytes (i2v when ltx2_images set)."""
        assert self.generator is not None

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.mp4")
            kwargs: dict = dict(
                save_video=True,
                return_frames=False,
                output_path=output_path,
                height=height,
                width=width,
                num_frames=num_frames,
                fps=fps,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )
            if seed is not None:
                kwargs["seed"] = seed
            if negative_prompt is not None:
                kwargs["negative_prompt"] = negative_prompt
            if ltx2_images:
                # i2v: FastVideo LTX-2 contract -- list[(path, frame_idx, strength)].
                # Same compiled shape as t2v; only the per-token denoise mask differs.
                kwargs["ltx2_images"] = ltx2_images
                kwargs["ltx2_image_crf"] = ltx2_image_crf

            result = self.generator.generate_video(prompt=prompt, **kwargs)
            result_dict = result if isinstance(result, dict) else {}
            generation_time = _coerce_optional_float(result_dict.get("generation_time"))
            e2e_latency = _coerce_optional_float(result_dict.get("e2e_latency"))
            logger.info("[%s] MP4 written to %s", video_id, output_path)
            if generation_time is not None:
                logger.info(
                    "[%s] Generation time: %.2f seconds", video_id, generation_time
                )
            else:
                logger.info("[%s] Generation time: unavailable", video_id)

            if e2e_latency is not None:
                logger.info("[%s] E2E latency: %.2f seconds", video_id, e2e_latency)
            else:
                logger.info("[%s] E2E latency: unavailable", video_id)

            time_start = time.perf_counter()
            with open(output_path, "rb") as f:
                data = f.read()
            time_end = time.perf_counter()
            logger.info(
                "[%s] File read time: %.2f seconds", video_id, time_end - time_start
            )

            return data

    # ── Preflight ─────────────────────────────────────────────────────────────

    async def preflight(self) -> None:
        """
        Warm in-memory torch.compile state for every shape the API admits,
        so the FIRST customer request lands at steady-state latency
        regardless of which shape arrives first.

        Opt-in via ``LTX2_PREFLIGHT=1`` (callers gate the call, not this
        method). Reads the shapes JSON via ``WARMUP_SHAPES_JSON_PATH``
        env var; the per-model entry point sets a default for this env
        before backend setup so subprocess workers inherit the right
        path too.

        Failures abort pod boot. If the cache shipped in the image
        doesn't cover a menu shape (or there is a code/version mismatch
        between the warmup that built the cache and this worker), pod
        crash-loops -- better than serving customer traffic with
        ~15-minute first-shape recompiles.
        """
        if self.pool_mode:
            # Eager warm-on-boot: spawn + warm every menu shape's persistent
            # subprocess via the SAME SubprocessPool.route() path serving uses
            # (the subprocess stays resident in the LRU pool afterwards). This
            # moves the 200-600s first-call in-memory warm to boot so the first
            # real customer request per shape is steady-state, and readiness is
            # gated on it. Failures abort boot (crash-loop) -- better than
            # serving cold or with a broken cache.
            if self.pool is None:
                raise RuntimeError(
                    "preflight called before initialize_model; this is a bug."
                )
            shapes_path = os.environ.get("WARMUP_SHAPES_JSON_PATH")
            if not shapes_path or not os.path.isfile(shapes_path):
                raise RuntimeError(
                    "preflight (pool): WARMUP_SHAPES_JSON_PATH unset/missing "
                    "(%r); cannot eager-warm. Fix the image build." % shapes_path
                )
            with open(shapes_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            shapes = cfg.get("shapes", [])
            fps = int(cfg.get("fps", 24))
            guidance_scale = float(cfg.get("guidance_scale", 1.0))
            # i2v is a SEPARATE compiled graph from t2v (measured 2026-06-19: it
            # adds its own ~8 fx graphs; it does NOT "ride the same compiled
            # shape" as an earlier comment wrongly claimed). So when the model
            # serves i2v we MUST warm it too, or the readiness gate lies and the
            # first i2v request eats a cold ~9-18min recompile. Opt in via
            # shapes.json "warm_i2v": true. Both modes warm into the SAME
            # resident per-shape process (same shape_key), so no extra pool slot.
            warm_i2v = bool(cfg.get("warm_i2v", False))
            if len(shapes) > VIDEO_POOL_MAX_SIZE:
                raise RuntimeError(
                    "eager warm-on-boot needs VIDEO_POOL_MAX_SIZE (%d) >= number of "
                    "menu shapes (%d): otherwise warming the last shape LRU-evicts "
                    "an earlier one and defeats warm-on-boot. Bump "
                    "VIDEO_POOL_MAX_SIZE or trim shapes.json."
                    % (VIDEO_POOL_MAX_SIZE, len(shapes))
                )
            logger.info(
                "preflight(pool): eager-warming %d shapes (i2v=%s) from %s",
                len(shapes),
                warm_i2v,
                shapes_path,
            )
            t_total = time.perf_counter()

            def _warm_req(label, w, h, nf, tmpdir):
                return {
                    "request_id": "preflight_%s_%dx%d@%df" % (label, w, h, nf),
                    "prompt": "warmup",
                    "width": w,
                    "height": h,
                    "num_frames": nf,
                    "fps": fps,
                    "num_inference_steps": 1,
                    "guidance_scale": guidance_scale,
                    "seed": 42,
                    "negative_prompt": None,
                    "output_path": os.path.join(
                        tmpdir, "preflight_%s_%dx%d@%df.mp4" % (label, w, h, nf)
                    ),
                }

            def _check(shape_key, label, result, out_path):
                # route() DONE is necessary but not sufficient: a broken bake can
                # report DONE yet write a missing/empty file. Fail boot loudly.
                if result.get("status") != "DONE":
                    raise RuntimeError(
                        "preflight %s failed for shape %s: %s; refusing to start. "
                        "Likely a baked-cache miss or a FastVideo version "
                        "mismatch. See ltx23/RUNBOOK.md."
                        % (label, shape_key, result.get("error", result.get("status")))
                    )
                if not os.path.isfile(out_path) or os.path.getsize(out_path) == 0:
                    raise RuntimeError(
                        "preflight %s: shape %s reported DONE but produced a "
                        "missing/empty output file (%s); refusing to start "
                        "(broken bake)." % (label, shape_key, out_path)
                    )

            with tempfile.TemporaryDirectory() as tmpdir:
                # Synthetic conditioning image for the i2v warm: content is
                # irrelevant (it only needs to be a valid image so the i2v code
                # path compiles); the pipeline resizes it to the target shape.
                i2v_cond_path = None
                if warm_i2v:
                    i2v_cond_path = os.path.join(tmpdir, "preflight_i2v_cond.png")
                    try:
                        from PIL import Image

                        Image.new("RGB", (512, 512), (128, 128, 128)).save(
                            i2v_cond_path
                        )
                    except Exception as exc:
                        raise RuntimeError(
                            "preflight: could not create i2v warm conditioning "
                            "image (%r); refusing to start." % exc
                        )
                for idx, shape in enumerate(shapes, 1):
                    w = int(shape["width"])
                    h = int(shape["height"])
                    nf = int(shape["num_frames"])
                    shape_key = "%dx%d@%df" % (w, h, nf)
                    t_shape = time.perf_counter()
                    # t2v warm. route() expects the caller to hold the global
                    # lock; it also serializes spawns (GPU is single-tenant).
                    t2v_req = _warm_req("t2v", w, h, nf, tmpdir)
                    async with self._generate_lock:
                        result = await self.pool.route(shape_key, t2v_req)
                    _check(shape_key, "t2v", result, t2v_req["output_path"])
                    # i2v warm: same shape_key (same resident process) but a
                    # distinct compiled graph, so it must be warmed explicitly.
                    if warm_i2v:
                        i2v_req = _warm_req("i2v", w, h, nf, tmpdir)
                        i2v_req["ltx2_images"] = [(i2v_cond_path, 0, 1.0)]
                        i2v_req["ltx2_image_crf"] = 0.0
                        async with self._generate_lock:
                            result = await self.pool.route(shape_key, i2v_req)
                        _check(shape_key, "i2v", result, i2v_req["output_path"])
                    logger.info(
                        "preflight(pool): %s warmed in %.1fs (%d/%d)%s",
                        shape_key,
                        time.perf_counter() - t_shape,
                        idx,
                        len(shapes),
                        " [t2v+i2v]" if warm_i2v else "",
                    )
            logger.info(
                "preflight(pool): complete in %.1fs", time.perf_counter() - t_total
            )
            return

        if self.generator is None:
            raise RuntimeError(
                "preflight called before initialize_model; this is a bug."
            )

        # Lazy-import torch here: the legacy preflight path is the only
        # caller in this module, and we don't want backend.py imports
        # to drag in torch for callers that never run preflight.
        import torch

        shapes_path = os.environ.get("WARMUP_SHAPES_JSON_PATH")
        if not shapes_path:
            raise RuntimeError(
                "preflight: WARMUP_SHAPES_JSON_PATH is unset. The per-model "
                "entry point must set a default before calling preflight."
            )
        if not os.path.isfile(shapes_path):
            logger.warning(
                "preflight: %s not found; skipping. Customer first-request "
                "latency will be uneven across shapes. Fix the image build, "
                "or set LTX2_PREFLIGHT_SKIP=1 to acknowledge.",
                shapes_path,
            )
            return

        with open(shapes_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        shapes = cfg.get("shapes", [])
        fps = int(cfg.get("fps", 24))
        guidance_scale = float(cfg.get("guidance_scale", 1.0))

        logger.info("preflight: warming %d shapes from %s", len(shapes), shapes_path)
        t_total = time.perf_counter()

        with tempfile.TemporaryDirectory() as tmpdir:
            for idx, shape in enumerate(shapes, 1):
                # Reset dynamo state so each preflight compile/load matches
                # what create_video() will see at request time. See
                # claude_plans/2026-05-13-ltx2-cache-order-dependence.md.
                torch._dynamo.reset()
                torch.cuda.empty_cache()
                w = int(shape["width"])
                h = int(shape["height"])
                nf = int(shape["num_frames"])
                tag = "%dx%d@%df" % (w, h, nf)
                out_path = os.path.join(tmpdir, "preflight_%s.mp4" % tag)

                t_shape = time.perf_counter()
                try:
                    await asyncio.to_thread(
                        self.generator.generate_video,
                        prompt="warmup",
                        save_video=True,
                        return_frames=False,
                        output_path=out_path,
                        width=w,
                        height=h,
                        num_frames=nf,
                        fps=fps,
                        num_inference_steps=1,
                        guidance_scale=guidance_scale,
                        seed=42,
                    )
                except Exception as exc:
                    elapsed = time.perf_counter() - t_shape
                    logger.error(
                        "preflight: %s FAILED after %.1fs",
                        tag,
                        elapsed,
                        exc_info=True,
                    )
                    raise RuntimeError(
                        "preflight failed for shape %s; refusing to start. "
                        "Likely cause: baked compile cache doesn't cover this "
                        "shape, or a code/version mismatch between the warmup "
                        "that built the cache and this worker. See "
                        "ltx2/RUNBOOK.md." % tag
                    ) from exc

                elapsed = time.perf_counter() - t_shape
                logger.info(
                    "preflight: %s warmed in %.1fs (%d/%d)",
                    tag,
                    elapsed,
                    idx,
                    len(shapes),
                )

        total = time.perf_counter() - t_total
        logger.info("preflight: complete in %.1fs", total)

    # ── Dynamo endpoint ───────────────────────────────────────────────────────

    async def _generate_mp4_via_pool(
        self,
        *,
        request: VideoCreateRequest,
        video_id: str,
        width: int,
        height: int,
        num_frames: int,
        fps: int,
        ltx2_images: list[tuple[str, int, float]] | None = None,
        ltx2_image_crf: float = 0.0,
    ) -> bytes:
        """Route one request through the subprocess pool. Caller holds the global lock.

        For i2v, ``ltx2_images`` carries (temp_path, frame, strength); the temp
        file is created by ``create_video`` (via ``resolve_image_bytes``) and
        stays valid until ``pool.route`` returns (same-pod fs, so the subprocess
        can read it); ``create_video`` unlinks it in its ``finally``."""
        assert self.pool is not None
        nvext = request.nvext
        shape_key = f"{width}x{height}@{num_frames}f"
        # Don't pre-create the file — the generator's generate_video
        # writes the path itself. Use a deterministic name keyed on
        # video_id.
        output_path = os.path.join(tempfile.gettempdir(), f"video-{video_id}.mp4")
        if os.path.exists(output_path):
            os.unlink(output_path)

        pool_request: dict = {
            "request_id": video_id,
            "prompt": request.prompt,
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "fps": fps,
            "num_inference_steps": nvext.num_inference_steps,
            "guidance_scale": nvext.guidance_scale,
            "seed": nvext.seed,
            "negative_prompt": nvext.negative_prompt,
            "output_path": output_path,
        }
        if ltx2_images:
            pool_request["ltx2_images"] = ltx2_images
            pool_request["ltx2_image_crf"] = ltx2_image_crf
        try:
            result = await self.pool.route(shape_key, pool_request)
            status = result["status"]
            if status == "ERROR":
                raise RuntimeError(
                    f"pool subprocess reported ERROR: {result.get('error')}"
                )
            if status == "FATAL":
                raise RuntimeError(
                    f"pool subprocess reported FATAL: {result.get('error')}"
                )
            if status != "DONE":
                raise RuntimeError(f"pool subprocess unexpected status: {status}")
            if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
                raise RuntimeError(
                    f"pool subprocess reported DONE but {output_path} "
                    f"is missing or empty"
                )
            logger.info(
                "[%s] pool subprocess gen %dms",
                video_id,
                result.get("elapsed_ms", 0),
            )
            with open(output_path, "rb") as f:
                return f.read()
        finally:
            try:
                if os.path.exists(output_path):
                    os.unlink(output_path)
            except OSError as exc:
                logger.warning(
                    "[%s] failed to delete %s: %s",
                    video_id,
                    output_path,
                    exc,
                )

    @dynamo_endpoint(VideoCreateRequest, VideoCreateResponse)
    async def create_video(self, request: VideoCreateRequest):
        """
        Non-streaming endpoint.

        Generates one video clip using the parameters from the request's nvext
        field, then yields a single VideoCreateResponse with data[0].b64_json
        containing the complete MP4 file encoded in base64.
        """
        if not self.pool_mode:
            # Legacy in-process path: reset dynamo state at the start of
            # each request so compile-cache lookups match the fresh-state
            # keys produced by warmup. Without this, state accumulates
            # across customer requests and the second+ request misses the
            # cache. Pool path doesn't need this because each subprocess
            # already starts with fresh dynamo state per its lifetime.
            import torch

            torch._dynamo.reset()
            torch.cuda.empty_cache()
            if self.generator is None:
                raise RuntimeError("Generator is not initialized")
        else:
            if self.pool is None:
                raise RuntimeError("Pool is not initialized")

        nvext = request.nvext
        try:
            width_str, height_str = request.size.lower().split("x", 1)
            width, height = int(width_str), int(height_str)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Invalid size format '{request.size}', expected 'WxH'"
            ) from exc

        if width <= 0 or height <= 0:
            raise ValueError(
                f"Invalid size '{request.size}', width and height must be positive"
            )

        num_frames = (
            nvext.num_frames
            if nvext.num_frames is not None
            else nvext.fps * request.seconds
        )
        if num_frames <= 0:
            raise ValueError("num_frames must be positive")

        fps = nvext.fps
        if fps <= 0:
            raise ValueError("fps must be positive")

        video_id = f"video_{uuid.uuid4().hex}"
        created_ts = int(time.time())

        logger.info(
            "[%s] create_video: prompt='%s...' size=%s frames=%d steps=%d pool_mode=%s",
            video_id,
            request.prompt[:60],
            request.size,
            num_frames,
            nvext.num_inference_steps,
            self.pool_mode,
        )
        logger.info(
            "[%s] Waiting for generate lock (locked=%s)",
            video_id,
            self._generate_lock.locked(),
        )
        # i2v: resolve the conditioning image (if any) to a temp file BEFORE
        # taking the single-tenant GPU lock. The network fetch runs OFF the
        # event loop (asyncio.to_thread) so it never stalls the loop or holds
        # the lock. Empty list -> t2v. Temp file cleaned up in `finally`.
        ltx2_images: list[tuple[str, int, float]] = []
        i2v_temp_path: str | None = None
        # The Dynamo frontend forwards an i2v image ONLY via the top-level
        # `input_reference` field (its `nvext` is a typed struct without an image
        # field, so nvext.image_url never survives the frontend hop). Prefer
        # input_reference; fall back to nvext.image_url for direct-to-worker
        # calls that bypass the frontend.
        image_ref = request.input_reference or nvext.image_url
        if image_ref:
            from .i2v_input import resolve_image_bytes

            img_bytes = await asyncio.to_thread(resolve_image_bytes, image_ref)
            fd, i2v_temp_path = tempfile.mkstemp(
                suffix=".img", prefix=f"i2v-{video_id}-"
            )
            with os.fdopen(fd, "wb") as f:
                f.write(img_bytes)
            ltx2_images = [
                (
                    i2v_temp_path,
                    int(nvext.image_frame_index),
                    float(nvext.image_strength),
                )
            ]
        ltx2_image_crf = float(nvext.image_crf)

        try:
            async with self._generate_lock:
                t = time.perf_counter()
                logger.info(
                    "[%s] Generating video (%dx%d, %d frames, %d steps, i2v=%s) ...",
                    video_id,
                    width,
                    height,
                    num_frames,
                    nvext.num_inference_steps,
                    bool(ltx2_images),
                )
                try:
                    if self.pool_mode:
                        mp4_bytes = await self._generate_mp4_via_pool(
                            request=request,
                            video_id=video_id,
                            width=width,
                            height=height,
                            num_frames=num_frames,
                            fps=fps,
                            ltx2_images=ltx2_images,
                            ltx2_image_crf=ltx2_image_crf,
                        )
                    else:
                        mp4_bytes = await asyncio.to_thread(
                            self._generate_mp4,
                            prompt=request.prompt,
                            video_id=video_id,
                            width=width,
                            height=height,
                            num_frames=num_frames,
                            fps=fps,
                            num_inference_steps=nvext.num_inference_steps,
                            guidance_scale=nvext.guidance_scale,
                            seed=nvext.seed,
                            negative_prompt=nvext.negative_prompt,
                            ltx2_images=ltx2_images,
                            ltx2_image_crf=ltx2_image_crf,
                        )
                except Exception as exc:
                    logger.exception("[%s] Generation failed", video_id)
                    raise RuntimeError(
                        f"Video generation failed for request {video_id}"
                    ) from exc

                elapsed = time.perf_counter() - t
                logger.info(
                    "[%s] Generation done in %.1fs — encoding %.2f MB MP4",
                    video_id,
                    elapsed,
                    len(mp4_bytes) / 1_048_576,
                )

                yield VideoCreateResponse(
                    id=video_id,
                    created=created_ts,
                    model=request.model,
                    data=[VideoData(b64_json=base64.b64encode(mp4_bytes).decode())],
                ).model_dump()
            logger.info("[%s] Generation request finished", video_id)
        finally:
            if i2v_temp_path is not None:
                try:
                    os.unlink(i2v_temp_path)
                except OSError as exc:
                    logger.warning(
                        "[%s] failed to delete i2v temp %s: %s",
                        video_id,
                        i2v_temp_path,
                        exc,
                    )
