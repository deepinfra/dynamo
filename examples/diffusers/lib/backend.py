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
    ) -> bytes:
        """Generate a video clip and return it as MP4 bytes."""
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
            # Pool mode spawns subprocesses lazily on first request per
            # shape; there's no in-process generator to warm. Eager
            # spawn-at-boot for all menu shapes is a future optimization.
            logger.info("preflight skipped: pool mode is enabled (lazy spawn)")
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
    ) -> bytes:
        """Route one request through the subprocess pool. Caller holds the global lock."""
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
        async with self._generate_lock:
            t = time.perf_counter()
            logger.info(
                "[%s] Generating video (%dx%d, %d frames, %d steps) ...",
                video_id,
                width,
                height,
                num_frames,
                nvext.num_inference_steps,
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
