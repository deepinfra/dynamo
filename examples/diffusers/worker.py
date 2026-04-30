#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
FastVideo Worker for Dynamo (non-streaming)

Registers a VideoGenerator as a Dynamo backend endpoint compatible with the
/v1/videos frontend endpoint.  The endpoint generates a full video
clip from the request parameters and returns it as a single response containing
the complete MP4 file base64-encoded in data[0].b64_json.

Generation parameters (size, fps, num_frames, etc.) are taken from the
request body's nvext field, so the same worker instance can serve requests
with different resolutions and quality settings without restarting.

One request at a time (asyncio.Lock — VideoGenerator is not re-entrant).

Usage:
  python worker.py [--model MODEL] [--num-gpus N] [--enable-optimizations]
                   [--attention-backend ATTENTION_BACKEND]

Options:
  --model          HuggingFace model path
                   (default: FastVideo/LTX2-Distilled-Diffusers)
  --num-gpus       Number of GPUs (default: 1)
  --enable-optimizations
                   Enable FP4 quantization (if available) and torch.compile
  --attention-backend
                   Attention backend (default: TORCH_SDPA)

Request format (sent to /v1/videos):
  prompt:   text description of the desired video
  model:    HuggingFace model path (must match what the worker registered)
  size:     "WxH" string, e.g. "1920x1088" (default: "1920x1088")
  seconds:  clip duration when nvext.num_frames is not set (default: 5)
  nvext:
    fps:                frames per second (default: 24)
    num_frames:         total frames; overrides fps * seconds when set (default: 121)
    num_inference_steps diffusion steps (default: 5)
    guidance_scale:     CFG scale (default: 1.0)
    seed:               RNG seed (default: 10)
    negative_prompt:    text to avoid (optional)
"""

import argparse
import asyncio
import base64
import hashlib
import json
import logging
import os
import tempfile
import time
import uuid

import torch
import uvloop
from fastvideo import VideoGenerator
from fastvideo.configs.pipelines.base import PipelineConfig
from fastvideo.platforms.interface import AttentionBackendEnum
from pydantic import BaseModel, Field

from dynamo.llm import ModelInput, ModelType, register_llm  # type: ignore[attr-defined]
from dynamo.runtime import DistributedRuntime, dynamo_endpoint

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "FastVideo/LTX2-Distilled-Diffusers"
DEFAULT_ATTENTION_BACKEND = "TORCH_SDPA"
# FastVideo exposes NO_ATTENTION in the enum, but it is not a selectable
# inference backend for this worker's FASTVIDEO_ATTENTION_BACKEND override.
ATTENTION_BACKEND_CHOICES = tuple(
    backend_name
    for backend_name in AttentionBackendEnum.__members__
    if backend_name != "NO_ATTENTION"
)

# ── Request / Response models ─────────────────────────────────────────────────


def _get_worker_namespace() -> str:
    """
    Resolve Dynamo namespace for endpoint registration.

    Kubernetes operator injects DYN_NAMESPACE (and optionally a rollout suffix).
    Compose/local runs keep using the historical "dynamo" default.
    """
    namespace = os.environ.get("DYN_NAMESPACE", "dynamo")
    suffix = os.environ.get("DYN_NAMESPACE_WORKER_SUFFIX")
    if suffix:
        namespace = f"{namespace}-{suffix}"
    return namespace


class NvExtVideoCreateRequest(BaseModel):
    fps: int = Field(default=24, description="Frames per second")
    num_frames: int | None = Field(
        default=121, description="Total frames; overrides fps * seconds"
    )
    num_inference_steps: int = Field(default=5, description="Diffusion inference steps")
    guidance_scale: float = Field(
        default=1.0, description="Classifier-free guidance scale"
    )
    seed: int | None = Field(default=10, description="RNG seed for reproducibility")
    negative_prompt: str | None = Field(
        default=None, description="Text to avoid in generation"
    )


class VideoCreateRequest(BaseModel):
    prompt: str = Field(description="Text description of the desired video")
    model: str = Field(description="HuggingFace model path")
    size: str = Field(default="1920x1088", description="Frame dimensions as 'WxH'")
    seconds: int = Field(
        default=5, description="Clip duration; used when nvext.num_frames is unset"
    )
    user: str | None = Field(default=None)
    nvext: NvExtVideoCreateRequest = Field(default_factory=NvExtVideoCreateRequest)


class VideoData(BaseModel):
    b64_json: str | None = Field(default=None, description="Base64-encoded MP4 video")
    mime_type: str = Field(default="video/mp4")


class VideoCreateResponse(BaseModel):
    id: str
    object: str = "video"
    created: int
    model: str
    status: str = "complete"
    data: list[VideoData]


# ── Backend ───────────────────────────────────────────────────────────────────


def _coerce_optional_float(value: object) -> float | None:
    """Best-effort conversion for optional numeric metrics from backend results."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class FastVideoBackend:
    def __init__(self, args: argparse.Namespace) -> None:
        self.model_name: str = args.model
        self.served_model_name: str = args.served_model_name or args.model
        self.num_gpus: int = args.num_gpus
        self.enable_optimizations: bool = args.enable_optimizations
        self.attention_backend: str = args.attention_backend

        # One request at a time — VideoGenerator is not re-entrant
        self._generate_lock = asyncio.Lock()
        self.generator: VideoGenerator | None = None

        os.environ["FASTVIDEO_ATTENTION_BACKEND"] = self.attention_backend
        os.environ["FASTVIDEO_STAGE_LOGGING"] = "1"
        os.environ["FASTVIDEO_ENABLE_RMSNORM_FP4_PREQUANT"] = "0"

    async def initialize_model(self) -> None:
        logger.info("Loading VideoGenerator model=%s", self.model_name)
        loop = asyncio.get_running_loop()

        def _load():
            # Import here so the module is found whether worker.py runs
            # from /opt/app inside the container or from a local checkout.
            from ltx2_config import standard_kwargs, fp4_kwargs

            pipeline_config = PipelineConfig.from_pretrained(self.model_name)

            # Default path: standard kwargs (matches the warmup cache the
            # image was baked against). FP4 path is opt-in and requires a
            # separately-baked image.
            if not self.enable_optimizations:
                optimization_kwargs = standard_kwargs()
            else:
                major, minor = torch.cuda.get_device_capability()
                if major < 10:
                    logger.warning(
                        "FP4 quantization is only supported on NVIDIA Blackwell GPUs "
                        "(compute capability 10.0+). Detected compute capability: %d.%d. "
                        "Continuing without FP4 optimizations.",
                        major, minor,
                    )
                    optimization_kwargs = standard_kwargs()
                else:
                    logger.info(
                        "Using FP4 quantization for VideoGenerator model=%s",
                        self.model_name,
                    )
                    try:
                        from fastvideo.layers.quantization.fp4_config import FP4Config
                    except ImportError as exc:
                        raise RuntimeError(
                            "FastVideo optimizations require "
                            "fastvideo.layers.quantization.fp4_config, but this "
                            "FastVideo build does not provide it. Re-run "
                            "worker.py without --enable-optimizations or install a "
                            "FastVideo version that includes fp4_config."
                        ) from exc
                    pipeline_config.dit_config.quant_config = FP4Config()
                    optimization_kwargs = fp4_kwargs()

            return VideoGenerator.from_pretrained(
                self.model_name,
                num_gpus=self.num_gpus,
                pipeline_config=pipeline_config,
                **optimization_kwargs,
            )

        self.generator = await loop.run_in_executor(None, _load)
        logger.info("VideoGenerator ready")

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

        Opt-in via LTX2_PREFLIGHT=1 (callers gate the call, not this method)
        because it costs ~25 min wall time for the soft-launch 10-shape menu
        even with a baked cache -- the per-shape compile / kernel-load /
        autotune-confirm work is mostly insensitive to num_inference_steps,
        so we can't make this fast just by making the inference cheap.

        Reads warmup_shapes.json (must be at /opt/app/warmup_shapes.json,
        configurable via WARMUP_SHAPES_JSON_PATH). For each shape, runs one
        generation with num_inference_steps=1 and a short dummy prompt.

        Failures abort pod boot. If the cache shipped in the image doesn't
        cover a menu shape (or there is a code/version mismatch between the
        warmup that built the cache and this worker), pod crash-loops --
        better than serving customer traffic with ~15-minute first-shape
        recompiles.
        """
        if self.generator is None:
            raise RuntimeError(
                "preflight called before initialize_model; this is a bug."
            )

        shapes_path = os.environ.get(
            "WARMUP_SHAPES_JSON_PATH", "/opt/app/warmup_shapes.json"
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
                        tag, elapsed, exc_info=True,
                    )
                    raise RuntimeError(
                        "preflight failed for shape %s; refusing to start. "
                        "Likely cause: baked compile cache doesn't cover this "
                        "shape, or a code/version mismatch between the warmup "
                        "that built the cache and this worker. See "
                        "dynamo/examples/diffusers/RUNBOOK.md."
                        % tag
                    ) from exc

                elapsed = time.perf_counter() - t_shape
                logger.info(
                    "preflight: %s warmed in %.1fs (%d/%d)",
                    tag, elapsed, idx, len(shapes),
                )

        total = time.perf_counter() - t_total
        logger.info("preflight: complete in %.1fs", total)

    # ── Dynamo endpoint ───────────────────────────────────────────────────────

    @dynamo_endpoint(VideoCreateRequest, VideoCreateResponse)
    async def create_video(self, request: VideoCreateRequest):
        """
        Non-streaming endpoint.

        Generates one video clip using the parameters from the request's nvext
        field, then yields a single VideoCreateResponse with data[0].b64_json
        containing the complete MP4 file encoded in base64.
        """
        # Reset dynamo state at the start of each request so this request's
        # compile-cache lookups match the fresh-state keys produced by
        # warmup. Without this, state accumulates across customer requests
        # and the second+ request misses the cache.
        torch._dynamo.reset()
        torch.cuda.empty_cache()
        if self.generator is None:
            raise RuntimeError("Generator is not initialized")

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
            "[%s] create_video: prompt='%s...' size=%s frames=%d steps=%d",
            video_id,
            request.prompt[:60],
            request.size,
            num_frames,
            nvext.num_inference_steps,
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


# ── Dynamo wiring ─────────────────────────────────────────────────────────────


async def _register_model(endpoint, served_name: str, model_path: str) -> None:
    try:
        await register_llm(
            ModelInput.Text,  # type: ignore[attr-defined]
            ModelType.Videos,
            endpoint,
            model_path,
            served_name,
        )
        logger.info("Successfully registered model: %s (path=%s)", served_name, model_path)
    except Exception as e:
        logger.error("Failed to register model: %s", e, exc_info=True)
        raise RuntimeError("Model registration failed") from e


async def backend_worker(runtime: DistributedRuntime, args: argparse.Namespace) -> None:
    namespace_name = _get_worker_namespace()
    component_name = "backend"
    endpoint_name = "generate"

    endpoint = runtime.endpoint(f"{namespace_name}.{component_name}.{endpoint_name}")
    logger.info(
        "Serving endpoint %s/%s/%s", namespace_name, component_name, endpoint_name
    )

    backend = FastVideoBackend(args)
    await backend.initialize_model()
    # Preflight is opt-in (LTX2_PREFLIGHT=1) because, even with a baked
    # cache, it pays full first-call cost per shape -- ~25 minutes for our
    # 10-shape menu. For low-volume soft-launch traffic that's worse than
    # accepting that the first customer per shape per pod pays the cold
    # cost. Re-enable when traffic patterns make the trade-off favor
    # uniform latency at the cost of long pod boots.
    if os.environ.get("LTX2_PREFLIGHT") == "1":
        await backend.preflight()

    await asyncio.gather(
        endpoint.serve_endpoint(backend.create_video),  # type: ignore[arg-type]
        _register_model(endpoint, backend.served_model_name, backend.model_name),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FastVideo Worker for Dynamo (non-streaming)"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"HuggingFace model path (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--served-model-name",
        default=None,
        help="Name advertised to the Dynamo discovery layer (default: same as --model)",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        dest="num_gpus",
        help="Number of GPUs (default: 1)",
    )
    parser.add_argument(
        "--enable-optimizations",
        action="store_true",
        dest="enable_optimizations",
        help="Enable FP4 quantization (if available) and torch.compile",
    )
    parser.add_argument(
        "--attention-backend",
        choices=ATTENTION_BACKEND_CHOICES,
        default=DEFAULT_ATTENTION_BACKEND,
        dest="attention_backend",
        help=(
            "Attention backend to set via FASTVIDEO_ATTENTION_BACKEND "
            f"(choices: {', '.join(ATTENTION_BACKEND_CHOICES)}; "
            f"default: {DEFAULT_ATTENTION_BACKEND})"
        ),
    )
    return parser.parse_args()


async def main(args: argparse.Namespace) -> None:
    loop = asyncio.get_running_loop()
    # Use Kubernetes discovery in-cluster and file discovery for local compose by default.
    discovery_backend = os.environ.get("DYN_DISCOVERY_BACKEND")
    if not discovery_backend:
        discovery_backend = (
            "kubernetes" if os.environ.get("KUBERNETES_SERVICE_HOST") else "file"
        )
    logger.info("Using discovery backend: %s", discovery_backend)
    logger.info("Resolved worker namespace: %s", _get_worker_namespace())
    # Pass enable_nats=False explicitly: the bundled ai-dynamo-runtime 1.0.0
    # is from before upstream commit af0ff07 ("remove enable_nats usage")
    # and so its DistributedRuntime ctor still requires the 4th positional
    # arg to gate the NATS client. Omitting it defaults to True, which
    # makes the worker hard-fail at startup with "Failed to connect to
    # NATS: Connection refused" because the cluster runs ZMQ-only
    # (DYN_EVENT_PLANE=zmq, deepinfra has no NATS).
    runtime = DistributedRuntime(loop, discovery_backend, "tcp", False)
    await backend_worker(runtime, args)


def _compute_menu_hash(shapes_json_path: str) -> tuple[str, int]:
    """
    Hash the canonical shape-tuple list from a warmup_shapes.json. Must
    match the algorithm in:
      - dynamo/examples/diffusers/run-benchmark.sh / docker bake step
      - backend/tests/test_ltx_shape_menu.py
    Anywhere this hash is referenced, it is the first 8 hex chars of
    sha256(json.dumps(sorted(shape_tuples), separators=(',',':'))).

    Returns (hash, shape_count) so callers can log both without re-reading.
    """
    with open(shapes_json_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    shapes = sorted(
        (int(s["width"]), int(s["height"]), int(s["num_frames"]))
        for s in cfg["shapes"]
    )
    canonical = json.dumps(shapes, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:8], len(shapes)


def _assert_shape_menu_hash_matches() -> None:
    """
    Refuse to start if the image's baked compile cache doesn't match the
    shape menu the worker is about to serve. The image's Dockerfile sets
    IMAGE_SHAPE_HASH at bake time; we recompute the same hash here from
    warmup_shapes.json and abort on mismatch.

    If IMAGE_SHAPE_HASH is unset (dev runs, or images built before this
    invariant landed), we log a warning and continue. Production images
    are expected to have it set.

    The procedure for resolving a mismatch is documented in:
      dynamo/examples/diffusers/RUNBOOK.md
    """
    expected = os.environ.get("IMAGE_SHAPE_HASH")
    if not expected:
        logger.warning(
            "IMAGE_SHAPE_HASH is unset; skipping image/menu hash check. "
            "This is expected for dev runs and for images built before "
            "the bake step started writing IMAGE_SHAPE_HASH. Production "
            "images must set it -- see the RUNBOOK."
        )
        return

    shapes_json = os.environ.get(
        "WARMUP_SHAPES_JSON_PATH", "/opt/app/warmup_shapes.json"
    )
    if not os.path.isfile(shapes_json):
        raise RuntimeError(
            f"warmup_shapes.json not found at {shapes_json}. "
            f"Cannot verify the image/menu hash. Set "
            f"WARMUP_SHAPES_JSON_PATH to the correct location, or "
            f"check your image build."
        )

    actual, shape_count = _compute_menu_hash(shapes_json)
    if actual != expected:
        raise RuntimeError(
            f"FATAL: image/menu shape-hash mismatch -- refusing to start.\n"
            f"\n"
            f"  Image baked with hash: {expected}  (env IMAGE_SHAPE_HASH, set at "
            f"bake time)\n"
            f"  Current menu hash:     {actual}  (recomputed from {shapes_json})\n"
            f"\n"
            f"This means the compile cache shipped in the image was produced "
            f"for a different shape menu than the one this worker is about to "
            f"serve. Customer requests for any shape in the gap would hit a "
            f"15+ minute torch.compile, violating our latency SLO.\n"
            f"\n"
            f"Either:\n"
            f"  A) Roll back the model config to the last image whose tag ends "
            f"in -{actual}, OR\n"
            f"  B) Build a new image for the current menu (~3 hours on B200).\n"
            f"\n"
            f"Procedures: dynamo/examples/diffusers/RUNBOOK.md"
        )

    logger.info(
        "image/menu shape-hash check OK (%s); %d shapes in menu",
        actual,
        shape_count,
    )


if __name__ == "__main__":
    _args = _parse_args()
    logging.basicConfig(
        level=(
            logging.DEBUG
            if os.environ.get("FASTVIDEO_LOG_LEVEL") == "DEBUG"
            else logging.INFO
        ),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )
    # Refuse to start if image cache and shape menu disagree. Runs before
    # any model load or networking so we fail fast with a clear message.
    _assert_shape_menu_hash_matches()
    uvloop.install()
    asyncio.run(main(_args))
