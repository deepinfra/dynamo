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
import collections
import hashlib
import json
import logging
import os
import signal
import sys
import tempfile
import time
import uuid
from contextlib import suppress
from multiprocessing import Pipe
from multiprocessing.connection import Connection

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

# ── Pool-mode configuration ──────────────────────────────────────────────────
# Pool mode is OPT-IN for Phase 2 soft launch. LTX2_POOL_MODE=1 routes every
# customer request through a per-shape persistent subprocess; default (unset
# or any other value) keeps the legacy in-process generator path.
# See ~/backend/claude_plans/2026-05-14-ltx2-phase2-subprocess-pool.md.
LTX2_POOL_MODE = os.environ.get("LTX2_POOL_MODE", "0") == "1"
# Max concurrent shape-pinned subprocesses. K=2 fits on B200 (178 GiB):
# each pool subprocess internally spawns a FastVideo multiproc_executor
# Worker child that holds the model -- the outer pool subprocess is
# ~600 MiB but the FastVideo Worker is ~70 GiB on LTX-2. Observed on
# di-slc-39 2026-05-14: at 2 hot shapes the GPU is ~80% utilized;
# adding a 3rd hot shape OOMs deterministically during model load.
# If more shapes need to be hot than fit, k8s scale-out is the
# answer, not more subprocesses per pod.
LTX2_POOL_MAX_SIZE = int(os.environ.get("LTX2_POOL_MAX_SIZE", "2"))
# Subprocess must print READY within this window after spawn (covers
# Python startup + torch import + model load from /data/default +
# cache hydrate).
LTX2_POOL_SPAWN_TIMEOUT_S = 180
# Per-request generation timeout; longest production shape is well
# under this.
LTX2_POOL_GEN_TIMEOUT_S = 1800
# SIGTERM grace before SIGKILL during eviction / shutdown.
LTX2_POOL_SIGTERM_GRACE_S = 30
# How many lines of subprocess stderr to retain for crash diagnostics.
LTX2_POOL_STDERR_RING_LINES = 50
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


# ── Shared model-load helper ─────────────────────────────────────────────────


def _load_generator(
    model_name: str,
    num_gpus: int,
    enable_optimizations: bool,
) -> VideoGenerator:
    """
    Pure factory: build a VideoGenerator for `model_name`.

    Shared by both the legacy in-process FastVideoBackend.initialize_model
    path and the pool-mode _pool_worker_main entrypoint, so the two paths
    produce byte-identical compile-cache keys (same kwargs → same key).
    """
    from ltx2_config import standard_kwargs, fp4_kwargs

    pipeline_config = PipelineConfig.from_pretrained(model_name)

    if not enable_optimizations:
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
                model_name,
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
        model_name,
        num_gpus=num_gpus,
        pipeline_config=pipeline_config,
        **optimization_kwargs,
    )


# ── Subprocess pool (Phase 2) ────────────────────────────────────────────────


class _SubprocessHandle:
    """
    State for one shape-pinned subprocess in the pool.

    Lifecycle: parent spawns via SubprocessPool._spawn (which creates a
    duplex multiprocessing.Pipe, hands the child end to the subprocess
    via pass_fds, and reads the first message on the parent end to
    confirm READY). Subsequent requests flow through the Connection
    as pickled dicts -- stdout/stderr are diagnostic-only and drained
    into a ring buffer so a crash leaves us with the last ~50 lines
    of context.

    diag_buf captures both stdout and stderr lines (library noise from
    FastVideo / vllm / torch.compile / multiproc workers). With the
    Connection-based protocol, library writes to stdout cannot corrupt
    the wire protocol, so we treat both pipes as pure diagnostics.
    """

    def __init__(
        self,
        shape_key: str,
        proc: asyncio.subprocess.Process,
        conn: Connection,
    ) -> None:
        self.shape_key = shape_key
        self.proc = proc
        self.conn = conn
        self.last_used = time.monotonic()
        self.diag_buf: collections.deque[str] = collections.deque(
            maxlen=LTX2_POOL_STDERR_RING_LINES
        )
        self.stdout_drainer: asyncio.Task | None = None
        self.stderr_drainer: asyncio.Task | None = None

    def diag_tail(self) -> str:
        return "\n".join(self.diag_buf)


class SubprocessPool:
    """
    LRU pool of shape-pinned subprocesses.

    The parent's FastVideoBackend._generate_lock already serializes all
    customer requests at the pod level (the GPU is single-tenant). So we
    don't need a per-handle lock — at most one request flows through
    pool.route() at a time. The pool's internal _pool_lock just protects
    the OrderedDict during spawn/evict.
    """

    def __init__(
        self,
        model_path: str,
        num_gpus: int,
        enable_optimizations: bool,
        attention_backend: str,
    ) -> None:
        self.model_path = model_path
        self.num_gpus = num_gpus
        self.enable_optimizations = enable_optimizations
        self.attention_backend = attention_backend
        self._handles: collections.OrderedDict[str, _SubprocessHandle] = (
            collections.OrderedDict()
        )
        self._pool_lock = asyncio.Lock()

    async def route(self, shape_key: str, request: dict) -> dict:
        """
        Run one request through the subprocess for `shape_key`, spawning
        one (and evicting LRU if needed) if absent. Caller is expected to
        hold the global generate lock.

        Returns a dict with at minimum 'status' in {'DONE', 'ERROR', 'FATAL'}
        plus 'request_id'. Raises RuntimeError if the subprocess dies,
        times out, or returns a desynced response.
        """
        handle = await self._get_or_spawn(shape_key)
        self._handles.move_to_end(shape_key)
        handle.last_used = time.monotonic()

        request_id = request["request_id"]
        payload = {"kind": "REQUEST", **request}

        try:
            await asyncio.to_thread(handle.conn.send, payload)
        except (BrokenPipeError, ConnectionResetError, OSError, EOFError) as exc:
            tail = handle.diag_tail()
            await self._discard(shape_key, handle)
            raise RuntimeError(
                f"subprocess for {shape_key} died before request "
                f"could be sent: {exc}. diag tail:\n{tail}"
            ) from exc

        try:
            msg = await self._recv_msg(handle, LTX2_POOL_GEN_TIMEOUT_S)
        except asyncio.TimeoutError:
            tail = handle.diag_tail()
            await self._discard(shape_key, handle)
            raise RuntimeError(
                f"subprocess for {shape_key} did not respond within "
                f"{LTX2_POOL_GEN_TIMEOUT_S}s. diag tail:\n{tail}"
            )
        except (EOFError, OSError, ConnectionError) as exc:
            tail = handle.diag_tail()
            await self._discard(shape_key, handle)
            raise RuntimeError(
                f"subprocess for {shape_key} EOF mid-request: {exc}. "
                f"diag tail:\n{tail}"
            ) from exc

        if not isinstance(msg, dict):
            tail = handle.diag_tail()
            await self._discard(shape_key, handle)
            raise RuntimeError(
                f"subprocess for {shape_key} returned non-dict message "
                f"{msg!r}. diag tail:\n{tail}"
            )

        resp_rid = msg.get("request_id")
        if resp_rid != request_id:
            tail = handle.diag_tail()
            await self._discard(shape_key, handle)
            raise RuntimeError(
                f"subprocess for {shape_key} returned request_id={resp_rid!r} "
                f"but expected {request_id!r} (desync). Killed. "
                f"diag tail:\n{tail}"
            )

        kind = msg.get("kind")
        if kind == "DONE":
            return {
                "status": "DONE",
                "request_id": request_id,
                "elapsed_ms": int(msg.get("elapsed_ms", 0)),
            }
        if kind == "ERROR":
            err = (
                f"{msg.get('exception_type', '?')} "
                f"{msg.get('exception_repr', '?')}"
            )
            return {"status": "ERROR", "request_id": request_id, "error": err}
        if kind == "FATAL":
            tail = handle.diag_tail()
            self._handles.pop(shape_key, None)
            # Subprocess is exiting on its own (sys.exit(2) after sending
            # FATAL). _kill waits up to SIGTERM grace, then SIGKILLs if
            # still alive, then cancels drainers and closes the conn.
            await self._kill(handle)
            err = (
                f"{msg.get('reason', '?')}: "
                f"{msg.get('exception_repr', '?')}"
            )
            logger.error(
                "[pool/%s] FATAL %s. diag tail:\n%s", shape_key, err, tail,
            )
            return {"status": "FATAL", "request_id": request_id, "error": err}

        tail = handle.diag_tail()
        await self._discard(shape_key, handle)
        raise RuntimeError(
            f"subprocess for {shape_key} returned unexpected message "
            f"kind={kind!r}: {msg!r}. diag tail:\n{tail}"
        )

    async def _recv_msg(
        self,
        handle: _SubprocessHandle,
        timeout: float,
    ) -> dict:
        """
        Receive one protocol message from the subprocess. Returns the
        pickled dict on the wire. Raises asyncio.TimeoutError on timeout
        (poll did not see data in `timeout` seconds), EOFError if the
        subprocess closed its end, or OSError if the pipe is broken.

        Runs the blocking poll/recv in a worker thread so the asyncio
        event loop stays responsive during long generation calls.
        """
        def _do() -> dict:
            if not handle.conn.poll(timeout):
                raise asyncio.TimeoutError()
            return handle.conn.recv()
        return await asyncio.to_thread(_do)

    async def _get_or_spawn(self, shape_key: str) -> _SubprocessHandle:
        async with self._pool_lock:
            handle = self._handles.get(shape_key)
            if handle is not None and handle.proc.returncode is None:
                return handle
            if handle is not None:
                # In dict but dead — clean up and respawn.
                self._handles.pop(shape_key, None)
                await self._kill(handle)
            while len(self._handles) >= LTX2_POOL_MAX_SIZE:
                await self._evict_lru()
            handle = await self._spawn(shape_key)
            self._handles[shape_key] = handle
            return handle

    async def _spawn(self, shape_key: str) -> _SubprocessHandle:
        inductor_dir = f"/cache/per-shape/{shape_key}/torchinductor"
        triton_dir = f"/cache/per-shape/{shape_key}/triton"
        env = os.environ.copy()
        env["TORCHINDUCTOR_CACHE_DIR"] = inductor_dir
        env["TRITON_CACHE_DIR"] = triton_dir

        # Duplex multiprocessing.Pipe for the wire protocol. The child
        # inherits child_conn's fd via pass_fds and reconstructs a
        # Connection in _pool_worker_main. No library code can write to
        # this fd because no library has a handle to it -- protocol
        # traffic is structurally isolated from stdout/stderr noise.
        parent_conn, child_conn = Pipe(duplex=True)

        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--pool-worker",
            "--shape-key", shape_key,
            "--model", self.model_path,
            "--num-gpus", str(self.num_gpus),
            "--attention-backend", self.attention_backend,
            "--protocol-fd", str(child_conn.fileno()),
        ]
        if self.enable_optimizations:
            cmd.append("--enable-optimizations")

        logger.info(
            "[pool/%s] spawning subprocess (cache=%s)",
            shape_key, inductor_dir,
        )
        try:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env,
                    pass_fds=(child_conn.fileno(),),
                )
            finally:
                # Close parent's view of the child end. If we don't,
                # parent holds the write side open, and parent_conn.recv()
                # never observes EOF when the subprocess dies -- it would
                # block until the generation timeout instead of failing
                # fast.
                child_conn.close()
        except BaseException:
            # Popen failed (or was cancelled). Close the parent end too
            # instead of relying on Connection.__del__ — GC-driven fd
            # cleanup in an error path is the kind of thing that bites
            # under stress.
            parent_conn.close()
            raise

        handle = _SubprocessHandle(shape_key, proc, parent_conn)
        try:
            handle.stdout_drainer = asyncio.create_task(
                self._drain_pipe(handle, proc.stdout, "stdout")
            )
            handle.stderr_drainer = asyncio.create_task(
                self._drain_pipe(handle, proc.stderr, "stderr")
            )

            t_spawn = time.monotonic()
            try:
                msg = await self._recv_msg(handle, LTX2_POOL_SPAWN_TIMEOUT_S)
            except asyncio.TimeoutError:
                tail = handle.diag_tail()
                raise RuntimeError(
                    f"subprocess for {shape_key} failed to reach READY "
                    f"within {LTX2_POOL_SPAWN_TIMEOUT_S}s. diag tail:\n{tail}"
                )
            except (EOFError, OSError, ConnectionError) as exc:
                tail = handle.diag_tail()
                raise RuntimeError(
                    f"subprocess for {shape_key} died before READY: {exc}. "
                    f"diag tail:\n{tail}"
                ) from exc

            if not isinstance(msg, dict) or msg.get("kind") != "READY":
                tail = handle.diag_tail()
                raise RuntimeError(
                    f"subprocess for {shape_key} sent unexpected first "
                    f"message {msg!r} (expected kind=READY). "
                    f"diag tail:\n{tail}"
                )

            logger.info(
                "[pool/%s] READY in %.1fs",
                shape_key, time.monotonic() - t_spawn,
            )
            return handle
        except BaseException:
            # Any failure after Popen — including unanticipated exceptions
            # from _recv_msg, drainer task creation, or cancellation — must
            # tear down the subprocess, drainers, and parent_conn before
            # propagating. _kill is idempotent and safe on partial state
            # (drainer fields may be None if create_task itself raised).
            await self._kill(handle)
            raise

    async def _drain_pipe(
        self,
        handle: _SubprocessHandle,
        pipe: asyncio.StreamReader,
        label: str,
    ) -> None:
        """
        Drain `pipe` (subprocess stdout or stderr) into the per-handle
        diagnostic ring buffer and forward each line to our logger.
        With the Connection-based protocol, both stdout and stderr are
        pure diagnostic channels; library noise on either is harmless
        and useful for post-mortem if a subprocess dies.

        Runs for the lifetime of the subprocess. Cancelled by _kill on
        eviction/shutdown.
        """
        try:
            while True:
                line = await pipe.readline()
                if not line:
                    return
                text = line.decode(errors="replace").rstrip("\n")
                handle.diag_buf.append(text)
                logger.info(
                    "[pool/%s/%s] %s", handle.shape_key, label, text,
                )
        except asyncio.CancelledError:
            return
        except Exception:
            logger.exception(
                "[pool/%s] %s drainer crashed", handle.shape_key, label,
            )

    async def _evict_lru(self) -> None:
        # Caller holds _pool_lock.
        if not self._handles:
            return
        shape_key, handle = next(iter(self._handles.items()))
        logger.info("[pool/%s] evicting LRU (idle %.0fs)",
                    shape_key, time.monotonic() - handle.last_used)
        self._handles.pop(shape_key, None)
        await self._kill(handle)

    async def _kill(self, handle: _SubprocessHandle) -> None:
        # Close the parent end of the protocol pipe FIRST. If the
        # subprocess is idle (blocked in conn.recv()), it'll see
        # EOFError and exit voluntarily — no SIGTERM needed. If it's
        # busy in generate_video(), the close is observed the next
        # time it touches conn; SIGTERM below still forces a kill.
        # Safe on an already-closed Connection (e.g. after FATAL
        # path where the child closed its end first).
        with suppress(Exception):
            await asyncio.to_thread(handle.conn.close)

        proc = handle.proc
        if proc.returncode is None:
            with suppress(ProcessLookupError):
                proc.terminate()
            try:
                await asyncio.wait_for(
                    proc.wait(), timeout=LTX2_POOL_SIGTERM_GRACE_S,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "[pool/%s] SIGTERM grace expired; SIGKILL",
                    handle.shape_key,
                )
                with suppress(ProcessLookupError):
                    proc.kill()
                with suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(proc.wait(), timeout=5)
        for drainer in (handle.stdout_drainer, handle.stderr_drainer):
            if drainer is not None:
                drainer.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await drainer

    async def _discard(self, shape_key: str, handle: _SubprocessHandle) -> None:
        """Remove from pool and kill. Used for unrecoverable subprocess errors."""
        self._handles.pop(shape_key, None)
        await self._kill(handle)

    async def shutdown(self) -> None:
        """Kill all subprocesses. Called on pod shutdown."""
        async with self._pool_lock:
            handles = list(self._handles.values())
            self._handles.clear()
        for handle in handles:
            await self._kill(handle)


class FastVideoBackend:
    def __init__(self, args: argparse.Namespace) -> None:
        self.model_name: str = args.model
        self.served_model_name: str = args.served_model_name or args.model
        self.num_gpus: int = args.num_gpus
        self.enable_optimizations: bool = args.enable_optimizations
        self.attention_backend: str = args.attention_backend

        # One request at a time — both the legacy in-process VideoGenerator
        # and the pool path serialize at the pod level via this lock (GPU
        # is single-tenant).
        self._generate_lock = asyncio.Lock()
        self.generator: VideoGenerator | None = None

        # Phase 2: opt-in subprocess pool. When enabled, customer requests
        # are routed to per-shape persistent subprocesses instead of the
        # in-process generator. See module-level constants.
        self.pool_mode: bool = LTX2_POOL_MODE
        self.pool: SubprocessPool | None = None

        os.environ["FASTVIDEO_ATTENTION_BACKEND"] = self.attention_backend
        os.environ["FASTVIDEO_STAGE_LOGGING"] = "1"
        os.environ["FASTVIDEO_ENABLE_RMSNORM_FP4_PREQUANT"] = "0"

    async def initialize_model(self) -> None:
        if self.pool_mode:
            logger.info(
                "pool mode enabled: creating SubprocessPool "
                "(max=%d, model=%s); no in-process generator load",
                LTX2_POOL_MAX_SIZE, self.model_name,
            )
            self.pool = SubprocessPool(
                model_path=self.model_name,
                num_gpus=self.num_gpus,
                enable_optimizations=self.enable_optimizations,
                attention_backend=self.attention_backend,
            )
            return

        logger.info("Loading VideoGenerator model=%s", self.model_name)
        loop = asyncio.get_running_loop()
        self.generator = await loop.run_in_executor(
            None,
            _load_generator,
            self.model_name,
            self.num_gpus,
            self.enable_optimizations,
        )
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
        if self.pool_mode:
            # Pool mode spawns subprocesses lazily on first request per
            # shape; there's no in-process generator to warm. Eager
            # spawn-at-boot for all menu shapes is a future optimization
            # (gated on a separate LTX2_POOL_EAGER_SPAWN flag).
            logger.info("preflight skipped: pool mode is enabled (lazy spawn)")
            return

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
        # Don't pre-create the file — fastvideo's generate_video writes
        # the path itself. Use a deterministic name keyed on video_id.
        output_path = os.path.join(
            tempfile.gettempdir(), f"ltx2-{video_id}.mp4"
        )
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
                video_id, result.get("elapsed_ms", 0),
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
                    video_id, output_path, exc,
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

    try:
        await asyncio.gather(
            endpoint.serve_endpoint(backend.create_video),  # type: ignore[arg-type]
            _register_model(endpoint, backend.served_model_name, backend.model_name),
        )
    finally:
        if backend.pool is not None:
            logger.info("shutting down subprocess pool")
            await backend.pool.shutdown()


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


# ── Pool-worker subprocess entrypoint (Phase 2) ──────────────────────────────


def _pool_worker_main(
    shape_key: str,
    model_path: str,
    num_gpus: int,
    enable_optimizations: bool,
    attention_backend: str,
    protocol_fd: int,
) -> int:
    """
    Subprocess entry. Pinned to one shape for the lifetime of the process.

    Wire protocol: pickled-dict messages over a multiprocessing.Connection.
    The parent created a duplex Pipe and passed the child end's fd via
    pass_fds; we reconstruct a Connection here. stdout/stderr are
    diagnostic-only -- library noise on either pipe is drained by the
    parent into a ring buffer but cannot corrupt the wire protocol
    because no library has a handle to `conn`'s fd.

    Messages (each is a dict with a 'kind' discriminator):

      child → parent
        {"kind": "READY",  "request_id": "_",  "shape_key": ...}
        {"kind": "DONE",   "request_id": <id>, "elapsed_ms": <int>}
        {"kind": "ERROR",  "request_id": <id>, "exception_type": ..., "exception_repr": ...}
        {"kind": "FATAL",  "request_id": <id>, "reason": "CUDA_FAULT", "exception_repr": ...}

      parent → child
        {"kind": "REQUEST", "request_id": <id>, ...generation parameters...}

    Per-shape cache dirs (TORCHINDUCTOR_CACHE_DIR, TRITON_CACHE_DIR) are
    set by the parent in the child's env before exec — torch reads them
    on first inductor use, so they must be in env at import time.

    Trust boundary: parent↔child messages are pickled. Both ends are the
    same codebase running on the same host as the same user; this is an
    intra-pod, intra-trust-boundary channel. Any future cross-tenant use
    of this pattern would need to switch to JSON-over-Connection.
    """
    # Defensive: parent should have already set these; setdefault keeps
    # things sane if someone runs this entrypoint by hand.
    os.environ.setdefault(
        "TORCHINDUCTOR_CACHE_DIR",
        f"/cache/per-shape/{shape_key}/torchinductor",
    )
    os.environ.setdefault(
        "TRITON_CACHE_DIR",
        f"/cache/per-shape/{shape_key}/triton",
    )
    os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", attention_backend)
    os.environ.setdefault("FASTVIDEO_STAGE_LOGGING", "1")
    os.environ.setdefault("FASTVIDEO_ENABLE_RMSNORM_FP4_PREQUANT", "0")

    conn = Connection(protocol_fd)

    def _on_sigterm(_signum, _frame):
        # Clean exit on SIGTERM (parent's eviction path); a blocked
        # conn.recv() would otherwise sit on the syscall forever.
        print(f"[pool-worker/{shape_key}] SIGTERM received, exiting",
              file=sys.stderr, flush=True)
        with suppress(Exception):
            conn.close()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _on_sigterm)

    print(
        f"[pool-worker/{shape_key}] loading model={model_path} "
        f"num_gpus={num_gpus} enable_optimizations={enable_optimizations}",
        file=sys.stderr, flush=True,
    )
    t_load = time.perf_counter()
    generator = _load_generator(model_path, num_gpus, enable_optimizations)
    load_s = time.perf_counter() - t_load
    print(
        f"[pool-worker/{shape_key}] generator ready in {load_s:.1f}s",
        file=sys.stderr, flush=True,
    )

    try:
        conn.send({"kind": "READY", "request_id": "_", "shape_key": shape_key})

        while True:
            try:
                req = conn.recv()
            except EOFError:
                # Parent closed the protocol channel; exit cleanly.
                return 0
            if not isinstance(req, dict) or req.get("kind") != "REQUEST":
                # Malformed message from parent (shouldn't happen with our
                # well-typed sender, but report and keep going).
                conn.send({
                    "kind": "ERROR",
                    "request_id": req.get("request_id", "_") if isinstance(req, dict) else "_",
                    "exception_type": "ProtocolError",
                    "exception_repr": f"unexpected message {req!r}",
                })
                continue
            request_id = req.get("request_id", "_")
            try:
                kwargs: dict = dict(
                    prompt=req["prompt"],
                    save_video=True,
                    return_frames=False,
                    output_path=req["output_path"],
                    width=req["width"],
                    height=req["height"],
                    num_frames=req["num_frames"],
                    fps=req["fps"],
                    num_inference_steps=req["num_inference_steps"],
                    guidance_scale=req["guidance_scale"],
                )
                if req.get("seed") is not None:
                    kwargs["seed"] = req["seed"]
                if req.get("negative_prompt") is not None:
                    kwargs["negative_prompt"] = req["negative_prompt"]

                t0 = time.perf_counter()
                generator.generate_video(**kwargs)
                elapsed_ms = int((time.perf_counter() - t0) * 1000)
                conn.send({
                    "kind": "DONE",
                    "request_id": request_id,
                    "elapsed_ms": elapsed_ms,
                })
            except Exception as exc:  # noqa: BLE001
                msg = str(exc)
                # CUDA-context-corrupting failures (OOM and other CUDA errors)
                # leave the context unrecoverable. Exit so the parent respawns
                # fresh; recovering in-place is worse than paying one cold
                # respawn. String matching because torch.cuda exception
                # hierarchy doesn't cleanly cover every context-corrupting
                # case (kernel asserts, illegal memory access).
                if "CUDA out of memory" in msg or "CUDA error" in msg:
                    # FATAL send may itself raise BrokenPipeError if the
                    # parent is gone; suppress so the CUDA exit path still
                    # runs. The parent detects subprocess death via the
                    # drainer / recv EOF either way.
                    with suppress(Exception):
                        conn.send({
                            "kind": "FATAL",
                            "request_id": request_id,
                            "reason": "CUDA_FAULT",
                            "exception_repr": repr(exc),
                        })
                    print(
                        f"[pool-worker/{shape_key}] CUDA fault, exiting",
                        file=sys.stderr, flush=True,
                    )
                    with suppress(Exception):
                        conn.close()
                    return 2
                # Recoverable per-request error (validation, bad prompt, etc.).
                # Report and stay alive — the persistent subprocess is the
                # whole point of the pool; don't respawn on soft failures.
                conn.send({
                    "kind": "ERROR",
                    "request_id": request_id,
                    "exception_type": type(exc).__name__,
                    "exception_repr": repr(exc),
                })
    except (BrokenPipeError, ConnectionResetError, OSError) as exc:
        # Parent disconnected mid-protocol (READY send, request-loop send,
        # or recv-side OSError). Exit cleanly with code 0 -- the parent
        # has already detected the subprocess as needing cleanup via its
        # drainer / recv EOF, and an uncaught BrokenPipeError would just
        # turn into a noisy exit-1 with a Python traceback that has no
        # consumer.
        print(
            f"[pool-worker/{shape_key}] parent connection lost: {exc}",
            file=sys.stderr, flush=True,
        )
        with suppress(Exception):
            conn.close()
        return 0

    # Unreachable: the `while True` loop above only exits via `return 0`
    # on parent EOF or `return 2` on CUDA fault. Explicit terminal
    # return makes the int-return contract obvious to future readers
    # and keeps mypy happy.
    return 0


def _pool_worker_dispatch_if_requested() -> None:
    """
    If invoked with --pool-worker, run as a pool subprocess and exit.
    Must run BEFORE the parent's logging.basicConfig / hash-check /
    Dynamo registration — pool subprocesses are internal compute slaves,
    not endpoint workers.
    """
    if "--pool-worker" not in sys.argv:
        return
    sub_parser = argparse.ArgumentParser(add_help=False)
    sub_parser.add_argument("--pool-worker", action="store_true")
    sub_parser.add_argument("--shape-key", required=True)
    sub_parser.add_argument("--model", required=True)
    sub_parser.add_argument("--num-gpus", type=int, default=1)
    sub_parser.add_argument("--enable-optimizations", action="store_true")
    sub_parser.add_argument(
        "--attention-backend", default=DEFAULT_ATTENTION_BACKEND,
    )
    sub_parser.add_argument("--protocol-fd", required=True, type=int)
    sub_args = sub_parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
        force=True,
    )
    sys.exit(_pool_worker_main(
        shape_key=sub_args.shape_key,
        model_path=sub_args.model,
        num_gpus=sub_args.num_gpus,
        enable_optimizations=sub_args.enable_optimizations,
        attention_backend=sub_args.attention_backend,
        protocol_fd=sub_args.protocol_fd,
    ))


if __name__ == "__main__":
    # Pool-worker subprocess dispatch MUST run before any parent-only
    # setup (logging config, image/menu hash check, uvloop install).
    # Pool subprocesses don't serve customers; they don't need (and
    # would fail on) the production-worker preconditions.
    _pool_worker_dispatch_if_requested()

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
