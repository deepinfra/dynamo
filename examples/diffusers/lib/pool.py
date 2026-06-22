# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generic LRU subprocess pool for video-pipeline workers.

Phase 2 of the LTX-2 rollout introduced per-shape persistent
subprocesses to amortize the ~3-minute cold-compile cost across all
same-shape requests. This module is the model-agnostic core: any
factory function with signature ``(model_path, num_gpus,
enable_optimizations) -> object-with-generate_video()`` plugs in.

Conventions:

- Constants live as ``VIDEO_POOL_*`` and read from env vars of the
  same name. Production deployments MUST set ``VIDEO_POOL_MODE=1`` to
  opt in; default-off keeps the legacy in-process generator path.
- Pool subprocesses are spawned by invoking the top-level
  ``worker.py`` shim with ``--pool-worker`` + ``--model-factory
  <dotted>``. The shim short-circuits to
  :func:`_pool_worker_dispatch_if_requested` which imports the factory
  dynamically and runs :func:`_pool_worker_main`.
- All metrics live on the parent. Subprocesses do not have a Dynamo
  endpoint and cannot emit. The parent's :meth:`SubprocessPool.route`
  FATAL branch is the single point that records ``cuda_fault``.
"""

import argparse
import asyncio
import collections
import importlib
import logging
import os
import signal
import sys
import time
from contextlib import suppress
from multiprocessing import Pipe
from multiprocessing.connection import Connection
from typing import Any, Callable

from .metrics import (
    POOL_COLD_SPAWN_SECONDS,
    POOL_EVICTION_TOTAL,
    POOL_REQUEST_LATENCY,
    POOL_REQUEST_TOTAL,
    POOL_SIZE,
    POOL_SPAWN_TOTAL,
    POOL_SUBPROCESS_FAILURE_TOTAL,
)

logger = logging.getLogger(__name__)

# ── Pool-mode configuration ──────────────────────────────────────────────────
# Pool mode is OPT-IN. VIDEO_POOL_MODE=1 routes every customer request
# through a per-shape persistent subprocess; default (unset or any other
# value) keeps the legacy in-process generator path.
# See ~/backend/claude_plans/2026-05-14-ltx2-phase2-subprocess-pool.md.
VIDEO_POOL_MODE = os.environ.get("VIDEO_POOL_MODE", "0") == "1"
# Max concurrent shape-pinned subprocesses. K=2 fits on B200 (178 GiB)
# for LTX-2: each pool subprocess internally spawns a FastVideo
# multiproc_executor Worker child that holds the model -- the outer
# pool subprocess is ~600 MiB but the FastVideo Worker is ~70 GiB on
# LTX-2. Observed on di-slc-39 2026-05-14: at 2 hot shapes the GPU is
# ~80% utilized; adding a 3rd hot shape OOMs deterministically during
# model load. If more shapes need to be hot than fit, k8s scale-out is
# the answer, not more subprocesses per pod.
VIDEO_POOL_MAX_SIZE = int(os.environ.get("VIDEO_POOL_MAX_SIZE", "2"))
# Subprocess must print READY within this window after spawn (covers
# Python startup + torch import + model load + cache hydrate).
VIDEO_POOL_SPAWN_TIMEOUT_S = int(os.environ.get("VIDEO_POOL_SPAWN_TIMEOUT_S", "180"))
# Per-request generation timeout; longest production shape is well
# under this.
VIDEO_POOL_GEN_TIMEOUT_S = int(os.environ.get("VIDEO_POOL_GEN_TIMEOUT_S", "1800"))
# SIGTERM grace before SIGKILL during eviction / shutdown.
VIDEO_POOL_SIGTERM_GRACE_S = int(os.environ.get("VIDEO_POOL_SIGTERM_GRACE_S", "30"))
# How many lines of subprocess stderr to retain for crash diagnostics.
VIDEO_POOL_STDERR_RING_LINES = int(os.environ.get("VIDEO_POOL_STDERR_RING_LINES", "50"))


# Top-level worker.py shim path. Pool subprocesses are spawned by
# invoking this with --pool-worker; the shim dispatches into
# _pool_worker_dispatch_if_requested. Computed relative to this file so
# it works regardless of cwd. In production the layout is
# /opt/app/worker.py (shim) and /opt/app/lib/pool.py (this file).
_WORKER_ENTRY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "worker.py",
)


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
    the model factory's transitive deps -- FastVideo / vllm /
    torch.compile / multiproc workers). With the Connection-based
    protocol, library writes to stdout cannot corrupt the wire
    protocol, so we treat both pipes as pure diagnostics.
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
            maxlen=VIDEO_POOL_STDERR_RING_LINES
        )
        self.stdout_drainer: asyncio.Task | None = None
        self.stderr_drainer: asyncio.Task | None = None

    def diag_tail(self) -> str:
        return "\n".join(self.diag_buf)


class SubprocessPool:
    """
    LRU pool of shape-pinned subprocesses.

    The parent's backend-level _generate_lock already serializes all
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
        model_factory_dotted: str,
        model_label: str,
    ) -> None:
        self.model_path = model_path
        self.num_gpus = num_gpus
        self.enable_optimizations = enable_optimizations
        self.attention_backend = attention_backend
        # Dotted reference (``pkg.module:func``) passed to subprocesses
        # via --model-factory. The subprocess imports it via importlib
        # and calls it as the model factory.
        self.model_factory_dotted = model_factory_dotted
        # Metric label value used for the ``model`` label on every
        # video_pool_* series this pool emits.
        self.model_label = model_label
        self._handles: collections.OrderedDict[
            str, _SubprocessHandle
        ] = collections.OrderedDict()
        self._pool_lock = asyncio.Lock()
        # Pre-initialize the labeled gauge so ``video_pool_size`` shows
        # up at /metrics from boot (with value 0). Without this, the
        # series wouldn't emit until first pool activity, which makes
        # it hard for operators to confirm pool mode is configured
        # before any traffic flows.
        POOL_SIZE.labels(model=self.model_label).set(0)

    async def route(self, shape_key: str, request: dict) -> dict:
        """
        Run one request through the subprocess for `shape_key`, spawning
        one (and evicting LRU if needed) if absent. Caller is expected to
        hold the global generate lock.

        Returns a dict with at minimum 'status' in {'DONE', 'ERROR', 'FATAL'}
        plus 'request_id'. Raises RuntimeError if the subprocess dies,
        times out, or returns a desynced response.
        """
        t_request_start = time.monotonic()
        handle = await self._get_or_spawn(shape_key)
        self._handles.move_to_end(shape_key)
        handle.last_used = time.monotonic()

        request_id = request["request_id"]
        payload = {"kind": "REQUEST", **request}

        try:
            await asyncio.to_thread(handle.conn.send, payload)
        except (BrokenPipeError, ConnectionResetError, OSError, EOFError) as exc:
            POOL_SUBPROCESS_FAILURE_TOTAL.labels(
                model=self.model_label, reason="send_failed"
            ).inc()
            tail = handle.diag_tail()
            await self._discard(shape_key, handle)
            raise RuntimeError(
                f"subprocess for {shape_key} died before request "
                f"could be sent: {exc}. diag tail:\n{tail}"
            ) from exc

        try:
            msg = await self._recv_msg(handle, VIDEO_POOL_GEN_TIMEOUT_S)
        except asyncio.TimeoutError:
            POOL_SUBPROCESS_FAILURE_TOTAL.labels(
                model=self.model_label, reason="gen_timeout"
            ).inc()
            tail = handle.diag_tail()
            await self._discard(shape_key, handle)
            raise RuntimeError(
                f"subprocess for {shape_key} did not respond within "
                f"{VIDEO_POOL_GEN_TIMEOUT_S}s. diag tail:\n{tail}"
            )
        except (EOFError, OSError, ConnectionError) as exc:
            POOL_SUBPROCESS_FAILURE_TOTAL.labels(
                model=self.model_label, reason="gen_eof"
            ).inc()
            tail = handle.diag_tail()
            await self._discard(shape_key, handle)
            raise RuntimeError(
                f"subprocess for {shape_key} EOF mid-request: {exc}. "
                f"diag tail:\n{tail}"
            ) from exc

        if not isinstance(msg, dict):
            POOL_SUBPROCESS_FAILURE_TOTAL.labels(
                model=self.model_label, reason="parse_error"
            ).inc()
            tail = handle.diag_tail()
            await self._discard(shape_key, handle)
            raise RuntimeError(
                f"subprocess for {shape_key} returned non-dict message "
                f"{msg!r}. diag tail:\n{tail}"
            )

        resp_rid = msg.get("request_id")
        if resp_rid != request_id:
            POOL_SUBPROCESS_FAILURE_TOTAL.labels(
                model=self.model_label, reason="desync"
            ).inc()
            tail = handle.diag_tail()
            await self._discard(shape_key, handle)
            raise RuntimeError(
                f"subprocess for {shape_key} returned request_id={resp_rid!r} "
                f"but expected {request_id!r} (desync). Killed. "
                f"diag tail:\n{tail}"
            )

        kind = msg.get("kind")
        if kind == "DONE":
            POOL_REQUEST_TOTAL.labels(
                model=self.model_label, shape_key=shape_key, status="DONE"
            ).inc()
            POOL_REQUEST_LATENCY.labels(
                model=self.model_label, shape_key=shape_key
            ).observe(time.monotonic() - t_request_start)
            return {
                "status": "DONE",
                "request_id": request_id,
                "elapsed_ms": int(msg.get("elapsed_ms", 0)),
            }
        if kind == "ERROR":
            POOL_REQUEST_TOTAL.labels(
                model=self.model_label, shape_key=shape_key, status="ERROR"
            ).inc()
            err = f"{msg.get('exception_type', '?')} {msg.get('exception_repr', '?')}"
            return {"status": "ERROR", "request_id": request_id, "error": err}
        if kind == "FATAL":
            POOL_REQUEST_TOTAL.labels(
                model=self.model_label, shape_key=shape_key, status="FATAL"
            ).inc()
            POOL_SUBPROCESS_FAILURE_TOTAL.labels(
                model=self.model_label, reason="cuda_fault"
            ).inc()
            tail = handle.diag_tail()
            self._handles.pop(shape_key, None)
            self._update_pool_size()
            # Subprocess is exiting on its own (sys.exit(2) after sending
            # FATAL). _kill waits up to SIGTERM grace, then SIGKILLs if
            # still alive, then cancels drainers and closes the conn.
            await self._kill(handle)
            err = f"{msg.get('reason', '?')}: {msg.get('exception_repr', '?')}"
            logger.error(
                "[pool/%s] FATAL %s. diag tail:\n%s",
                shape_key,
                err,
                tail,
            )
            return {"status": "FATAL", "request_id": request_id, "error": err}

        POOL_SUBPROCESS_FAILURE_TOTAL.labels(
            model=self.model_label, reason="parse_error"
        ).inc()
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

    def _update_pool_size(self) -> None:
        POOL_SIZE.labels(model=self.model_label).set(len(self._handles))

    async def _get_or_spawn(self, shape_key: str) -> _SubprocessHandle:
        async with self._pool_lock:
            handle = self._handles.get(shape_key)
            if handle is not None and handle.proc.returncode is None:
                return handle
            if handle is not None:
                # In dict but dead — clean up and respawn.
                self._handles.pop(shape_key, None)
                self._update_pool_size()
                await self._kill(handle)
            while len(self._handles) >= VIDEO_POOL_MAX_SIZE:
                await self._evict_lru()
            handle = await self._spawn(shape_key)
            self._handles[shape_key] = handle
            self._update_pool_size()
            return handle

    async def _spawn(self, shape_key: str) -> _SubprocessHandle:
        POOL_SPAWN_TOTAL.labels(model=self.model_label, shape_key=shape_key).inc()
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
            _WORKER_ENTRY_PATH,
            "--pool-worker",
            "--shape-key",
            shape_key,
            "--model",
            self.model_path,
            "--num-gpus",
            str(self.num_gpus),
            "--attention-backend",
            self.attention_backend,
            "--model-factory",
            self.model_factory_dotted,
            "--protocol-fd",
            str(child_conn.fileno()),
        ]
        if self.enable_optimizations:
            cmd.append("--enable-optimizations")

        logger.info(
            "[pool/%s] spawning subprocess (cache=%s)",
            shape_key,
            inductor_dir,
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
                msg = await self._recv_msg(handle, VIDEO_POOL_SPAWN_TIMEOUT_S)
            except asyncio.TimeoutError:
                POOL_SUBPROCESS_FAILURE_TOTAL.labels(
                    model=self.model_label, reason="spawn_timeout"
                ).inc()
                tail = handle.diag_tail()
                raise RuntimeError(
                    f"subprocess for {shape_key} failed to reach READY "
                    f"within {VIDEO_POOL_SPAWN_TIMEOUT_S}s. diag tail:\n{tail}"
                )
            except (EOFError, OSError, ConnectionError) as exc:
                POOL_SUBPROCESS_FAILURE_TOTAL.labels(
                    model=self.model_label, reason="spawn_eof"
                ).inc()
                tail = handle.diag_tail()
                raise RuntimeError(
                    f"subprocess for {shape_key} died before READY: {exc}. "
                    f"diag tail:\n{tail}"
                ) from exc

            if not isinstance(msg, dict) or msg.get("kind") != "READY":
                POOL_SUBPROCESS_FAILURE_TOTAL.labels(
                    model=self.model_label, reason="spawn_parse_error"
                ).inc()
                tail = handle.diag_tail()
                raise RuntimeError(
                    f"subprocess for {shape_key} sent unexpected first "
                    f"message {msg!r} (expected kind=READY). "
                    f"diag tail:\n{tail}"
                )

            cold_spawn_s = time.monotonic() - t_spawn
            POOL_COLD_SPAWN_SECONDS.labels(
                model=self.model_label, shape_key=shape_key
            ).observe(cold_spawn_s)
            logger.info("[pool/%s] READY in %.1fs", shape_key, cold_spawn_s)
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
                    "[pool/%s/%s] %s",
                    handle.shape_key,
                    label,
                    text,
                )
        except asyncio.CancelledError:
            return
        except Exception:
            logger.exception(
                "[pool/%s] %s drainer crashed",
                handle.shape_key,
                label,
            )

    async def _evict_lru(self) -> None:
        # Caller holds _pool_lock.
        if not self._handles:
            return
        shape_key, handle = next(iter(self._handles.items()))
        POOL_EVICTION_TOTAL.labels(model=self.model_label, shape_key=shape_key).inc()
        logger.info(
            "[pool/%s] evicting LRU (idle %.0fs)",
            shape_key,
            time.monotonic() - handle.last_used,
        )
        self._handles.pop(shape_key, None)
        self._update_pool_size()
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
                    proc.wait(),
                    timeout=VIDEO_POOL_SIGTERM_GRACE_S,
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
        drainers = [
            d for d in (handle.stdout_drainer, handle.stderr_drainer) if d is not None
        ]
        for d in drainers:
            d.cancel()
        if drainers:
            # Await the cancellations so the drain tasks finish unwinding before
            # we return; return_exceptions swallows the expected CancelledError
            # (and any late drain error) instead of failing the kill path.
            await asyncio.gather(*drainers, return_exceptions=True)

    async def _discard(self, shape_key: str, handle: _SubprocessHandle) -> None:
        """Remove from pool and kill. Used for unrecoverable subprocess errors."""
        self._handles.pop(shape_key, None)
        self._update_pool_size()
        await self._kill(handle)

    async def shutdown(self) -> None:
        """Kill all subprocesses. Called on pod shutdown."""
        async with self._pool_lock:
            handles = list(self._handles.values())
            self._handles.clear()
            self._update_pool_size()
        for handle in handles:
            await self._kill(handle)


# ── Pool-worker subprocess entrypoint ────────────────────────────────────────


def _set_parent_death_signal(sig: int = signal.SIGTERM) -> None:
    """
    Ask the kernel to send `sig` to this process when its parent thread
    dies. Linux-only via prctl(PR_SET_PDEATHSIG). Closes the orphan
    window where a parent crash would leave the generator's Worker child
    holding GPU memory until container teardown.

    Note: PR_SET_PDEATHSIG fires on parent THREAD death, not necessarily
    parent PROCESS death. The dynamo worker parent is single-threaded
    Python (uvloop event loop in the main thread), so the two are
    equivalent for us. If the parent ever sprouts a long-lived non-main
    thread that this subprocess inherits as its parent, this guarantee
    weakens; revisit at that point.
    """
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        PR_SET_PDEATHSIG = 1
        if libc.prctl(PR_SET_PDEATHSIG, sig, 0, 0, 0) != 0:
            errno = ctypes.get_errno()
            print(
                f"[pool-worker] prctl(PR_SET_PDEATHSIG) failed errno={errno}",
                file=sys.stderr,
                flush=True,
            )
    except (OSError, AttributeError) as exc:
        print(
            f"[pool-worker] PR_SET_PDEATHSIG unavailable: {exc}",
            file=sys.stderr,
            flush=True,
        )


def _megacache_blob_path(shape_key: str) -> str | None:
    """Per-shape Mega-Cache blob path, or None if the feature is off.

    Opt-in via ``LTX_MEGACACHE_DIR``. When set, the dir holds one
    ``<shape_key>.megacache.bin`` per shape: a portable ``torch.compiler``
    artifact bundle (FX graphs + Triton + autotune best_configs). At bake
    time the dir is writable and starts empty (worker compiles cold, then
    saves the blob); the blobs are then COPYd into the image, and at serve
    time the worker LOADs them so a fresh pod skips the autotune recompile.
    Unset => no-op (legacy behavior: compile cold per process).
    """
    base = os.environ.get("LTX_MEGACACHE_DIR")
    if not base:
        return None
    return os.path.join(base, f"{shape_key}.megacache.bin")


def _export_megacache_env(shape_key: str) -> None:
    """Export the per-shape Mega-Cache blob path for the FastVideo worker child.

    The actual ``torch.compiler.save/load_cache_artifacts`` calls MUST run in
    the process that compiles -- and FastVideo's MultiprocExecutor compiles in a
    spawned worker child, NOT here. So this pool-worker only resolves the
    per-shape path and exports it as ``LTX_MEGACACHE_BLOB`` BEFORE the factory
    builds the generator (which spawns that child). The child inherits the env
    and does the load (before first forward) / save (after first forward). See
    fastvideo/worker/gpu_worker.py. No-op when ``LTX_MEGACACHE_DIR`` is unset.
    """
    path = _megacache_blob_path(shape_key)
    if path:
        os.environ["LTX_MEGACACHE_BLOB"] = path
        print(
            f"[pool-worker/{shape_key}] megacache blob path -> {path} "
            f"(load/save happen in the fastvideo worker child)",
            file=sys.stderr,
            flush=True,
        )


def _pool_worker_main(
    factory_func: Callable[[str, int, bool], Any],
    shape_key: str,
    model_path: str,
    num_gpus: int,
    enable_optimizations: bool,
    attention_backend: str,
    protocol_fd: int,
) -> int:
    """
    Subprocess entry. Pinned to one shape for the lifetime of the process.

    ``factory_func`` is the model-specific factory imported by the
    dispatcher; signature is ``(model_path, num_gpus,
    enable_optimizations) -> object-with-generate_video()``. Heavy
    imports (torch, fastvideo, ...) MUST happen inside the factory so
    the dispatcher's argv-parse-and-import path stays cheap.

    Wire protocol: pickled-dict messages over a multiprocessing.Connection.
    The parent created a duplex Pipe and passed the child end's fd via
    pass_fds; we reconstruct a Connection here. stdout/stderr are
    diagnostic-only -- library noise on either pipe is drained by the
    parent into a ring buffer but cannot corrupt the wire protocol
    because no library has a handle to ``conn``'s fd.

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
    # First action: ask the kernel to SIGTERM us if the parent dies.
    # Must run before any model load -- a parent crash during the
    # ~150-second cold-spawn window would otherwise orphan a child
    # holding GPU memory until container teardown.
    _set_parent_death_signal(signal.SIGTERM)
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
    # Persist the CuTe DSL on-disk cache (the ~270 FlashAttention-4 + QuACK
    # @cute.jit kernels) and the CUDA PTX JIT cache to a stable location, so a
    # fresh process reuses them instead of recompiling every boot. These are NOT
    # covered by torch Mega-Cache (which only handles inductor/autotune/aot/pgo).
    # CuTe defaults to an ephemeral /tmp dir if unset -> recompile every process.
    # Path under /cache so the bake populates it and it can be carried into the
    # image; override via env. See ltx23_cache_investigation_report.md.
    os.environ.setdefault("CUTE_DSL_CACHE_DIR", "/cache/cutedsl")
    os.environ.setdefault("CUDA_CACHE_PATH", "/cache/cuda")
    # QuACK (RMSNorm/softmax CuTe kernels) has its own persistent .o cache plus
    # an autotuning-results cache. Both default to ephemeral/off: QUACK_CACHE_DIR
    # falls back to $HOME (wiped per container) and QUACK_CACHE_AUTOTUNING is off.
    # Pin the dir under /cache and persist autotuning so a fresh process reuses
    # both instead of recompiling + re-autotuning QuACK kernels.
    os.environ.setdefault("QUACK_CACHE_DIR", "/cache/quack")
    os.environ.setdefault("QUACK_CACHE_AUTOTUNING", "1")
    for _cache_dir in (
        os.environ["CUTE_DSL_CACHE_DIR"],
        os.environ["CUDA_CACHE_PATH"],
        os.environ["QUACK_CACHE_DIR"],
    ):
        try:
            os.makedirs(_cache_dir, exist_ok=True)
        except OSError:
            # Best-effort: these are optional compile/autotune caches. If the
            # dir can't be created the libraries fall back to ephemeral/off, so
            # a failure here must not block worker startup.
            pass

    conn = Connection(protocol_fd)

    def _on_sigterm(_signum, _frame):
        # Clean exit on SIGTERM (parent's eviction path); a blocked
        # conn.recv() would otherwise sit on the syscall forever.
        print(
            f"[pool-worker/{shape_key}] SIGTERM received, exiting",
            file=sys.stderr,
            flush=True,
        )
        with suppress(Exception):
            conn.close()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _on_sigterm)

    # Export the per-shape Mega-Cache blob path BEFORE building the generator:
    # factory_func spawns the FastVideo worker child that actually compiles, and
    # the child inherits this env to load/save the compile-artifact blob.
    _export_megacache_env(shape_key)

    print(
        f"[pool-worker/{shape_key}] loading model={model_path} "
        f"num_gpus={num_gpus} enable_optimizations={enable_optimizations}",
        file=sys.stderr,
        flush=True,
    )
    t_load = time.perf_counter()
    generator = factory_func(model_path, num_gpus, enable_optimizations)
    load_s = time.perf_counter() - t_load
    print(
        f"[pool-worker/{shape_key}] generator ready in {load_s:.1f}s",
        file=sys.stderr,
        flush=True,
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
                conn.send(
                    {
                        "kind": "ERROR",
                        "request_id": req.get("request_id", "_")
                        if isinstance(req, dict)
                        else "_",
                        "exception_type": "ProtocolError",
                        "exception_repr": f"unexpected message {req!r}",
                    }
                )
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
                if req.get("ltx2_images"):
                    # i2v: forward conditioning image(s) to the per-shape worker.
                    # Same compiled shape as t2v (per-token mask, not a shape change).
                    kwargs["ltx2_images"] = req["ltx2_images"]
                    kwargs["ltx2_image_crf"] = req.get("ltx2_image_crf", 0.0)

                t0 = time.perf_counter()
                generator.generate_video(**kwargs)
                elapsed_ms = int((time.perf_counter() - t0) * 1000)
                conn.send(
                    {
                        "kind": "DONE",
                        "request_id": request_id,
                        "elapsed_ms": elapsed_ms,
                    }
                )
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
                        conn.send(
                            {
                                "kind": "FATAL",
                                "request_id": request_id,
                                "reason": "CUDA_FAULT",
                                "exception_repr": repr(exc),
                            }
                        )
                    print(
                        f"[pool-worker/{shape_key}] CUDA fault, exiting",
                        file=sys.stderr,
                        flush=True,
                    )
                    with suppress(Exception):
                        conn.close()
                    return 2
                # Recoverable per-request error (validation, bad prompt, etc.).
                # Report and stay alive — the persistent subprocess is the
                # whole point of the pool; don't respawn on soft failures.
                conn.send(
                    {
                        "kind": "ERROR",
                        "request_id": request_id,
                        "exception_type": type(exc).__name__,
                        "exception_repr": repr(exc),
                    }
                )
    except (BrokenPipeError, ConnectionResetError, OSError) as exc:
        # Parent disconnected mid-protocol (READY send, request-loop send,
        # or recv-side OSError). Exit cleanly with code 0 -- the parent
        # has already detected the subprocess as needing cleanup via its
        # drainer / recv EOF, and an uncaught BrokenPipeError would just
        # turn into a noisy exit-1 with a Python traceback that has no
        # consumer.
        print(
            f"[pool-worker/{shape_key}] parent connection lost: {exc}",
            file=sys.stderr,
            flush=True,
        )
        with suppress(Exception):
            conn.close()
        return 0

    # Unreachable: the `while True` loop above only exits via `return 0`
    # on parent EOF or `return 2` on CUDA fault. Explicit terminal
    # return makes the int-return contract obvious to future readers
    # and keeps mypy happy.
    return 0


def _import_factory(dotted: str) -> Callable[[str, int, bool], Any]:
    """
    Import a model factory referenced as ``pkg.module:func``. Heavy
    imports (torch / fastvideo / ...) happen inside the function body
    so this import-and-resolve step itself is cheap.
    """
    if ":" not in dotted:
        raise ValueError(
            f"--model-factory must be `pkg.module:func` form, got {dotted!r}"
        )
    module_name, fn_name = dotted.split(":", 1)
    mod = importlib.import_module(module_name)
    fn = getattr(mod, fn_name, None)
    if not callable(fn):
        raise ValueError(
            f"--model-factory {dotted!r}: {fn_name!r} not callable in {module_name!r}"
        )
    return fn


def _pool_worker_dispatch_if_requested() -> None:
    """
    If invoked with --pool-worker, run as a pool subprocess and exit.
    Must run BEFORE any parent-only setup (logging.basicConfig /
    hash-check / Dynamo registration) -- pool subprocesses are internal
    compute slaves, not endpoint workers.
    """
    if "--pool-worker" not in sys.argv:
        return
    sub_parser = argparse.ArgumentParser(add_help=False)
    sub_parser.add_argument("--pool-worker", action="store_true")
    sub_parser.add_argument("--shape-key", required=True)
    sub_parser.add_argument("--model", required=True)
    sub_parser.add_argument("--num-gpus", type=int, default=1)
    sub_parser.add_argument("--enable-optimizations", action="store_true")
    sub_parser.add_argument("--attention-backend", default="FLASH_ATTN")
    sub_parser.add_argument("--model-factory", required=True)
    sub_parser.add_argument("--protocol-fd", required=True, type=int)
    sub_args, _ = sub_parser.parse_known_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
        force=True,
    )
    factory = _import_factory(sub_args.model_factory)
    sys.exit(
        _pool_worker_main(
            factory_func=factory,
            shape_key=sub_args.shape_key,
            model_path=sub_args.model,
            num_gpus=sub_args.num_gpus,
            enable_optimizations=sub_args.enable_optimizations,
            attention_backend=sub_args.attention_backend,
            protocol_fd=sub_args.protocol_fd,
        )
    )
