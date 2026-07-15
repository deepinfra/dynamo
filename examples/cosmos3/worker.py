# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Cosmos 3 Dynamo videogen worker.

Loads a Cosmos 3 checkpoint once at startup and serves text/image-to-video
generation. Two operating modes:

**Dynamo mode** (production, default):
  Registers as a Dynamo backend endpoint. A separate Dynamo frontend
  (``python -m dynamo.frontend``) handles HTTP and routes requests to
  this worker via Dynamo RPC. The frontend exposes ``/v1/videos`` and
  this worker registers with ``ModelType.Videos``.

**Standalone mode** (``--standalone``, for local testing):
  Runs a plain FastAPI HTTP server directly on the worker, no frontend
  needed. Useful for ``docker run`` one-shot testing.

In both modes the underlying inference is identical:

    setup_args = OmniSetupOverrides(...).build_setup(world_size=WORLD_SIZE)
    pipe = OmniInference.create(setup_args)
    sample_args = OmniSampleOverrides(...).build_sample(model_config=pipe.model_config)
    pipe.generate([sample_args])

Multi-GPU (CFG- and context-parallel) inference:

``pipe.generate`` is a *collective* — every rank must call it with the same
sample args. For ``--num-gpus > 1`` the worker (re-)launches itself under
``torchrun --nproc-per-node=N``; ``init_script()`` then initializes the process
group. Rank 0 runs the endpoint and, for each request, broadcasts the request
to all ranks; every rank runs ``generate`` together, and only rank 0 reads back
the produced media file.

Usage:
  # Dynamo mode (requires a frontend running alongside):
  python3 worker.py --served-model-name nvidia/Cosmos3-Nano --model Cosmos3-Nano

  # Standalone HTTP mode (no frontend needed):
  python3 worker.py --standalone --served-model-name nvidia/Cosmos3-Nano --model Cosmos3-Nano
"""

# init_script() configures logging/CUDA and, when WORLD_SIZE > 1 (i.e. under
# torchrun), initializes torch.distributed. It must run before any heavy
# cosmos_framework/torch imports, exactly as in cosmos_framework/scripts/inference.py.
from cosmos_framework.inference.common.init import init_script

init_script()

import argparse  # noqa: E402
import asyncio  # noqa: E402
import base64  # noqa: E402
import mimetypes  # noqa: E402
import os  # noqa: E402
import random  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import uuid  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import torch  # noqa: E402
from cosmos_framework.inference.args import (  # noqa: E402
    OmniSampleOverrides,
    OmniSetupOverrides,
)
from cosmos_framework.inference.common.init import (  # noqa: E402
    get_local_rank,
    get_rank,
    init_output_dir,
    is_rank0,
)
from cosmos_framework.inference.inference import OmniInference  # noqa: E402
from cosmos_framework.utils import distributed, log  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

# ---------------------------------------------------------------------------
# Control-plane process group (multi-GPU request dispatch)
# ---------------------------------------------------------------------------
# The default process group is NCCL, whose watchdog aborts any collective
# pending longer than the init timeout (default 30 min). The idle
# _worker_rank_loop blocks on broadcast_object() waiting for the *next*
# request, so a 2-GPU worker that receives no traffic for 30 minutes is
# killed by its own watchdog (SeqNum stuck at the first idle wait; pager
# incident 2026-06-07). Request dispatch therefore goes over a gloo (CPU)
# group with a generous timeout; NCCL remains in charge of the actual
# generation collectives inside _run_generate().
#
# new_group() is itself collective: every rank executes this at import time
# (module import is symmetric under torchrun), keeping creation in lockstep.
_CONTROL_PG = None
if torch.distributed.is_available() and torch.distributed.is_initialized():
    import datetime as _dt

    _CONTROL_PG = torch.distributed.new_group(
        backend="gloo", timeout=_dt.timedelta(days=7)
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RESOLUTION_ALIASES = {"256p": "256", "480p": "480", "720p": "720", "1080p": "1080"}
_DATA_URI_RE = re.compile(r"^data:([^;,]+)?(?:;base64)?,(.*)$", re.DOTALL)

# Reverse lookup: (width, height) -> (resolution, aspect_ratio).
# Built from cosmos_framework's VIDEO_RES_SIZE_INFO (same as IMAGE_RES_SIZE_INFO
# for resolutions we support). Used to decode the Dynamo frontend's "size" field
# ("WxH") back to the resolution + aspect_ratio the inference engine expects.
_SIZE_TO_RES_AR: dict[tuple[int, int], tuple[str, str]] = {}


def _init_size_lookup() -> None:
    from cosmos_framework.data.generator.utils import VIDEO_RES_SIZE_INFO

    for res, ar_map in VIDEO_RES_SIZE_INFO.items():
        for ar, (w, h) in ar_map.items():
            _SIZE_TO_RES_AR[(w, h)] = (res, ar)


_init_size_lookup()


def _normalize_resolution(resolution: str) -> str:
    return _RESOLUTION_ALIASES.get(resolution.strip().lower(), resolution.strip())


def _decode_data_uri(uri: str, output_dir: Path) -> str:
    """Decode a ``data:`` URI to a local file and return the path.

    Cosmos 3's ``vision_path`` only accepts ``http(s)://``, ``s3://``, or
    local file paths. This helper bridges the gap for base64-encoded payloads
    sent by the API gateway.
    """
    m = _DATA_URI_RE.match(uri)
    if m is None:
        raise ValueError(f"Invalid data URI: {uri[:80]}...")
    mime_type = m.group(1) or "application/octet-stream"
    ext = mimetypes.guess_extension(mime_type) or ".bin"
    data = base64.b64decode(m.group(2))
    path = output_dir / f"input_vision{ext}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return str(path)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
# These match the JSON format that deepinfra/deepapi/handler.py sends to
# the /v1/videos endpoint and the response it expects back.


# Valid model_mode values for action modalities.
_ACTION_MODES = frozenset({"forward_dynamics", "inverse_dynamics", "policy"})


class NvExtFields(BaseModel):
    """Subset of Dynamo's NvExt that we care about."""

    model_config = {"extra": "ignore"}
    fps: int | None = None
    num_frames: int | None = None
    seed: int | None = None
    num_steps: int | None = None


class Cosmos3VideoRequest(BaseModel):
    model: str | None = None
    prompt: str
    # Direct fields (handler → worker pod, or standalone mode)
    num_frames: int = 189
    resolution: str = "720p"
    aspect_ratio: str | None = None
    fps: int = 24
    seed: int | None = None
    image_url: str | None = None
    video_url: str | None = None
    # Dynamo frontend passthrough fields (NvCreateVideoRequest)
    size: str | None = None
    input_reference: str | None = None
    nvext: NvExtFields | None = None
    # Action modality fields
    model_mode: str | None = None
    action_url: str | None = None
    domain_name: str | None = None
    action_chunk_size: int | None = None
    raw_action_dim: int | None = None
    image_size: int | None = None

    def resolve(self) -> tuple[int, str, str | None, int, int | None, str | None]:
        """Resolve effective (num_frames, resolution, aspect_ratio, fps, seed, vision_url).

        Dynamo frontend fields (nvext, size, input_reference) take priority when
        present, falling back to direct fields for standalone/backward-compat.
        """
        nf = self.num_frames
        res = self.resolution
        ar = self.aspect_ratio
        fps = self.fps
        seed = self.seed
        if self.nvext is not None:
            if self.nvext.num_frames is not None:
                nf = self.nvext.num_frames
            if self.nvext.fps is not None:
                fps = self.nvext.fps
            if self.nvext.seed is not None:
                seed = self.nvext.seed
        if self.size is not None:
            parts = self.size.lower().split("x")
            if len(parts) == 2:
                try:
                    w, h = int(parts[0]), int(parts[1])
                    if (w, h) in _SIZE_TO_RES_AR:
                        res, ar = _SIZE_TO_RES_AR[(w, h)]
                except ValueError:
                    log.warning(
                        "ignoring unparseable size %r; using resolution defaults",
                        self.size,
                    )
        vision_url = self.video_url or self.image_url or self.input_reference
        return nf, res, ar, fps, seed, vision_url


class Cosmos3VideoDataItem(BaseModel):
    output_format: str = "mp4"
    b64_json: str | None = None
    url: str | None = None
    seed: int | None = None
    media_type: str = "video"
    action: list | None = None


class Cosmos3VideoResponse(BaseModel):
    """Matches Dynamo's NvVideosResponse so the frontend can parse it."""

    id: str = Field(default_factory=lambda: f"video-{uuid.uuid4().hex}")
    object: str = "video"
    model: str = ""
    status: str = "completed"
    progress: int = 100
    created: int = Field(default_factory=lambda: int(time.time()))
    data: list[Cosmos3VideoDataItem] = Field(default_factory=list)
    error: str | None = None
    inference_time_s: float | None = None


# ---------------------------------------------------------------------------
# Inference engine (shared between both modes)
# ---------------------------------------------------------------------------

_pipe: OmniInference | None = None
_served_model_name: str = ""
_output_root: Path = Path("/tmp/cosmos3_worker_outputs")
_generate_lock = threading.Lock()
# Default denoising steps for VIDEO when a request doesn't specify one. Driven by
# the COSMOS3_NUM_STEPS env (set via model-config extra_env) so step count is
# tunable without an image rebuild. Parsed defensively: a bad value degrades to the
# framework default instead of crashing the worker at import. Applied to video only
# (see the payload builders) — image modes keep their own framework default (e.g.
# text2image defaults to 50, unvalidated at a lower count).
def _parse_default_num_steps() -> int | None:
    raw = os.environ.get("COSMOS3_NUM_STEPS")
    if not raw:
        return None
    try:
        n = int(raw)
    except ValueError:
        log.warning("ignoring non-integer COSMOS3_NUM_STEPS=%r", raw)
        return None
    if n <= 0:
        log.warning("ignoring non-positive COSMOS3_NUM_STEPS=%r", raw)
        return None
    return n


_default_num_steps: int | None = _parse_default_num_steps()


def _run_generate(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Run one generation collectively on all ranks.

    Returns the response dict on rank 0; ``None`` on other ranks.
    """
    assert _pipe is not None
    rank_dir = _output_root / payload["job_id"] / f"rank{get_rank()}"
    rank_dir.mkdir(parents=True, exist_ok=True)
    try:
        vision_url = payload["vision_url"]
        if vision_url and vision_url.startswith("data:"):
            vision_url = _decode_data_uri(vision_url, rank_dir / "inputs")

        override_kwargs: dict[str, Any] = {
            "name": "sample",
            "output_dir": rank_dir,
            "prompt": payload["prompt"],
            "num_frames": payload["num_frames"],
            "resolution": payload["resolution"],
            "fps": payload["fps"],
            "seed": payload["seed"],
            "vision_path": vision_url,
        }
        if payload.get("aspect_ratio") is not None:
            override_kwargs["aspect_ratio"] = payload["aspect_ratio"]
        # Action modality fields (only set when present).
        model_mode = payload.get("model_mode")
        if model_mode is not None:
            override_kwargs["model_mode"] = model_mode
        if payload.get("action_url") is not None:
            override_kwargs["action_path"] = payload["action_url"]
        for key in ("domain_name", "action_chunk_size", "raw_action_dim", "image_size"):
            if payload.get(key) is not None:
                override_kwargs[key] = payload[key]
        if payload.get("num_steps") is not None:
            override_kwargs["num_steps"] = payload["num_steps"]

        overrides = OmniSampleOverrides(**override_kwargs)
        overrides.download(rank_dir / "inputs")
        sample_args = overrides.build_sample(model_config=_pipe.model_config)
        results = _pipe.generate([sample_args])

        if not is_rank0():
            return None
        if not results:
            raise RuntimeError("No output produced")
        result = results[0]
        if result.status != "success":
            raise RuntimeError(result.message or f"Generation {result.status}")

        # Extract action output if present (inverse_dynamics, policy modes).
        action_output: list | None = None
        for out in result.outputs:
            if "action" in out.content:
                action_output = out.content["action"]
                break

        files = [f for out in result.outputs for f in out.files]
        if not files:
            raise RuntimeError("Output produced no media file")
        output_file = Path(files[0])
        if not output_file.is_absolute():
            output_file = rank_dir / output_file

        b64 = base64.b64encode(output_file.read_bytes()).decode("ascii")
        media_type = (
            "image"
            if output_file.suffix.lower() in (".jpg", ".jpeg", ".png")
            else "video"
        )
        resp: dict[str, Any] = {
            "b64_json": b64,
            "seed": payload["seed"],
            "media_type": media_type,
        }
        if action_output is not None:
            resp["action"] = action_output
        return resp
    finally:
        shutil.rmtree(rank_dir, ignore_errors=True)


def _generate_blocking(request: Cosmos3VideoRequest) -> Cosmos3VideoResponse:
    """Thread-safe synchronous generation. Called from async context via to_thread."""
    num_frames, resolution, aspect_ratio, fps, seed_val, vision_url = request.resolve()
    seed = seed_val if seed_val is not None else random.randint(0, 2**31 - 1)
    payload: dict[str, Any] = {
        "job_id": uuid.uuid4().hex,
        "prompt": request.prompt,
        "num_frames": num_frames,
        "resolution": _normalize_resolution(resolution),
        "fps": fps,
        "seed": seed,
        "vision_url": vision_url,
    }
    if aspect_ratio is not None:
        payload["aspect_ratio"] = aspect_ratio
    ns = None
    if request.nvext is not None and request.nvext.num_steps is not None:
        ns = request.nvext.num_steps
    elif num_frames > 1:  # video only; image modes keep their framework default
        ns = _default_num_steps
    if ns is not None:
        payload["num_steps"] = ns
    # Forward action fields when present.
    if request.model_mode is not None:
        payload["model_mode"] = request.model_mode
    if request.action_url is not None:
        payload["action_url"] = request.action_url
    for key in ("domain_name", "action_chunk_size", "raw_action_dim", "image_size"):
        val = getattr(request, key)
        if val is not None:
            payload[key] = val

    with _generate_lock:
        distributed.broadcast_object(payload, src=0, group=_CONTROL_PG)
        result = _run_generate(payload)
    if result is None:
        raise RuntimeError("rank 0 got None result")
    media_type = result.get("media_type", "video")
    output_format = "jpeg" if media_type == "image" else "mp4"
    return Cosmos3VideoResponse(
        model=_served_model_name,
        data=[
            Cosmos3VideoDataItem(
                output_format=output_format,
                b64_json=result["b64_json"],
                seed=result["seed"],
                media_type=media_type,
                action=result.get("action"),
            )
        ],
    )


# ---------------------------------------------------------------------------
# Multi-GPU: non-rank-0 worker loop
# ---------------------------------------------------------------------------


def _worker_rank_loop() -> None:
    """Non-zero ranks: wait for broadcasts from rank 0 and join each generation."""
    torch.cuda.set_device(get_local_rank())
    while True:
        # Blocks until rank 0 dispatches the next request. Runs on the gloo
        # control group: an idle wait here may legitimately last hours.
        payload = distributed.broadcast_object(None, src=0, group=_CONTROL_PG)
        if payload is None:
            continue
        try:
            _run_generate(payload)
        except Exception as e:  # noqa: BLE001
            log.error(f"rank {get_rank()} generation error: {e}")


# ---------------------------------------------------------------------------
# Mode A: Dynamo backend (production)
# ---------------------------------------------------------------------------


def _get_worker_namespace() -> str:
    """Resolve Dynamo namespace for endpoint registration."""
    namespace = os.environ.get("DYN_NAMESPACE", "dynamo")
    suffix = os.environ.get("DYN_NAMESPACE_WORKER_SUFFIX")
    if suffix:
        namespace = f"{namespace}-{suffix}"
    return namespace


async def _register_model(endpoint, served_name: str, model_path: str) -> None:
    from dynamo.llm import (  # type: ignore[attr-defined]
        ModelInput,
        ModelType,
        register_llm,
    )

    try:
        await register_llm(
            ModelInput.Text,
            ModelType.Videos,
            endpoint,
            model_path,
            served_name,
        )
        log.info(f"Registered model with Dynamo: {served_name} (path={model_path})")
    except Exception as e:
        log.error(f"Failed to register model with Dynamo: {e}")
        raise RuntimeError("Model registration failed") from e


async def _dynamo_main(args: argparse.Namespace) -> None:
    """Dynamo backend main loop (rank 0 only)."""
    from dynamo.runtime import DistributedRuntime, dynamo_endpoint

    loop = asyncio.get_running_loop()
    discovery_backend = os.environ.get("DYN_DISCOVERY_BACKEND")
    if not discovery_backend:
        discovery_backend = (
            "kubernetes" if os.environ.get("KUBERNETES_SERVICE_HOST") else "file"
        )
    namespace_name = _get_worker_namespace()
    log.info(f"Dynamo discovery backend: {discovery_backend}")
    log.info(f"Dynamo namespace: {namespace_name}")

    # DistributedRuntime(loop, discovery_backend, request_plane, enable_nats)
    # enable_nats=False: DeepInfra clusters use ZMQ, not NATS.
    runtime = DistributedRuntime(loop, discovery_backend, "tcp", False)

    component_name = "backend"
    endpoint_name = "generate"
    endpoint = runtime.endpoint(f"{namespace_name}.{component_name}.{endpoint_name}")
    log.info(
        f"Serving Dynamo endpoint: {namespace_name}/{component_name}/{endpoint_name}"
    )

    # Decorate the generate function with dynamo_endpoint for serialization.
    @dynamo_endpoint(Cosmos3VideoRequest, Cosmos3VideoResponse)
    async def create_video(request: Cosmos3VideoRequest):
        if request.model is not None and request.model != _served_model_name:
            raise ValueError(
                f"Model {request.model!r} not found. This worker serves {_served_model_name!r}."
            )
        result = await asyncio.to_thread(_generate_blocking, request)
        yield result.model_dump()

    # Log system port status for observability.
    _system_port = os.environ.get("DYN_SYSTEM_PORT", "-1")
    try:
        _system_port_int = int(_system_port)
    except ValueError:
        _system_port_int = -1
    if _system_port_int < 0:
        log.warning(
            f"DYN_SYSTEM_PORT not set (or is {_system_port!r}); metrics will NOT be exposed on a system port."
        )
    else:
        log.info(f"Dynamo system port enabled: {_system_port_int}")

    model_path = args.model
    await asyncio.gather(
        endpoint.serve_endpoint(create_video),
        _register_model(endpoint, _served_model_name, model_path),
    )


# ---------------------------------------------------------------------------
# Mode B: Standalone HTTP server (local testing)
# ---------------------------------------------------------------------------


def _run_standalone_http(args: argparse.Namespace) -> None:
    """Run a plain FastAPI HTTP server (no Dynamo frontend needed)."""
    import queue

    import uvicorn
    from fastapi import FastAPI, HTTPException

    request_queue: "queue.Queue[dict[str, Any]]" = queue.Queue()
    app = FastAPI(title="Cosmos 3 inference worker (standalone)")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/videos")
    def create_video(req: Cosmos3VideoRequest) -> dict:
        if req.model is not None and req.model != _served_model_name:
            raise HTTPException(
                status_code=404,
                detail=f"Model {req.model!r} not found. This worker serves {_served_model_name!r}.",
            )
        num_frames, resolution, aspect_ratio, fps, seed_val, vision_url = req.resolve()
        seed = seed_val if seed_val is not None else random.randint(0, 2**31 - 1)
        payload = {
            "job_id": uuid.uuid4().hex,
            "prompt": req.prompt,
            "num_frames": num_frames,
            "resolution": _normalize_resolution(resolution),
            "fps": fps,
            "seed": seed,
            "vision_url": vision_url,
        }
        if aspect_ratio is not None:
            payload["aspect_ratio"] = aspect_ratio
        ns = None
        if req.nvext is not None and req.nvext.num_steps is not None:
            ns = req.nvext.num_steps
        elif num_frames > 1:  # video only; image modes keep their framework default
            ns = _default_num_steps
        if ns is not None:
            payload["num_steps"] = ns
        # Forward action fields when present.
        if req.model_mode is not None:
            payload["model_mode"] = req.model_mode
        if req.action_url is not None:
            payload["action_url"] = req.action_url
        for key in ("domain_name", "action_chunk_size", "raw_action_dim", "image_size"):
            val = getattr(req, key)
            if val is not None:
                payload[key] = val
        job: dict[str, Any] = {
            "payload": payload,
            "event": threading.Event(),
            "result": None,
            "error": None,
        }
        request_queue.put(job)
        job["event"].wait()

        if job["error"] is not None:
            raise HTTPException(status_code=400, detail=str(job["error"]))
        return job["result"]

    def inference_thread() -> None:
        torch.cuda.set_device(get_local_rank())
        while True:
            job = request_queue.get()
            try:
                distributed.broadcast_object(job["payload"], src=0, group=_CONTROL_PG)
                result = _run_generate(job["payload"])
                media_type = result.get("media_type", "video") if result else "video"
                output_format = "jpeg" if media_type == "image" else "mp4"
                item = {
                    "output_format": output_format,
                    "b64_json": result["b64_json"],
                    "seed": result["seed"],
                    "media_type": media_type,
                }
                if result.get("action") is not None:
                    item["action"] = result["action"]
                job["result"] = Cosmos3VideoResponse(
                    model=_served_model_name,
                    data=[Cosmos3VideoDataItem(**item)],
                ).model_dump()
            except Exception as e:  # noqa: BLE001
                log.error(f"Generation failed: {e}")
                job["error"] = e
            finally:
                job["event"].set()

    threading.Thread(target=inference_thread, daemon=True).start()
    log.success(f"Starting standalone HTTP server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cosmos 3 Dynamo videogen worker")
    parser.add_argument(
        "--served-model-name",
        required=True,
        help="Public model name clients request, e.g. nvidia/Cosmos3-Nano",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Checkpoint path or registered name, e.g. Cosmos3-Nano or /data/default",
    )
    parser.add_argument(
        "--num-gpus", type=int, default=1, help="Number of GPUs for this worker"
    )
    parser.add_argument(
        "--standalone",
        action="store_true",
        help="Run as standalone HTTP server (no Dynamo frontend required)",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host (standalone mode only)")
    parser.add_argument(
        "--port", type=int, default=80, help="Port (standalone mode only)"
    )
    return parser.parse_args()


def _relaunch_under_torchrun(num_gpus: int) -> None:
    """Re-exec under torchrun for multi-GPU collective inference."""
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={num_gpus}",
        os.path.abspath(__file__),
        *sys.argv[1:],
    ]
    log.info(f"Relaunching under torchrun for {num_gpus} GPUs: {' '.join(cmd)}")
    os.execvp(cmd[0], cmd)


def main() -> None:
    global _pipe, _served_model_name, _output_root

    args = parse_args()

    # Multi-GPU: relaunch under torchrun unless already a torchrun child.
    if args.num_gpus > 1 and "RANK" not in os.environ:
        _relaunch_under_torchrun(args.num_gpus)

    _served_model_name = args.served_model_name
    _output_root = Path(
        os.environ.get("COSMOS3_OUTPUT_DIR", "/tmp/cosmos3_worker_outputs")
    ).absolute()
    init_output_dir(_output_root)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    log.info(
        f"Loading Cosmos 3 model {_served_model_name!r} from checkpoint {args.model!r} (world_size={world_size})..."
    )
    setup_args = OmniSetupOverrides(
        checkpoint_path=args.model,
        output_dir=_output_root,
        # Content guardrails disabled. The bundled Qwen3Guard text guard is unfit:
        # a single coarse "Sexual Content" category (no CSAM/minor distinction),
        # demonstrable demographic bias (blocks benign prompts by subject race),
        # and it runs the prompt check AFTER generation so every rejection wastes
        # a full clip. Content filtering will be handled by a platform-level
        # video-gen guardrail applied uniformly across models, not per-model here.
        guardrails=False,
    ).build_setup(world_size=world_size)
    _pipe = OmniInference.create(setup_args)
    log.success("Model loaded.")

    if not is_rank0():
        _worker_rank_loop()
        return

    # Rank 0: serve requests.
    if args.standalone:
        _run_standalone_http(args)
    else:
        import uvloop

        log.success("Starting Dynamo backend...")
        uvloop.install()
        asyncio.run(_dynamo_main(args))


if __name__ == "__main__":
    main()
