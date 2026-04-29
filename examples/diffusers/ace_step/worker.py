#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
ACE-Step Worker for Dynamo (non-streaming)

Registers an ACE-Step music generation pipeline as a Dynamo backend endpoint.
The endpoint generates a single music clip from the request parameters and
returns it as a base64-encoded audio file in data[0].b64_json.

ACE-Step is a two-stage pipeline:
  1. LM planner (vLLM-backed) turns the user query into a song blueprint
     (caption metadata + lyrics with timestamps).
  2. Diffusion Transformer + VAE synthesizes the audio from the blueprint.

Both stages run inside this worker process — there is no separate decoupling
into vLLM-Omni at the platform level. This mirrors the upstream
ACEStepHandler / LLMHandler API.

One request at a time (asyncio.Lock — handlers are not re-entrant).

Usage:
  python worker.py [--model MODEL_NAME] [--dit-config DIT] [--lm-model LM]
                   [--checkpoint-dir DIR] [--num-gpus N]

Defaults target the largest "best quality" tier: v1.5 XL DiT + 4B LM.

Request format (NvCreateMusicRequest):
  prompt:   caption / text description of the desired music
  model:    model name registered with Dynamo
  lyrics:   vocal text, or '[Instrumental]' to suppress vocals
  duration: clip duration in seconds (-1 for model-chosen)
  nvext:
    bpm:                tempo (30-300)
    keyscale:           musical key, e.g. 'C major'
    num_inference_steps DiT denoising steps
    guidance_scale:     CFG scale
    seed:               RNG seed (None / -1 for random)
    thinking:           enable LM planner reasoning (default True)
    batch_size:         clips per request (default 1)
    ref_audio:          reference audio for style (URL or base64)
    negative_prompt:    text to avoid
"""

import argparse
import asyncio
import base64
import logging
import os
import time
import uuid

import uvloop

from dynamo.common.protocols.music_protocol import (
    MusicData,
    NvCreateMusicRequest,
    NvMusicResponse,
)
from dynamo.llm import ModelInput, ModelType, register_llm  # type: ignore[attr-defined]
from dynamo.runtime import DistributedRuntime, dynamo_endpoint

logger = logging.getLogger(__name__)

# Largest / best-quality tier. Override via CLI args at deploy time.
DEFAULT_MODEL_NAME = "ACE-Step/acestep-v15-xl"
DEFAULT_DIT_CONFIG = "acestep-v15-xl-sft"
DEFAULT_LM_MODEL = "acestep-5Hz-lm-4B"
DEFAULT_LM_BACKEND = "vllm"
DEFAULT_AUDIO_FORMAT = "flac"


def _get_worker_namespace() -> str:
    """Resolve Dynamo namespace for endpoint registration.

    Kubernetes operator injects DYN_NAMESPACE (and optionally a rollout suffix).
    Local / compose runs keep the historical "dynamo" default.
    """
    namespace = os.environ.get("DYN_NAMESPACE", "dynamo")
    suffix = os.environ.get("DYN_NAMESPACE_WORKER_SUFFIX")
    if suffix:
        namespace = f"{namespace}-{suffix}"
    return namespace


# ── Backend ───────────────────────────────────────────────────────────────────


class AceStepBackend:
    def __init__(self, args: argparse.Namespace) -> None:
        self.model_name: str = args.model
        self.dit_config: str = args.dit_config
        self.lm_model: str = args.lm_model
        self.lm_backend: str = args.lm_backend
        self.checkpoint_dir: str = args.checkpoint_dir
        self.project_root: str = args.project_root
        self.num_gpus: int = args.num_gpus
        self.device: str = args.device

        # One request at a time — upstream handlers are not re-entrant.
        self._generate_lock = asyncio.Lock()
        self.dit_handler = None
        self.lm_handler = None

    async def initialize_model(self) -> None:
        logger.info(
            "Loading ACE-Step pipeline (dit=%s lm=%s lm_backend=%s)",
            self.dit_config,
            self.lm_model,
            self.lm_backend,
        )
        loop = asyncio.get_running_loop()

        def _load():
            from acestep.handler import AceStepHandler
            from acestep.llm_inference import LLMHandler

            dit = AceStepHandler()
            dit.initialize_service(
                project_root=self.project_root,
                config_path=self.dit_config,
                device=self.device,
            )

            lm = LLMHandler()
            lm.initialize(
                checkpoint_dir=self.checkpoint_dir,
                lm_model_path=self.lm_model,
                backend=self.lm_backend,
                device=self.device,
            )
            return dit, lm

        self.dit_handler, self.lm_handler = await loop.run_in_executor(None, _load)
        logger.info("ACE-Step pipeline ready")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _generate_clip(
        self,
        request_id: str,
        prompt: str,
        lyrics: str,
        duration: float,
        bpm: int | None,
        keyscale: str | None,
        num_inference_steps: int | None,
        guidance_scale: float | None,
        seed: int | None,
        thinking: bool,
        batch_size: int,
        negative_prompt: str | None,
        audio_format: str,
    ) -> tuple[bytes, int, float, int | None, str | None]:
        """Run the upstream pipeline and return (audio_bytes, sample_rate,
        duration_s, seed_used, lrc)."""
        assert self.dit_handler is not None and self.lm_handler is not None
        from acestep.inference import (
            GenerationConfig,
            GenerationParams,
            generate_music,
        )

        params_kwargs: dict = {
            "caption": prompt,
            "lyrics": lyrics,
            "duration": duration,
            "thinking": thinking,
        }
        if bpm is not None:
            params_kwargs["bpm"] = bpm
        if keyscale is not None:
            params_kwargs["keyscale"] = keyscale
        if num_inference_steps is not None:
            params_kwargs["inference_steps"] = num_inference_steps
        if guidance_scale is not None:
            params_kwargs["guidance_scale"] = guidance_scale
        if seed is not None:
            params_kwargs["seed"] = seed
        if negative_prompt is not None:
            params_kwargs["negative_prompt"] = negative_prompt

        params = GenerationParams(**params_kwargs)
        config = GenerationConfig(
            batch_size=batch_size,
            audio_format=audio_format,
            use_random_seed=(seed is None or seed < 0),
            seeds=[seed] if seed is not None and seed >= 0 else None,
        )

        result = generate_music(
            dit_handler=self.dit_handler,
            llm_handler=self.lm_handler,
            params=params,
            config=config,
            save_dir=None,
        )

        if not getattr(result, "success", False):
            raise RuntimeError(
                f"[{request_id}] ACE-Step generation failed: "
                f"{getattr(result, 'error', 'unknown error')}"
            )

        audios = getattr(result, "audios", None) or []
        if not audios:
            raise RuntimeError(f"[{request_id}] ACE-Step returned no audio clips")

        clip = audios[0]
        path = clip.get("path")
        sample_rate = int(clip.get("sample_rate") or 0)
        clip_seed = clip.get("params", {}).get("seed")
        lrc = (
            (getattr(result, "extra_outputs", None) or {})
            .get("lm_metadata", {})
            .get("lrc")
        )

        if not path or not os.path.exists(path):
            raise RuntimeError(
                f"[{request_id}] ACE-Step result path missing or unreadable: {path!r}"
            )

        with open(path, "rb") as f:
            audio_bytes = f.read()

        # Best-effort duration; librosa is optional at runtime so we fall back
        # to the requested duration when unavailable.
        duration_s: float = float(duration) if duration and duration > 0 else 0.0
        try:
            import io

            import soundfile as sf  # lightweight; pulled in by acestep

            with sf.SoundFile(io.BytesIO(audio_bytes)) as sfh:
                duration_s = sfh.frames / float(sfh.samplerate)
                if not sample_rate:
                    sample_rate = sfh.samplerate
        except Exception:  # noqa: BLE001 — duration is informational
            pass

        return audio_bytes, sample_rate, duration_s, clip_seed, lrc

    # ── Dynamo endpoint ───────────────────────────────────────────────────────

    @dynamo_endpoint(NvCreateMusicRequest, NvMusicResponse)
    async def create_music(self, request: NvCreateMusicRequest):
        """Non-streaming endpoint.

        Generates one music clip and yields a single NvMusicResponse with
        data[0].b64_json containing the encoded audio.
        """
        if self.dit_handler is None or self.lm_handler is None:
            raise RuntimeError("ACE-Step handlers are not initialized")

        nvext = request.nvext
        bpm = nvext.bpm if nvext else None
        keyscale = nvext.keyscale if nvext else None
        num_inference_steps = nvext.num_inference_steps if nvext else None
        guidance_scale = nvext.guidance_scale if nvext else None
        seed = nvext.seed if nvext else None
        thinking = bool(nvext.thinking) if nvext and nvext.thinking is not None else True
        batch_size = nvext.batch_size if nvext and nvext.batch_size else 1
        negative_prompt = nvext.negative_prompt if nvext else None

        lyrics = request.lyrics or "[Instrumental]"
        duration = request.duration if request.duration is not None else -1.0
        audio_format = request.response_format or DEFAULT_AUDIO_FORMAT

        request_id = f"music_{uuid.uuid4().hex}"
        created_ts = int(time.time())

        logger.info(
            "[%s] create_music: prompt='%s...' duration=%s bpm=%s key=%s steps=%s",
            request_id,
            request.prompt[:60],
            duration,
            bpm,
            keyscale,
            num_inference_steps,
        )
        logger.info(
            "[%s] Waiting for generate lock (locked=%s)",
            request_id,
            self._generate_lock.locked(),
        )

        async with self._generate_lock:
            t = time.perf_counter()
            try:
                (
                    audio_bytes,
                    sample_rate,
                    duration_s,
                    clip_seed,
                    lrc,
                ) = await asyncio.to_thread(
                    self._generate_clip,
                    request_id=request_id,
                    prompt=request.prompt,
                    lyrics=lyrics,
                    duration=duration,
                    bpm=bpm,
                    keyscale=keyscale,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    seed=seed,
                    thinking=thinking,
                    batch_size=batch_size,
                    negative_prompt=negative_prompt,
                    audio_format=audio_format,
                )
            except Exception as exc:
                logger.exception("[%s] Generation failed", request_id)
                raise RuntimeError(
                    f"Music generation failed for request {request_id}"
                ) from exc

            elapsed = time.perf_counter() - t
            logger.info(
                "[%s] Generation done in %.1fs — encoding %.2f MB %s",
                request_id,
                elapsed,
                len(audio_bytes) / 1_048_576,
                audio_format,
            )

            yield NvMusicResponse(
                id=request_id,
                created=created_ts,
                model=request.model,
                inference_time_s=elapsed,
                data=[
                    MusicData(
                        b64_json=base64.b64encode(audio_bytes).decode(),
                        mime_type=f"audio/{audio_format}",
                        sample_rate=sample_rate or None,
                        duration_s=duration_s or None,
                        seed=clip_seed,
                        lrc=lrc,
                    )
                ],
            ).model_dump()
        logger.info("[%s] Generation request finished", request_id)


# ── Dynamo wiring ─────────────────────────────────────────────────────────────


async def _register_model(endpoint, model_name: str) -> None:
    try:
        await register_llm(
            ModelInput.Text,  # type: ignore[attr-defined]
            ModelType.Audios,
            endpoint,
            model_name,
            model_name,
        )
        logger.info("Successfully registered model: %s", model_name)
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

    backend = AceStepBackend(args)
    await backend.initialize_model()

    await asyncio.gather(
        endpoint.serve_endpoint(backend.create_music),  # type: ignore[arg-type]
        _register_model(endpoint, backend.model_name),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ACE-Step Music Generation Worker for Dynamo (non-streaming)"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help=(
            "Model name to register with Dynamo (clients pass this as "
            f"`model` in the request body). Default: {DEFAULT_MODEL_NAME}"
        ),
    )
    parser.add_argument(
        "--dit-config",
        default=DEFAULT_DIT_CONFIG,
        dest="dit_config",
        help=(
            "ACE-Step DiT config name passed to "
            "AceStepHandler.initialize_service. "
            f"Default: {DEFAULT_DIT_CONFIG} (4B XL — best quality tier)"
        ),
    )
    parser.add_argument(
        "--lm-model",
        default=DEFAULT_LM_MODEL,
        dest="lm_model",
        help=(
            "ACE-Step LM model path passed to LLMHandler.initialize. "
            f"Default: {DEFAULT_LM_MODEL}"
        ),
    )
    parser.add_argument(
        "--lm-backend",
        default=DEFAULT_LM_BACKEND,
        dest="lm_backend",
        choices=("vllm", "transformers"),
        help=f"LM inference backend (default: {DEFAULT_LM_BACKEND})",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=os.environ.get("ACESTEP_CHECKPOINT_DIR", "/models/acestep"),
        dest="checkpoint_dir",
        help="Directory containing ACE-Step LM checkpoints (default: $ACESTEP_CHECKPOINT_DIR or /models/acestep)",
    )
    parser.add_argument(
        "--project-root",
        default=os.environ.get("ACESTEP_PROJECT_ROOT", "/opt/ACE-Step-1.5"),
        dest="project_root",
        help="ACE-Step source checkout root (default: $ACESTEP_PROJECT_ROOT or /opt/ACE-Step-1.5)",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        dest="num_gpus",
        help="Number of GPUs (default: 1)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device (default: cuda)",
    )
    return parser.parse_args()


async def main(args: argparse.Namespace) -> None:
    loop = asyncio.get_running_loop()
    discovery_backend = os.environ.get("DYN_DISCOVERY_BACKEND")
    if not discovery_backend:
        discovery_backend = (
            "kubernetes" if os.environ.get("KUBERNETES_SERVICE_HOST") else "file"
        )
    logger.info("Using discovery backend: %s", discovery_backend)
    logger.info("Resolved worker namespace: %s", _get_worker_namespace())
    runtime = DistributedRuntime(loop, discovery_backend, "tcp")
    await backend_worker(runtime, args)


if __name__ == "__main__":
    _args = _parse_args()
    logging.basicConfig(
        level=(
            logging.DEBUG
            if os.environ.get("ACESTEP_LOG_LEVEL") == "DEBUG"
            else logging.INFO
        ),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )
    uvloop.install()
    asyncio.run(main(_args))
