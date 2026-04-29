#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Benchmark a baked ACE-Step ship image: generate N music clips against a
small prompt set, measure end-to-end wall time per request, save audio
files for inspection.

Mirrors examples/diffusers/benchmark.py (fastvideo) — designed to run
inside the baked container WITHOUT bind-mounting an external cache, so
the test exercises whatever the image actually ships. Loads the
ACE-Step pipeline ONCE and reuses it across generations, mirroring what
a production worker sees: one model load, many sequential requests.

The script imports `acestep` directly — NO Dynamo runtime, NO HTTP
frontend. That matches the team's engineering loop for fastvideo, where
the dynamo wiring is validated separately as a future production step.

Usage (inside the container):
  python3 benchmark.py \\
    --gpu-uuid GPU-xxxx-yyyy-zzzz \\
    --output-dir /tmp/benchmark-outputs \\
    --csv /tmp/benchmark-outputs/timings.csv

Output:
  /tmp/benchmark-outputs/<prompt-id>_<seed>.flac   (one per generation)
  /tmp/benchmark-outputs/timings.csv               (one row per generation)
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

# Small fixed prompt set; deterministic seeds so reruns are comparable.
PROMPTS = [
    (
        "lofi",
        "upbeat lo-fi hip hop beat with mellow piano, warm vinyl crackle, "
        "and a relaxed jazz drum loop",
        "[Instrumental]",
    ),
    (
        "synthwave",
        "energetic 1980s synthwave with arpeggiated leads, gated reverb "
        "drums, and a driving four-on-the-floor bass",
        "[Instrumental]",
    ),
    (
        "cinematic",
        "uplifting cinematic strings with brass swells and timpani, "
        "orchestral, slow build, hopeful",
        "[Instrumental]",
    ),
]

DEFAULT_DURATION_S = 15.0
DEFAULT_SEED = 42


def _select_gpu(gpu_uuid: str) -> None:
    """Pin to the GPU identified by `gpu_uuid` and fail fast if absent.

    Sets CUDA_VISIBLE_DEVICES *before* torch is imported so torch only
    sees this one device. Even if the container was started with
    `docker --gpus device=<UUID>` already, this still wins (and gets
    logged) — same pattern as the fastvideo benchmark.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_uuid

    import torch  # imported here so CUDA_VISIBLE_DEVICES is honoured

    if not torch.cuda.is_available():
        print(
            f"[benchmark] FATAL: no CUDA device visible. CUDA_VISIBLE_DEVICES="
            f"{gpu_uuid!r}. Check that the UUID is valid and that the "
            f"container was started with the matching `docker --gpus` flag.",
            file=sys.stderr,
        )
        sys.exit(2)

    visible = torch.cuda.device_count()
    if visible != 1:
        print(
            f"[benchmark] WARNING: {visible} CUDA devices visible despite "
            f"--gpu-uuid pinning. Will still use device 0.",
            flush=True,
        )

    name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    cc = (props.major, props.minor)
    mem_gb = props.total_memory / 1e9
    print(
        f"[benchmark] using GPU 0: {name} (sm_{cc[0]}{cc[1]}, "
        f"{mem_gb:.0f} GiB). CUDA_VISIBLE_DEVICES={gpu_uuid}",
        flush=True,
    )


def _load_pipeline(
    dit_config: str,
    lm_model: str,
    lm_backend: str,
    checkpoint_dir: str,
    project_root: str,
    device: str,
):
    """Load the ACE-Step DiT + LM handlers once."""
    from acestep.handler import AceStepHandler
    from acestep.llm_inference import LLMHandler

    print(
        f"[benchmark] loading ACE-Step (dit={dit_config} lm={lm_model} "
        f"lm_backend={lm_backend})",
        flush=True,
    )
    t0 = time.perf_counter()

    dit = AceStepHandler()
    dit.initialize_service(
        project_root=project_root,
        config_path=dit_config,
        device=device,
    )

    lm = LLMHandler()
    lm.initialize(
        checkpoint_dir=checkpoint_dir,
        lm_model_path=lm_model,
        backend=lm_backend,
        device=device,
    )

    elapsed = time.perf_counter() - t0
    print(f"[benchmark] pipeline ready in {elapsed:.1f}s", flush=True)
    return dit, lm


def _generate_one(
    dit_handler,
    lm_handler,
    prompt_id: str,
    caption: str,
    lyrics: str,
    duration: float,
    seed: int,
    output_dir: Path,
    audio_format: str,
) -> tuple[Path, float, int, float]:
    """Run one generation. Returns (output_path, sample_rate, seed_used, gen_time_s)."""
    from acestep.inference import (
        GenerationConfig,
        GenerationParams,
        generate_music,
    )

    params = GenerationParams(
        caption=caption,
        lyrics=lyrics,
        duration=duration,
        seed=seed,
        thinking=True,
    )
    config = GenerationConfig(
        batch_size=1,
        audio_format=audio_format,
        use_random_seed=False,
        seeds=[seed],
    )

    t0 = time.perf_counter()
    result = generate_music(
        dit_handler=dit_handler,
        llm_handler=lm_handler,
        params=params,
        config=config,
        save_dir=str(output_dir),
    )
    elapsed = time.perf_counter() - t0

    if not getattr(result, "success", False):
        raise RuntimeError(
            f"[{prompt_id}] generation failed: "
            f"{getattr(result, 'error', 'unknown error')}"
        )

    audios = getattr(result, "audios", None) or []
    if not audios:
        raise RuntimeError(f"[{prompt_id}] returned no audio")

    clip = audios[0]
    src_path = Path(clip["path"])
    sample_rate = int(clip.get("sample_rate") or 0)
    seed_used = int(clip.get("params", {}).get("seed", seed))

    # Normalize filename so reruns don't accumulate randomly-named files.
    dst = output_dir / f"{prompt_id}_seed{seed_used}.{audio_format}"
    if src_path.resolve() != dst.resolve():
        dst.write_bytes(src_path.read_bytes())

    return dst, sample_rate, seed_used, elapsed


def _validate_audio(path: Path, requested_duration: float) -> tuple[float, float, int]:
    """Decode the audio and return (duration_s, peak_amplitude, sample_rate).
    Asserts non-silent and within ±2s of requested duration.
    """
    import soundfile as sf

    with sf.SoundFile(str(path)) as fh:
        sample_rate = fh.samplerate
        frames = fh.read(dtype="float32", always_2d=False)
    duration_s = len(frames) / float(sample_rate) if sample_rate else 0.0
    peak = float(abs(frames).max()) if len(frames) else 0.0

    if peak <= 0.001:
        raise AssertionError(f"{path}: audio appears silent (peak={peak:.6f})")
    if requested_duration > 0 and abs(duration_s - requested_duration) > 2.0:
        raise AssertionError(
            f"{path}: duration {duration_s:.2f}s outside ±2s of "
            f"requested {requested_duration}s"
        )
    return duration_s, peak, sample_rate


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark a baked ACE-Step ship image."
    )
    parser.add_argument(
        "--gpu-uuid",
        required=True,
        dest="gpu_uuid",
        help="GPU UUID to pin (find with `nvidia-smi -L`).",
    )
    parser.add_argument(
        "--output-dir",
        default="/tmp/benchmark-outputs",
        dest="output_dir",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Path for timings CSV (default: <output-dir>/timings.csv).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION_S,
        help=f"Clip duration in seconds (default: {DEFAULT_DURATION_S}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"RNG seed (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--audio-format",
        default="flac",
        dest="audio_format",
        choices=("flac", "wav", "mp3"),
    )
    # ACE-Step pipeline knobs (defaults match the worker scaffold).
    parser.add_argument("--dit-config", default="acestep-v15-xl-sft", dest="dit_config")
    parser.add_argument("--lm-model", default="acestep-5Hz-lm-4B", dest="lm_model")
    parser.add_argument(
        "--lm-backend", default="vllm", dest="lm_backend", choices=("vllm", "transformers")
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=os.environ.get("ACESTEP_CHECKPOINT_DIR", "/models/acestep"),
        dest="checkpoint_dir",
    )
    parser.add_argument(
        "--project-root",
        default=os.environ.get("ACESTEP_PROJECT_ROOT", "/opt/ACE-Step-1.5"),
        dest="project_root",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.csv) if args.csv else output_dir / "timings.csv"

    _select_gpu(args.gpu_uuid)

    dit_handler, lm_handler = _load_pipeline(
        dit_config=args.dit_config,
        lm_model=args.lm_model,
        lm_backend=args.lm_backend,
        checkpoint_dir=args.checkpoint_dir,
        project_root=args.project_root,
        device=args.device,
    )

    rows = []
    failed = 0
    for prompt_id, caption, lyrics in PROMPTS:
        print(f"[benchmark] -> {prompt_id}: {caption[:60]}...", flush=True)
        try:
            audio_path, sr, seed_used, gen_s = _generate_one(
                dit_handler=dit_handler,
                lm_handler=lm_handler,
                prompt_id=prompt_id,
                caption=caption,
                lyrics=lyrics,
                duration=args.duration,
                seed=args.seed,
                output_dir=output_dir,
                audio_format=args.audio_format,
            )
            duration_s, peak, sr_decoded = _validate_audio(audio_path, args.duration)
            print(
                f"[benchmark]   ok: {audio_path.name} {duration_s:.2f}s "
                f"@ {sr_decoded} Hz peak={peak:.3f} gen={gen_s:.1f}s",
                flush=True,
            )
            rows.append(
                {
                    "prompt_id": prompt_id,
                    "seed": seed_used,
                    "requested_duration_s": args.duration,
                    "actual_duration_s": round(duration_s, 3),
                    "sample_rate_hz": sr_decoded,
                    "peak_amplitude": round(peak, 4),
                    "generation_time_s": round(gen_s, 2),
                    "audio_file": str(audio_path),
                }
            )
        except Exception as exc:  # noqa: BLE001 — one failure shouldn't abort the run
            failed += 1
            print(f"[benchmark]   FAIL: {prompt_id}: {exc}", flush=True)
            rows.append(
                {
                    "prompt_id": prompt_id,
                    "seed": args.seed,
                    "requested_duration_s": args.duration,
                    "actual_duration_s": "",
                    "sample_rate_hz": "",
                    "peak_amplitude": "",
                    "generation_time_s": "",
                    "audio_file": "",
                    "error": str(exc),
                }
            )

    fieldnames = [
        "prompt_id",
        "seed",
        "requested_duration_s",
        "actual_duration_s",
        "sample_rate_hz",
        "peak_amplitude",
        "generation_time_s",
        "audio_file",
        "error",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            r.setdefault("error", "")
            w.writerow(r)

    total = len(PROMPTS)
    ok = total - failed
    print(
        f"[benchmark] done: {ok}/{total} ok, {failed} failed. "
        f"timings: {csv_path}",
        flush=True,
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
