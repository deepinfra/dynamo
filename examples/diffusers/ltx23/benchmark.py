#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Benchmark a baked FastVideo ship image: generate N videos per shape
against a small prompt set, measure end-to-end wall time per request,
save MP4s for quality inspection.

Designed to run inside the baked image WITHOUT bind-mounting the host's
/cache, so the test exercises the cache that's actually shipped in the
image. If a baked artifact is missing, you'll see a 600s+ cold-compile
on the first request for that shape, which is the diagnostic signal.

VideoGenerator is loaded ONCE and reused across all generations -- this
mirrors what a production pod sees (one model load, many sequential
requests).

The script itself is model-agnostic; it loads whatever HF model id you
pass via --model and uses the shapes file you pass via --shapes. The
optimization_kwargs in _load_generator() are tuned for LTX-2.3-style
distilled models (refine + tiling + compile); a different model family
may want to edit those defaults.

Usage (inside the container):
  python3 ltx23/benchmark.py \\
    --shapes ltx23/shapes.json \\
    --output-dir /tmp/benchmark-outputs \\
    --csv /tmp/benchmark-outputs/timings.csv \\
    --gpu-uuid GPU-xxxx-yyyy-zzzz

Output:
  /tmp/benchmark-outputs/<shape>_<prompt-id>.mp4   (one per generation)
  /tmp/benchmark-outputs/timings.csv               (one row per generation)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

PROMPTS = [
    (
        "dog",
        "A close-up tracking shot of a golden retriever sprinting toward the "
        "camera through a sunlit alpine meadow at golden hour, paws kicking up "
        "wildflowers, ears flapping, tongue out in joyful pants, shallow depth "
        "of field, shot on ARRI Alexa 65, 4K, photorealistic",
    ),
    (
        "city",
        "Aerial drone shot over Tokyo at twilight, neon-lit skyscrapers "
        "reflected in rain-slicked streets, cinematic anamorphic lens with "
        "soft bokeh falloff, dramatic lighting, photorealistic",
    ),
    (
        "nature",
        "A waterfall cascading down moss-covered cliffs in a tropical "
        "rainforest, mist catching shafts of golden sunlight, slow-motion, "
        "photorealistic, shot on RED Komodo",
    ),
]


def _ensure_cache_env() -> None:
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/cache/torchinductor")
    os.environ.setdefault("TRITON_CACHE_DIR", "/cache/triton")


def _select_gpu(gpu_uuid: str) -> None:
    """
    Pin the run to the GPU identified by `gpu_uuid` and log which physical
    device that resolves to. Sets CUDA_VISIBLE_DEVICES *before* torch is
    imported so torch only sees this one device. If the environment also
    has the device pinned externally (e.g. via `docker --gpus`), the
    explicit UUID still wins and is recorded in the log.
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


def _load_generator(model: str):
    os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")
    os.environ.setdefault("FASTVIDEO_STAGE_LOGGING", "1")
    os.environ.setdefault("FASTVIDEO_ENABLE_RMSNORM_FP4_PREQUANT", "0")

    import torch  # noqa: F401
    from fastvideo import VideoGenerator
    from fastvideo.configs.pipelines.base import PipelineConfig

    # Put examples/diffusers/ on sys.path so the package import resolves
    # regardless of where the script is invoked from.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ltx23.config import standard_kwargs

    print(f"[benchmark] loading VideoGenerator model={model}", flush=True)
    t0 = time.perf_counter()
    pipeline_config = PipelineConfig.from_pretrained(model)
    generator = VideoGenerator.from_pretrained(
        model,
        num_gpus=1,
        pipeline_config=pipeline_config,
        **standard_kwargs(),
    )
    print(
        f"[benchmark] generator ready in {time.perf_counter() - t0:.1f}s",
        flush=True,
    )
    return generator


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark a baked FastVideo ship image: N shapes x M prompts "
        "generations, with per-generation timings and saved MP4s.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--shapes",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "shapes.json"),
        help="path to the shapes JSON (must match the menu the image was baked for)",
    )
    parser.add_argument(
        "--output-dir",
        default="/tmp/benchmark-outputs",
        help="directory where MP4s will be written; one MP4 per (shape, prompt)",
    )
    parser.add_argument(
        "--csv",
        default="/tmp/benchmark-outputs/timings.csv",
        help="path to the timings CSV; one row per generation",
    )
    parser.add_argument(
        "--model",
        default="FastVideo/LTX-2.3-Distilled-Diffusers",
        help="HuggingFace model identifier to load",
    )
    parser.add_argument(
        "--gpu-uuid",
        required=True,
        help="GPU UUID to pin the run to (e.g. 'GPU-d1062f6e-...'); sets "
        "CUDA_VISIBLE_DEVICES so torch sees only this device, and "
        "is recorded in the script's startup log so the benchmark "
        "output unambiguously identifies which physical GPU it ran on.",
    )
    parser.add_argument(
        "--prompt-major",
        action="store_true",
        help="Iterate prompts in the OUTER loop (every shape gets prompt 1, "
        "then every shape gets prompt 2, etc.). Forces a shape switch "
        "between every two consecutive generations. Useful for measuring "
        "whether torch.compile's in-memory state survives shape revisits "
        "or gets evicted -- mimics production where requests come in "
        "arbitrary shape order. Default (without flag): shape-major, "
        "which measures first-hit-per-shape and same-shape steady state.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Call torch._dynamo.reset() before each generation. Clears "
        "dynamo's accumulated guard / specialization state so each "
        "compile happens in 'fresh process' state, producing "
        "deterministic cache keys regardless of access order. Use this "
        "to test whether the baked compile cache hits when the access "
        "pattern (shape order) differs from how the cache was built. If "
        "every generation lands at ~steady-state with --reset, the "
        "cache is fully usable and we can use the same reset() in "
        "production worker.py for predictable performance.",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=None,
        help="If set, permute shape access order with this seed before the "
        "iteration loop. Tests whether the cache survives customer "
        "access patterns that differ from shapes.json order. "
        "Default (None): no shuffle, matches warmup access order -- "
        "best-case cache hit. Suggested post-bake validation: one pass "
        "without this flag (matched-order baseline) plus one pass with "
        "--shuffle-seed 42 (customer-like permutation); if the shuffled "
        "pass shows any shape with cold-compile latency, the cache is "
        "order-dependent and the bake is partially broken.",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=None,
        help="If set, use only the first N prompts from the hardcoded "
        "PROMPTS list (default: all %d). For validation use 1 -- "
        "sufficient to detect cold-compile latency per shape without "
        "paying for redundant warm-state generations." % len(PROMPTS),
    )
    args = parser.parse_args()

    prompts = PROMPTS
    if args.num_prompts is not None:
        if not 1 <= args.num_prompts <= len(PROMPTS):
            parser.error(
                f"--num-prompts must be in [1, {len(PROMPTS)}], "
                f"got {args.num_prompts}"
            )
        prompts = PROMPTS[: args.num_prompts]
        print(
            f"[benchmark] limited to first {args.num_prompts} of "
            f"{len(PROMPTS)} prompts",
            flush=True,
        )

    _ensure_cache_env()
    _select_gpu(args.gpu_uuid)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.shapes, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    shapes = cfg["shapes"]
    fps = int(cfg.get("fps", 24))
    num_inference_steps = int(cfg.get("num_inference_steps", 5))
    guidance_scale = float(cfg.get("guidance_scale", 1.0))

    if args.shuffle_seed is not None:
        import random

        rng = random.Random(args.shuffle_seed)
        permuted = list(range(len(shapes)))
        rng.shuffle(permuted)
        print(
            f"[benchmark] shuffled shape order with seed={args.shuffle_seed}: "
            f"{permuted}",
            flush=True,
        )
        shapes = [shapes[i] for i in permuted]

    total = len(shapes) * len(prompts)
    print(
        f"[benchmark] {len(shapes)} shapes x {len(prompts)} prompts = "
        f"{total} generations; output -> {output_dir}",
        flush=True,
    )
    print(
        "[benchmark] to monitor, open another shell on the SAME host where "
        "you launched this benchmark (NOT your laptop, and NOT inside the "
        "container) and run one of:",
        flush=True,
    )
    print(
        "[benchmark]   tail -f ~/benchmark.log | grep -F '[benchmark]'",
        flush=True,
    )
    print(
        f'[benchmark]   watch -n 30 "tail -20 ~/benchmark.log; echo; '
        f"echo done: \\$(grep -cF '-> ' ~/benchmark.log) / {total}\"",
        flush=True,
    )
    print(
        "[benchmark] (substitute your log path if you nohup'd to "
        "somewhere other than ~/benchmark.log)",
        flush=True,
    )

    generator = _load_generator(args.model)

    with open(args.csv, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "shape",
                "width",
                "height",
                "num_frames",
                "prompt_id",
                "seed",
                "wall_seconds",
                "output_mp4",
                "mp4_bytes",
            ]
        )
        csv_file.flush()

        n = 0
        if args.prompt_major:
            # Outer loop over prompts: forces a shape switch between consecutive
            # generations. Useful for measuring whether torch.compile's in-memory
            # state is preserved across shape revisits, or evicted.
            order = [
                (s_idx, p_idx)
                for p_idx in range(len(prompts))
                for s_idx in range(len(shapes))
            ]
        else:
            # Default shape-major: 3 prompts on each shape before moving to the
            # next. Measures first-hit-per-shape and same-shape steady state.
            order = [
                (s_idx, p_idx)
                for s_idx in range(len(shapes))
                for p_idx in range(len(prompts))
            ]

        for shape_idx, prompt_idx in order:
            shape = shapes[shape_idx]
            w, h, nf = (
                int(shape["width"]),
                int(shape["height"]),
                int(shape["num_frames"]),
            )
            tag = f"{w}x{h}@{nf}f"
            prompt_id, prompt_text = prompts[prompt_idx]

            n += 1
            seed = 1000 + shape_idx * 10 + prompt_idx  # deterministic, distinct
            out_path = output_dir / f"{tag}_{prompt_id}.mp4"

            print(
                f"[benchmark] ({n}/{total}) {tag} prompt={prompt_id} "
                f"seed={seed} -> {out_path}",
                flush=True,
            )

            if args.reset:
                # Clear dynamo's accumulated guard / specialization state so the
                # next compile happens in fresh state and produces deterministic
                # cache keys regardless of access order.
                import torch._dynamo

                torch._dynamo.reset()

            t0 = time.perf_counter()
            try:
                generator.generate_video(
                    prompt=prompt_text,
                    save_video=True,
                    return_frames=False,
                    output_path=str(out_path),
                    width=w,
                    height=h,
                    num_frames=nf,
                    fps=fps,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    seed=seed,
                )
                elapsed = time.perf_counter() - t0
                size = out_path.stat().st_size if out_path.exists() else 0
                print(
                    f"[benchmark]   -> {elapsed:.1f}s  " f"({size / 1_048_576:.1f} MB)",
                    flush=True,
                )
                writer.writerow(
                    [
                        tag,
                        w,
                        h,
                        nf,
                        prompt_id,
                        seed,
                        f"{elapsed:.2f}",
                        str(out_path),
                        size,
                    ]
                )
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                print(
                    f"[benchmark]   FAIL after {elapsed:.1f}s: {exc!r}",
                    flush=True,
                )
                writer.writerow(
                    [
                        tag,
                        w,
                        h,
                        nf,
                        prompt_id,
                        seed,
                        f"{elapsed:.2f}",
                        "",
                        0,
                    ]
                )
            csv_file.flush()

    print(f"[benchmark] done. timings -> {args.csv}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
