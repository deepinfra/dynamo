#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Pre-compile LTX-2 torch.compile / triton / inductor caches for every
production shape so the first production request is fast.

Two driver modes, with very different cache properties:

SINGLE-PROCESS MODE (default). One VideoGenerator load, all shapes
generated sequentially in one process. As of 2026-05-14 this mode
is known to produce an ORDER-DEPENDENT cache: `_dynamo.reset()`
between shapes clears dynamo state but not the in-process Triton /
CUDA module / autotuner state (we haven't isolated which one), so
each shape's compile-cache key reflects state left by earlier shapes.
A fresh production worker asked to serve shape N first cannot
reproduce that key and recompiles. Kept for diagnostic / benchmark
use; NOT the path that builds the ship cache.

SUBPROCESS-PER-SHAPE MODE (--legacy-subprocess). Each shape runs in
a fresh Python subprocess pointed at its own per-shape cache dirs
under /cache/per-shape/<shape_key>/{torchinductor,triton}. Fresh
process state plus isolated cache dirs means each shape's cache key
is computed from clean kernel state and can be reproduced by a fresh
production worker pointed at the same per-shape dir. As of 2026-05-14
THIS IS THE PRODUCTION CACHE-BUILD PATH. Flag name retained for diff
size; rename deferred. Phase 2 (separate change) teaches the worker
to switch TORCHINDUCTOR_CACHE_DIR / TRITON_CACHE_DIR per request.

See ~/backend/claude_plans/2026-05-14-ltx2-per-shape-cache.md.

Usage (single-process, diagnostic):
  python warmup.py --shapes warmup_shapes.json --output-dir /tmp/warmup

Usage (subprocess-per-shape, production cache build):
  python warmup.py --legacy-subprocess --shapes warmup_shapes.json

Usage (single-shape worker, internal -- only used by --legacy-subprocess):
  python warmup.py --single-shape --width 1920 --height 1088 \\
      --num-frames 121 --fps 24 --num-inference-steps 5 \\
      --guidance-scale 1.0 --seed 42 \\
      --model FastVideo/LTX2-Distilled-Diffusers \\
      --output /tmp/warmup/shape_1920x1088@121f.mp4
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import time

import torch
from pathlib import Path


PROMPT = (
    "A close-up tracking shot of a golden retriever sprinting toward the camera "
    "through a sunlit alpine meadow at golden hour, paws kicking up wildflowers, "
    "ears flapping, tongue out in joyful pants, shallow depth of field, shot on "
    "ARRI Alexa 65, 4K, photorealistic"
)


DEFAULT_INDUCTOR_CACHE = "/cache/torchinductor"
DEFAULT_TRITON_CACHE = "/cache/triton"


def _ensure_cache_env() -> tuple[str, str]:
    """
    Pin torch-inductor and triton cache directories so compile artifacts
    persist across subprocesses (and, once baked into the image, across
    container runs). Must run before `import torch` in any process that
    will compile; torch reads these on first inductor use and defaults to
    `/tmp/torchinductor_<user>` otherwise.
    """
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", DEFAULT_INDUCTOR_CACHE)
    os.environ.setdefault("TRITON_CACHE_DIR", DEFAULT_TRITON_CACHE)
    inductor = os.environ["TORCHINDUCTOR_CACHE_DIR"]
    triton = os.environ["TRITON_CACHE_DIR"]
    Path(inductor).mkdir(parents=True, exist_ok=True)
    Path(triton).mkdir(parents=True, exist_ok=True)
    return inductor, triton


def _load_generator(model: str):
    """
    Load and return a VideoGenerator configured for LTX-2.

    This must be called AFTER _ensure_cache_env() in any process that will
    compile. The optimization_kwargs are the canonical configuration; both
    warmup and production worker use this exact set so cache keys match.
    """
    os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")
    os.environ.setdefault("FASTVIDEO_STAGE_LOGGING", "1")
    os.environ.setdefault("FASTVIDEO_ENABLE_RMSNORM_FP4_PREQUANT", "0")

    import torch  # noqa: F401  (ensure CUDA init before FastVideo)
    from fastvideo import VideoGenerator
    from fastvideo.configs.pipelines.base import PipelineConfig

    # Import here (after _ensure_cache_env) so the local layout works whether
    # warmup.py is invoked at /opt/app/warmup.py (in-container) or from a
    # checkout (the parent dir is on sys.path either way).
    from ltx2_config import standard_kwargs

    print(f"[warmup] loading VideoGenerator model={model}", flush=True)
    t_load = time.perf_counter()
    pipeline_config = PipelineConfig.from_pretrained(model)
    generator = VideoGenerator.from_pretrained(
        model,
        num_gpus=1,
        pipeline_config=pipeline_config,
        **standard_kwargs(),
    )
    print(
        f"[warmup] generator ready in {time.perf_counter() - t_load:.1f}s",
        flush=True,
    )
    return generator


def _generate_one(
    generator,
    *,
    prompt: str,
    width: int,
    height: int,
    num_frames: int,
    fps: int,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
    output_path: str,
) -> None:
    """One generate_video call. Caller times it."""
    generator.generate_video(
        prompt=prompt,
        save_video=True,
        return_frames=False,
        output_path=output_path,
        width=width,
        height=height,
        num_frames=num_frames,
        fps=fps,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
    )


# ── single-shape mode (used by --legacy-subprocess driver) ────────────────────


def _run_single_shape(args: argparse.Namespace) -> int:
    """Load VideoGenerator, generate one shape, save MP4, exit."""
    _ensure_cache_env()
    generator = _load_generator(args.model)

    t_gen = time.perf_counter()
    _generate_one(
        generator,
        prompt=PROMPT,
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        fps=args.fps,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        output_path=args.output,
    )
    print(
        f"[warmup] shape {args.width}x{args.height}@{args.num_frames}f "
        f"generated in {time.perf_counter() - t_gen:.1f}s -> {args.output}",
        flush=True,
    )
    return 0


# ── driver mode ──────────────────────────────────────────────────────────────


def _du_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            # File may have vanished between os.walk and stat; safe to skip.
            with contextlib.suppress(OSError):
                total += (Path(root) / f).stat().st_size
    return total


def _human(n: int) -> str:
    for unit in ("B", "K", "M", "G"):
        if n < 1024 or unit == "G":
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}G"


def _read_shapes_config(args: argparse.Namespace) -> dict:
    with open(args.shapes, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not cfg.get("shapes"):
        print("[warmup] shapes list is empty -- nothing to do", file=sys.stderr)
        sys.exit(2)
    return cfg


def _final_summary(
    successes: list[str],
    failures: list[tuple[str, str]],
    total_shapes: int,
    cache_before: int,
    cache_after: int,
    min_cache_growth_bytes: int,
) -> int:
    cache_growth = cache_after - cache_before
    print(
        f"[warmup] done. success={len(successes)}/{total_shapes} "
        f"failures={[t for t, _ in failures]} "
        f"cache_before={_human(cache_before)} cache_after={_human(cache_after)} "
        f"cache_growth={_human(cache_growth)}",
        flush=True,
    )
    if failures:
        return 1
    if cache_growth < min_cache_growth_bytes:
        print(
            f"[warmup] ERROR cache grew by {_human(cache_growth)} "
            f"(< --min-cache-growth-bytes {min_cache_growth_bytes}). "
            f"Did compile actually run?",
            file=sys.stderr,
            flush=True,
        )
        return 3
    return 0


def _run_driver_single_process(args: argparse.Namespace) -> int:
    """
    Single-process driver: load VideoGenerator once, generate all shapes
    in one process, in the order given by warmup_shapes.json.

    DIAGNOSTIC USE ONLY. The cache produced here is order-dependent:
    `_dynamo.reset()` clears dynamo state between shapes but not the
    in-process Triton / CUDA module / autotuner state (we haven't
    isolated which one), so each shape's compile-cache key reflects
    state left by earlier shapes. A fresh production worker cannot
    reproduce those keys. Use --legacy-subprocess to build the ship
    cache.

    A shape that raises an exception aborts the whole run -- single-process
    has no isolation.
    """
    _ensure_cache_env()
    cfg = _read_shapes_config(args)

    model = cfg.get("model", "FastVideo/LTX2-Distilled-Diffusers")
    fps = int(cfg.get("fps", 24))
    num_inference_steps = int(cfg.get("num_inference_steps", 5))
    guidance_scale = float(cfg.get("guidance_scale", 1.0))
    seed = int(cfg.get("seed", 42))
    shapes = cfg["shapes"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    inductor_cache = Path(os.environ["TORCHINDUCTOR_CACHE_DIR"])
    triton_cache = Path(os.environ["TRITON_CACHE_DIR"])
    cache_before = _du_bytes(inductor_cache) + _du_bytes(triton_cache)
    print(
        f"[warmup] driver start (single-process): {len(shapes)} shape(s), "
        f"model={model}, cache_before={_human(cache_before)} "
        f"(inductor={inductor_cache} triton={triton_cache})",
        flush=True,
    )

    generator = _load_generator(model)

    successes: list[str] = []
    failures: list[tuple[str, str]] = []

    for idx, shape in enumerate(shapes, 1):
        # Reset dynamo state between shapes so compile-cache keys are keyed
        # only on shape parameters, not on guards accumulated from prior
        # shapes. NOTE (2026-05-14): _dynamo.reset() does not clear the
        # in-process Triton / CUDA module / autotuner state (we haven't
        # isolated which one), so single-process warmup still produces
        # order-dependent caches. The real fix is --legacy-subprocess
        # (subprocess-per-shape, per-shape cache dirs). See
        # claude_plans/2026-05-14-ltx2-per-shape-cache.md.
        torch._dynamo.reset()
        torch.cuda.empty_cache()
        w = int(shape["width"])
        h = int(shape["height"])
        nf = int(shape["num_frames"])
        tag = f"{w}x{h}@{nf}f"
        out_path = output_dir / f"shape_{tag}.mp4"
        if out_path.exists():
            out_path.unlink()

        print(
            f"[warmup] ({idx}/{len(shapes)}) generating {tag} -> {out_path}",
            flush=True,
        )
        t0 = time.perf_counter()
        try:
            _generate_one(
                generator,
                prompt=PROMPT,
                width=w,
                height=h,
                num_frames=nf,
                fps=fps,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                output_path=str(out_path),
            )
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            msg = f"{type(exc).__name__}: {exc!r}"
            print(
                f"[warmup] FAIL {tag}: {msg} (after {elapsed:.1f}s)",
                flush=True,
            )
            failures.append((tag, msg))
            # Single-process mode: do not continue. dynamo / fastvideo state
            # may be inconsistent after a crash; failing fast keeps the run
            # honest and avoids producing a partially-poisoned cache.
            break

        elapsed = time.perf_counter() - t0
        if out_path.exists() and out_path.stat().st_size > 0:
            size_mb = out_path.stat().st_size / 1_048_576
            print(
                f"[warmup] OK   {tag} in {elapsed:.1f}s ({size_mb:.1f}MB)",
                flush=True,
            )
            successes.append(tag)
        else:
            msg = (
                f"no output file (or zero bytes) "
                f"file_exists={out_path.exists()} "
                f"size={out_path.stat().st_size if out_path.exists() else 0}"
            )
            print(
                f"[warmup] FAIL {tag}: {msg} (after {elapsed:.1f}s)",
                flush=True,
            )
            failures.append((tag, msg))
            break

    cache_after = _du_bytes(inductor_cache) + _du_bytes(triton_cache)
    return _final_summary(
        successes, failures, len(shapes),
        cache_before, cache_after, args.min_cache_growth_bytes,
    )


def _run_driver_subprocess(args: argparse.Namespace) -> int:
    """
    Subprocess-per-shape driver: spawn one fresh Python subprocess per
    shape, each pointed at its own /cache/per-shape/<shape_key>/{torch-
    inductor,triton} cache directory.

    PRODUCTION CACHE-BUILD PATH (as of 2026-05-14). Fresh-process state
    plus isolated cache dirs gives each shape a compile-cache key derived
    from clean kernel state, reproducible by a fresh production worker
    pointed at the same per-shape dir (Phase 2). Flag name retained for
    compatibility; rename deferred.
    """
    # Parent is just a dispatcher; per-shape cache dirs are set on each
    # child via env= below, so we do NOT call _ensure_cache_env() here.
    cfg = _read_shapes_config(args)

    model = cfg.get("model", "FastVideo/LTX2-Distilled-Diffusers")
    fps = int(cfg.get("fps", 24))
    num_inference_steps = int(cfg.get("num_inference_steps", 5))
    guidance_scale = float(cfg.get("guidance_scale", 1.0))
    seed = int(cfg.get("seed", 42))
    shapes = cfg["shapes"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_shape_root = Path("/cache/per-shape")
    cache_before = _du_bytes(per_shape_root)
    print(
        f"[warmup] driver start (subprocess-per-shape): {len(shapes)} shape(s), "
        f"model={model}, cache_before={_human(cache_before)} "
        f"root={per_shape_root}",
        flush=True,
    )

    successes: list[str] = []
    failures: list[tuple[str, str]] = []

    for idx, shape in enumerate(shapes, 1):
        w = int(shape["width"])
        h = int(shape["height"])
        nf = int(shape["num_frames"])
        tag = f"{w}x{h}@{nf}f"
        out_path = output_dir / f"shape_{tag}.mp4"
        if out_path.exists():
            out_path.unlink()

        shape_inductor = per_shape_root / tag / "torchinductor"
        shape_triton = per_shape_root / tag / "triton"
        shape_inductor.mkdir(parents=True, exist_ok=True)
        shape_triton.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            __file__,
            "--single-shape",
            "--model", model,
            "--width", str(w),
            "--height", str(h),
            "--num-frames", str(nf),
            "--fps", str(fps),
            "--num-inference-steps", str(num_inference_steps),
            "--guidance-scale", str(guidance_scale),
            "--seed", str(seed),
            "--output", str(out_path),
        ]
        print(
            f"[warmup] ({idx}/{len(shapes)}) launching {tag} -> {out_path}",
            flush=True,
        )
        print(
            f"[warmup] ({idx}/{len(shapes)}) per-shape cache dir: "
            f"{per_shape_root}/{tag}",
            flush=True,
        )
        child_env = os.environ.copy()
        child_env["TORCHINDUCTOR_CACHE_DIR"] = str(shape_inductor)
        child_env["TRITON_CACHE_DIR"] = str(shape_triton)
        t0 = time.perf_counter()
        try:
            result = subprocess.run(
                cmd,
                check=False,
                timeout=args.per_shape_timeout,
                env=child_env,
            )
        except subprocess.TimeoutExpired:
            elapsed = time.perf_counter() - t0
            msg = f"timeout after {elapsed:.1f}s"
            print(f"[warmup] FAIL {tag}: {msg}", flush=True)
            failures.append((tag, msg))
            continue

        elapsed = time.perf_counter() - t0
        if result.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0:
            size_mb = out_path.stat().st_size / 1_048_576
            print(
                f"[warmup] OK   {tag} in {elapsed:.1f}s ({size_mb:.1f}MB)",
                flush=True,
            )
            successes.append(tag)
        else:
            msg = (
                f"rc={result.returncode} "
                f"file_exists={out_path.exists()} "
                f"size={out_path.stat().st_size if out_path.exists() else 0}"
            )
            print(f"[warmup] FAIL {tag}: {msg} (after {elapsed:.1f}s)", flush=True)
            failures.append((tag, msg))
            if args.fail_fast:
                break

    cache_after = _du_bytes(per_shape_root)
    return _final_summary(
        successes, failures, len(shapes),
        cache_before, cache_after, args.min_cache_growth_bytes,
    )


# ── argument parsing ─────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LTX-2 warmup / compile-cache populator. Default: single-process "
                    "(matches production access). Use --legacy-subprocess for "
                    "crash-isolated debug runs.",
    )
    p.add_argument(
        "--single-shape", action="store_true",
        help="(internal) run one shape and exit; only used by --legacy-subprocess",
    )

    # driver args
    p.add_argument(
        "--legacy-subprocess", action="store_true",
        help="legacy driver mode: spawn one Python subprocess per shape. "
             "Produces a cache keyed by fresh-process dynamo state, which "
             "does NOT match production single-process access. Use only for "
             "crash-isolation debugging.",
    )
    p.add_argument("--shapes", default="warmup_shapes.json",
                   help="path to shapes JSON")
    p.add_argument("--output-dir", default="/tmp/warmup",
                   help="where to save rendered MP4s")
    p.add_argument("--per-shape-timeout", type=int, default=1800,
                   help="per-shape subprocess timeout in seconds (legacy-subprocess only)")
    p.add_argument("--fail-fast", action="store_true",
                   help="stop on first shape failure (legacy-subprocess only; "
                        "single-process always fails fast)")
    p.add_argument("--min-cache-growth-bytes", type=int, default=0,
                   help="fail if combined cache grew by fewer bytes than this")

    # single-shape args
    p.add_argument("--model", default="FastVideo/LTX2-Distilled-Diffusers")
    p.add_argument("--width", type=int)
    p.add_argument("--height", type=int)
    p.add_argument("--num-frames", type=int)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--num-inference-steps", type=int, default=5)
    p.add_argument("--guidance-scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str)

    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if args.single_shape:
        missing = [
            name for name, val in [
                ("--width", args.width),
                ("--height", args.height),
                ("--num-frames", args.num_frames),
                ("--output", args.output),
            ] if val is None
        ]
        if missing:
            print(f"[warmup] single-shape mode missing: {missing}", file=sys.stderr)
            return 2
        return _run_single_shape(args)
    if args.legacy_subprocess:
        return _run_driver_subprocess(args)
    return _run_driver_single_process(args)


if __name__ == "__main__":
    sys.exit(main())
