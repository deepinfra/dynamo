#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Pre-compile FastWan2.2-TI2V-5B torch.compile / triton / inductor caches for every
production shape so the first production request is fast.

Routes each shape through the same ``lib.pool.SubprocessPool`` code path
the serving worker uses, so the cache-building subprocess is byte-identical
in code path to the runtime serving subprocess. Per-shape
``TORCHINDUCTOR_CACHE_DIR`` / ``TRITON_CACHE_DIR`` are set by the pool itself
on the child env (/cache/per-shape/<shape_key>/{torchinductor,triton}),
matching what production reads at serve time.

CROSS-PROCESS CACHE PORTABILITY: the implicit on-disk per-shape cache (these
dirs) does NOT port to a fresh process -- a fresh worker recomputes a different
fxgraph key and RECOMPILES. The per-shape dirs here are only the in-process
compile scratch for THIS bake run.

To actually make a fresh pod warm, use torch **Mega-Cache**
(save/load_cache_artifacts -- wired in fastvideo gpu_worker + lib/pool.py): it
DOES port across processes. The remaining residual is the torch.compile
FRONT-END (dynamo + AOTAutograd re-traces every process to produce the cache
key) -- not cacheable short of AOTInductor. The shared machinery and the full
investigation (instrumented on LTX-2.3, 2026-06-18) live in
**ltx23/CACHING.md** + ~/ltx23_cache_investigation_report.md.

Usage:
  python fastwan/warmup.py --shapes fastwan22_5b/shapes.json \\
      --output-dir /tmp/warmup --model /data/default
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import sys
import time
from pathlib import Path

# Put the parent directory (examples/diffusers/) on sys.path so we can
# import ``lib.pool`` regardless of where this script is invoked from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROMPT = (
    "A close-up tracking shot of a golden retriever sprinting toward the camera "
    "through a sunlit alpine meadow at golden hour, paws kicking up wildflowers, "
    "ears flapping, tongue out in joyful pants, shallow depth of field, shot on "
    "ARRI Alexa 65, 4K, photorealistic"
)


# ── helpers ──────────────────────────────────────────────────────────────────


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


# ── driver ───────────────────────────────────────────────────────────────────


def _run_driver(args: argparse.Namespace) -> int:
    """
    Route each shape through the production ``lib.pool.SubprocessPool``
    code path (spawning ``worker.py --pool-worker``). A fresh K=1 pool is
    built per shape, the request is routed once, and the pool is shut
    down before the next shape so each shape gets a fresh pool subprocess
    writing to its own ``/cache/per-shape/<tag>/`` dir.

    Using the production pool subprocess as the cache-building subprocess
    is the load-bearing property: torch.compile / inductor fx_graph_cache
    keys are sensitive to invocation context (__main__ identity, argv,
    sys.modules layout). The cache-building subprocess IS the cache-reading
    subprocess by construction, so keys produced here match what the
    runtime worker asks for at serve time.
    """
    from lib.pool import SubprocessPool

    cfg = _read_shapes_config(args)

    model = cfg.get("model", "/data/default")
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
        f"[warmup] driver start: {len(shapes)} shape(s), "
        f"model={model}, cache_before={_human(cache_before)} "
        f"root={per_shape_root}",
        flush=True,
    )

    successes: list[str] = []
    failures: list[tuple[str, str]] = []

    async def _route_one_shape_with_timeout(
        tag: str, request: dict, timeout: float
    ) -> dict:
        pool = SubprocessPool(
            model_path=model,
            num_gpus=1,
            # Bake the cache so it MATCHES the serving path: fastwan22_5b.factory
            # always loads FP8 (the QAD checkpoint's native format), so the same
            # graph is compiled here and at serve time. Denoise steps (3) come
            # from the shapes file. enable_optimizations is passed through but
            # the fastwan22_5b factory ignores it (no gated optimizations).
            enable_optimizations=True,
            attention_backend="FLASH_ATTN",
            model_factory_dotted="fastwan22_5b.factory:load_model",
            model_label="fastwan22-ti2v-5b",
        )
        try:
            return await asyncio.wait_for(pool.route(tag, request), timeout=timeout)
        finally:
            await pool.shutdown()

    for idx, shape in enumerate(shapes, 1):
        w = int(shape["width"])
        h = int(shape["height"])
        nf = int(shape["num_frames"])
        tag = f"{w}x{h}@{nf}f"
        out_path = output_dir / f"shape_{tag}.mp4"
        if out_path.exists():
            out_path.unlink()

        # The pool's _spawn sets TORCHINDUCTOR_CACHE_DIR / TRITON_CACHE_DIR
        # on the child env using the shape_key, so we only need the dirs
        # to exist (the pool subprocess writes into them).
        shape_inductor = per_shape_root / tag / "torchinductor"
        shape_triton = per_shape_root / tag / "triton"
        shape_inductor.mkdir(parents=True, exist_ok=True)
        shape_triton.mkdir(parents=True, exist_ok=True)

        request = {
            "request_id": f"warmup_{tag}",
            "prompt": PROMPT,
            "width": w,
            "height": h,
            "num_frames": nf,
            "fps": fps,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "negative_prompt": None,
            "output_path": str(out_path),
        }

        print(
            f"[warmup] ({idx}/{len(shapes)}) launching {tag} -> {out_path}",
            flush=True,
        )
        print(
            f"[warmup] ({idx}/{len(shapes)}) per-shape cache dir: "
            f"{per_shape_root}/{tag}",
            flush=True,
        )

        t0 = time.perf_counter()
        try:
            result = asyncio.run(
                _route_one_shape_with_timeout(tag, request, args.per_shape_timeout)
            )
        except asyncio.TimeoutError:
            elapsed = time.perf_counter() - t0
            msg = f"timeout after {elapsed:.1f}s"
            print(f"[warmup] FAIL {tag}: {msg}", flush=True)
            failures.append((tag, msg))
            if args.fail_fast:
                break
            continue
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            msg = f"{type(exc).__name__}: {exc}"
            print(f"[warmup] FAIL {tag}: {msg} (after {elapsed:.1f}s)", flush=True)
            failures.append((tag, msg))
            if args.fail_fast:
                break
            continue

        elapsed = time.perf_counter() - t0
        status = result.get("status")
        if status == "DONE" and out_path.exists() and out_path.stat().st_size > 0:
            size_mb = out_path.stat().st_size / 1_048_576
            print(
                f"[warmup] OK   {tag} in {elapsed:.1f}s ({size_mb:.1f}MB)",
                flush=True,
            )
            successes.append(tag)
        else:
            msg = (
                f"status={status} "
                f"error={result.get('error', '?')} "
                f"file_exists={out_path.exists()} "
                f"size={out_path.stat().st_size if out_path.exists() else 0}"
            )
            print(f"[warmup] FAIL {tag}: {msg} (after {elapsed:.1f}s)", flush=True)
            failures.append((tag, msg))
            if args.fail_fast:
                break

    cache_after = _du_bytes(per_shape_root)
    return _final_summary(
        successes,
        failures,
        len(shapes),
        cache_before,
        cache_after,
        args.min_cache_growth_bytes,
    )


# ── argument parsing ─────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FastWan2.2-TI2V-5B warmup / compile-cache populator. Routes each "
        "shape through lib.pool.SubprocessPool so the cache-building "
        "subprocess matches the runtime serving subprocess in code path. "
        "Cache keys produced here match what the runtime asks for at "
        "serve time.",
    )

    p.add_argument(
        "--shapes",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "shapes.json"),
        help="path to shapes JSON (default: fastwan22_5b/shapes.json next to this script)",
    )
    p.add_argument(
        "--output-dir", default="/tmp/warmup", help="where to save rendered MP4s"
    )
    p.add_argument(
        "--per-shape-timeout",
        type=int,
        default=1800,
        help="per-shape subprocess timeout in seconds",
    )
    p.add_argument(
        "--fail-fast",
        action="store_true",
        help="stop on first shape failure (default: continue, report at end)",
    )
    p.add_argument(
        "--min-cache-growth-bytes",
        type=int,
        default=0,
        help="fail if combined cache grew by fewer bytes than this",
    )

    # Override for the model path baked into shapes.json. The driver
    # reads ``cfg.get("model", "/data/default")`` from shapes.json; this
    # flag is currently a no-op (kept as a forward-compat hook). The
    # production default is the local /data/default mount.
    p.add_argument("--model", default="/data/default")

    return p.parse_args()


def main() -> int:
    args = _parse_args()
    return _run_driver(args)


if __name__ == "__main__":
    sys.exit(main())
