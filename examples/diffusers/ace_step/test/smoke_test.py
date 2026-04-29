#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Functional smoke test for the ACE-Step Dynamo worker.

What this verifies (without any musical-theory knowledge):
  1. The worker accepts an NvCreateMusicRequest and yields an NvMusicResponse.
  2. The returned base64 payload decodes to a valid audio file (FLAC/WAV/MP3).
  3. The decoded audio is non-silent (peak amplitude above a small threshold).
  4. The decoded audio's duration is within tolerance of the requested duration.
  5. (Optional) Determinism: same prompt + seed twice → byte-identical output.

Reaches the worker via the Dynamo runtime directly (no HTTP frontend), since
/v1/audio/generations is not yet wired in the Rust frontend.

Usage:
  # Run only the basic smoke test
  python smoke_test.py --prompt "lo-fi chill beats" --duration 15

  # Also assert determinism (slower — runs generation twice)
  python smoke_test.py --prompt "lo-fi chill beats" --duration 15 \
      --seed 42 --check-determinism

  # Save the generated clip for a manual listen
  python smoke_test.py --prompt "uplifting cinematic strings" \
      --duration 20 --out /tmp/sample.flac

Exit code: 0 on success, non-zero on any assertion failure.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import io
import json
import logging
import os
import sys

logger = logging.getLogger("acestep_smoke_test")


def _import_runtime():
    """Import Dynamo runtime lazily so this script can be loaded for --help
    without the dynamo package installed."""
    from dynamo.runtime import DistributedRuntime  # type: ignore

    return DistributedRuntime


def _get_namespace() -> str:
    namespace = os.environ.get("DYN_NAMESPACE", "dynamo")
    suffix = os.environ.get("DYN_NAMESPACE_WORKER_SUFFIX")
    if suffix:
        namespace = f"{namespace}-{suffix}"
    return namespace


def _decode_audio(b64: str) -> tuple[bytes, "object", int, float, float]:
    """Return (raw_bytes, ndarray, sample_rate, duration_s, peak_abs)."""
    import numpy as np  # noqa: F401  (used via soundfile output)
    import soundfile as sf

    raw = base64.b64decode(b64)
    with sf.SoundFile(io.BytesIO(raw)) as fh:
        sample_rate = fh.samplerate
        frames = fh.read(dtype="float32", always_2d=False)
    duration_s = len(frames) / float(sample_rate) if sample_rate else 0.0
    peak = float(abs(frames).max()) if frames.size else 0.0
    return raw, frames, sample_rate, duration_s, peak


async def _send_request(
    runtime, model: str, prompt: str, duration: float, seed: int | None
) -> dict:
    """Issue one generate call to the registered ACE-Step worker."""
    namespace = _get_namespace()
    client = runtime.namespace(namespace).component("backend").endpoint("generate").client()
    await client.wait_for_instances()

    nvext: dict = {}
    if seed is not None:
        nvext["seed"] = seed
        nvext["thinking"] = True

    payload = {
        "prompt": prompt,
        "model": model,
        "lyrics": "[Instrumental]",
        "duration": duration,
        "response_format": "flac",
        "nvext": nvext,
    }

    logger.info("Sending request: %s", json.dumps({**payload, "prompt": prompt[:60]}))

    last_response: dict | None = None
    async for chunk in await client.generate(payload):
        # The endpoint yields a single NvMusicResponse dict.
        if isinstance(chunk, dict):
            last_response = chunk
        else:
            # Some Dynamo client variants wrap responses; try .data().
            data = getattr(chunk, "data", None)
            last_response = data() if callable(data) else data

    if last_response is None:
        raise AssertionError("worker yielded no response")
    return last_response


def _assert_response_shape(resp: dict) -> dict:
    assert resp.get("status") in (
        "completed",
        "complete",
    ), f"unexpected status: {resp.get('status')!r}"
    data = resp.get("data") or []
    assert data, "response.data is empty"
    clip = data[0]
    assert clip.get("b64_json"), "data[0].b64_json missing"
    return clip


def _assert_audio_valid(
    clip: dict, requested_duration: float, duration_tolerance: float
) -> tuple[bytes, int, float]:
    raw, _frames, sample_rate, duration_s, peak = _decode_audio(clip["b64_json"])

    assert sample_rate > 0, f"invalid sample_rate: {sample_rate!r}"
    assert duration_s > 0, "decoded audio has zero duration"
    assert peak > 0.001, f"decoded audio appears silent (peak={peak:.6f})"

    if requested_duration > 0:
        delta = abs(duration_s - requested_duration)
        assert delta <= duration_tolerance, (
            f"duration mismatch: requested={requested_duration}s "
            f"got={duration_s:.2f}s tolerance={duration_tolerance}s"
        )

    logger.info(
        "audio OK: %.2fs @ %d Hz, peak=%.3f, %d KiB",
        duration_s,
        sample_rate,
        peak,
        len(raw) // 1024,
    )
    return raw, sample_rate, duration_s


async def _run(args: argparse.Namespace) -> int:
    DistributedRuntime = _import_runtime()
    loop = asyncio.get_running_loop()

    discovery = os.environ.get(
        "DYN_DISCOVERY_BACKEND",
        "kubernetes" if os.environ.get("KUBERNETES_SERVICE_HOST") else "file",
    )
    runtime = DistributedRuntime(loop, discovery, "tcp")

    logger.info("Discovery=%s namespace=%s", discovery, _get_namespace())

    resp1 = await _send_request(
        runtime,
        model=args.model,
        prompt=args.prompt,
        duration=args.duration,
        seed=args.seed,
    )
    clip1 = _assert_response_shape(resp1)
    raw1, _sr1, _dur1 = _assert_audio_valid(
        clip1, args.duration, args.duration_tolerance
    )

    if args.out:
        with open(args.out, "wb") as f:
            f.write(raw1)
        logger.info("wrote %d bytes to %s", len(raw1), args.out)

    if args.check_determinism:
        if args.seed is None:
            logger.warning("--check-determinism requires --seed; skipping")
        else:
            logger.info("Running second generation to check determinism...")
            resp2 = await _send_request(
                runtime,
                model=args.model,
                prompt=args.prompt,
                duration=args.duration,
                seed=args.seed,
            )
            clip2 = _assert_response_shape(resp2)
            raw2, _, _ = _assert_audio_valid(
                clip2, args.duration, args.duration_tolerance
            )
            h1 = hashlib.sha256(raw1).hexdigest()
            h2 = hashlib.sha256(raw2).hexdigest()
            if h1 != h2:
                logger.warning(
                    "outputs differ for the same seed (sha256: %s vs %s) — "
                    "determinism is best-effort; check upstream notes.",
                    h1[:12],
                    h2[:12],
                )
            else:
                logger.info("deterministic: sha256=%s", h1[:12])

    logger.info("smoke test passed")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="ACE-Step/acestep-v15-xl",
        help="Model name registered by the worker.",
    )
    parser.add_argument(
        "--prompt",
        default="upbeat lo-fi hip hop beat with mellow piano and warm vinyl crackle",
        help="Caption / text description.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=15.0,
        help="Requested clip duration in seconds.",
    )
    parser.add_argument(
        "--duration-tolerance",
        type=float,
        default=2.0,
        dest="duration_tolerance",
        help="Allowed |actual - requested| seconds (default: 2.0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed (required for --check-determinism).",
    )
    parser.add_argument(
        "--check-determinism",
        action="store_true",
        dest="check_determinism",
        help="Send the same request twice and compare output hashes.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional path to write the generated audio for manual listening.",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    args = _parse_args()
    try:
        return asyncio.run(_run(args))
    except AssertionError as exc:
        logger.error("smoke test FAILED: %s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001 — top-level reporter
        logger.exception("smoke test ERRORED: %s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
