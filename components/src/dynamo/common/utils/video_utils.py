# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Video utilities for video diffusion.

Provides helpers for parsing video request parameters and encoding numpy
video frames to MP4 format.
"""

import io
import logging
import os
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


DEFAULT_VIDEO_WIDTH = 832
DEFAULT_VIDEO_HEIGHT = 480
DEFAULT_VIDEO_FPS = 16
DEFAULT_VIDEO_NUM_FRAMES = 97


def parse_size(
    size: str | None,
    default_w: int = DEFAULT_VIDEO_WIDTH,
    default_h: int = DEFAULT_VIDEO_HEIGHT,
) -> Tuple[int, int]:
    """Parse a 'WxH' string into (width, height).

    Falls back to default_w x default_h when size is None or malformed.
    """
    if not size:
        return default_w, default_h
    try:
        w, h = size.split("x")
        return int(w), int(h)
    except (ValueError, AttributeError):
        logger.warning("Invalid size format: %s, using defaults", size)
        return default_w, default_h


def compute_num_frames(
    num_frames: int | None = None,
    seconds: int | None = None,
    fps: int | None = None,
    default_fps: int = DEFAULT_VIDEO_FPS,
    default_num_frames: int = DEFAULT_VIDEO_NUM_FRAMES,
) -> int:
    """Compute the number of video frames.

    Priority: num_frames > seconds x fps > default_num_frames.
    """
    if num_frames is not None:
        return num_frames
    if seconds is not None or fps is not None:
        _seconds = seconds if seconds is not None else 4
        _fps = fps if fps is not None else default_fps
        return _seconds * _fps
    return default_num_frames


def normalize_video_frames(images: list) -> list:
    """Normalize stage_output.images into a frame list for export_to_video.

    Args:
        images: stage_output.images -- a list that may contain a single
            torch.Tensor or np.ndarray representing the full video.

    Returns:
        List of frames suitable for diffusers export_to_video.
    """
    frames = images[0] if len(images) == 1 else images

    if isinstance(frames, np.ndarray):
        if frames.ndim == 5:
            frames = frames[0]
        return list(frames)

    return list(frames)


def frames_to_numpy(images: list) -> np.ndarray:
    """Convert a list of PIL Images to a numpy array suitable for video encoding.

    Args:
        images: List of PIL Image objects (video frames).

    Returns:
        Numpy array of shape ``(num_frames, height, width, 3)`` with dtype
        ``uint8`` and values in ``[0, 255]``.

    Raises:
        ValueError: If no images are provided or images have inconsistent sizes.
    """
    if not images:
        raise ValueError("No images provided for video encoding")

    frames = []
    for img in images:
        arr = np.array(img.convert("RGB"))
        frames.append(arr)

    # Validate consistent sizes
    shapes = {f.shape for f in frames}
    if len(shapes) > 1:
        raise ValueError(
            f"Inconsistent frame sizes detected: {shapes}. "
            "All frames must have the same dimensions."
        )

    return np.stack(frames, axis=0)


def encode_to_mp4(
    frames: np.ndarray,
    output_dir: str,
    request_id: str,
    fps: int = 16,
) -> str:
    """Encode numpy frames to MP4 file.

    Args:
        frames: Video frames as numpy array of shape (num_frames, height, width, 3)
            with uint8 values 0-255.
        output_dir: Directory to save the output video.
        request_id: Unique identifier for the request (used in filename).
        fps: Frames per second for the output video.

    Returns:
        Path to the saved MP4 file.

    Raises:
        ImportError: If imageio is not available.
        RuntimeError: If encoding fails.
    """
    try:
        import imageio.v3 as iio
    except ImportError:
        try:
            import imageio as iio  # type: ignore[no-redef]
        except ImportError:
            raise ImportError(
                "imageio is required for video encoding. "
                "Install with: pip install imageio[ffmpeg]"
            )

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{request_id}.mp4")

    logger.info(f"Encoding {len(frames)} frames to {output_path} at {fps} fps")

    try:
        # Use imageio to write MP4
        # imageio.v3 API
        if hasattr(iio, "imwrite"):
            iio.imwrite(output_path, frames, fps=fps, codec="libx264")
        else:
            # Fall back to v2 API
            writer = iio.get_writer(output_path, fps=fps, codec="libx264")  # type: ignore[attr-defined]
            try:
                for frame in frames:
                    writer.append_data(frame)
            finally:
                writer.close()

        logger.info(f"Video saved to {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Failed to encode video: {e}")
        raise RuntimeError(f"Video encoding failed: {e}") from e


# Video workers have three distinct CPU consumers that peak in different phases and want
# different thread counts, so each gets its own env-overridable knob (defaults below):
#   - Inductor compile pool (boot / first request: parallel Triton kernel compilation)
#   - torch intra-op pool   (generation: CPU-side ops around a GPU-bound diffusion)
#   - libx264 encode        (encode: ~12 is the measured PyAV sweet spot for 720p)
# On shared GPU nodes an uncapped worker is a noisy neighbor: torch sizes its pool to the
# physical core count (~112 on a 224-thread box) and libx264 grabs every core, which both
# thrashes (a 720p/81f encode took 244s vs ~1s) and starves co-located pods. The three
# phases are sequential, so with OMP_WAIT_POLICY=PASSIVE / KMP_BLOCKTIME=0 an idle pool
# sleeps and hands its cores to the active phase instead of spinning — they don't sum.
DEFAULT_VIDEO_TORCH_THREADS = 12
DEFAULT_VIDEO_ENCODE_THREADS = 12
DEFAULT_VIDEO_COMPILE_THREADS = 32


def video_encode_threads() -> int:
    """CPU threads for the libx264 software encode (env: DI_VIDEO_ENCODE_THREADS)."""
    return int(os.getenv("DI_VIDEO_ENCODE_THREADS", str(DEFAULT_VIDEO_ENCODE_THREADS)))


def limit_video_worker_threads() -> None:
    """Cap a video worker's CPU threads so it stays a good neighbor on shared nodes.

    Call once at the very top of a video worker's startup, before torch loads. Sets the
    OpenMP wait policy (idle pools sleep between the sequential compile -> generate ->
    encode phases rather than spinning), bounds the Inductor compile pool, and caps
    torch's intra-op pool. The encoder is capped separately via video_encode_threads().
    Every video worker entry calls this once — that is the general pattern. No-op if
    torch is unavailable.
    """
    # Import-time knobs: set before torch imports so OpenMP / Inductor pick them up. Idle
    # pools then sleep instead of spin (covers both libgomp and Intel OpenMP / MKL).
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    os.environ.setdefault("KMP_BLOCKTIME", "0")
    os.environ.setdefault(
        "TORCHINDUCTOR_COMPILE_THREADS",
        os.getenv("DI_VIDEO_COMPILE_THREADS", str(DEFAULT_VIDEO_COMPILE_THREADS)),
    )
    torch_threads = int(
        os.getenv("DI_VIDEO_TORCH_THREADS", str(DEFAULT_VIDEO_TORCH_THREADS))
    )
    try:
        import torch

        torch.set_num_threads(torch_threads)
    except Exception:  # noqa: BLE001
        logger.warning("could not cap torch CPU threads to %s", torch_threads)
    logger.info(
        "video worker CPU budget: torch=%s encode=%s compile=%s (OMP_WAIT_POLICY=PASSIVE)",
        torch_threads,
        video_encode_threads(),
        os.environ.get("TORCHINDUCTOR_COMPILE_THREADS"),
    )


def encode_to_video_bytes(
    frames: np.ndarray,
    fps: int = 16,
    output_format: str = "mp4",
) -> bytes:
    """Encode numpy frames to video bytes (in-memory).

    Args:
        frames: Video frames as numpy array of shape (num_frames, height, width, 3)
            with uint8 values 0-255.
        fps: Frames per second for the output video.
        output_format: Container format — "mp4", "webm".

    Returns:
        Encoded video as bytes.

    Raises:
        ImportError: If imageio is not available.
        RuntimeError: If encoding fails.
    """
    import tempfile

    import av

    # Defensive squeeze: MediaOutput.video is (B, T, H, W, C) since TRT-LLM rc9.
    if frames.ndim == 5 and frames.shape[0] == 1:
        frames = frames[0]
    frames = np.ascontiguousarray(frames, dtype=np.uint8)
    num_frames, height, width, _ = frames.shape

    if output_format == "mp4":
        codec = "libx264"
    elif output_format == "webm":
        codec = "libvpx-vp9"
    else:
        raise ValueError(f"No codec specified for response format: {output_format}")

    logger.info(
        f"Encoding {num_frames} frames to {output_format} ({codec}, software) at {fps} fps"
    )

    # Software encode via PyAV: the worker runs on NVENC-less datacenter GPUs
    # (B200/H100/A100) where h264_nvenc has no capable device, and the in-tree
    # imageio ffmpeg has no software h264. PyAV bundles libx264. Encode to a temp
    # file (not an in-memory pipe) for reliability, then read the bytes back.
    tmp = tempfile.NamedTemporaryFile(suffix=f".{output_format}", delete=False)
    tmp.close()
    try:
        container = av.open(tmp.name, mode="w")
        try:
            stream = container.add_stream(codec, rate=fps)
            stream.width = width
            stream.height = height
            stream.pix_fmt = "yuv420p"
            stream.codec_context.thread_count = video_encode_threads()
            if codec == "libx264":
                stream.options = {"crf": "18", "preset": "veryfast"}
            # BT.709 color tags so players don't render the clip washed-out.
            try:
                cc = stream.codec_context
                cc.color_primaries = 1  # AVCOL_PRI_BT709
                cc.color_trc = 1        # AVCOL_TRC_BT709
                cc.colorspace = 1       # AVCOL_SPC_BT709
                cc.color_range = 1      # AVCOL_RANGE_MPEG (limited / "tv")
            except Exception:  # noqa: BLE001
                logger.warning("could not set BT.709 color tags on the stream")
            for i in range(num_frames):
                frame = av.VideoFrame.from_ndarray(frames[i], format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():  # flush encoder
                container.mux(packet)
        finally:
            container.close()
        with open(tmp.name, "rb") as fh:
            video_bytes = fh.read()
        logger.info(f"Encoded video to {len(video_bytes)} bytes")
        return video_bytes
    except Exception as e:
        logger.error(f"Failed to encode video to bytes: {e}")
        raise RuntimeError(f"Video encoding to bytes failed: {e}") from e
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
