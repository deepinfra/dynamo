# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for dynamo.common.utils.video_utils module."""

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def make_frames(n=3, h=16, w=16) -> np.ndarray:
    """Return a small uint8 frame array (n, h, w, 3). Even dims for libx264."""
    return np.zeros((n, h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# encode_to_video_bytes — PyAV/libx264 software encode (no NVENC, no imageio)
# ---------------------------------------------------------------------------


class TestEncodeToVideoBytes:
    """Tests for encode_to_video_bytes(): software-encodes via PyAV so it works on
    NVENC-less datacenter GPUs. These exercise the real encoder (skipped if PyAV is
    unavailable) rather than mocking, since the whole point is that the bytes decode."""

    def test_mp4_returns_h264_bytes(self):
        av = pytest.importorskip("av")
        from dynamo.common.utils.video_utils import encode_to_video_bytes

        out = encode_to_video_bytes(make_frames(n=5), fps=8, output_format="mp4")
        assert isinstance(out, bytes) and len(out) > 0
        with av.open(io.BytesIO(out)) as container:
            assert container.streams.video[0].codec_context.name == "h264"

    def test_webm_returns_vp9_bytes(self):
        av = pytest.importorskip("av")
        from dynamo.common.utils.video_utils import encode_to_video_bytes

        out = encode_to_video_bytes(make_frames(n=5), fps=8, output_format="webm")
        assert isinstance(out, bytes) and len(out) > 0
        with av.open(io.BytesIO(out)) as container:
            assert container.streams.video[0].codec_context.name in (
                "vp9",
                "libvpx-vp9",
            )

    def test_mp4_tags_bt709(self):
        av = pytest.importorskip("av")
        from dynamo.common.utils.video_utils import encode_to_video_bytes

        out = encode_to_video_bytes(make_frames(n=5), fps=16, output_format="mp4")
        with av.open(io.BytesIO(out)) as container:
            cc = container.streams.video[0].codec_context
            # BT.709 primaries/transfer/colorspace so players don't render washed-out.
            assert int(cc.color_primaries) == 1
            assert int(cc.color_trc) == 1
            assert int(cc.colorspace) == 1

    def test_squeezes_5d_batch_dim(self):
        av = pytest.importorskip("av")
        from dynamo.common.utils.video_utils import encode_to_video_bytes

        # MediaOutput.video is (B, T, H, W, C) since TRT-LLM rc9; encode must squeeze B=1.
        frames_5d = make_frames(n=5)[None]  # (1, 5, 16, 16, 3)
        out = encode_to_video_bytes(frames_5d, fps=8, output_format="mp4")
        with av.open(io.BytesIO(out)) as container:
            assert container.streams.video[0].codec_context.name == "h264"

    def test_unsupported_format_raises_value_error(self):
        pytest.importorskip("av")
        from dynamo.common.utils.video_utils import encode_to_video_bytes

        with pytest.raises(ValueError):
            encode_to_video_bytes(make_frames(), output_format="avi")


# ---------------------------------------------------------------------------
# limit_video_worker_threads / video_encode_threads — shared video-worker caps
# ---------------------------------------------------------------------------


class TestVideoWorkerThreadCaps:
    """The three env-overridable CPU-thread knobs + the OpenMP wait policy."""

    def test_encode_threads_default(self, monkeypatch):
        monkeypatch.delenv("DI_VIDEO_ENCODE_THREADS", raising=False)
        from dynamo.common.utils.video_utils import (
            DEFAULT_VIDEO_ENCODE_THREADS,
            video_encode_threads,
        )

        assert video_encode_threads() == DEFAULT_VIDEO_ENCODE_THREADS

    def test_encode_threads_env_override(self, monkeypatch):
        monkeypatch.setenv("DI_VIDEO_ENCODE_THREADS", "7")
        from dynamo.common.utils.video_utils import video_encode_threads

        assert video_encode_threads() == 7

    def test_limit_sets_wait_policy_compile_and_torch(self, monkeypatch):
        import os

        for k in (
            "OMP_WAIT_POLICY",
            "KMP_BLOCKTIME",
            "TORCHINDUCTOR_COMPILE_THREADS",
            "DI_VIDEO_COMPILE_THREADS",
            "DI_VIDEO_TORCH_THREADS",
        ):
            monkeypatch.delenv(k, raising=False)
        from dynamo.common.utils import video_utils

        fake_torch = MagicMock()
        with patch.dict("sys.modules", {"torch": fake_torch}):
            video_utils.limit_video_worker_threads()

        assert os.environ["OMP_WAIT_POLICY"] == "PASSIVE"
        assert os.environ["KMP_BLOCKTIME"] == "0"
        assert os.environ["TORCHINDUCTOR_COMPILE_THREADS"] == str(
            video_utils.DEFAULT_VIDEO_COMPILE_THREADS
        )
        fake_torch.set_num_threads.assert_called_once_with(
            video_utils.DEFAULT_VIDEO_TORCH_THREADS
        )

    def test_limit_torch_threads_env_override(self, monkeypatch):
        monkeypatch.setenv("DI_VIDEO_TORCH_THREADS", "9")
        from dynamo.common.utils import video_utils

        fake_torch = MagicMock()
        with patch.dict("sys.modules", {"torch": fake_torch}):
            video_utils.limit_video_worker_threads()

        fake_torch.set_num_threads.assert_called_once_with(9)
