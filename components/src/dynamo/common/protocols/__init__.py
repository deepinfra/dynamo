# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared protocol types used across multiple Dynamo backends.

This module provides protocol types for various modalities:
- video_protocol: NvCreateVideoRequest, NvVideosResponse for video generation
- music_protocol: NvCreateMusicRequest, NvMusicResponse for music generation
"""

from dynamo.common.protocols.music_protocol import (
    MusicData,
    MusicNvExt,
    NvCreateMusicRequest,
    NvMusicResponse,
)
from dynamo.common.protocols.video_protocol import (
    NvCreateVideoRequest,
    NvVideosResponse,
    VideoData,
)

__all__ = [
    "MusicData",
    "MusicNvExt",
    "NvCreateMusicRequest",
    "NvCreateVideoRequest",
    "NvMusicResponse",
    "NvVideosResponse",
    "VideoData",
]
