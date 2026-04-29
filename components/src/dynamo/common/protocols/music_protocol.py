# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Protocol types for music generation.

Distinct from the TTS protocol in audio_protocol.py — music generation has
its own request shape (caption + lyrics + tempo/key/duration) and is served
by a different worker (e.g. ACE-Step). The endpoint is /v1/audio/generations,
parallel to /v1/audio/speech for TTS.

Note: until the Rust HTTP frontend grows native /v1/audio/generations support,
clients reach the worker via the Dynamo runtime directly. These Pydantic
models are the source of truth for the request/response contract.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class MusicNvExt(BaseModel):
    """NVIDIA extensions for music generation requests."""

    bpm: Optional[int] = Field(default=None, ge=30, le=300)
    """Tempo in beats per minute."""

    keyscale: Optional[str] = None
    """Musical key/scale, e.g. 'C major', 'A minor'."""

    num_inference_steps: Optional[int] = None
    """DiT denoising steps (default depends on model variant)."""

    guidance_scale: Optional[float] = None
    """Classifier-free guidance scale."""

    seed: Optional[int] = None
    """RNG seed for reproducibility (-1 or None for random)."""

    thinking: Optional[bool] = None
    """Enable LM planner reasoning."""

    batch_size: Optional[int] = Field(default=None, ge=1, le=8)
    """Number of clips to generate per request."""

    ref_audio: Optional[str] = None
    """Reference audio (URL or base64) used for style conditioning."""

    negative_prompt: Optional[str] = None
    """Optional negative caption."""


class NvCreateMusicRequest(BaseModel):
    """Request for music generation (/v1/audio/generations endpoint)."""

    prompt: str
    """Caption / text description of the desired music."""

    model: str
    """The music generation model to use."""

    lyrics: Optional[str] = None
    """Vocal lyrics; pass '[Instrumental]' to suppress vocals."""

    duration: Optional[float] = Field(default=None, ge=-1, le=600)
    """Clip duration in seconds (-1 for model-chosen, max 600s)."""

    response_format: Optional[Literal["wav", "flac", "mp3"]] = "flac"
    """Output audio container."""

    user: Optional[str] = None
    """Optional user identifier."""

    nvext: Optional[MusicNvExt] = None
    """NVIDIA extensions."""


class MusicData(BaseModel):
    """Single generated music clip."""

    url: Optional[str] = None
    """URL of the generated audio (if response_format is 'url')."""

    b64_json: Optional[str] = None
    """Base64-encoded audio payload."""

    mime_type: str = "audio/flac"
    """MIME type of the encoded audio."""

    sample_rate: Optional[int] = None
    """Sample rate of the generated audio."""

    duration_s: Optional[float] = None
    """Actual duration of the generated clip in seconds."""

    seed: Optional[int] = None
    """Seed used to generate this clip."""

    lrc: Optional[str] = None
    """Timestamped lyrics in LRC format, when produced by the LM planner."""


class NvMusicResponse(BaseModel):
    """Response structure for music generation."""

    id: str
    """Unique identifier for the response."""

    object: str = "audio.music"
    """Object type."""

    model: str
    """Model used for generation."""

    status: str = "completed"
    """Generation status."""

    progress: int = 100
    """Progress percentage (0-100)."""

    created: int
    """Unix timestamp of creation."""

    data: list[MusicData] = []
    """List of generated music clips."""

    error: Optional[str] = None
    """Error message if generation failed."""

    inference_time_s: Optional[float] = None
    """Inference time in seconds."""
