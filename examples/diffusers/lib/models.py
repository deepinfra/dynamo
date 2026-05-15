# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pydantic request/response models for the ``/v1/videos`` endpoint.

These are model-agnostic: any video-pipeline backend (LTX-2 today,
future models) accepts the same request envelope.
"""

from pydantic import BaseModel, Field


class NvExtVideoCreateRequest(BaseModel):
    fps: int = Field(default=24, description="Frames per second")
    num_frames: int | None = Field(
        default=121, description="Total frames; overrides fps * seconds"
    )
    num_inference_steps: int = Field(default=5, description="Diffusion inference steps")
    guidance_scale: float = Field(
        default=1.0, description="Classifier-free guidance scale"
    )
    seed: int | None = Field(default=10, description="RNG seed for reproducibility")
    negative_prompt: str | None = Field(
        default=None, description="Text to avoid in generation"
    )


class VideoCreateRequest(BaseModel):
    prompt: str = Field(description="Text description of the desired video")
    model: str = Field(description="HuggingFace model path")
    size: str = Field(default="1920x1088", description="Frame dimensions as 'WxH'")
    seconds: int = Field(
        default=5, description="Clip duration; used when nvext.num_frames is unset"
    )
    user: str | None = Field(default=None)
    nvext: NvExtVideoCreateRequest = Field(default_factory=NvExtVideoCreateRequest)


class VideoData(BaseModel):
    b64_json: str | None = Field(default=None, description="Base64-encoded MP4 video")
    mime_type: str = Field(default="video/mp4")


class VideoCreateResponse(BaseModel):
    id: str
    object: str = "video"
    created: int
    model: str
    status: str = "complete"
    data: list[VideoData]
