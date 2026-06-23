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
    # i2v conditioning (LTX-2.3): omit image_url for text-to-video. i2v shares
    # the same shape_key / pool process as t2v (same WxH@frames) BUT compiles a
    # SEPARATE graph (measured 2026-06-19: ~8 extra fx graphs; an earlier comment
    # wrongly said it "rides the same compiled shape"). So boot-warm must warm
    # BOTH modes and each mode needs its own compile-cache blob. See
    # ltx23/CACHING.md.
    image_url: str | None = Field(
        default=None,
        description=(
            "i2v conditioning image: HTTP(S) URL, data: URI, or raw base64. "
            "Omit for text-to-video."
        ),
    )
    image_frame_index: int = Field(
        default=0, description="Frame to anchor the conditioning image at (i2v)."
    )
    image_strength: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Conditioning strength (i2v)."
    )
    image_crf: float = Field(
        default=0.0,
        description=(
            "JPEG re-encode quality for the conditioning image; 0 skips "
            "re-encoding already-compressed input (FastVideo default is 33)."
        ),
    )


class VideoCreateRequest(BaseModel):
    prompt: str = Field(description="Text description of the desired video")
    model: str = Field(description="HuggingFace model path")
    size: str = Field(default="1920x1088", description="Frame dimensions as 'WxH'")
    seconds: int = Field(
        default=5, description="Clip duration; used when nvext.num_frames is unset"
    )
    user: str | None = Field(default=None)
    # i2v conditioning image, TOP-LEVEL. This mirrors the Dynamo HTTP frontend's
    # NvCreateVideoRequest.input_reference -- the ONLY channel the frontend
    # forwards an i2v image through. The frontend's `nvext` is a typed struct
    # (VideoNvExt) that has no image field, so `nvext.image_url` is silently
    # dropped at the frontend; the image MUST arrive here. The frontend passes
    # this through opaquely (no fetch/decode), so create_video resolves it via
    # i2v_input.resolve_image_bytes (URL / data: URI / raw base64).
    input_reference: str | None = Field(default=None)
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
