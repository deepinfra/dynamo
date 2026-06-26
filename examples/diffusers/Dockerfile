# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Shared runtime image for Dynamo frontend and FastVideo workers.
#
# This image mirrors FastVideo's OWN validated environment for LTX-2.3 (their
# apps/dreamverse/docker/Dockerfile + pyproject.toml @ the pinned SHA), rather
# than forcing FastVideo onto a different CUDA stack. Earlier we built on
# CUDA 13.1 / torch cu130 and spent enormous effort fighting torch/torchaudio
# index mismatches AND still got morphing/brightness-pulsing video. FastVideo
# validates on CUDA 12.9 / torch cu128 / transformers 4.57.3 / their XOR-op FA4
# fork. On a cu128 base, PyPI's default torch is also cu128, so NO index
# constraints are needed. Match their stack exactly to reproduce their result.
FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04

RUN apt-get update \
 && apt-get install -yq libucx0 python3-dev python3-pip python3-venv git protobuf-compiler curl ffmpeg libclang-dev \
 && apt-get clean

COPY --from=ghcr.io/astral-sh/uv:0.10.11 /uv /uvx /bin/
ENV UV_LINK_MODE=copy

# Version pins applied to EVERY uv resolution in this image. (cu128 base, so no
# +cuXXX index gymnastics needed -- these are plain version pins.)
#  - torch==2.11.0: FastVideo's pinned torch. WITHOUT this, the flash-attn
#    install (XOR-op repo) re-resolves and DOWNGRADES torch to 2.10.0, which
#    then ABI-mismatches the 2.11 that FastVideo re-pins -> broken. The pin holds
#    torch at 2.11.0 through every later install.
#  - transformers==4.57.3: FastVideo's VALIDATED version. Their pyproject floor
#    is >=4.57.3, so a fresh install otherwise pulls 5.x (a major bump that
#    changes text-encoder/tokenizer defaults; not what their published LTX-2.3
#    result was produced with).
RUN printf 'torch==2.11.0\ntransformers==4.57.3\n' > /opt/uv-constraint.txt
ENV UV_CONSTRAINT=/opt/uv-constraint.txt

# torch 2.11.0 cu128, matching FastVideo's pyproject [tool.uv.sources]
# (pytorch-cu128). Installed before flash-attn (its build needs torch present).
# cu128 == PyPI's default CUDA build, so downstream installs (flashinfer,
# ai-dynamo, FastVideo) don't drift the torch CUDA version.
RUN uv venv /opt/dynamo/venv --python 3.12 \
 && . /opt/dynamo/venv/bin/activate \
 && uv pip install pip setuptools packaging ninja psutil uvloop wheel \
 && uv pip install torch==2.11.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 \
 && uv pip install flashinfer-python

ENV VIRTUAL_ENV=/opt/dynamo/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# Override at build time to target a different GPU architecture, e.g.:
#   docker build --build-arg TORCH_CUDA_ARCH_LIST="9.0a 10.0a" ...
ARG TORCH_CUDA_ARCH_LIST="10.0a"

# Lower MAX_JOBS if the build OOMs (machines with <96GB RAM and many CPU cores).
ARG MAX_JOBS=4

# Persist arch list; CMAKE_ARGS pins fastvideo-kernel to sm_100 (Blackwell;
# CUDA 12.9 supports it) and disables the ThunderKittens kernel build (TK has
# crashed at worker startup on B200 in earlier revisions).
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=100a -DFASTVIDEO_KERNEL_BUILD_TK=OFF"

# flash-attention FA4 from FastVideo's exact fork (XOR-op/flash-attention@
# fa4-compile -- the flash-attn-cute source in their pyproject). The cute
# kernels ARE the spatial-temporal attention. flash-attn ignores
# TORCH_CUDA_ARCH_LIST and uses FLASH_ATTN_CUDA_ARCHS ("100a").
RUN export FLASH_ATTN_CUDA_ARCHS=$(echo "${TORCH_CUDA_ARCH_LIST}" | sed 's/ /;/g; s/\.//g') \
 && . /opt/dynamo/venv/bin/activate \
 && echo "Building flash-attention (XOR-op fa4-compile) for FLASH_ATTN_CUDA_ARCHS=${FLASH_ATTN_CUDA_ARCHS} MAX_JOBS=${MAX_JOBS}" \
 && git clone https://github.com/XOR-op/flash-attention \
 && cd flash-attention \
 && git switch fa4-compile \
 && uv pip install . --no-build-isolation \
 && uv pip install ./flash_attn/cute \
 && rm -rf ../flash-attention

# Install Dynamo with /v1/videos support (our serving layer).
RUN uv pip install ai-dynamo==1.1.1

# Pin FastVideo to a known-good SHA (LTX-2.3 + NVFP4 + i2v + audio).
ARG FASTVIDEO_SHA=633d39356804e63478d242611e992dc8e1af3caa
ENV FASTVIDEO_SHA=${FASTVIDEO_SHA}

# deepinfra Mega-Cache patch for FastVideo's worker child (the process that
# compiles): adds torch.compiler save/load_cache_artifacts so a fresh pod
# reuses baked compile artifacts instead of recompiling. COPY'd here (before
# the FastVideo build) so `git apply` below picks it up; the full app tree is
# only COPYd to /opt/app later. See examples/diffusers/ltx23/CACHING.md.
COPY patches/ltx23_gpu_worker_megacache.patch /tmp/megacache.patch

# deepinfra encode patch: cap libx264 -threads so the CPU mp4 encode can't grab
# the whole shared node (no NVENC on B200 -> CPU libx264). git apply'd below;
# default 32, env-overridable via FASTVIDEO_X264_THREADS. See PR description.
COPY patches/x264-threads-cap.patch /tmp/x264-threads-cap.patch

# Install FastVideo at the pinned SHA. No pyproject sed needed: the base is now
# cu128, which is exactly what FastVideo's [tool.uv.sources] targets, so the
# torch family resolves natively without any index gymnastics.
RUN echo "Building FastVideo @ ${FASTVIDEO_SHA} for TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}" \
 && . /opt/dynamo/venv/bin/activate \
 && uv pip install setuptools_scm scikit-build-core cmake ninja \
 && git clone https://github.com/hao-ai-lab/FastVideo.git /tmp/FastVideo \
 && cd /tmp/FastVideo \
 && git checkout ${FASTVIDEO_SHA} \
 && git submodule update --init --recursive \
 && git apply --verbose /tmp/megacache.patch \
 && git apply --verbose /tmp/x264-threads-cap.patch \
 && cd / \
 && uv pip install --no-build-isolation /tmp/FastVideo \
 && rm -rf /tmp/FastVideo

ENV FASTVIDEO_VIDEO_CODEC=libx264
ENV FASTVIDEO_X264_PRESET=ultrafast

# NOTE: do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True here --
# incompatible with FastVideo's multiprocess CUDA tensor sharing without
# CAP_SYS_PTRACE (see prior revisions).

WORKDIR /opt/app
COPY . /opt/app/

# Page-cache warming + LD_LIBRARY_PATH unset happen in entrypoint.sh.
ENTRYPOINT ["/opt/app/entrypoint.sh"]
