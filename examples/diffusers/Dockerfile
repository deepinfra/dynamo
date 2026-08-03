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

# Persist arch list; CMAKE_ARGS pins the fastvideo-kernel CUDA arch and disables
# the ThunderKittens kernel build (TK has crashed at worker startup on B200 in
# earlier revisions). CUDA_ARCH_CMAKE must match TORCH_CUDA_ARCH_LIST's target:
#   - 100a (default) -> Blackwell B200/sm_100 (LTX-2.3 fleet)
#   - 90a            -> Hopper H100/sm_90 (FastWan-QAD fleet; build with
#                       --build-arg TORCH_CUDA_ARCH_LIST=9.0a --build-arg CUDA_ARCH_CMAKE=90a)
# It's a separate arg (not derived) because the cmake form drops the dot ("9.0a"
# -> "90a") and we keep that mapping explicit rather than sed-munging it.
ARG CUDA_ARCH_CMAKE=100a
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH_CMAKE} -DFASTVIDEO_KERNEL_BUILD_TK=OFF"

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

# FA4 (flash-attn-cute) source swap: the XOR-op fork installed above is stale
# against cutlass-dsl 4.5+ (cute.core.ThrMma renamed away -> AttributeError at
# import, which kills any worker that selects FLASH_ATTN). FastVideo hit the
# same break and fixed it upstream (hao-ai-lab/FastVideo#1564) by pinning
# flash-attn-4 from Dao-AILab/flash-attention at a CuTe-DSL-4.6-compatible
# commit. Reproduce that exact pin (same as Dockerfile.dreamverse); do not
# bump the ref without re-validating.
ARG FA4_CUTE_REF=82d6441eec5d4dfec120153db2c0145ae855a083
RUN . /opt/dynamo/venv/bin/activate \
 && uv pip uninstall flash-attn-cute \
 && git clone https://github.com/Dao-AILab/flash-attention.git /tmp/fa-upstream \
 && cd /tmp/fa-upstream && git checkout ${FA4_CUTE_REF} \
 && cd / \
 && uv pip install --no-deps --no-build-isolation /tmp/fa-upstream/flash_attn/cute \
 && rm -rf /tmp/fa-upstream
# --no-deps: the pin declares nvidia-cutlass-dsl==4.6.0.dev0 (pre-release),
# which uv refuses and which would needlessly downgrade the venv's 4.6.0
# release; the environment already satisfies the real requirement.

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

# The upstream FA4 pin returns 4-tuples from _flash_attn_fwd; fastvideo's
# wrapper unpacks 2. Same [:2] fix FastVideo ships today (see
# Dockerfile.dreamverse, where this patch has been applied since the LTX-2.3
# FA4 work).
COPY patches/flash-attn-cute-fa4-tuple-fix.patch /tmp/flash-attn-cute-fa4-tuple-fix.patch

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
 && git apply --verbose /tmp/flash-attn-cute-fa4-tuple-fix.patch \
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

# FastWan decodes with the TAEHV tiny autoencoder (the ~2.4s path) instead of
# the full Wan VAE (~5.8s). Bake the single-file taehv module + the Wan2.1
# checkpoint into the fastwan family dir; fastwan.factory loads them. Gated so
# LTX images (which use the full VAE) don't carry it. Build the FastWan image
# with --build-arg INCLUDE_FASTWAN_TAEHV=1.
ARG INCLUDE_FASTWAN_TAEHV=0
RUN if [ "$INCLUDE_FASTWAN_TAEHV" = "1" ]; then \
      . /opt/dynamo/venv/bin/activate && \
      uv pip install imageio imageio-ffmpeg && \
      curl -fsSL https://raw.githubusercontent.com/madebyollin/taehv/main/taehv.py \
        -o /opt/app/fastwan/taehv.py && \
      curl -fsSL https://github.com/madebyollin/taehv/raw/main/taew2_1.pth \
        -o /opt/app/fastwan/taew2_1.pth && \
      test -s /opt/app/fastwan/taew2_1.pth ; \
    fi

# Page-cache warming + LD_LIBRARY_PATH unset happen in entrypoint.sh.
ENTRYPOINT ["/opt/app/entrypoint.sh"]
