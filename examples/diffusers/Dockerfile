# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Shared runtime image for Dynamo frontend and FastVideo workers.
FROM nvidia/cuda:13.1.1-devel-ubuntu24.04

RUN apt-get update \
 && apt-get install -yq libucx0 python3-dev python3-pip python3-venv git protobuf-compiler curl ffmpeg libclang-dev \
 && apt-get clean

COPY --from=ghcr.io/astral-sh/uv:0.10.11 /uv /uvx /bin/
ENV UV_LINK_MODE=copy

RUN uv venv /opt/dynamo/venv --python 3.12 \
 && . /opt/dynamo/venv/bin/activate \
 && uv pip install pip setuptools packaging ninja psutil uvloop wheel \
 && uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130 \
 && uv pip install flashinfer-python

ENV VIRTUAL_ENV=/opt/dynamo/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# Override at build time to target a different GPU architecture, e.g.:
#   docker build --build-arg TORCH_CUDA_ARCH_LIST="9.0a 10.0a" ...
ARG TORCH_CUDA_ARCH_LIST="10.0a"

# Lower MAX_JOBS if the build OOMs (machines with <96GB RAM and many CPU cores).
#   docker build --build-arg MAX_JOBS=2 ...
ARG MAX_JOBS=4

# Persist arch list into the final image env for introspection, and let the
# FastVideo install below see it. CMAKE_ARGS pins fastvideo-kernel to the same
# arch and disables the ThunderKittens (TK) kernel build — TK kernels have
# crashed during worker startup on B200 (SIGQUIT) in earlier image revisions.
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=100a -DFASTVIDEO_KERNEL_BUILD_TK=OFF"

# flash-attention ignores TORCH_CUDA_ARCH_LIST and uses its own FLASH_ATTN_CUDA_ARCHS variable.
# Translate from PyTorch format ("10.0a", space-separated with dots) to flash-attention
# format ("100a", semicolon-separated without dots).
RUN export FLASH_ATTN_CUDA_ARCHS=$(echo "${TORCH_CUDA_ARCH_LIST}" | sed 's/ /;/g; s/\.//g') \
 && echo "Building flash-attention for TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} FLASH_ATTN_CUDA_ARCHS=${FLASH_ATTN_CUDA_ARCHS} MAX_JOBS=${MAX_JOBS}" \
 && git clone https://github.com/RandNMR73/flash-attention \
 && cd flash-attention \
 && git switch fa4-compile \
 && uv pip install . --no-build-isolation \
 && uv pip install ./flash_attn/cute \
 && rm -rf ../flash-attention

# Install Dynamo with /v1/videos support.
RUN uv pip install ai-dynamo==1.1.1

# Pin FastVideo to a known-good SHA. Updating this is a deliberate act:
# the runtime image's compile cache is keyed on FastVideo's API surface
# (kwargs to from_pretrained, generate_video signature). Bumping this SHA
# requires re-baking the cache and re-tagging the image. Procedure in
# RUNBOOK § "Updating FastVideo".
ARG FASTVIDEO_SHA=70ee5d230cda9651b88ada04be97ed94179bc482

# Stamp the SHA into the runtime env so a running container can self-id:
#   docker exec <container> printenv FASTVIDEO_SHA
ENV FASTVIDEO_SHA=${FASTVIDEO_SHA}

# Install FastVideo at the pinned SHA from the public upstream repository.
RUN echo "Building FastVideo @ ${FASTVIDEO_SHA} for TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}" \
 && . /opt/dynamo/venv/bin/activate \
 && uv pip install setuptools_scm scikit-build-core cmake ninja \
 && git clone https://github.com/hao-ai-lab/FastVideo.git /tmp/FastVideo \
 && cd /tmp/FastVideo \
 && git checkout ${FASTVIDEO_SHA} \
 && git submodule update --init --recursive \
 && cd / \
 && uv pip install --no-build-isolation /tmp/FastVideo

ENV FASTVIDEO_VIDEO_CODEC=libx264
ENV FASTVIDEO_X264_PRESET=ultrafast

# NOTE: do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True here.
# That allocator uses cuMemMap-based virtual memory, which is incompatible
# with FastVideo's multiprocess CUDA tensor sharing -- the parent process's
# pipe.recv() of a CUDA tensor from the GPU worker subprocess hits
# `pidfd_getfd: Operation not permitted` because the container lacks
# CAP_SYS_PTRACE. Causes every generation to fail at the multiproc executor
# layer. If we ever switch the allocator, also need to add
# --cap-add=SYS_PTRACE to the container or stop using FastVideo's MP executor.

WORKDIR /opt/app
COPY . /opt/app/

# Pre-warm the page cache for /cache (compile artifacts) and /data/default
# (model weights) before the worker starts. Cuts ~50-100s off first inference
# per shape because torch.compile's .so loads and the model's weight reads
# hit RAM instead of disk. See entrypoint.sh for details.
ENTRYPOINT ["/opt/app/entrypoint.sh"]
