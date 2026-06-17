# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Shared runtime image for Dynamo frontend and FastVideo workers.
FROM nvidia/cuda:13.1.1-devel-ubuntu24.04

RUN apt-get update \
 && apt-get install -yq libucx0 python3-dev python3-pip python3-venv git protobuf-compiler curl ffmpeg libclang-dev \
 && apt-get clean

COPY --from=ghcr.io/astral-sh/uv:0.10.11 /uv /uvx /bin/
ENV UV_LINK_MODE=copy

# Pin the torch family to the CUDA-13.0 build across EVERY uv resolution in
# this image. The base is CUDA 13.1, and flash-attention's cpp_extension build
# rejects a torch whose compiled CUDA major differs. A later unpinned install
# (flashinfer-python, ai-dynamo, FastVideo) otherwise re-resolves torch from
# PyPI and silently swaps in the cu128 (CUDA 12.8) wheel -> flash-attn fails
# with "detected CUDA version (13.1) mismatches ... (12.8)". torchaudio is the
# same trap one layer deeper: FastVideo's transformers 5.x imports torchaudio
# at module load, and the PyPI torchaudio wheel is cu128-linked (dlopen of
# libcudart.so.12, absent here) -> `import fastvideo` dies. Pin torch +
# torchvision + torchaudio to +cu130 (torchcodec/torchdata already resolve
# correctly). The constraint forbids any non-+cu130 build; UV_EXTRA_INDEX_URL
# keeps the +cu130 wheels reachable from installs whose primary index is PyPI;
# unsafe-first-match lets uv fall through to that index when PyPI's wheel
# violates the constraint, while every other package still resolves from PyPI
# first (as prod's frozen LTX-2 image does). These versions match prod exactly.
RUN printf 'torch==2.11.0+cu130\ntorchvision==0.26.0+cu130\ntorchaudio==2.11.0+cu130\n' > /opt/torch-constraint.txt
ENV UV_CONSTRAINT=/opt/torch-constraint.txt
ENV UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu130
ENV UV_INDEX_STRATEGY=unsafe-first-match

RUN uv venv /opt/dynamo/venv --python 3.12 \
 && . /opt/dynamo/venv/bin/activate \
 && uv pip install pip setuptools packaging ninja psutil uvloop wheel \
 && uv pip install torch==2.11.0+cu130 torchvision==0.26.0+cu130 --index-url https://download.pytorch.org/whl/cu130 \
 && uv pip install flashinfer-python==0.6.11

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
 # Use FastVideo's exact FA4 fork (XOR-op/flash-attention@fa4-compile, the
 # flash-attn-cute source in their pyproject [dreamverse] extra) instead of the
 # RandNMR73 fork (a separate Dao-AILab fork). The FA4 cute kernels ARE the
 # spatial-temporal attention; a divergent fork is the prime suspect for
 # morphing-instead-of-motion / artifacts. Aligning to their validated fork.
 && git clone https://github.com/XOR-op/flash-attention \
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
# LTX-2.3 build: bumped from 70ee5d23 (#1239, 2026-04-23, pre-LTX-2.3) to
# 633d3935 (main, 2026-06-12) which has LTX-2.3 + NVFP4(renamed) + i2v + audio.
# The LTX-2 image is FROZEN on the old SHA (retiring); only LTX-2.3 builds from
# this branch. (One shared Dockerfile can't build both families -- different
# FastVideo versions -- which is fine since LTX-2 is never rebuilt.)
ARG FASTVIDEO_SHA=633d39356804e63478d242611e992dc8e1af3caa

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
 # FastVideo's [tool.uv.sources] pins the torch family (torch/torchvision/
 # torchaudio) to the "pytorch-cu128" index (download.pytorch.org/whl/cu128) --
 # uv honors that project-level source over any --index-url we pass. But this
 # image is CUDA 13.1 with cu130 torch: flash-attn is already compiled against
 # cu130 (matching prod's proven B200 stack), and a cu128 torchaudio dlopens
 # libcudart.so.12 (absent here). torch 2.11.0 is the same code either way --
 # +cuXXX only selects the bundled CUDA runtime -- so retarget FastVideo's torch
 # index to cu130 to match this base instead of letting it downgrade the stack.
 && sed -i 's|download.pytorch.org/whl/cu128|download.pytorch.org/whl/cu130|g' /tmp/FastVideo/pyproject.toml \
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
