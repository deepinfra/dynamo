# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Runtime image for the ACE-Step music generation worker.
# Mirrors examples/diffusers/Dockerfile (FastVideo) but installs ACE-Step.

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

RUN apt-get update \
 && apt-get install -yq libucx0 python3-dev python3-pip python3-venv git \
    protobuf-compiler curl ffmpeg libsndfile1 libclang-dev \
 && apt-get clean

COPY --from=ghcr.io/astral-sh/uv:0.10.11 /uv /uvx /bin/
ENV UV_LINK_MODE=copy

RUN uv venv /opt/dynamo/venv --python 3.12 \
 && . /opt/dynamo/venv/bin/activate \
 && uv pip install pip setuptools packaging ninja psutil uvloop \
 && uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 \
 && uv pip install flashinfer-python

ENV VIRTUAL_ENV=/opt/dynamo/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# Override at build time to target a different GPU architecture, e.g.:
#   docker build --build-arg TORCH_CUDA_ARCH_LIST="9.0 9.0a" ...
# H100 = "9.0 9.0a"; A100 = "8.0"; Blackwell = "10.0 10.0a".
ARG TORCH_CUDA_ARCH_LIST="8.0 9.0 9.0a"

# Install Dynamo with audio support.
RUN uv pip install ai-dynamo==1.0.0

# Install ACE-Step from upstream. The project is distributed as a uv-managed
# source checkout; we install it as a regular pip package out of the cloned
# tree so the worker can `from acestep ...`.
ARG ACESTEP_REF=main
ENV ACESTEP_PROJECT_ROOT=/opt/ACE-Step-1.5
RUN echo "Installing ACE-Step ${ACESTEP_REF} for TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}" \
 && . /opt/dynamo/venv/bin/activate \
 && uv pip install hatchling setuptools_scm scikit-build-core cmake ninja \
 && uv pip install vllm \
 && git clone --depth 1 --branch "${ACESTEP_REF}" \
      https://github.com/ace-step/ACE-Step-1.5.git "${ACESTEP_PROJECT_ROOT}" \
 && uv pip install --no-build-isolation "${ACESTEP_PROJECT_ROOT}" \
 && uv pip install soundfile librosa

# Cache directory for HuggingFace weights (mount a PVC at this path in k8s).
ENV HF_HOME=/models/huggingface
ENV ACESTEP_CHECKPOINT_DIR=/models/acestep

WORKDIR /opt/app
COPY . /opt/app/
