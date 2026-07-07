# DeepInfra Cosmos3 Dynamo videogen worker image.
#
# Overlay build: clones NVIDIA/cosmos-framework at a pinned ref, applies the
# reviewed patches from ./patches, then follows upstream's own Dockerfile steps
# (same base image, uv-managed env) and bakes in the DeepInfra worker.
# There is no DeepInfra copy of the framework repo — this directory plus the
# pin below is the complete, reviewable definition of the image.
#
# Build (from this directory; token file must hold an HF token whose account
# has accepted the nvidia/Cosmos-Guardrail1 auto-gate — the secret mount keeps
# it out of image layers and history):
#   docker build --secret id=hf_token,src=/path/to/token-file \
#     -t localhost:30500/cosmos3-worker:v1 .

ARG CUDA_VERSION=13.0.2
ARG BASE_IMAGE=nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        ffmpeg \
        git \
        git-lfs \
        tree \
        wget

COPY --from=ghcr.io/astral-sh/uv:0.10.8 /uv /uvx /usr/local/bin/
ENV UV_LINK_MODE=copy
ENV UV_PYTHON_CACHE_DIR=/root/.cache/uv/python
ENV PATH="/root/.local/bin:$PATH"

# Upstream pin. Bumping this ref is THE upstream-update mechanism; if a patch
# in ./patches stops applying, the build fails loudly and the patch must be
# rebased (or dropped because upstream absorbed the fix).
ARG COSMOS_FRAMEWORK_REPO=https://github.com/NVIDIA/cosmos-framework.git
ARG COSMOS_FRAMEWORK_REF=463ac142b47c637098a7e86d0631252f82b65418

WORKDIR /workspace
RUN git clone --filter=blob:none ${COSMOS_FRAMEWORK_REPO} /workspace && \
    git -C /workspace checkout ${COSMOS_FRAMEWORK_REF}

# DeepInfra patches to upstream code (reviewed in deepinfra/dynamo).
COPY patches /workspace/deepinfra-patches
RUN git -C /workspace apply --verbose deepinfra-patches/*.patch

# Upstream build steps (mirrors NVIDIA/cosmos-framework Dockerfile @ the pin,
# minus bind-mounts: the sources are already in the layer from the clone).
RUN --mount=type=cache,target=/root/.cache/uv uv python install
RUN echo "$CUDA_VERSION" | sed -E 's/^([0-9]+)\.([0-9]+).*/cu\1\2/' > /root/.cuda-name
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-editable --all-extras --group=$(cat /root/.cuda-name)
ENV PATH="/workspace/.venv/bin:$PATH"

# Triton bundled ptxas doesn't support latest GPU architectures
ENV TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas"

# Serving deps. fastapi/uvicorn for standalone mode; ai-dynamo + uvloop for
# production Dynamo mode. ai-dynamo provides the DistributedRuntime, frontend
# binary, and endpoint registration used by the videogen worker pattern.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install fastapi uvicorn ai-dynamo==1.1.1 uvloop

# Pre-install the HF CLI tool so that `uvx hf@1.13.0` reuses this
# installation instead of creating an ephemeral env missing `click`.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv tool install hf@1.13.0 --with click

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --no-deps -e /workspace

# Pre-fetch HuggingFace dependencies needed by cosmos_framework at model init
# time: Qwen tokenizer configs (*.json, *.txt — few MB), Wan VAE checkpoints,
# and the full Cosmos-Guardrail1 snapshot (the one gated repo — auto-approve
# gate; the token account must have accepted it). The token comes in via a
# BuildKit secret mount so it never persists in a layer or in image history.
# This eliminates any runtime dependency on HuggingFace availability.
# Revisions MUST match the pins in
# cosmos_framework/inference/common/checkpoints.py and
# cosmos_framework/auxiliary/guardrail/common/core.py at the ref above — the
# runtime looks up these exact revisions; a mismatch silently falls back to a
# live (unauthenticated, rate-limited) HF fetch at every pod boot.
RUN --mount=type=secret,id=hf_token \
    export HF_TOKEN=$(cat /run/secrets/hf_token 2>/dev/null || true) && \
    uvx hf@1.13.0 download --format=json \
        Qwen/Qwen3-0.6B \
        --repo-type model \
        --revision c1899de289a04d12100db370d81485cdf75e47ca \
        --include '*.json' --include '*.txt' && \
    uvx hf@1.13.0 download --format=json \
        Qwen/Qwen3-VL-2B-Instruct \
        --repo-type model \
        --revision 89644892e4d85e24eaac8bacfd4f463576704203 \
        --include '*.json' --include '*.txt' && \
    uvx hf@1.13.0 download --format=json \
        Qwen/Qwen3-VL-8B-Instruct \
        --repo-type model \
        --revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
        --include '*.json' --include '*.txt' && \
    uvx hf@1.13.0 download --format=json \
        Qwen/Qwen3-VL-32B-Instruct \
        --repo-type model \
        --revision 0cfaf48183f594c314753d30a4c4974bc75f3ccb \
        --include '*.json' --include '*.txt' && \
    uvx hf@1.13.0 download --format=json \
        Wan-AI/Wan2.1-T2V-14B \
        --repo-type model \
        --revision a064a6c71f5be440641209c07bf2a5ce7a2ff5e4 \
        Wan2.1_VAE.pth && \
    uvx hf@1.13.0 download --format=json \
        Wan-AI/Wan2.2-TI2V-5B \
        --repo-type model \
        --revision 921dbaf3f1674a56f47e83fb80a34bac8a8f203e \
        Wan2.2_VAE.pth && \
    uvx hf@1.13.0 download --format=json \
        nvidia/Cosmos-Guardrail1 \
        --repo-type model \
        --revision d6d4bfa899a71454a700907664f3e88f503950cf

# DeepInfra runtime settings.
# - LD_LIBRARY_PATH is cleared so the venv/CUDA libs resolve without host leakage.
# - torch.compile (inductor) and triton write kernel caches to a persistent
#   mountable location instead of an ephemeral home dir.
ENV LD_LIBRARY_PATH=""
ENV TORCHINDUCTOR_CACHE_DIR="/cache/torchinductor"
ENV TRITON_CACHE_DIR="/cache/triton"
RUN mkdir -p /cache/torchinductor /cache/triton

# DeepInfra Dynamo videogen worker. Runtime args
# (--served-model-name, --model, --num-gpus, ...) are appended to this command.
# worker.py self-relaunches under torchrun when --num-gpus > 1.
COPY worker.py /workspace/worker.py
ENTRYPOINT ["python3", "worker.py"]
