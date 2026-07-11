# DeepInfra Cosmos3 Dynamo videogen worker image.
#
# Overlay build: clones NVIDIA/cosmos-framework at a pinned ref, applies the
# reviewed patches from ./patches, then follows upstream's own Dockerfile steps
# (same base image, uv-managed env) and bakes in the DeepInfra worker.
# There is no DeepInfra copy of the framework repo — this directory plus the
# pin below is the complete, reviewable definition of the image.
#
# Build (from this directory; no HF token needed — all prefetched repos are
# public now that the gated Cosmos-Guardrail1 snapshot is no longer fetched):
#   docker build -t localhost:30500/cosmos3-worker:v1 .

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

# Upstream build steps (mirrors NVIDIA/cosmos-framework Dockerfile @ the pin,
# minus bind-mounts: the sources are already in the layer from the clone).
# NOTE: patches are applied LAST (near the entrypoint), not here. cosmos_framework
# is installed editable (`uv pip install -e`), so patching /workspace afterward is
# picked up live at runtime — and keeping the patch layer last means editing a
# patch rebuilds in seconds instead of re-running the HF prefetch below.
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

# Pre-fetch the public HuggingFace dependencies cosmos_framework needs at model
# init time: Qwen tokenizer configs (*.json, *.txt — few MB) and the Wan VAE
# checkpoints. All are public repos, so the build needs NO HF token. This
# eliminates any runtime dependency on HuggingFace availability.
# Revisions MUST match the pins in
# cosmos_framework/inference/common/checkpoints.py at the ref above — the
# runtime looks up these exact revisions; a mismatch silently falls back to a
# live (unauthenticated, rate-limited) HF fetch at every pod boot.
# NOTE: the gated nvidia/Cosmos-Guardrail1 snapshot is intentionally NOT fetched
# — guardrails are disabled in worker.py (guardrails=False), so it is never
# loaded at runtime. Re-add its download here (and the BuildKit hf_token secret
# it needs) if guardrails are ever turned back on.
RUN uvx hf@1.13.0 download --format=json \
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
        Wan2.2_VAE.pth

# DeepInfra runtime settings.
# - LD_LIBRARY_PATH is cleared so the venv/CUDA libs resolve without host leakage.
# - torch.compile (inductor) and triton write kernel caches to a persistent
#   mountable location instead of an ephemeral home dir.
ENV LD_LIBRARY_PATH=""
ENV TORCHINDUCTOR_CACHE_DIR="/cache/torchinductor"
ENV TRITON_CACHE_DIR="/cache/triton"
RUN mkdir -p /cache/torchinductor /cache/triton

# DeepInfra patches to upstream code (reviewed in deepinfra/dynamo). Applied
# last so a patch edit doesn't invalidate the expensive prefetch layer above;
# the editable install makes them live at runtime. Fails loudly if a patch no
# longer applies against COSMOS_FRAMEWORK_REF (the signal to rebase or drop it).
COPY patches /workspace/deepinfra-patches
RUN git -C /workspace apply --verbose deepinfra-patches/*.patch

# DeepInfra Dynamo videogen worker. MM2's dynamo deployment runs
# `python3 worker.py` with workingDir=/opt/app (the videogen-worker convention),
# so worker.py MUST live at /opt/app/worker.py — NOT /workspace (where the
# framework is cloned). cosmos_framework is installed editable, so it imports
# fine regardless of cwd. Runtime args (--served-model-name, --model=/data/default,
# --num-gpus) are appended by the deployment; worker.py self-relaunches under
# torchrun when --num-gpus > 1.
COPY worker.py /opt/app/worker.py
WORKDIR /opt/app
ENTRYPOINT ["python3", "worker.py"]
