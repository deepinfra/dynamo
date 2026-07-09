# Cosmos3 Dynamo videogen worker (public cosmos-framework)

DeepInfra serving overlay for `nvidia/Cosmos3-Nano` and `nvidia/Cosmos3-Super`
on NVIDIA's **public** [cosmos-framework](https://github.com/NVIDIA/cosmos-framework),
replacing the retired private EA fork (`deepinfra/cosmos3`, archived — see
provenance below).

## What lives here

- `worker.py` — the Dynamo videogen worker: registers with the Dynamo frontend
  (`ModelType.Videos`), translates deepapi requests to
  `cosmos_framework.inference` calls, multi-GPU via torchrun with a gloo
  control-plane group (idle NCCL-watchdog fix). `--standalone` runs a plain
  FastAPI server for testing without any Dynamo infra.
- `Dockerfile` — overlay build: clones cosmos-framework at the pinned
  `COSMOS_FRAMEWORK_REF`, applies `patches/`, runs upstream's own build steps,
  bakes worker + HF prefetch (incl. gated `nvidia/Cosmos-Guardrail1`; the build
  `HF_TOKEN` account must have accepted that repo's auto-approve gate).
- `patches/` — the complete set of in-tree changes to upstream:
  - `0001-cap-encoder-threads.patch` — cap libx264 `-threads` (default 32,
    `DI_VIDEO_ENCODE_THREADS` to override) so mp4 encode doesn't grab every
    core on shared multi-GPU nodes.

## Updating upstream

Bump `COSMOS_FRAMEWORK_REF` in the Dockerfile and rebuild. If a patch fails to
apply, the build fails: rebase the patch, or delete it if upstream absorbed
the fix.

## Provenance

The EA-era history (19 PRs) lives in the archived private repo
`deepinfra/cosmos3`. Sixteen of those PRs were shims to load the public
diffusers weights on the private EA code — all obsolete: the public framework
loads the public checkpoints natively, ships the audio (AVAE) tokenizer,
fixes the discarded-parallelism-config bug (`inference/model.py`, fixed
upstream), and enables guardrails by default. What survives is `worker.py`
(EA PRs #1/#18), the encoder-thread cap (EA PR #19, now `patches/0001`), and
the Dockerfile's prefetch/bake strategy.
