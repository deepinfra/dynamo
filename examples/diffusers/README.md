<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# FastVideo Video Diffusion Example

Dynamo backend worker for FastVideo-style video-diffusion models. Today
serves LTX-2; the structure is designed so additional video models
(future onboarding) plug in as new per-model packages without
duplicating the pool / IPC / metrics infrastructure.

## Layout

```
examples/diffusers/
├── worker.py            top-level shim — dispatches --pool-worker
│                        invocations into lib.pool, otherwise calls
│                        the LTX-2 worker's main_cli
├── lib/                 generic video-pipeline infrastructure
│   ├── pool.py            SubprocessPool, Connection-based IPC,
│   │                      _pool_worker_main, dispatch entry,
│   │                      _set_parent_death_signal
│   ├── backend.py         GenericVideoBackend: Dynamo endpoint,
│   │                      legacy in-process path, pool routing path
│   ├── metrics.py         video_pool_* Prometheus series (with the
│   │                      `model` label sourced per-pod)
│   ├── models.py          Pydantic request/response models
│   ├── menu.py            shape-menu hash algorithm + boot-assertion
│   └── dynamo_wiring.py   get_worker_namespace, register_model
├── ltx2/                LTX-2-specific code
│   ├── worker.py          main_cli: CLI parse, backend setup,
│   │                      Dynamo registration
│   ├── factory.py         load_model(): VideoGenerator.from_pretrained
│   │                      shared between legacy + pool paths
│   ├── config.py          canonical kwargs (cache-keying)
│   ├── shapes.json        shape menu
│   ├── warmup.py          per-shape compile-cache producer
│   ├── benchmark.py       post-bake validation harness
│   ├── preflight_test.py  standalone preflight smoke-test
│   ├── ARCHITECTURE.md    why the worker is shaped this way
│   ├── RUNBOOK.md         operational procedures
│   ├── test_config.py     pin ship-path kwargs
│   └── test_shapes.py     pin the shape-menu hash
├── benchmark.py is now under ltx2/ — see above
├── Dockerfile, entrypoint.sh, run-benchmark.sh, preflight_test.py
└── deploy/, local/        unchanged k8s / compose manifests
```

## Further reading

- LTX-2 architecture + design rationale: [`ltx2/ARCHITECTURE.md`](ltx2/ARCHITECTURE.md)
- LTX-2 operational procedures: [`ltx2/RUNBOOK.md`](ltx2/RUNBOOK.md)
- Upstream Dynamo docs:
  [FastVideo - Dynamo Docs](https://docs.nvidia.com/dynamo/dev/user-guides/diffusion/fastvideo)
- [FastVideo - GitHub](../../docs/features/diffusion/fastvideo.md)
