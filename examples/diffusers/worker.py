#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Top-level shim. The production deployment invokes ``python3 worker.py``;
this module dispatches to the LTX-2 worker entry point.

Pool-worker subprocesses are also spawned by invoking this file with
``--pool-worker``; in that case we short-circuit into
``lib.pool._pool_worker_dispatch_if_requested`` BEFORE importing
``ltx2.worker`` (which pulls in dynamo.runtime / fastvideo / etc.).
Skipping those imports is the difference between a snappy subprocess
cold-start and one that pays for the full parent-side import tree.

ALL entry-point logic lives inside the ``if __name__ == "__main__":``
guard. Without that guard, Python's ``multiprocessing`` module would
re-execute the dispatch path every time a downstream library (e.g.
FastVideo's ``multiproc_executor``) spawns a subprocess that
re-imports ``__main__`` -- the second-level worker children would
inherit ``--pool-worker`` in ``sys.argv`` and bomb on missing args.

When onboarding additional video models, this shim grows a dispatcher
(env-var-selected ``from <family>.worker import main_cli``). For now
single-model so single import.
"""
import pathlib
import sys

# Make /opt/app discoverable so the ``lib`` and ``ltx2`` packages
# resolve regardless of how the entry point is invoked. Safe to run
# at import time -- it only mutates sys.path.
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))


if __name__ == "__main__":
    if "--pool-worker" in sys.argv:
        # Pool subprocesses load only what they need: lib.pool plus
        # whatever the --model-factory dotted reference imports lazily
        # inside its body. No dynamo.runtime, no fastvideo at this
        # layer.
        from lib.pool import _pool_worker_dispatch_if_requested

        _pool_worker_dispatch_if_requested()  # never returns; calls sys.exit
    else:
        # Family dispatcher (per this shim's design note). The LTX-2.3 image
        # sets VIDEO_MODEL_FAMILY=ltx23 (via the model config's extra_env);
        # default stays ltx2 so the existing LTX-2 image is unaffected.
        import importlib
        import os

        family = os.environ.get("VIDEO_MODEL_FAMILY", "ltx2")
        main_cli = importlib.import_module(f"{family}.worker").main_cli
        main_cli()
