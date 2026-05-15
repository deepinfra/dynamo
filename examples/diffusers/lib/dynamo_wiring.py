# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dynamo registration helpers shared across video-pipeline workers."""

import logging
import os

from dynamo.llm import ModelInput, ModelType, register_llm  # type: ignore[attr-defined]

logger = logging.getLogger(__name__)


def get_worker_namespace() -> str:
    """
    Resolve Dynamo namespace for endpoint registration.

    Kubernetes operator injects DYN_NAMESPACE (and optionally a rollout suffix).
    Compose/local runs keep using the historical "dynamo" default.
    """
    namespace = os.environ.get("DYN_NAMESPACE", "dynamo")
    suffix = os.environ.get("DYN_NAMESPACE_WORKER_SUFFIX")
    if suffix:
        namespace = f"{namespace}-{suffix}"
    return namespace


async def register_model(endpoint, served_name: str, model_path: str) -> None:
    try:
        await register_llm(
            ModelInput.Text,  # type: ignore[attr-defined]
            ModelType.Videos,
            endpoint,
            model_path,
            served_name,
        )
        logger.info(
            "Successfully registered model: %s (path=%s)", served_name, model_path
        )
    except Exception as e:
        logger.error("Failed to register model: %s", e, exc_info=True)
        raise RuntimeError("Model registration failed") from e
