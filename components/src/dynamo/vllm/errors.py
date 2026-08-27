# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Translate vLLM request errors into Dynamo's HTTP error boundary."""

from vllm.exceptions import VLLMNotFoundError, VLLMUnprocessableEntityError

try:
    from vllm.exceptions import VLLMClientError
except ImportError:
    # vLLM < 0.27 has no VLLMError/VLLMClientError/VLLMServerError hierarchy
    # (added upstream 2026-08-18); VLLMNotFoundError/VLLMUnprocessableEntityError
    # exist standalone instead of inheriting from it. This shim keeps request
    # handling working against those older vLLM builds -- it just won't catch
    # vLLM's own NotFound/UnprocessableEntity errors as VLLMClientError there,
    # since they don't subclass this fallback.
    class VLLMClientError(Exception):  # type: ignore[no-redef]
        """Compatibility shim for vLLM < 0.27, which lacks this class."""

from dynamo.llm.exceptions import HttpError


def vllm_client_error_to_http_error(exc: VLLMClientError) -> HttpError:
    """Preserve the HTTP status assigned by vLLM's client-error hierarchy."""
    if isinstance(exc, VLLMUnprocessableEntityError):
        status_code = 422
    elif isinstance(exc, VLLMNotFoundError):
        status_code = 404
    else:
        status_code = 400
    return HttpError(status_code, str(exc))
