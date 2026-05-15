# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generic video-pipeline infrastructure for diffusers workers.

This package is model-agnostic. Per-model code (LTX-2, etc.) lives in
sibling packages (``ltx2/``, future models) and calls into ``lib``.
"""
