# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LTX-2.3 i2v conditioning-image input resolver.

Resolves an API-supplied conditioning image (HTTP(S) URL, ``data:`` URI, or raw
base64) to raw bytes. The caller (``lib.backend.create_video``) runs this off
the event loop (``asyncio.to_thread``), writes the bytes to a temp file, and
builds the FastVideo LTX-2 i2v contract:
    generate_video(..., ltx2_images=[(path, frame_index, strength)],
                        ltx2_image_crf=<crf>)
Tuple shape (str, int, float) and kwarg names verified against the pinned target
FastVideo SHA (pipeline_batch_info.py: ltx2_images / ltx2_image_crf). See the
LTX-2.3 distilled i2v example (basic_ltx2_3_distilled_i2v.py).
"""
from __future__ import annotations

import base64
import binascii
import io
import ipaddress
import logging
import socket
import urllib.request
from urllib.parse import urlparse

from PIL import Image

logger = logging.getLogger(__name__)

_MAX_IMAGE_BYTES = 32 * 1024 * 1024
_DOWNLOAD_TIMEOUT_S = 20


def _decode_data_uri_or_b64(value: str) -> bytes:
    payload = value
    if value.startswith("data:"):
        _, _, payload = value.partition(",")  # data:[<mediatype>][;base64],<data>
    try:
        return base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("conditioning image is not valid base64 / data-URI") from exc


def _assert_public_host(url: str) -> None:
    """SSRF guard: reject URLs that resolve to a non-public address.

    The conditioning-image URL is customer-supplied and fetched server-side, so
    without this an attacker could point it at cloud metadata (169.254.169.254),
    localhost, or internal/private ranges. Resolve the host and require every
    resolved address to be globally routable. (Residual: DNS-rebinding TOCTOU
    between resolve and connect is not covered by this check.)
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("conditioning image URL must be http or https")
    host = parsed.hostname
    if not host:
        raise ValueError("conditioning image URL has no host")
    try:
        infos = socket.getaddrinfo(
            host, parsed.port or (443 if parsed.scheme == "https" else 80)
        )
    except socket.gaierror as exc:
        raise ValueError("conditioning image URL host does not resolve") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global or ip.is_multicast:
            raise ValueError("conditioning image URL resolves to a non-public address")


class _SsrfSafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Re-validate the target host on every redirect hop (blocks redirect-to-internal)."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        _assert_public_host(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_SSRF_SAFE_OPENER = urllib.request.build_opener(_SsrfSafeRedirectHandler())


def _fetch_url(url: str) -> bytes:
    _assert_public_host(url)  # validate before connecting; redirects re-validated by the opener
    req = urllib.request.Request(url, headers={"User-Agent": "deepinfra-ltx2"})
    with _SSRF_SAFE_OPENER.open(req, timeout=_DOWNLOAD_TIMEOUT_S) as resp:
        data = resp.read(_MAX_IMAGE_BYTES + 1)
    if len(data) > _MAX_IMAGE_BYTES:
        raise ValueError("conditioning image exceeds %d bytes" % _MAX_IMAGE_BYTES)
    return data


def _validate_decodable_image(data: bytes) -> None:
    """Confirm ``data`` is a decodable image BEFORE it reaches the GPU subprocess.

    A non-image URL (HTML / redirect / 404 body) or a corrupt payload would
    otherwise raise ``PIL.UnidentifiedImageError`` *inside* the resident pool
    worker, killing that subprocess and forcing an ~8.5min cold recompile on the
    next request -- i.e. a user could trigger a recompile with bad input.
    Validating here (the main process, before the GPU subprocess and the generate
    lock) turns that into a fast, clean ``ValueError`` instead.
    """
    if not data:
        raise ValueError("conditioning image is empty")
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.verify()
    except Exception as exc:  # UnidentifiedImageError, OSError, ...
        raise ValueError("conditioning image is not a decodable image") from exc


def resolve_image_bytes(image: str) -> bytes:
    """Resolve an image reference (URL | data-URI | raw base64) to raw bytes.

    Blocking (network for URLs) -- callers run it via ``asyncio.to_thread`` so it
    never stalls the event loop. Enforces a size cap and a download timeout, and
    validates the result is a decodable image so malformed input fails fast in the
    main process instead of crashing the GPU subprocess (which would force a
    recompile).
    """
    if image.startswith(("http://", "https://")):
        data = _fetch_url(image)
    else:
        data = _decode_data_uri_or_b64(image)
    _validate_decodable_image(data)
    return data
