# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Weight/input repacking for the conv2d kernels.

Several conv kernels don't consume the raw OIHW weight (or NCHW input) layout
directly — they need it reshaped into a kernel-local format for coalesced loads:
K-major padded tiles for the 1x1/general GEMM, [K_out, 9, C_pad] for the 3x3
kernels, channel-blocked NCHWc for the cblocked path, and the G·g·Gᵀ filter
transform for Winograd F(4x4,3x3). These packs are pure functions of the weight
tensor, so the results are LRU-cached keyed on (storage ptr, shape, dtype,
block, version): a weight repacks once and every later call with the same
weight is a cache hit, making the steady-state repack cost negligible.
"""

import os
from collections import OrderedDict

import torch

from aiter.ops.triton.conv._utils import BLOCK_K, _storage_ptr
from aiter.ops.triton.utils.conv_config_utils import has_conv_config

_DEFAULT_PACK_CACHE_MAXSIZE = 256


def _read_pack_cache_maxsize(default: int = _DEFAULT_PACK_CACHE_MAXSIZE) -> int:
    """Read AITER_TRITON_CONV_PACK_CACHE_SIZE, falling back to `default` for
    missing, non-integer, or non-positive values."""
    raw = os.environ.get("AITER_TRITON_CONV_PACK_CACHE_SIZE")
    if raw is None:
        return default
    try:
        value = int(raw.strip())
    except ValueError:
        return default
    return value if value > 0 else default


_PACK_CACHE_MAXSIZE = _read_pack_cache_maxsize()


class _LRUPackCache:
    """Bounded LRU for weight prepacks. Stores (src_tensor, item) — the
    strong ref to src keeps storage alive so the storage_ptr in the key
    cannot be reused by a different tensor while this entry lives."""

    def __init__(self, maxsize: int = _PACK_CACHE_MAXSIZE):
        self._d: OrderedDict[tuple, tuple] = OrderedDict()
        self._max = max(1, maxsize)

    def get(self, key):
        entry = self._d.get(key)
        if entry is None:
            return None
        self._d.move_to_end(key)
        return entry

    def put(self, key, src, item):
        self._d[key] = (src, item)
        self._d.move_to_end(key)
        while len(self._d) > self._max:
            self._d.popitem(last=False)


_PACK_CACHE = _LRUPackCache()
_PACK_CACHE_3x3 = _LRUPackCache()
_PACK_CACHE_WINOGRAD_F4X3 = _LRUPackCache()


def prepack_oihw_to_kmajor(w_oihw: torch.Tensor, block_k: int = BLOCK_K):
    K_out, C, R, S = w_oihw.shape
    K_red = C * R * S
    K_pad = ((K_red + block_k - 1) // block_k) * block_k
    w_rs = w_oihw.reshape(K_out, K_red)
    if K_pad != K_red:
        pad = torch.zeros(
            (K_out, K_pad - K_red), device=w_oihw.device, dtype=w_oihw.dtype
        )
        w_rs = torch.cat([w_rs, pad], dim=1)
    return w_rs.contiguous(), K_pad


def get_or_make_weight_pack(w_oihw: torch.Tensor, block_k: int = BLOCK_K):
    key = (
        _storage_ptr(w_oihw),
        tuple(w_oihw.shape),
        w_oihw.dtype,
        block_k,
    )
    entry = _PACK_CACHE.get(key)
    if entry is not None:
        return entry[1]
    item = prepack_oihw_to_kmajor(w_oihw, block_k)
    _PACK_CACHE.put(key, w_oihw, item)
    return item


def prepack_oihw_to_3x3(w_oihw: torch.Tensor, block_c: int = BLOCK_K):
    """Pack weights as [K_out, 9, C_pad] for 3x3 specialized kernel."""
    K_out, C, R, S = w_oihw.shape
    assert R == 3 and S == 3
    C_pad = ((C + block_c - 1) // block_c) * block_c
    w_rs = w_oihw.reshape(K_out, C, 9).permute(0, 2, 1).contiguous()  # [K_out, 9, C]
    if C_pad != C:
        pad = torch.zeros(
            (K_out, 9, C_pad - C), device=w_oihw.device, dtype=w_oihw.dtype
        )
        w_rs = torch.cat([w_rs, pad], dim=2)
    return w_rs.contiguous(), C_pad


def get_or_make_weight_pack_3x3(w_oihw: torch.Tensor, block_c: int = BLOCK_K):
    key = (
        _storage_ptr(w_oihw),
        tuple(w_oihw.shape),
        w_oihw.dtype,
        block_c,
    )
    cached = _PACK_CACHE_3x3.get(key)
    if cached is not None:
        return cached[1]
    item = prepack_oihw_to_3x3(w_oihw, block_c)
    _PACK_CACHE_3x3.put(key, w_oihw, item)
    return item


# Minimum input pixel count (N*H*W) at which the 3x3 NCHW-direct kernel is used
# instead of materializing an NCHWc pack. Measured on gfx1201 (see
# conv/DESIGN.md §5.3a): the direct kernel's channel-strided tile loads do not
# vectorize, which costs ~30% of kernel-only throughput on tiny spatial maps
# (N*H*W <= 196: 49x49 and 14x14 ResNet tails) but is far cheaper than the full
# extra read+write pass of the pack once the activation is large. The measured
# crossover sits between N*H*W = 196 (direct loses) and 784 (direct wins).
_NCHW_DIRECT_MIN_PIXELS = 512


def _nchw_direct_is_profitable(N: int, H: int, W: int) -> bool:
    """Whether the 3x3 NCHW-direct kernel should read the activation in place
    instead of paying for a materialized NCHWc pack.

    Two conditions, both routing decisions rather than tuning values:

    * the activation is large enough for the saved pass to outweigh the
      direct kernel's channel-strided loads (see the constant above), and
    * a ``CONV-3X3-NCHW`` tuning table ships for the running arch. The table
      is only measured on the arch it was swept on; where it is absent the
      router degrades to the materialized pack + ``CONV-3X3-CBLOCKED``, which
      is tuned everywhere. Without this probe the missing file would raise
      inside ``get_conv_config`` instead of falling back.
    """
    return N * H * W >= _NCHW_DIRECT_MIN_PIXELS and has_conv_config("CONV-3X3-NCHW")


def is_nchw_direct_view(x_blocked: torch.Tensor) -> bool:
    """True when ``x_blocked`` is the zero-copy NCHW *view* produced by
    :func:`prepack_nchw_to_cblocked` rather than a materialized NCHWc pack.

    The view carries the channel axis on the outer stride (``H*W``); a real
    NCHWc pack always has ``stride(-1) == 1``. When ``H*W == 1`` the two
    layouts are bit-identical, so mis-classifying that degenerate case is
    harmless.
    """
    return x_blocked.stride(-1) != 1


def prepack_nchw_to_cblocked(x: torch.Tensor, block_c: int = BLOCK_K):
    """Present an NCHW input as [N, C_blocks, H, W, Cb] for the 3x3 kernels.

    Two representations share one return contract — both have shape
    ``[N, C_blocks, H, W, Cb]`` with ``C_blocks * Cb == C_pad``, and
    ``.contiguous()`` on either yields the identical channel-blocked buffer:

    * **Zero-copy NCHW view** (large activations). ``as_strided`` re-expresses
      the caller's NCHW tensor with the channel axis split into
      ``(C_blocks, Cb)`` and moved last. Nothing is copied; the 3x3 NCHW-direct
      kernel reads the original activation in place. This removes an entire
      read+write pass over the activation on every call — the input pack is the
      one repack that cannot be cached, since the activation changes per call.
    * **Materialized NCHWc pack** (small activations). The original behaviour:
      channels are physically gathered so each pixel carries ``Cb`` contiguous
      channels, which is what ``_conv2d_3x3_cblocked_kernel`` and the Winograd
      cblocked input transform want.

    Callers that need real NCHWc storage (the Winograd cblocked path) call
    ``.contiguous()`` on the result; see ``_launch_winograd_f4x3_cblocked``.
    """
    N, C, H, W = x.shape
    Cb = block_c
    C_blocks = (C + Cb - 1) // Cb
    C_pad = C_blocks * Cb

    if not x.is_contiguous():
        x = x.contiguous()

    if C_pad != C:
        # The channel tail is zeroed (not left uninitialized) so that
        # `.contiguous()` on the view below reproduces the materialized pack
        # exactly, for consumers that do not mask on `c < C`.
        x_padded = torch.zeros((N, C_pad, H, W), device=x.device, dtype=x.dtype)
        x_padded[:, :C, :, :] = x
    else:
        x_padded = x

    if _nchw_direct_is_profitable(N, H, W):
        x_view = x_padded.as_strided(
            (N, C_blocks, H, W, Cb),
            (C_pad * H * W, Cb * H * W, W, 1, H * W),
        )
        return x_view, C_pad

    x_blocked = (
        x_padded.reshape(N, C_blocks, Cb, H, W).permute(0, 1, 3, 4, 2).contiguous()
    )
    return x_blocked, C_pad


def prepack_winograd_filter_f4x3(w_oihw: torch.Tensor, block_c: int = BLOCK_K):
    """Transform 3x3 filters for Winograd F(4x4,3x3). G @ g @ G^T for each (k,c).
    Input: [K_out, C, 3, 3] fp16.  Output: [36, K_out, C_pad] fp16."""
    K_out, C, R, S = w_oihw.shape
    assert R == 3 and S == 3
    C_pad = ((C + block_c - 1) // block_c) * block_c
    # G matrix (6x3)
    G = torch.tensor(
        [
            [1.0 / 4, 0.0, 0.0],
            [-1.0 / 6, -1.0 / 6, -1.0 / 6],
            [-1.0 / 6, 1.0 / 6, -1.0 / 6],
            [1.0 / 24, 1.0 / 12, 1.0 / 6],
            [1.0 / 24, -1.0 / 12, 1.0 / 6],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device=w_oihw.device,
    )

    g = w_oihw.float()  # [K_out, C, 3, 3]
    u = torch.einsum("ij,kcjl,lm->kcim", G, g, G.t())
    u = u.reshape(K_out, C, 36).permute(2, 0, 1).contiguous()
    if C_pad != C:
        pad = torch.zeros(
            (36, K_out, C_pad - C), device=w_oihw.device, dtype=torch.float32
        )
        u = torch.cat([u, pad], dim=2)
    return u.to(w_oihw.dtype).contiguous(), C_pad


def get_or_make_winograd_filter_f4x3(w_oihw: torch.Tensor, block_c: int = BLOCK_K):
    key = (
        _storage_ptr(w_oihw),
        tuple(w_oihw.shape),
        w_oihw.dtype,
        block_c,
    )
    cached = _PACK_CACHE_WINOGRAD_F4X3.get(key)
    if cached is not None:
        return cached[1]
    item = prepack_winograd_filter_f4x3(w_oihw, block_c)
    _PACK_CACHE_WINOGRAD_F4X3.put(key, w_oihw, item)
    return item
