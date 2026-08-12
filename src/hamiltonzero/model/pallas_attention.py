# Copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from ._pallas_attn import mhsea_with_tuned_ad


_SUPPORTED_MHSEA_HEAD_DIMS = frozenset((8, 16, 32, 64))


def _require_fp32(*values: Array) -> None:
    for value in values:
        if jnp.dtype(value.dtype) != jnp.dtype(jnp.float32):
            raise TypeError(f"attention tensors must be float32, got {value.dtype}")


def reference_edge_attention(
    Q: Float[Array, "n h d"],
    K: Float[Array, "n h d"],
    V: Float[Array, "n h d"],
    edge_bias: Float[Array, "n n h"],
    mask: Int[Array, "n"],
) -> Float[Array, "n h d"]:
    _require_fp32(Q, K, V, edge_bias)
    _n, _n_heads, d_head = Q.shape
    sm_scale = float(d_head) ** -0.5
    logits = jnp.einsum("ihd,jhd->hij", Q, K) * sm_scale
    logits = logits + jnp.transpose(edge_bias, (2, 0, 1))
    key_mask = (1.0 - mask.astype(Q.dtype))[None, None, :] * -1e9
    logits = logits + key_mask
    alpha = jax.nn.softmax(logits, axis=-1)
    return jnp.einsum("hij,jhd->ihd", alpha, V)


def mhsea_tuned_edge_attention(
    Q: Float[Array, "n h d"],
    K: Float[Array, "n h d"],
    V: Float[Array, "n h d"],
    edge_bias: Float[Array, "n n h"],
    mask: Int[Array, "n"],
) -> Float[Array, "n h d"]:
    _require_fp32(Q, K, V, edge_bias)
    _n, _n_heads, d_head = Q.shape
    sm_scale = float(d_head) ** -0.5
    square = (
        K.shape == Q.shape
        and V.shape == Q.shape
        and edge_bias.shape == (_n, _n, _n_heads)
        and mask.shape == (_n,)
    )
    supports_tuned = square and _n >= 64 and _n % 32 == 0
    supports_full_ad = square and 0 < _n < 64 and not (_n & (_n - 1))
    if d_head not in _SUPPORTED_MHSEA_HEAD_DIMS or not (
        supports_tuned or supports_full_ad
    ):
        return reference_edge_attention(Q, K, V, edge_bias, mask)
    Q4 = (Q * jnp.asarray(sm_scale, dtype=Q.dtype))[None]
    K4 = K[None]
    V4 = V[None]
    e4 = jnp.transpose(edge_bias, (0, 2, 1))[None]
    mask4 = mask.astype(jnp.bool_)[None]
    return mhsea_with_tuned_ad(Q4, K4, e4, V4, mask4)[0]


__all__ = ["mhsea_tuned_edge_attention", "reference_edge_attention"]
