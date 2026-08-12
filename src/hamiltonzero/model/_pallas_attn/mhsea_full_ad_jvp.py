# Copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from .utils import compiler_params


def _mask_value() -> jax.Array:
    return jnp.float32(-10000.0)


def _mhsea_jvp_kernel(
    q_ref,
    k_ref,
    e_ref,
    v_ref,
    mask_ref,
    qd_ref,
    kd_ref,
    ed_ref,
    vd_ref,
    o_ref,
    od_ref,
    *,
    precision,
):
    mask = mask_ref[:]
    square_mask = mask[:, None] & mask[None, :]

    q = jnp.where(mask[:, None], q_ref[:, :], jnp.float32(0.0))
    k = jnp.where(mask[:, None], k_ref[:, :], jnp.float32(0.0))
    v = jnp.where(mask[:, None], v_ref[:, :], jnp.float32(0.0))
    e = e_ref[:, :]

    qd = jnp.where(mask[:, None], qd_ref[:, :], jnp.float32(0.0))
    kd = jnp.where(mask[:, None], kd_ref[:, :], jnp.float32(0.0))
    vd = jnp.where(mask[:, None], vd_ref[:, :], jnp.float32(0.0))
    ed = ed_ref[:, :]

    scores_raw = pl.dot(q, k, trans_b=True, precision=precision) + e
    scores = jnp.where(square_mask, scores_raw, _mask_value())
    scores_max = jnp.max(scores, axis=-1, keepdims=True)
    probabilities_unscaled = jnp.exp(scores - scores_max)
    probabilities = probabilities_unscaled / jnp.sum(
        probabilities_unscaled, axis=-1, keepdims=True
    )

    scores_d_raw = (
        pl.dot(qd, k, trans_b=True, precision=precision)
        + pl.dot(q, kd, trans_b=True, precision=precision)
        + ed
    )
    scores_d = jnp.where(square_mask, scores_d_raw, jnp.float32(0.0))
    scores_d_mean = jnp.sum(probabilities * scores_d, axis=-1, keepdims=True)
    probabilities_d = probabilities * (scores_d - scores_d_mean)

    output = pl.dot(probabilities, v, precision=precision)
    output_d = pl.dot(probabilities_d, v, precision=precision) + pl.dot(
        probabilities, vd, precision=precision
    )
    o_ref[:, :] = output.astype(o_ref.dtype)
    od_ref[:, :] = output_d.astype(od_ref.dtype)


def mhsea_jvp_pallas(q, k, e, v, mask, qd, kd, ed, vd, *, precision):
    batch_size, seq_len, num_heads, head_dim = q.shape
    qkv_shape = (batch_size, seq_len, num_heads, head_dim)
    edge_shape = (batch_size, seq_len, num_heads, seq_len)
    qkv_spec = pl.BlockSpec(
        (None, seq_len, None, head_dim),
        lambda batch, head: (batch, 0, head, 0),
    )
    edge_spec = pl.BlockSpec(
        (None, seq_len, None, seq_len),
        lambda batch, head: (batch, 0, head, 0),
    )
    mask_spec = pl.BlockSpec((None, seq_len), lambda batch, _head: (batch, 0))
    kernel = pl.pallas_call(
        partial(_mhsea_jvp_kernel, precision=precision),
        grid=(batch_size, num_heads),
        in_specs=[
            qkv_spec,
            qkv_spec,
            edge_spec,
            qkv_spec,
            mask_spec,
            qkv_spec,
            qkv_spec,
            edge_spec,
            qkv_spec,
        ],
        out_specs=[qkv_spec, qkv_spec],
        out_shape=[
            jax.ShapeDtypeStruct(qkv_shape, jnp.float32),
            jax.ShapeDtypeStruct(qkv_shape, jnp.float32),
        ],
        compiler_params=compiler_params(num_warps=4, num_stages=1),
        debug=False,
        interpret=False,
        name="mhsea_full_ad_jvp",
    )
    return kernel(q, k, e, v, mask, qd, kd, ed, vd)


__all__ = ["mhsea_jvp_pallas"]
