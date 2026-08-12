# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Modifications copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: MIT

import jax.numpy as jnp
from jax.experimental import pallas as pl
from .utils import big_number


def mhsea_kernel(
    q_ref, k_ref, e_ref, v_ref, mask_ref, o_ref, lse_ref, q_block_len: int, precision
):
    q_idx = pl.program_id(1)
    kv_mask = mask_ref[:]
    q_slice = pl.dslice(q_idx * q_block_len, q_block_len)
    q_mask = mask_ref[q_slice]
    square_mask = q_mask[:, None] & kv_mask[None, :]
    q = jnp.where(q_mask[:, None], q_ref[:, :], jnp.asarray(0, dtype=q_ref.dtype))
    k = jnp.where(kv_mask[:, None], k_ref[:, :], jnp.asarray(0, dtype=k_ref.dtype))
    e = e_ref[:, :]
    v = jnp.where(kv_mask[:, None], v_ref[:, :], jnp.asarray(0, dtype=v_ref.dtype))
    s = jnp.where(
        square_mask, pl.dot(q, k, trans_b=True, precision=precision) + e, big_number()
    )
    max_val = jnp.max(s, axis=1, keepdims=False)
    lse = max_val + jnp.log(jnp.sum(jnp.exp(s - max_val[:, None]), axis=1))
    p = jnp.exp(s - lse[:, None])
    lse_ref[:] = lse.astype(lse_ref.dtype)
    o = pl.dot(p, v.astype(p.dtype), precision=precision)
    o_ref[:, :] = o.astype(o_ref.dtype)
