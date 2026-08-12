# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Modifications copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: MIT

from functools import partial
from typing import Tuple
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from .mhsea import mhsea_kernel
from .utils import (
    big_number,
    compiler_params,
    create_grid,
    get_key_value_block_spec,
    get_lse_block_spec,
    get_mask_block_spec,
    get_query_block_spec,
    sum_columns,
)


def mhsea_forward(
    q: jax.Array,
    k: jax.Array,
    e: jax.Array,
    v: jax.Array,
    mask: jax.Array,
    *,
    q_block_len: int,
    num_warps: int,
    num_stages: int,
    precision,
) -> Tuple[
    jax.Array,
    Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
]:
    batch_len, seq_len, num_heads, head_len = q.shape
    v_dim = v.shape[-1]
    qdtype = q.dtype
    dtype = jnp.float32
    kernel_fn = pl.pallas_call(
        partial(mhsea_kernel, q_block_len=q_block_len, precision=precision),
        grid=create_grid(batch_len, seq_len, num_heads, q_block_len),
        in_specs=[
            get_query_block_spec(q_block_len, head_len),
            get_key_value_block_spec(seq_len, head_len),
            get_query_block_spec(q_block_len, seq_len),
            get_key_value_block_spec(seq_len, v_dim),
            get_mask_block_spec(seq_len),
        ],
        out_specs=[
            get_query_block_spec(q_block_len, v_dim),
            get_lse_block_spec(q_block_len),
        ],
        out_shape=[
            jax.ShapeDtypeStruct(
                shape=(batch_len, seq_len, num_heads, v_dim), dtype=dtype
            ),
            jax.ShapeDtypeStruct(shape=(batch_len, seq_len, num_heads), dtype=dtype),
        ],
        compiler_params=compiler_params(num_warps=num_warps, num_stages=num_stages),
        debug=False,
        interpret=False,
        name="mhea_forward",
    )
    o, lse = kernel_fn(
        q.astype(dtype),
        k.astype(dtype),
        e.astype(dtype),
        v.astype(dtype),
        mask.astype(jnp.bool_),
    )
    return (o.astype(qdtype), (q, k, e, v, mask, lse, o))


def mhsea_backward(
    q_block_len: int,
    num_warps: int,
    num_stages: int,
    precision,
    fwd_cache: Tuple[
        jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array
    ],
    o_vjp: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    q, k, e, v, mask, lse, o = fwd_cache
    mask = mask.astype(jnp.bool_)
    batch_len, seq_len, num_heads, head_len = q.shape
    block_len = q_block_len
    dtype = jnp.float32
    dq, de = pl.pallas_call(
        partial(mhsea_q_vjp_kernel, block_len=block_len, precision=precision),
        grid=(batch_len, seq_len // block_len, num_heads),
        in_specs=[
            pl.BlockSpec(
                index_map=lambda i, j, k: (i, j, k, 0),
                block_shape=(None, block_len, None, head_len),
            ),
            pl.BlockSpec(
                index_map=lambda i, j, k: (i, 0, k, 0),
                block_shape=(None, seq_len, None, head_len),
            ),
            pl.BlockSpec(
                index_map=lambda i, j, k: (i, j, k, 0),
                block_shape=(None, block_len, None, seq_len),
            ),
            pl.BlockSpec(
                index_map=lambda i, j, k: (i, 0, k, 0),
                block_shape=(None, seq_len, None, head_len),
            ),
            pl.BlockSpec(index_map=lambda i, j, k: (i, 0), block_shape=(None, seq_len)),
            pl.BlockSpec(
                index_map=lambda i, j, k: (i, j, k), block_shape=(None, block_len, None)
            ),
            pl.BlockSpec(
                index_map=lambda i, j, k: (i, j, k, 0),
                block_shape=(None, block_len, None, head_len),
            ),
            pl.BlockSpec(
                index_map=lambda i, j, k: (i, j, k, 0),
                block_shape=(None, block_len, None, head_len),
            ),
        ],
        out_specs=[
            pl.BlockSpec(
                index_map=lambda i, j, k: (i, j, k, 0),
                block_shape=(None, block_len, None, head_len),
            ),
            pl.BlockSpec(
                index_map=lambda i, j, k: (i, j, k, 0),
                block_shape=(None, block_len, None, seq_len),
            ),
        ],
        out_shape=[
            jax.ShapeDtypeStruct(
                shape=(batch_len, seq_len, num_heads, head_len), dtype=dtype
            ),
            jax.ShapeDtypeStruct(
                shape=(batch_len, seq_len, num_heads, seq_len), dtype=dtype
            ),
        ],
        compiler_params=compiler_params(num_warps=num_warps, num_stages=num_stages),
        debug=False,
        interpret=False,
        name="mhsea_backward_q_vjp",
    )(
        q.astype(dtype),
        k.astype(dtype),
        e.astype(dtype),
        v.astype(dtype),
        mask,
        lse.astype(dtype),
        o.astype(dtype),
        o_vjp.astype(dtype),
    )
    dk, dv = pl.pallas_call(
        partial(mhsea_kv_vjp_kernel, block_len=block_len, precision=precision),
        grid=(batch_len, seq_len // block_len, num_heads),
        in_specs=[
            pl.BlockSpec(
                index_map=lambda i, j, k: (i, 0, k, 0),
                block_shape=(None, seq_len, None, head_len),
            ),
            pl.BlockSpec(
                index_map=lambda i, j, k: (i, j, k, 0),
                block_shape=(None, block_len, None, head_len),
            ),
            pl.BlockSpec(
                index_map=lambda i, j, k: (i, 0, k, j),
                block_shape=(None, seq_len, None, block_len),
            ),
            pl.BlockSpec(
                index_map=lambda i, j, k: (i, j, k, 0),
                block_shape=(None, block_len, None, head_len),
            ),
            pl.BlockSpec(index_map=lambda i, j, k: (i, 0), block_shape=(None, seq_len)),
            pl.BlockSpec(
                index_map=lambda i, j, k: (i, 0, k), block_shape=(None, seq_len, None)
            ),
            pl.BlockSpec(
                index_map=lambda i, j, k: (i, 0, k, 0),
                block_shape=(None, seq_len, None, head_len),
            ),
            pl.BlockSpec(
                index_map=lambda i, j, k: (i, 0, k, 0),
                block_shape=(None, seq_len, None, head_len),
            ),
        ],
        out_specs=[
            pl.BlockSpec(
                index_map=lambda i, j, k: (i, j, k, 0),
                block_shape=(None, block_len, None, head_len),
            ),
            pl.BlockSpec(
                index_map=lambda i, j, k: (i, j, k, 0),
                block_shape=(None, block_len, None, head_len),
            ),
        ],
        out_shape=[
            jax.ShapeDtypeStruct(
                shape=(batch_len, seq_len, num_heads, head_len), dtype=dtype
            ),
            jax.ShapeDtypeStruct(
                shape=(batch_len, seq_len, num_heads, head_len), dtype=dtype
            ),
        ],
        compiler_params=compiler_params(num_warps=num_warps, num_stages=num_stages),
        debug=False,
        interpret=False,
        name="mhsea_backward_kv_vjp",
    )(
        q.astype(dtype),
        k.astype(dtype),
        e.astype(dtype),
        v.astype(dtype),
        mask.astype(jnp.bool_),
        lse.astype(dtype),
        o.astype(dtype),
        o_vjp.astype(dtype),
    )
    return (
        dq.astype(o_vjp.dtype),
        dk.astype(o_vjp.dtype),
        de.astype(o_vjp.dtype),
        dv.astype(o_vjp.dtype),
    )


def mhsea_q_vjp_kernel(
    q_ref,
    k_ref,
    e_ref,
    v_ref,
    mask_ref,
    lse_ref,
    o_ref,
    o_vjp_ref,
    q_vjp_ref,
    e_vjp_ref,
    block_len,
    precision,
):
    qix = pl.program_id(1)
    q_vjp = jnp.zeros(q_vjp_ref.shape, dtype=q_vjp_ref.dtype)
    q_slice = pl.dslice(qix * block_len, block_len)

    def _kaxis_loop(kix, q_vjp):
        mask_q = mask_ref[q_slice]
        k_slice = pl.dslice(kix * block_len, block_len)
        mask_k = mask_ref[k_slice]
        square_mask = mask_q[:, None] & mask_k[None, :]
        q = jnp.where(mask_q[:, None], q_ref[:, :], jnp.asarray(0, dtype=q_ref.dtype))
        k = jnp.where(
            mask_k[:, None], k_ref[k_slice, :], jnp.asarray(0, dtype=k_ref.dtype)
        )
        v = jnp.where(
            mask_k[:, None], v_ref[k_slice, :], jnp.asarray(0, dtype=v_ref.dtype)
        )
        e = e_ref[:, k_slice]
        lse = lse_ref[:]
        s = jnp.where(
            square_mask,
            pl.dot(q, k, trans_b=True, precision=precision) + e,
            big_number(),
        )
        p = jnp.exp(s - lse[:, None])
        o = o_ref[:, :]
        o_vjp = jnp.where(
            mask_q[:, None], o_vjp_ref[:, :], jnp.asarray(0, dtype=o_vjp_ref.dtype)
        )
        s_vjp = (
            pl.dot(o_vjp, v, trans_b=True, precision=precision) - sum_columns(o * o_vjp)
        ) * p
        s_vjp *= square_mask.astype(s_vjp.dtype)
        q_vjp_kblock = pl.dot(s_vjp, k.astype(s_vjp.dtype), precision=precision)
        q_vjp += q_vjp_kblock
        e_vjp_ref[:, k_slice] = s_vjp.astype(e_vjp_ref.dtype)
        return q_vjp.astype(jnp.float32)

    q_vjp = jax.lax.fori_loop(
        0, k_ref.shape[0] // block_len, _kaxis_loop, q_vjp.astype(jnp.float32)
    )
    q_vjp_ref[:, :] = q_vjp.astype(q_vjp_ref.dtype)


def mhsea_kv_vjp_kernel(
    q_ref,
    k_ref,
    e_ref,
    v_ref,
    mask_ref,
    lse_ref,
    o_ref,
    o_vjp_ref,
    k_vjp_ref,
    v_vjp_ref,
    block_len,
    precision,
):
    kix = pl.program_id(1)
    k_vjp = jnp.zeros((block_len, k_vjp_ref.shape[-1]), dtype=k_vjp_ref.dtype)
    v_vjp = jnp.zeros((block_len, v_vjp_ref.shape[-1]), dtype=v_vjp_ref.dtype)
    k_slice = pl.dslice(kix * block_len, block_len)

    def _qaxis_loop(qix, store):
        k_vjp, v_vjp = store
        q_slice = pl.dslice(qix * block_len, block_len)
        mask_q = mask_ref[q_slice]
        mask_k = mask_ref[k_slice]
        square_mask = mask_q[:, None] & mask_k[None, :]
        q = jnp.where(
            mask_q[:, None], q_ref[q_slice, :], jnp.asarray(0, dtype=q_ref.dtype)
        )
        k = jnp.where(mask_k[:, None], k_ref[:, :], jnp.asarray(0, dtype=k_ref.dtype))
        v = jnp.where(mask_k[:, None], v_ref[:, :], jnp.asarray(0, dtype=v_ref.dtype))
        e = e_ref[q_slice, :]
        lse = lse_ref[q_slice]
        s = jnp.where(
            square_mask,
            pl.dot(q, k, trans_b=True, precision=precision) + e,
            big_number(),
        )
        p = jnp.exp(s - lse[:, None])
        o = o_ref[q_slice, :]
        o_vjp = jnp.where(
            mask_q[:, None],
            o_vjp_ref[q_slice, :],
            jnp.asarray(0, dtype=o_vjp_ref.dtype),
        )
        s_vjp = (
            pl.dot(o_vjp, v, trans_b=True, precision=precision) - sum_columns(o * o_vjp)
        ) * p
        s_vjp *= square_mask.astype(s_vjp.dtype)
        k_vjp += pl.dot(s_vjp, q.astype(s_vjp.dtype), trans_a=True, precision=precision)
        v_vjp += pl.dot(p, o_vjp.astype(p.dtype), trans_a=True, precision=precision)
        return (k_vjp.astype(jnp.float32), v_vjp.astype(jnp.float32))

    k_vjp, v_vjp = jax.lax.fori_loop(
        0,
        q_ref.shape[0] // block_len,
        _qaxis_loop,
        (k_vjp.astype(jnp.float32), v_vjp.astype(jnp.float32)),
    )
    k_vjp_ref[:, :] = k_vjp.astype(k_vjp_ref.dtype)
    v_vjp_ref[:, :] = v_vjp.astype(v_vjp_ref.dtype)
