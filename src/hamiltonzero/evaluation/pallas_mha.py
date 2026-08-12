# SPDX-License-Identifier: Apache-2.0

# Copyright 2023 The JAX Authors.
# Modifications copyright (c) 2026 Simulacra Research Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import annotations

import dataclasses
import functools
import math
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu


@dataclasses.dataclass(frozen=True, slots=True)
class BlockSizes:
    block_q: int
    block_k: int

    @classmethod
    def get_default(cls):
        return cls(block_q=128, block_k=128)


def noncausal_bias_mha_forward_kernel(
    q_ref,
    k_ref,
    v_ref,
    bias_ref,
    o_ref: Any,
    *,
    block_q: int,
    block_k: int,
    head_dim: int,
):
    seq_len = k_ref.shape[0]
    start_q = pl.program_id(0)
    head_dim_padded = q_ref.shape[-1]
    m_i = jnp.zeros(block_q, dtype=jnp.float32) - float("inf")
    l_i = jnp.zeros(block_q, dtype=jnp.float32)
    o = jnp.zeros((block_q, head_dim_padded), dtype=jnp.float32)
    curr_q_slice = pl.dslice(start_q * block_q, block_q)
    head_mask = (jnp.arange(head_dim_padded) < head_dim)[None, :]
    q = plgpu.load(q_ref, mask=head_mask, other=0.0)

    def body(start_k, carry):
        o_prev, m_prev, l_prev = carry
        curr_k_slice = pl.dslice(start_k * block_k, block_k)
        k = plgpu.load(k_ref.at[curr_k_slice, :], mask=head_mask, other=0.0)
        qk = pl.dot(q, k.T)
        bias = plgpu.load(bias_ref.at[curr_q_slice, curr_k_slice])
        qk += bias
        qk *= math.log2(math.e)
        qk = qk.astype(q_ref.dtype)
        m_curr = jnp.max(qk, axis=-1)
        m_next = jnp.maximum(m_prev, m_curr)
        correction = jnp.exp2(m_prev - m_next)
        l_prev_corr = correction * l_prev
        s_curr = jnp.exp2(qk - m_next[:, None])
        l_curr = s_curr.sum(axis=-1)
        l_next = l_prev_corr + l_curr
        o_prev_corr = correction[:, None] * o_prev
        v = plgpu.load(v_ref.at[curr_k_slice, :], mask=head_mask)
        o_curr = pl.dot(s_curr.astype(v.dtype), v)
        o_next = o_prev_corr + o_curr
        return o_next, m_next, l_next

    upper_bound = pl.cdiv(seq_len, block_k)
    o, _m_i, l_i = lax.fori_loop(0, upper_bound, body, (o, m_i, l_i))
    o /= l_i[:, None]
    plgpu.store(o_ref.at[:, : o.shape[-1]], o.astype(o_ref.dtype), mask=head_mask)


@functools.partial(jax.jit, static_argnames=["block_sizes"])
def noncausal_bias_mha(
    q,
    k,
    v,
    bias,
    *,
    block_sizes: BlockSizes = BlockSizes.get_default(),
):
    batch_size, q_seq_len, num_heads, head_dim = q.shape
    kv_seq_len = k.shape[1]
    block_q = min(block_sizes.block_q, q_seq_len)
    block_k = min(block_sizes.block_k, kv_seq_len)
    head_dim_padded = pl.next_power_of_2(head_dim)
    if (q.shape[-1] != k.shape[-1]) or (q.shape[-1] != v.shape[-1]):
        raise ValueError(
            "This kernel expects q, k, and v to have the same head dimension, "
            f"but found {q.shape=}, {k.shape=}, {v.shape=}."
        )
    if bias.shape != (batch_size, q_seq_len, kv_seq_len, num_heads):
        raise ValueError(
            f"bias must have shape [batch, query, key, heads]; got {bias.shape}"
        )
    if q_seq_len % block_q != 0:
        raise ValueError(f"{q_seq_len=} must be a multiple of {block_q=}")
    if kv_seq_len % block_k != 0:
        raise ValueError(f"{kv_seq_len=} must be a multiple of {block_k=}")
    grid = (pl.cdiv(q_seq_len, block_q), batch_size, num_heads)
    num_warps = 4 if head_dim <= 64 else 8
    kernel = functools.partial(
        noncausal_bias_mha_forward_kernel,
        block_q=block_q,
        block_k=block_k,
        head_dim=head_dim,
    )
    in_specs = [
        pl.BlockSpec(
            (None, block_q, None, head_dim_padded),
            lambda i, j, k: (j, i, k, 0),
        ),
        pl.BlockSpec(
            (None, kv_seq_len, None, head_dim_padded),
            lambda i, j, k: (j, 0, k, 0),
        ),
        pl.BlockSpec(
            (None, kv_seq_len, None, head_dim_padded),
            lambda i, j, k: (j, 0, k, 0),
        ),
        pl.BlockSpec(
            (None, block_q, kv_seq_len, None),
            lambda i, j, k: (j, i, 0, k),
        ),
    ]
    out_shape = [q]
    out_specs = [
        pl.BlockSpec(
            (None, block_q, None, head_dim_padded),
            lambda i, j, k: (j, i, k, 0),
        )
    ]
    out = pl.pallas_call(
        kernel,
        grid=grid,
        in_specs=in_specs,
        out_specs=out_specs,
        compiler_params=plgpu.CompilerParams(num_warps=num_warps, num_stages=2),
        out_shape=out_shape,
        name="mha_forward",
    )(q, k, v, bias)
    return out[0]


__all__ = ["BlockSizes", "noncausal_bias_mha"]
