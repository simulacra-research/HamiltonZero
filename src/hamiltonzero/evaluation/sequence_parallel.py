# Copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import jax
import jax.numpy as jnp

from .pallas_mha import BlockSizes, noncausal_bias_mha


def _validate_rectangular_attention_inputs(
    q_local: jax.Array,
    k_global: jax.Array,
    v_global: jax.Array,
    edge_bias_local: jax.Array,
    key_mask: jax.Array,
) -> None:
    if q_local.ndim != 3 or k_global.ndim != 3 or v_global.ndim != 3:
        raise ValueError("q, k, and v must have shapes [sequence, heads, head_dim]")
    q_len, n_heads, head_dim = q_local.shape
    kv_len = k_global.shape[0]
    if q_len < 1 or kv_len < 1:
        raise ValueError("query and key sequence lengths must both be positive")
    if k_global.shape != v_global.shape:
        raise ValueError(
            f"k and v shapes must match; got {k_global.shape} and {v_global.shape}"
        )
    if k_global.shape[1:] != (n_heads, head_dim):
        raise ValueError(
            "q, k, and v must have the same head count and head dimension; "
            f"got {q_local.shape}, {k_global.shape}, and {v_global.shape}"
        )
    if edge_bias_local.shape != (q_len, kv_len, n_heads):
        raise ValueError(
            "edge bias must have shape [local_queries, global_keys, heads]; "
            f"got {edge_bias_local.shape}, expected {(q_len, kv_len, n_heads)}"
        )
    if key_mask.shape != (kv_len,):
        raise ValueError(f"key mask must have shape {(kv_len,)}, got {key_mask.shape}")
    arrays = (q_local, k_global, v_global, edge_bias_local)
    if any(value.dtype != jnp.float32 for value in arrays):
        raise TypeError(
            "large-N rectangular attention is fp32-only; got "
            + ", ".join(str(value.dtype) for value in arrays)
        )


def _dividing_block_size(length: int, requested: int | None) -> int:
    block = min(length, 128 if requested is None else requested)
    if block < 1:
        raise ValueError(f"block size must be positive, got {block}")
    while length % block:
        block //= 2
    return block


def pallas_rectangular_edge_attention(
    q_local: jax.Array,
    k_global: jax.Array,
    v_global: jax.Array,
    edge_bias_local: jax.Array,
    key_mask: jax.Array,
    *,
    block_k: int | None = None,
) -> jax.Array:

    _validate_rectangular_attention_inputs(
        q_local, k_global, v_global, edge_bias_local, key_mask
    )
    sm_scale = float(q_local.shape[-1]) ** -0.5
    q_len = q_local.shape[0]
    kv_len = k_global.shape[0]
    bq = _dividing_block_size(q_len, None)
    bk = _dividing_block_size(kv_len, block_k)
    block_sizes = BlockSizes(block_q=bq, block_k=bk)

    masked_bias = jnp.where(
        key_mask[None, :, None].astype(bool),
        edge_bias_local,
        jnp.asarray(-1.0e30, dtype=jnp.float32),
    )
    return noncausal_bias_mha(
        (q_local * jnp.asarray(sm_scale, dtype=jnp.float32))[None],
        k_global[None],
        v_global[None],
        masked_bias[None],
        block_sizes=block_sizes,
    )[0]


def ring_learned_fwl2_local(
    a_local: jax.Array,
    b_local: jax.Array,
    *,
    axis_name: str,
    axis_size: int,
) -> jax.Array:

    local_rows, global_columns, channels = a_local.shape
    if b_local.shape != (local_rows, global_columns, channels):
        raise ValueError(
            f"local 2-FWL shapes must match; got {a_local.shape}, {b_local.shape}"
        )
    return ring_learned_fwl2_columns_local(
        a_local,
        b_local,
        axis_name=axis_name,
        axis_size=axis_size,
    )


def ring_learned_fwl2_columns_local(
    a_local: jax.Array,
    b_local_columns: jax.Array,
    *,
    axis_name: str,
    axis_size: int,
) -> jax.Array:

    if a_local.ndim != 3 or b_local_columns.ndim != 3:
        raise ValueError(
            "local 2-FWL operands must both be rank three; got "
            f"{a_local.shape} and {b_local_columns.shape}"
        )
    local_rows, global_columns, channels = a_local.shape
    if b_local_columns.shape[0] != local_rows:
        raise ValueError(
            "local A/B row counts must match; got "
            f"{local_rows} and {b_local_columns.shape[0]}"
        )
    if b_local_columns.shape[2] != channels:
        raise ValueError(
            "local A/B channel counts must match; got "
            f"{channels} and {b_local_columns.shape[2]}"
        )
    if a_local.dtype != b_local_columns.dtype:
        raise TypeError(
            "local A/B dtypes must match; got "
            f"{a_local.dtype} and {b_local_columns.dtype}"
        )
    if global_columns != local_rows * axis_size:
        raise ValueError(
            "2-FWL row shards must evenly tile the contracted axis; "
            f"got local_rows={local_rows}, columns={global_columns}, "
            f"axis_size={axis_size}"
        )

    def contribution(b_panel, origin):
        a_panel = jax.lax.dynamic_slice_in_dim(
            a_local, origin * local_rows, local_rows, axis=1
        )
        return jnp.einsum("ikc,kjc->ijc", a_panel, b_panel)

    origin0 = jax.lax.axis_index(axis_name).astype(jnp.int32)
    accumulator0 = contribution(b_local_columns, origin0)
    ring_permutation = [(lane, (lane + 1) % axis_size) for lane in range(axis_size)]

    def ring_step(carry, _):
        b_panel, origin, accumulator = carry
        b_panel = jax.lax.ppermute(b_panel, axis_name=axis_name, perm=ring_permutation)
        origin = (origin - jnp.asarray(1, jnp.int32)) % axis_size
        accumulator = accumulator + contribution(b_panel, origin)
        return (b_panel, origin, accumulator), None

    (_, _, result), _ = jax.lax.scan(
        ring_step,
        (b_local_columns, origin0, accumulator0),
        xs=None,
        length=axis_size - 1,
    )
    return result
