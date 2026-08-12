# Copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from collections.abc import Callable
from functools import partial
from math import gcd

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from hamiltonzero.compiled.types import SharedTrunk, TrunkCompilerKernel
from hamiltonzero.model.fused_silu import fused_silu
from hamiltonzero.model.global_ladder import (
    BoundaryGlobalUpdate,
    ResidualGlobalUpdate,
    TreeGlobalUpdate,
)
from hamiltonzero.model.readout_leaf_context import (
    PhysicalReadoutContext,
    PhysicalReadoutContextLayer,
    RouterContext,
    RouterContextLayer,
    lca_alibi_bias,
    lca_fixed_slopes,
)
from .sequence_parallel import (
    pallas_rectangular_edge_attention,
    ring_learned_fwl2_columns_local,
    ring_learned_fwl2_local,
)
from hamiltonzero.model.tree import _tree_ngpt_residual, _tree_sphere


def _local_rows(x: jax.Array, *, axis_name: str, local_size: int) -> jax.Array:
    start = jax.lax.axis_index(axis_name) * local_size
    return jax.lax.dynamic_slice_in_dim(x, start, local_size, axis=0)


def _global_row_indices(*, axis_name: str, local_size: int) -> jax.Array:
    start = jax.lax.axis_index(axis_name) * local_size
    return start + jnp.arange(local_size, dtype=jnp.int32)


def _gather_rows(x: jax.Array, *, axis_name: str) -> jax.Array:

    return jax.lax.all_gather(x, axis_name=axis_name, axis=0, tiled=True)


def ring_permute_rows_local(
    values: jax.Array,
    permutation: jax.Array,
    *,
    axis_name: str,
    axis_size: int,
) -> jax.Array:

    local_size = values.shape[0]
    lane = jax.lax.axis_index(axis_name).astype(jnp.int32)
    output_ids = jax.lax.dynamic_slice_in_dim(
        permutation,
        lane * local_size,
        local_size,
        axis=0,
    ).astype(jnp.int32)
    owners = output_ids // local_size
    offsets = output_ids % local_size
    output = jnp.zeros_like(values)

    def select(panel, origin, current):
        selected = panel[offsets]
        take = owners == origin
        while take.ndim < selected.ndim:
            take = take[..., None]
        return jnp.where(take, selected, current)

    origin = lane
    output = select(values, origin, output)
    ring = tuple((i, (i + 1) % axis_size) for i in range(axis_size))

    def step(carry, _):
        panel, panel_origin, current = carry
        panel = jax.lax.ppermute(panel, axis_name, ring)
        panel_origin = (panel_origin - jnp.asarray(1, dtype=jnp.int32)) % axis_size
        return (panel, panel_origin, select(panel, panel_origin, current)), None

    (_, _, output), _ = jax.lax.scan(
        step,
        (values, origin, output),
        xs=None,
        length=axis_size - 1,
    )
    return output


def permute_pair_rows_and_columns_local(
    edge_rows: jax.Array,
    permutation: jax.Array,
    *,
    axis_name: str,
    axis_size: int,
) -> jax.Array:

    rows = ring_permute_rows_local(
        edge_rows,
        permutation,
        axis_name=axis_name,
        axis_size=axis_size,
    )
    return rows[:, permutation]


def transpose_pair_rows_local(
    edge_rows: jax.Array,
    *,
    axis_name: str,
    axis_size: int,
) -> jax.Array:

    if axis_size == 1:
        return jnp.swapaxes(edge_rows, 0, 1)
    local_size, n = edge_rows.shape[:2]
    if n != local_size * axis_size:
        raise ValueError(
            "pair transpose requires N == local_rows * axis_size, got "
            f"N={n}, local_rows={local_size}, axis_size={axis_size}"
        )
    trailing = edge_rows.shape[2:]
    send = edge_rows.reshape((local_size, axis_size, local_size) + trailing)
    send = jnp.swapaxes(send, 0, 1)
    received = jax.lax.all_to_all(
        send,
        axis_name,
        split_axis=0,
        concat_axis=0,
    )
    received = jnp.transpose(
        received,
        (2, 0, 1) + tuple(range(3, received.ndim)),
    )
    return received.reshape((local_size, n) + trailing)


def _linear(module, x: jax.Array) -> jax.Array:
    out = x @ module.weight
    bias = getattr(module, "bias", None)
    return out if bias is None else out + bias


def _norm(module, x: jax.Array) -> jax.Array:

    weight = getattr(module, "weight", None)
    if weight is None:
        return x
    stats_dtype = jnp.promote_types(jnp.float32, x.dtype)
    x_hi = x.astype(stats_dtype) if x.dtype != stats_dtype else x
    bias = getattr(module, "bias", None)
    if bias is None:
        stat = jnp.mean(x_hi * x_hi, axis=-1, keepdims=True)
        normalized = x_hi * jax.lax.rsqrt(stat + module.eps)
    else:
        mean = jnp.mean(x_hi, axis=-1, keepdims=True)
        centered = x_hi - mean
        stat = jnp.mean(centered * centered, axis=-1, keepdims=True)
        normalized = centered * jax.lax.rsqrt(stat + module.eps)
    normalized = normalized.astype(x.dtype)
    out = normalized * weight.astype(x.dtype)
    return out if bias is None else out + bias.astype(x.dtype)


def _raw_norm(x, scale, shift, *, eps: float):
    stats_dtype = jnp.promote_types(jnp.float32, x.dtype)
    x_hi = x.astype(stats_dtype)
    if shift is None:
        normalized = x_hi * jax.lax.rsqrt(
            jnp.mean(x_hi * x_hi, axis=-1, keepdims=True) + eps
        )
    else:
        centered = x_hi - jnp.mean(x_hi, axis=-1, keepdims=True)
        normalized = centered * jax.lax.rsqrt(
            jnp.mean(centered * centered, axis=-1, keepdims=True) + eps
        )
    out = normalized.astype(x.dtype) * scale.astype(x.dtype)
    return out if shift is None else out + shift.astype(x.dtype)


def _mlp_after_input_projection(mlp, hidden: jax.Array) -> jax.Array:
    for norm, l1, l2 in zip(mlp.block_norms, mlp.block_l1s, mlp.block_l2s, strict=True):
        hidden = hidden + mlp.inner_gain * _linear(
            l2, mlp._act(_linear(l1, _norm(norm, hidden)))
        )
    return _linear(mlp.out_proj, _norm(mlp.out_norm, hidden))


def _mlp(mlp, x: jax.Array) -> jax.Array:
    return _mlp_after_input_projection(mlp, _linear(mlp.in_proj, x))


def _unnormalized_mlp(mlp, x: jax.Array) -> jax.Array:
    hidden = _linear(mlp.in_proj, x)
    for l1, l2 in zip(mlp.block_l1s, mlp.block_l2s, strict=True):
        hidden = hidden + mlp.inner_gain * _linear(l2, mlp._act(_linear(l1, hidden)))
    return _linear(mlp.out_proj, hidden)


def _split_linear(module, parts: tuple[jax.Array, ...]) -> jax.Array:

    offset = 0
    out = module.bias
    for part in parts:
        width = part.shape[-1]
        out = out + part @ module.weight[offset : offset + width]
        offset += width
    if offset != module.weight.shape[0]:
        raise ValueError(
            f"split input width {offset} does not match weight {module.weight.shape[0]}"
        )
    return out


def _mlp_split_input(mlp, parts: tuple[jax.Array, ...]) -> jax.Array:
    return _mlp_after_input_projection(mlp, _split_linear(mlp.in_proj, parts))


def _g_descriptor_pool(pool, g, xs, mask):

    n = xs.shape[0]
    xn = _norm(pool.ln_in, xs)
    q = (g @ pool.W_q).reshape(pool.n_heads, pool.d_k)
    k = _linear(pool.K, xn).reshape(n, pool.n_heads, pool.d_k)
    v = _linear(pool.V, xn).reshape(n, pool.n_heads, pool.d_v)
    scores = jnp.einsum("hd,nhd->hn", q, k) / jnp.sqrt(
        jnp.asarray(pool.d_k, dtype=xs.dtype)
    )
    scores = jnp.where(
        mask[None, :] > 0,
        scores,
        jnp.asarray(-1.0e30, dtype=scores.dtype),
    )
    weights = jax.nn.softmax(scores, axis=-1)
    return jnp.einsum("hn,nhv->hv", weights, v).reshape(-1)


def _g_descriptor_pool_rows(pool, g, xs_rows, key_mask, *, tile_size: int = 128):

    r, n = xs_rows.shape[:2]
    if tile_size < 1:
        raise ValueError("descriptor-pool tile_size must be positive")
    tile_width = gcd(n, min(n, int(tile_size)))
    tile_count = n // tile_width
    q = (g @ pool.W_q).reshape(pool.n_heads, pool.d_k)
    scale = jnp.sqrt(jnp.asarray(pool.d_k, dtype=xs_rows.dtype))
    scores0 = jnp.zeros((r, pool.n_heads, n), dtype=xs_rows.dtype)

    def score_tile(tile_index, scores):
        start = tile_index * tile_width
        xs_tile = jax.lax.dynamic_slice_in_dim(xs_rows, start, tile_width, axis=1)
        xn_tile = _norm(pool.ln_in, xs_tile)
        k_tile = _linear(pool.K, xn_tile).reshape(r, tile_width, pool.n_heads, pool.d_k)
        tile_scores = jnp.einsum("hd,rthd->rht", q, k_tile) / scale
        mask_tile = jax.lax.dynamic_slice_in_dim(key_mask, start, tile_width, axis=0)
        tile_scores = jnp.where(
            mask_tile[None, None, :] > 0,
            tile_scores,
            jnp.asarray(-1.0e30, dtype=tile_scores.dtype),
        )
        return jax.lax.dynamic_update_slice_in_dim(scores, tile_scores, start, axis=2)

    scores = jax.lax.fori_loop(0, tile_count, score_tile, scores0)
    weights = jax.nn.softmax(scores, axis=-1)
    pooled0 = jnp.zeros((r, pool.n_heads, pool.d_v), dtype=xs_rows.dtype)

    def value_tile(tile_index, pooled):
        start = tile_index * tile_width
        xs_tile = jax.lax.dynamic_slice_in_dim(xs_rows, start, tile_width, axis=1)
        xn_tile = _norm(pool.ln_in, xs_tile)
        v_tile = _linear(pool.V, xn_tile).reshape(r, tile_width, pool.n_heads, pool.d_v)
        weight_tile = jax.lax.dynamic_slice_in_dim(weights, start, tile_width, axis=2)
        return pooled + jnp.einsum("rht,rthv->rhv", weight_tile, v_tile)

    pooled = jax.lax.fori_loop(0, tile_count, value_tile, pooled0)
    return pooled.reshape(r, -1)


def _g_update(update, g, pooled):
    g_input = g @ update.g_tap_w
    x = jnp.concatenate((g_input, pooled.astype(g.dtype)))
    stats = jnp.mean(jnp.square(x), keepdims=True)
    x = x * jax.lax.rsqrt(stats + 1.0e-5) * update.ln_s
    hidden = fused_silu(x @ update.w1 + update.b1)
    delta = hidden @ update.w2 + update.b2
    if isinstance(update, ResidualGlobalUpdate):
        return g + update.residual_gain * delta
    if isinstance(update, BoundaryGlobalUpdate):
        return _tree_sphere(g + delta)
    if not isinstance(update, TreeGlobalUpdate):
        raise TypeError(f"unsupported global update {type(update)!r}")
    skip = _tree_sphere(g)
    proposal = _tree_sphere(delta)
    gain = update.alpha_max * jax.nn.sigmoid(update.alpha)
    return _tree_sphere(skip + gain * (proposal - skip))


def _edge_row_col_global_update(
    module,
    g,
    edge_rows,
    mask,
    row_mask,
    *,
    axis_name: str,
    tile_size: int = 128,
):
    row_desc_rows = _g_descriptor_pool_rows(
        module.row_pool, g, edge_rows, mask, tile_size=tile_size
    )
    row_desc = _gather_rows(row_desc_rows, axis_name=axis_name)

    edge_column_rows = transpose_pair_rows_local(
        edge_rows,
        axis_name=axis_name,
        axis_size=edge_rows.shape[1] // edge_rows.shape[0],
    )
    col_desc_rows = _g_descriptor_pool_rows(
        module.col_pool, g, edge_column_rows, mask, tile_size=tile_size
    )
    col_desc = _gather_rows(col_desc_rows, axis_name=axis_name)
    descriptors = jnp.concatenate((row_desc, col_desc), axis=0)
    descriptor_mask = jnp.concatenate((mask, mask), axis=0)
    pooled = _g_descriptor_pool(module.set_pool, g, descriptors, descriptor_mask)
    return _g_update(module.update, g, pooled)


def sequence_parallel_edge_global_update_local(
    module,
    g,
    edge_rows,
    mask,
    *,
    axis_name: str,
    tile_size: int = 128,
):

    row_indices = _global_row_indices(
        axis_name=axis_name, local_size=edge_rows.shape[0]
    )
    return _edge_row_col_global_update(
        module,
        g,
        edge_rows,
        mask,
        mask[row_indices],
        axis_name=axis_name,
        tile_size=tile_size,
    )


def _psi_project(edge_update, parts, *, left: bool):
    if left:
        linear_in = edge_update.psi_L_in
        linear_out = edge_update.psi_L_out
    else:
        linear_in = edge_update.psi_R_in
        linear_out = edge_update.psi_R_out
    value = _split_linear(linear_in, parts)
    hidden = fused_silu(value)
    return _linear(linear_out, hidden)


def _context_edge_update(
    edge_update,
    edge_rows,
    even_rows,
    even_all,
    mask,
    row_mask,
    *,
    axis_name: str,
    axis_size: int,
):

    edge_ln = _norm(edge_update.ln_edge, edge_rows)
    even_rows_ln = _norm(edge_update.ln_even, even_rows)
    even_all_ln = _norm(edge_update.ln_even, even_all)
    if edge_update.node_ctx_proj is not None:
        even_rows_ln = _linear(edge_update.node_ctx_proj, even_rows_ln)
        even_all_ln = _linear(edge_update.node_ctx_proj, even_all_ln)
    row_endpoint = even_rows_ln[:, None, :]
    column_endpoint = even_all_ln[None, :, :]

    pair_parts = (edge_ln, row_endpoint, column_endpoint)
    left = _psi_project(edge_update, pair_parts, left=True)
    right = _psi_project(edge_update, pair_parts, left=False)
    left = left * (
        row_mask[:, None, None].astype(left.dtype)
        * mask[None, :, None].astype(left.dtype)
    )
    right = right * mask[None, :, None].astype(right.dtype)
    path = ring_learned_fwl2_local(
        left,
        right,
        axis_name=axis_name,
        axis_size=axis_size,
    )
    n_eff = jnp.maximum(jnp.sum(mask), 1.0).astype(path.dtype)
    path = _norm(edge_update.ln_path, path / jnp.sqrt(n_eff))
    return _mlp_split_input(edge_update.ffn, (*pair_parts, path))


def _rectangular_block_attention(
    attention,
    even_rows,
    edge_rows,
    mask,
    *,
    axis_name: str,
    block_k: int,
):

    r = even_rows.shape[0]
    qkv_rows = _linear(attention.W_QKV, even_rows).reshape(
        r, 3, attention.n_heads_kernel, attention.d_head
    )
    qkv_all = _gather_rows(qkv_rows, axis_name=axis_name)
    query = qkv_rows[:, 0]
    key = qkv_all[:, 1]
    value = qkv_all[:, 2]

    edge_pre = _norm(attention.ln_edge, edge_rows)
    bias = _unnormalized_mlp(attention.bias_mlp, edge_pre)
    bias = bias / jnp.sqrt(jnp.asarray(attention.d_head, bias.dtype))
    out = pallas_rectangular_edge_attention(
        query,
        key,
        value,
        bias,
        mask,
        block_k=block_k,
    )
    gate = out[:, : attention.n_heads]
    value_out = out[:, attention.n_heads :]
    out = jax.nn.sigmoid(gate) * value_out
    return _linear(attention.W_O, out.reshape(r, -1))


def _sequence_transformer_block(
    block,
    even_rows,
    edge_rows,
    g,
    mask,
    row_mask,
    *,
    axis_name: str,
    axis_size: int,
    attention_block_k: int,
):
    even_all = _gather_rows(even_rows, axis_name=axis_name)
    edge_delta = _context_edge_update(
        block.edge_update_ctx,
        edge_rows,
        even_rows,
        even_all,
        mask,
        row_mask,
        axis_name=axis_name,
        axis_size=axis_size,
    )
    edge_rows = edge_rows + block.residual_gain * edge_delta

    even_pre = _norm(block.ln_attn, even_rows)
    attention_delta = _rectangular_block_attention(
        block.attn,
        even_pre,
        edge_rows,
        mask,
        axis_name=axis_name,
        block_k=attention_block_k,
    )
    even_rows = even_rows + block.residual_gain * attention_delta

    even_pre = _norm(block.ln_ffn, even_rows)
    even_pre = even_pre + (g @ block.g_ffn_proj_w)[None].astype(even_pre.dtype)
    ffn_delta = _linear(block.ffn.l2, fused_silu(_linear(block.ffn.l1, even_pre)))
    even_rows = even_rows + block.residual_gain * ffn_delta

    even_all = _gather_rows(even_rows, axis_name=axis_name)
    pooled = _g_descriptor_pool(block.g_pool, g, even_all, mask)
    g = _g_update(block.g_update, g, pooled)
    return even_rows, edge_rows, g


def _sequence_trunk_local(
    trunk,
    local_rows,
    edge_rows,
    g,
    mask,
    row_mask,
    *,
    axis_name: str,
    axis_size: int,
    attention_block_k: int,
):
    step = partial(
        _sequence_transformer_block,
        mask=mask,
        row_mask=row_mask,
        axis_name=axis_name,
        axis_size=axis_size,
        attention_block_k=attention_block_k,
    )
    dynamic, static = eqx.partition(trunk.blocks, eqx.is_array)

    def scan_step(carry, layer_dynamic):
        block = eqx.combine(layer_dynamic, static)
        return step(block, *carry), None

    (local_rows, edge_rows, g), _ = jax.lax.scan(
        scan_step, (local_rows, edge_rows, g), dynamic
    )
    return local_rows, edge_rows, g


def _pair_mlp_tiled(mlp, values, *, tile_size: int):

    n = values.shape[1]
    tile_width = min(n, int(tile_size))
    if tile_width < 1:
        raise ValueError("pair MLP tile_size must be positive")
    full_tiles = n // tile_width
    tail_start = full_tiles * tile_width
    output = jnp.zeros(
        values.shape[:2] + (int(mlp.out_proj.weight.shape[1]),),
        dtype=values.dtype,
    )

    def update_tile(start, width, current):
        tile = jax.lax.dynamic_slice_in_dim(values, start, width, axis=1)
        return jax.lax.dynamic_update_slice_in_dim(
            current, _mlp(mlp, tile), start, axis=1
        )

    output = jax.lax.fori_loop(
        0,
        full_tiles,
        lambda tile_index, current: update_tile(
            tile_index * tile_width, tile_width, current
        ),
        output,
    )
    if tail_start < n:
        output = update_tile(tail_start, n - tail_start, output)
    return output


def _pair_mlp_parts_tiled(mlp, parts, *, tile_size: int):

    parts = tuple(parts)
    if not parts:
        raise ValueError("pair MLP requires at least one input part")
    pair_shape = parts[0].shape[:2]
    if any(part.shape[:2] != pair_shape for part in parts):
        raise ValueError("pair MLP input parts must share their [R,N] axes")
    n = pair_shape[1]
    tile_width = min(n, int(tile_size))
    if tile_width < 1:
        raise ValueError("pair MLP tile_size must be positive")
    full_tiles = n // tile_width
    tail_start = full_tiles * tile_width
    output = jnp.zeros(
        pair_shape + (int(mlp.out_proj.weight.shape[1]),),
        dtype=parts[0].dtype,
    )

    def update_tile(start, width, current):
        tile = jnp.concatenate(
            tuple(
                jax.lax.dynamic_slice_in_dim(part, start, width, axis=1)
                for part in parts
            ),
            axis=-1,
        )
        return jax.lax.dynamic_update_slice_in_dim(
            current, _mlp(mlp, tile), start, axis=1
        )

    output = jax.lax.fori_loop(
        0,
        full_tiles,
        lambda tile_index, current: update_tile(
            tile_index * tile_width, tile_width, current
        ),
        output,
    )
    if tail_start < n:
        output = update_tile(tail_start, n - tail_start, output)
    return output


def _sequence_context_attention(
    layer,
    c_rows,
    edge_rows,
    bmask,
    row_indices,
    *,
    axis_name: str,
    block_k: int,
    tile_size: int,
):

    r = c_rows.shape[0]
    n = bmask.shape[0]
    qkv_rows = _linear(layer.W_QKV, c_rows).reshape(
        r, 3, layer.n_heads_kernel, layer.d_head
    )
    qkv_all = _gather_rows(qkv_rows, axis_name=axis_name)
    query = qkv_rows[:, 0]
    key = qkv_all[:, 1]
    value = qkv_all[:, 2]
    col_indices = jnp.arange(n, dtype=jnp.int32)
    rel = row_indices[:, None] - col_indices[None, :]
    if isinstance(layer, PhysicalReadoutContextLayer):
        direction = jnp.where(rel < 0, 1.0, jnp.where(rel > 0, -1.0, 0.0)).astype(
            edge_rows.dtype
        )[..., None]
    elif isinstance(layer, RouterContextLayer):
        direction = jnp.zeros((r, n, 1), dtype=edge_rows.dtype)
    else:
        raise TypeError("unsupported contextualizer layer")
    bias = _pair_mlp_parts_tiled(
        layer.bias_mlp,
        (edge_rows, direction),
        tile_size=tile_size,
    )
    bias = bias / jnp.sqrt(jnp.asarray(layer.d_head, dtype=bias.dtype))
    if isinstance(layer, PhysicalReadoutContextLayer):
        bias = bias + jnp.transpose(
            lca_alibi_bias(
                row_indices,
                col_indices,
                lca_fixed_slopes(layer.n_heads_kernel, dtype=bias.dtype),
            ),
            (1, 2, 0),
        )
    out = pallas_rectangular_edge_attention(
        query,
        key,
        value,
        bias,
        bmask,
        block_k=block_k,
    )
    out = jax.nn.sigmoid(out[:, : layer.n_heads]) * out[:, layer.n_heads :]
    return _linear(layer.W_O, out.reshape(r, -1))


def _sequence_context_layer(
    layer,
    c_rows,
    edge_rows,
    g,
    bmask,
    *,
    axis_name: str,
    axis_size: int,
    tile_size: int,
    attention_block_k: int,
):

    r, n = edge_rows.shape[:2]
    dtype = edge_rows.dtype
    row_indices = _global_row_indices(axis_name=axis_name, local_size=r)
    row_mask = bmask[row_indices]
    edge_n = _norm(layer.ln_edge, edge_rows)
    if isinstance(layer, PhysicalReadoutContextLayer):
        reverse_n = transpose_pair_rows_local(
            edge_n,
            axis_name=axis_name,
            axis_size=axis_size,
        )
        clock_rows = _local_rows(
            layer._slot_clock(n, dtype, bmask),
            axis_name=axis_name,
            local_size=r,
        )
        summary = layer.edge_summary_tiled(
            edge_n,
            bmask.astype(dtype),
            edge_reverse_rows=reverse_n,
            row_indices=row_indices,
            tile_size=tile_size,
        )
        parts = [
            _norm(layer.ln_c, c_rows + clock_rows),
            _norm(layer.ln_summary, summary),
        ]
    elif isinstance(layer, RouterContextLayer):
        clock_rows = None
        parts = [_norm(layer.ln_c, c_rows)]
    else:
        raise TypeError("unsupported contextualizer layer")
    delta_ctx = layer.residual_scale * _mlp(
        layer.ctx_mlp, jnp.concatenate(parts, axis=-1)
    )
    c1 = jnp.where(
        row_mask[:, None].astype(bool),
        c_rows + delta_ctx,
        jnp.zeros_like(c_rows),
    )

    c1_all = _gather_rows(c1, axis_name=axis_name)
    g = _g_update(
        layer.g_update,
        g,
        _g_descriptor_pool(layer.g_pool, g, c1_all, bmask),
    )

    edge_ctx_rows = _linear(
        layer.edge_node_ctx_proj,
        _norm(layer.ln_edge_ctx, c1),
    )
    edge_ctx_all = _gather_rows(edge_ctx_rows, axis_name=axis_name)
    edge1 = layer.edge_update_tiled(
        edge_rows,
        edge_n,
        edge_ctx_rows,
        edge_ctx_all,
        bmask.astype(dtype),
        row_indices=row_indices,
        g=g,
        tile_size=tile_size,
    )

    attention_source = c1
    if clock_rows is not None:
        attention_source = attention_source + clock_rows
    delta_attn = layer.residual_scale * _sequence_context_attention(
        layer,
        _norm(layer.ln_attn, attention_source),
        _norm(layer.ln_edge_attn, edge1),
        bmask,
        row_indices,
        axis_name=axis_name,
        block_k=attention_block_k,
        tile_size=tile_size,
    )
    c_out = jnp.where(
        row_mask[:, None].astype(bool),
        c1 + delta_attn,
        jnp.zeros_like(c1),
    )
    return c_out, edge1, g


def sequence_parallel_contextualizer_local(
    contextualizer,
    node_rows,
    edge_rows,
    real_mask,
    structural_mask,
    g=None,
    *,
    axis_name: str,
    axis_size: int,
    tile_size: int = 128,
    attention_block_k: int = 128,
):

    if not isinstance(contextualizer, (PhysicalReadoutContext, RouterContext)):
        raise TypeError("unsupported contextualizer")
    node_rows = node_rows.astype(jnp.float32)
    edge_rows = edge_rows.astype(jnp.float32)
    r, n = edge_rows.shape[:2]
    if node_rows.shape[0] != r or n != r * axis_size:
        raise ValueError("contextualizer inputs do not match the seq row layout")
    if real_mask.shape != (n,) or structural_mask.shape != (n,):
        raise ValueError("contextualizer masks must have replicated shape [N]")
    row_indices = _global_row_indices(axis_name=axis_name, local_size=r)
    real_rows = real_mask[row_indices].astype(bool)
    active_rows = structural_mask[row_indices].astype(bool)
    virtual_rows = active_rows & ~real_rows
    real = real_mask.astype(bool)
    active = structural_mask.astype(bool)
    virtual = active & ~real

    virtual_node = contextualizer.virtual_node[0].astype(node_rows.dtype)
    node_rows = jnp.where(
        real_rows[:, None],
        node_rows,
        jnp.where(
            virtual_rows[:, None],
            virtual_node[None, :],
            jnp.zeros_like(node_rows),
        ),
    )
    row_real_pair = real_rows[:, None] & real[None, :]
    mixed_pair = (real_rows[:, None] & virtual[None, :]) | (
        virtual_rows[:, None] & real[None, :]
    )
    virtual_pair = virtual_rows[:, None] & virtual[None, :]
    edge_rows = jnp.where(
        row_real_pair[..., None],
        edge_rows,
        jnp.where(
            mixed_pair[..., None],
            contextualizer.edge_empty_nonempty[0].astype(edge_rows.dtype),
            jnp.where(
                virtual_pair[..., None],
                contextualizer.edge_empty_empty[0].astype(edge_rows.dtype),
                jnp.zeros_like(edge_rows),
            ),
        ),
    )

    step = partial(
        _sequence_context_layer,
        bmask=structural_mask,
        axis_name=axis_name,
        axis_size=axis_size,
        tile_size=tile_size,
        attention_block_k=attention_block_k,
    )
    dynamic, static = eqx.partition(contextualizer.layers, eqx.is_array)

    def scan_step(carry, layer_dynamic):
        layer = eqx.combine(layer_dynamic, static)
        return step(layer, *carry), None

    (node_rows, edge_rows, g), _ = jax.lax.scan(
        scan_step, (node_rows, edge_rows, g), dynamic
    )
    return node_rows, edge_rows, g


def _sequence_tree_fwl(
    module,
    edge_rows,
    c_rows,
    c_all,
    mask,
    row_mask,
    *,
    axis_name: str,
    axis_size: int,
    tile_size: int,
):

    width = int(edge_rows.shape[1])
    if tile_size < 1:
        raise ValueError("tree FWL tile_size must be positive")

    tile_width = gcd(width, min(width, int(tile_size)))
    n_tiles = width // tile_width
    c_rows_ctx = _norm(module.ln_c, c_rows)
    c_all_ctx = _norm(module.ln_c, c_all)
    if module.node_ctx_proj is not None:
        c_rows_ctx = _linear(module.node_ctx_proj, c_rows_ctx)
        c_all_ctx = _linear(module.node_ctx_proj, c_all_ctx)

    left0 = jnp.zeros(
        (
            edge_rows.shape[0],
            width,
            int(module.psi_L_out.weight.shape[1]),
        ),
        dtype=edge_rows.dtype,
    )

    def project_left_tile(tile_index, left):
        start = tile_index * tile_width
        edge_tile = jax.lax.dynamic_slice_in_dim(edge_rows, start, tile_width, axis=1)
        c_columns = jax.lax.dynamic_slice_in_dim(c_all_ctx, start, tile_width, axis=0)
        mask_columns = jax.lax.dynamic_slice_in_dim(mask, start, tile_width, axis=0)
        pair_parts = (
            _norm(module.ln_edge, edge_tile),
            c_rows_ctx[:, None, :],
            c_columns[None, :, :],
        )
        projected = _psi_project(module, pair_parts, left=True)
        projected = projected * (
            row_mask[:, None, None].astype(projected.dtype)
            * mask_columns[None, :, None].astype(projected.dtype)
        )
        return jax.lax.dynamic_update_slice_in_dim(left, projected, start, axis=1)

    left = jax.lax.fori_loop(0, n_tiles, project_left_tile, left0)
    n_eff = jnp.maximum(jnp.sum(mask), 1.0).astype(edge_rows.dtype)

    def update_destination_tile(tile_index, updated_edges):
        start = tile_index * tile_width
        edge_tile = jax.lax.dynamic_slice_in_dim(
            updated_edges, start, tile_width, axis=1
        )
        c_columns = jax.lax.dynamic_slice_in_dim(c_all_ctx, start, tile_width, axis=0)
        mask_columns = jax.lax.dynamic_slice_in_dim(mask, start, tile_width, axis=0)
        pair_parts = (
            _norm(module.ln_edge, edge_tile),
            c_rows_ctx[:, None, :],
            c_columns[None, :, :],
        )
        right = _psi_project(module, pair_parts, left=False)
        right = right * mask_columns[None, :, None].astype(right.dtype)
        path = ring_learned_fwl2_columns_local(
            left,
            right,
            axis_name=axis_name,
            axis_size=axis_size,
        )
        path = path / jnp.sqrt(n_eff)
        path = _norm(module.ln_path, path)
        hidden = fused_silu(_split_linear(module.ffn_in, (*pair_parts, path)))
        delta = _linear(module.ffn_out, hidden)
        update_mask = row_mask[:, None].astype(bool) & mask_columns[None, :].astype(
            bool
        )
        edge_tile = _tree_ngpt_residual(
            edge_tile,
            delta,
            module.alpha,
            max_gain=module.ngpt_alpha_max,
            tag_id="",
            update_mask=update_mask,
        )
        return jax.lax.dynamic_update_slice_in_dim(
            updated_edges, edge_tile, start, axis=1
        )

    return jax.lax.fori_loop(0, n_tiles, update_destination_tile, edge_rows)


def _sequence_level_edge_attention(
    module,
    c_rows,
    edge_rows,
    mask,
    row_mask,
    *,
    axis_name: str,
    level: int,
    tile_size: int,
    attention_block_k: int,
):

    r, n = edge_rows.shape[:2]
    x = _raw_norm(
        c_rows,
        module.ln_scale,
        None,
        eps=module.ln_eps,
    )
    qkv_rows = (x @ module.w_qkv).reshape(r, 3, module.n_heads_kernel, module.d_head)
    qkv_all = _gather_rows(qkv_rows, axis_name=axis_name)
    query = qkv_rows[:, 0]
    key = qkv_all[:, 1]
    value = qkv_all[:, 2]
    bias = _pair_mlp_tiled(module.bias_mlp, edge_rows, tile_size=tile_size)
    bias = bias / jnp.sqrt(jnp.asarray(module.d_head, dtype=bias.dtype))
    row_indices = _global_row_indices(axis_name=axis_name, local_size=r)
    bias = bias + jnp.transpose(
        lca_alibi_bias(
            row_indices,
            jnp.arange(n, dtype=jnp.int32),
            lca_fixed_slopes(module.n_heads_kernel, dtype=bias.dtype),
        ),
        (1, 2, 0),
    )
    out = pallas_rectangular_edge_attention(
        query,
        key,
        value,
        bias,
        mask,
        block_k=attention_block_k,
    )
    out = jax.nn.sigmoid(out[:, : module.n_heads]) * out[:, module.n_heads :]
    proposal_attn = row_mask[:, None].astype(out.dtype) * (
        out.reshape(r, -1) @ module.w_o
    )
    c_attn = _tree_ngpt_residual(
        c_rows,
        proposal_attn,
        module.alpha_attn,
        max_gain=module.ngpt_alpha_max,
        tag_id="",
        update_mask=row_mask,
    )
    x_ffn = _raw_norm(
        c_attn,
        module.ffn_ln_scale,
        None,
        eps=module.ln_eps,
    )
    proposal_ffn = row_mask[:, None].astype(x_ffn.dtype) * (
        fused_silu(x_ffn @ module.ffn_w1 + module.ffn_b1) @ module.ffn_w2
        + module.ffn_b2
    )
    return _tree_ngpt_residual(
        c_attn,
        proposal_ffn,
        module.alpha_ffn,
        max_gain=module.ngpt_alpha_max,
        tag_id="",
        update_mask=row_mask,
    )


def sequence_parallel_physical_leaf_local(
    kernel,
    contextualized_node_rows,
    global_stream,
    *,
    axis_name: str,
):

    from hamiltonzero.compiled.tree import _project_global, compile_target_leaf_h

    leaf_g_emb = _project_global(
        kernel.leaf_projection,
        global_stream,
        dense_tag="gladder.to_gemb",
        norm_tag="gladder.gemb_ln",
    )
    node_all = _gather_rows(contextualized_node_rows, axis_name=axis_name)
    leaf_h = compile_target_leaf_h(kernel.leaf, node_all, leaf_g_emb)
    c_rows = _linear(kernel.leaf.P_c, contextualized_node_rows)
    c_rows = _tree_sphere(c_rows)
    return leaf_h, c_rows


def sequence_parallel_reduce_physical_local(
    kernel,
    edge_rows,
    leaf_h,
    c_rows,
    leaf_real,
    structural_mask,
    global_stream,
    permutation,
    *,
    axis_name: str,
    axis_size: int,
    replicate_threshold: int = 512,
    contextualizer_tile_size: int = 128,
    attention_block_k: int = 128,
):

    from hamiltonzero.compiled.tree import (
        compile_merge_h,
        compile_physical_tree_from_reduced_state,
    )
    from hamiltonzero.compiled.types import CARRY_LEFT, CARRY_RIGHT, EMPTY, MERGE
    from hamiltonzero.model.tree import (
        _tree_active_clock_depth,
        _tree_depth_count_features,
        edge_merge_masked,
    )

    if replicate_threshold < 1:
        raise ValueError("replicate_threshold must be positive")
    local_size, n = edge_rows.shape[:2]
    if n != local_size * axis_size or n & (n - 1):
        raise ValueError("physical sequence compiler requires power-of-two N")
    if local_size & (local_size - 1):
        raise ValueError("each seq lane must own a power-of-two row count")

    g = global_stream

    merge = kernel.merge
    m = leaf_real.astype(c_rows.dtype)
    k = structural_mask.astype(c_rows.dtype)
    counts = m
    n_total = jnp.sum(m)
    feature_n_levels = _tree_active_clock_depth(m)
    clock_depth = _tree_active_clock_depth(k)
    edge_rows = _tree_sphere(edge_rows)

    early_merge_h = []
    early_opcodes = []
    width = n
    rows_per_lane = local_size
    level = 0
    while width > int(replicate_threshold):
        if rows_per_lane < 2:
            raise ValueError(
                "replicate_threshold is too small for the available seq lanes"
            )
        c_all = _gather_rows(c_rows, axis_name=axis_name)
        c_a_rows, c_b_rows = c_rows[0::2], c_rows[1::2]
        m_a, m_b = m[0::2], m[1::2]
        k_a, k_b = k[0::2], k[1::2]
        cnt_a, cnt_b = counts[0::2], counts[1::2]
        m_rows = _local_rows(m, axis_name=axis_name, local_size=rows_per_lane)
        k_rows = _local_rows(k, axis_name=axis_name, local_size=rows_per_lane)
        m_a_rows, m_b_rows = m_rows[0::2], m_rows[1::2]
        k_a_rows, k_b_rows = k_rows[0::2], k_rows[1::2]
        both_struct = k_a * k_b
        both_struct_rows = _local_rows(
            both_struct,
            axis_name=axis_name,
            local_size=rows_per_lane // 2,
        )
        pair_base = jnp.maximum(
            jnp.sum((k_a + k_b - k_a * k_b).astype(jnp.int32)),
            jnp.asarray(2, dtype=jnp.int32),
        )
        depth = _tree_depth_count_features(
            cnt_a,
            cnt_b,
            n_total,
            level,
            feature_n_levels,
            c_rows.dtype,
        )
        depth_rows = _local_rows(
            depth,
            axis_name=axis_name,
            local_size=rows_per_lane // 2,
        )

        edge_blocks = edge_rows.reshape(
            rows_per_lane // 2,
            2,
            width // 2,
            2,
            edge_rows.shape[-1],
        )
        local_parent = jnp.arange(rows_per_lane // 2, dtype=jnp.int32)
        global_parent = (
            jax.lax.axis_index(axis_name) * (rows_per_lane // 2) + local_parent
        )
        sibling_lr = edge_blocks[local_parent, 0, global_parent, 1]
        sibling_rl = edge_blocks[local_parent, 1, global_parent, 0]
        level_active = jnp.any(both_struct.astype(bool))
        g_level = g @ kernel.tree_projection_weight + kernel.tree_projection_bias

        def candidate_one(ca, cb, elr, erl, dep, pidx, active):
            return merge.context_candidate(
                ca,
                cb,
                g_level,
                sibling_edge_lr=elr,
                sibling_edge_rl=erl,
                level_idx=jnp.int32(level),
                pair_idx=pidx,
                pair_base=pair_base,
                clock_depth=clock_depth,
                depth_feats=dep,
                kfac_structural_mask=active,
                kfac_g_structural_mask=level_active,
                kfac_scan_shared=False,
            )

        candidate = jax.vmap(candidate_one)(
            c_a_rows,
            c_b_rows,
            sibling_lr,
            sibling_rl,
            depth_rows,
            global_parent,
            both_struct_rows,
        )
        early_merge_h.append(
            _gather_rows(
                compile_merge_h(
                    merge,
                    candidate,
                    depth_rows,
                ),
                axis_name=axis_name,
            )
        )
        early_opcodes.append(
            jnp.where(
                m_a.astype(bool),
                jnp.where(m_b.astype(bool), MERGE, CARRY_LEFT),
                jnp.where(m_b.astype(bool), CARRY_RIGHT, EMPTY),
            ).astype(jnp.uint8)
        )

        gate_a_mask, gate_b_mask = k_a_rows, k_b_rows
        gate_both = gate_a_mask * gate_b_mask
        c_rows = (
            gate_both[:, None] * candidate
            + (gate_a_mask * (1.0 - gate_b_mask))[:, None] * c_a_rows
            + ((1.0 - gate_a_mask) * gate_b_mask)[:, None] * c_b_rows
        )

        c_a_all, c_b_all = c_all[0::2], c_all[1::2]

        parent_width = width // 2
        edge_tile_width = gcd(
            parent_width,
            min(parent_width, int(contextualizer_tile_size)),
        )
        edge_tile_count = parent_width // edge_tile_width
        edge_new0 = jnp.zeros(
            (rows_per_lane // 2, parent_width, edge_rows.shape[-1]),
            dtype=edge_rows.dtype,
        )

        def merge_destination_tile(tile_index, edge_output):
            start = tile_index * edge_tile_width

            def take_columns(x):
                return jax.lax.dynamic_slice_in_dim(x, start, edge_tile_width, axis=0)

            m_a_tile, m_b_tile = take_columns(m_a), take_columns(m_b)
            k_a_tile, k_b_tile = take_columns(k_a), take_columns(k_b)
            c_a_tile, c_b_tile = take_columns(c_a_all), take_columns(c_b_all)

            edge_block_tile = jax.lax.dynamic_slice_in_dim(
                edge_blocks, start, edge_tile_width, axis=2
            )

            def edge_row(e0, e1, e2, e3, ma, mb, ka, kb, ca, cb):
                return jax.vmap(
                    lambda x0, x1, x2, x3, mqa, mqb, kqa, kqb, cqa, cqb: (
                        edge_merge_masked(
                            x0,
                            x1,
                            x2,
                            x3,
                            ma,
                            mb,
                            mqa,
                            mqb,
                            ca,
                            cb,
                            cqa,
                            cqb,
                            merge.edge_merge,
                            k_2i=ka,
                            k_2i1=kb,
                            k_2j=kqa,
                            k_2j1=kqb,
                            kfac_scan_shared=False,
                        )[0]
                    )
                )(
                    e0,
                    e1,
                    e2,
                    e3,
                    m_a_tile,
                    m_b_tile,
                    k_a_tile,
                    k_b_tile,
                    c_a_tile,
                    c_b_tile,
                )

            edge_tile = jax.vmap(edge_row)(
                edge_block_tile[:, 0, :, 0],
                edge_block_tile[:, 0, :, 1],
                edge_block_tile[:, 1, :, 0],
                edge_block_tile[:, 1, :, 1],
                m_a_rows,
                m_b_rows,
                k_a_rows,
                k_b_rows,
                c_a_rows,
                c_b_rows,
            )
            return jax.lax.dynamic_update_slice_in_dim(
                edge_output, edge_tile, start, axis=1
            )

        edge_new = jax.lax.fori_loop(
            0, edge_tile_count, merge_destination_tile, edge_new0
        )
        c_all_new = _gather_rows(c_rows, axis_name=axis_name)
        edge_new = _sequence_tree_fwl(
            merge.tree_edge_fwl,
            edge_new,
            c_rows,
            c_all_new,
            both_struct,
            both_struct_rows,
            axis_name=axis_name,
            axis_size=axis_size,
            tile_size=contextualizer_tile_size,
        )
        edge_keep = both_struct_rows[:, None] * both_struct[None, :]

        def finalize_edge_tile(tile_index, current_edges):
            start = tile_index * edge_tile_width
            current_tile = jax.lax.dynamic_slice_in_dim(
                current_edges, start, edge_tile_width, axis=1
            )
            fallback_tile = jax.lax.dynamic_slice_in_dim(
                edge_blocks, start, edge_tile_width, axis=2
            )[:, 0, :, 0]
            keep_tile = jax.lax.dynamic_slice_in_dim(
                edge_keep, start, edge_tile_width, axis=1
            )[..., None].astype(bool)
            current_tile = _tree_sphere(current_tile)
            current_tile = jnp.where(keep_tile, current_tile, fallback_tile)
            return jax.lax.dynamic_update_slice_in_dim(
                current_edges, current_tile, start, axis=1
            )

        edge_rows = jax.lax.fori_loop(0, edge_tile_count, finalize_edge_tile, edge_new)
        c_skip = c_rows
        c_rows = _sequence_level_edge_attention(
            merge.level_edge_attn,
            c_rows,
            edge_rows,
            both_struct,
            both_struct_rows,
            axis_name=axis_name,
            level=level,
            tile_size=contextualizer_tile_size,
            attention_block_k=attention_block_k,
        )
        c_rows = jnp.where(
            both_struct_rows[:, None].astype(bool),
            _tree_sphere(c_rows),
            c_skip,
        )

        m = m_a + m_b - m_a * m_b
        k = k_a + k_b - k_a * k_b
        counts = cnt_a + cnt_b
        level_mask = k
        c_all_new = _gather_rows(c_rows, axis_name=axis_name)
        updated_g = _g_update(
            kernel.tree_update,
            g,
            _g_descriptor_pool(kernel.tree_pool, g, c_all_new, level_mask),
        )
        g = jnp.where(level_active, updated_g, g)
        width //= 2
        rows_per_lane //= 2
        level += 1

    c_reduced = _gather_rows(c_rows, axis_name=axis_name)
    edge_reduced = _gather_rows(edge_rows, axis_name=axis_name)
    return compile_physical_tree_from_reduced_state(
        kernel,
        perm=permutation,
        leaf_real=leaf_real,
        leaf_h=leaf_h,
        c_reduced=c_reduced,
        edge_reduced=edge_reduced,
        real_reduced=m,
        structural_reduced=k,
        counts_reduced=counts,
        g_reduced=g,
        early_merge_h=tuple(early_merge_h),
        early_opcodes=tuple(early_opcodes),
        full_structural_mask=structural_mask,
    )


def sequence_parallel_shared_trunk_local(
    kernel: TrunkCompilerKernel,
    j_double_prime_rows: jax.Array,
    h_prime: jax.Array,
    real_mask: jax.Array,
    balanced_mask: jax.Array,
    *,
    axis_name: str,
    axis_size: int,
    featurizer_tile_size: int = 128,
    attention_block_k: int = 128,
) -> SharedTrunk:

    if j_double_prime_rows.ndim != 3 or j_double_prime_rows.shape[-1] != 10:
        raise ValueError("J rows must have shape [N/P,N,10]")
    local_size, n = j_double_prime_rows.shape[:2]
    if n != local_size * axis_size:
        raise ValueError(
            f"sequence rows must evenly tile N: local={local_size}, "
            f"N={n}, shards={axis_size}"
        )
    if h_prime.shape != (n, 3):
        raise ValueError(f"h_prime must have shape {(n, 3)}, got {h_prime.shape}")
    if real_mask.shape != (n,) or balanced_mask.shape != (n,):
        raise ValueError("real and balanced masks must both have global shape [N]")
    if j_double_prime_rows.dtype != jnp.float32:
        raise TypeError("fresh sequence compiler currently requires fp32 graph inputs")

    row_indices = _global_row_indices(axis_name=axis_name, local_size=local_size)
    row_mask = real_mask[row_indices]
    featurizer = kernel.featurizer
    bond_rows, descriptor_rows = featurizer.eval_embed_local_rows(
        j_double_prime_rows,
        row_mask,
        real_mask,
        tile_size=featurizer_tile_size,
    )
    descriptor_all = _gather_rows(descriptor_rows, axis_name=axis_name)

    sum_j2, count_j = featurizer.eval_jh_stats_rows(
        j_double_prime_rows,
        row_mask,
        real_mask,
        row_indices=row_indices,
    )
    jh_stats = (
        jax.lax.psum(sum_j2, axis_name),
        jax.lax.psum(count_j, axis_name),
    )

    local_rows, global_raw = featurizer.eval_finalize_local_rows(
        J_double_prime_rows=j_double_prime_rows,
        local_desc_rows=descriptor_rows,
        local_desc_all=descriptor_all,
        row_indices=row_indices,
        mask=real_mask,
        h_prime=h_prime,
        jh_stats=jh_stats,
    )
    local_all = _gather_rows(local_rows, axis_name=axis_name)
    edge_rows = featurizer.eval_edge_rows(
        bond_emb_rows=bond_rows,
        local_rows=local_rows,
        local_final_all=local_all,
        global_feat=global_raw,
        row_indices=row_indices,
        mask=real_mask,
        tile_size=featurizer_tile_size,
    )

    g = _tree_sphere(global_raw.astype(local_rows.dtype))
    local_rows, edge_rows, g = _sequence_trunk_local(
        kernel.trunk,
        local_rows,
        edge_rows,
        g,
        real_mask,
        row_mask,
        axis_name=axis_name,
        axis_size=axis_size,
        attention_block_k=attention_block_k,
    )
    global_stream = _edge_row_col_global_update(
        kernel.shared_global,
        g,
        edge_rows,
        real_mask,
        row_mask,
        axis_name=axis_name,
    )
    return SharedTrunk(
        node_raw=local_rows,
        edge_raw=edge_rows,
        global_raw=global_raw,
        global_stream=global_stream,
        real_mask=real_mask,
        balanced_mask=balanced_mask,
    )


def build_sequence_parallel_shared_trunk(
    *,
    mesh: Mesh,
    kernel_template: TrunkCompilerKernel,
    axis_name: str = "seq",
    featurizer_tile_size: int = 128,
    attention_block_k: int = 128,
) -> Callable[
    [TrunkCompilerKernel, jax.Array, jax.Array, jax.Array, jax.Array],
    SharedTrunk,
]:

    if tuple(mesh.axis_names) != (axis_name,):
        raise ValueError(
            f"sequence trunk requires a one-dimensional {axis_name!r} mesh; "
            f"got {mesh.axis_names}"
        )
    axis_size = int(mesh.shape[axis_name])
    replicated_spec = P()
    edge_spec = P(axis_name, None, None)
    kernel_specs = jax.tree_util.tree_map(lambda _: replicated_spec, kernel_template)
    output_specs = SharedTrunk(
        node_raw=P(axis_name, None),
        edge_raw=edge_spec,
        global_raw=replicated_spec,
        global_stream=replicated_spec,
        real_mask=replicated_spec,
        balanced_mask=replicated_spec,
    )

    local = partial(
        sequence_parallel_shared_trunk_local,
        axis_name=axis_name,
        axis_size=axis_size,
        featurizer_tile_size=featurizer_tile_size,
        attention_block_k=attention_block_k,
    )
    mapped = jax.shard_map(
        local,
        mesh=mesh,
        in_specs=(
            kernel_specs,
            edge_spec,
            replicated_spec,
            replicated_spec,
            replicated_spec,
        ),
        out_specs=output_specs,
        check_vma=False,
    )

    replicated = NamedSharding(mesh, replicated_spec)
    edge_sharding = NamedSharding(mesh, edge_spec)
    output_shardings = SharedTrunk(
        node_raw=NamedSharding(mesh, P(axis_name, None)),
        edge_raw=edge_sharding,
        global_raw=replicated,
        global_stream=replicated,
        real_mask=replicated,
        balanced_mask=replicated,
    )
    return jax.jit(
        mapped,
        in_shardings=(
            jax.tree_util.tree_map(lambda _: replicated, kernel_template),
            edge_sharding,
            replicated,
            replicated,
            replicated,
        ),
        out_shardings=output_shardings,
    )


def build_sequence_parallel_contextualizer(
    *,
    mesh: Mesh,
    contextualizer_template,
    g_template: jax.Array,
    axis_name: str = "seq",
    tile_size: int = 128,
    attention_block_k: int = 128,
) -> Callable:

    if tuple(mesh.axis_names) != (axis_name,):
        raise ValueError(
            f"contextualizer requires a one-dimensional {axis_name!r} mesh"
        )
    axis_size = int(mesh.shape[axis_name])
    rep_spec = P()
    node_spec = P(axis_name, None)
    edge_spec = P(axis_name, None, None)
    context_specs = jax.tree_util.tree_map(lambda _: rep_spec, contextualizer_template)
    local = partial(
        sequence_parallel_contextualizer_local,
        axis_name=axis_name,
        axis_size=axis_size,
        tile_size=tile_size,
        attention_block_k=attention_block_k,
    )
    mapped = jax.shard_map(
        local,
        mesh=mesh,
        in_specs=(
            context_specs,
            node_spec,
            edge_spec,
            rep_spec,
            rep_spec,
            rep_spec,
        ),
        out_specs=(node_spec, edge_spec, rep_spec),
        check_vma=False,
    )
    rep = NamedSharding(mesh, rep_spec)
    node_sharding = NamedSharding(mesh, node_spec)
    edge_sharding = NamedSharding(mesh, edge_spec)
    return jax.jit(
        mapped,
        in_shardings=(
            jax.tree_util.tree_map(lambda _: rep, contextualizer_template),
            node_sharding,
            edge_sharding,
            rep,
            rep,
            rep,
        ),
        out_shardings=(node_sharding, edge_sharding, rep),
    )


def build_sequence_pair_permute(*, mesh: Mesh, axis_name: str = "seq") -> Callable:

    if tuple(mesh.axis_names) != (axis_name,):
        raise ValueError(
            f"pair permutation requires a one-dimensional {axis_name!r} mesh"
        )
    axis_size = int(mesh.shape[axis_name])
    node_spec = P(axis_name, None)
    edge_spec = P(axis_name, None, None)

    def local(node_rows, edge_rows, permutation):
        return (
            ring_permute_rows_local(
                node_rows,
                permutation,
                axis_name=axis_name,
                axis_size=axis_size,
            ),
            permute_pair_rows_and_columns_local(
                edge_rows,
                permutation,
                axis_name=axis_name,
                axis_size=axis_size,
            ),
        )

    return jax.jit(
        jax.shard_map(
            local,
            mesh=mesh,
            in_specs=(node_spec, edge_spec, P()),
            out_specs=(node_spec, edge_spec),
            check_vma=False,
        )
    )


def build_sequence_parallel_edge_global_update(
    *,
    mesh: Mesh,
    module_template,
    axis_name: str = "seq",
    tile_size: int = 128,
) -> Callable:

    if tuple(mesh.axis_names) != (axis_name,):
        raise ValueError(
            f"global edge update requires a one-dimensional {axis_name!r} mesh"
        )
    rep_spec = P()
    edge_spec = P(axis_name, None, None)
    mapped = jax.shard_map(
        partial(
            sequence_parallel_edge_global_update_local,
            axis_name=axis_name,
            tile_size=tile_size,
        ),
        mesh=mesh,
        in_specs=(
            jax.tree_util.tree_map(lambda _: rep_spec, module_template),
            rep_spec,
            edge_spec,
            rep_spec,
        ),
        out_specs=rep_spec,
        check_vma=False,
    )
    rep = NamedSharding(mesh, rep_spec)
    return jax.jit(
        mapped,
        in_shardings=(
            jax.tree_util.tree_map(lambda _: rep, module_template),
            rep,
            NamedSharding(mesh, edge_spec),
            rep,
        ),
        out_shardings=rep,
    )


def build_sequence_parallel_physical_leaf(
    *,
    mesh: Mesh,
    kernel_template,
    axis_name: str = "seq",
) -> Callable:

    if tuple(mesh.axis_names) != (axis_name,):
        raise ValueError(
            f"physical leaf projection requires a one-dimensional {axis_name!r} mesh"
        )
    rep_spec = P()
    node_spec = P(axis_name, None)
    kernel_specs = jax.tree_util.tree_map(lambda _: rep_spec, kernel_template)
    mapped = jax.shard_map(
        partial(sequence_parallel_physical_leaf_local, axis_name=axis_name),
        mesh=mesh,
        in_specs=(kernel_specs, node_spec, rep_spec),
        out_specs=((rep_spec,), node_spec),
        check_vma=False,
    )
    rep = NamedSharding(mesh, rep_spec)
    node_sharding = NamedSharding(mesh, node_spec)
    return jax.jit(
        mapped,
        in_shardings=(
            jax.tree_util.tree_map(lambda _: rep, kernel_template),
            node_sharding,
            rep,
        ),
        out_shardings=((rep,), node_sharding),
    )


def build_sequence_parallel_physical_reducer(
    *,
    mesh: Mesh,
    kernel_template,
    edge_template,
    axis_name: str = "seq",
    replicate_threshold: int = 512,
    contextualizer_tile_size: int = 128,
    attention_block_k: int = 128,
) -> Callable:

    from hamiltonzero.compiled.types import CompiledTree

    if tuple(mesh.axis_names) != (axis_name,):
        raise ValueError(
            f"physical reducer requires a one-dimensional {axis_name!r} mesh"
        )
    n = int(edge_template.shape[0])
    if tuple(edge_template.shape[:2]) != (n, n):
        raise ValueError("edge_template must have square global pair axes")
    axis_size = int(mesh.shape[axis_name])
    if n % axis_size or n & (n - 1):
        raise ValueError("global N must be a power of two divisible by seq lanes")
    rep_spec = P()
    node_spec = P(axis_name, None)
    edge_spec = P(axis_name, None, None)
    kernel_specs = jax.tree_util.tree_map(lambda _: rep_spec, kernel_template)
    n_levels = n.bit_length() - 1
    output_specs = CompiledTree(
        perm=rep_spec,
        inv_perm=rep_spec,
        leaf_real=rep_spec,
        leaf_h=(rep_spec,),
        leaf_combiner_h=(),
        merge_h=(rep_spec,) * n_levels,
        opcodes=(rep_spec,) * n_levels,
        readout_h=(rep_spec,),
        readout_combiner_h=(),
    )
    local = partial(
        sequence_parallel_reduce_physical_local,
        axis_name=axis_name,
        axis_size=axis_size,
        replicate_threshold=replicate_threshold,
        contextualizer_tile_size=contextualizer_tile_size,
        attention_block_k=attention_block_k,
    )
    mapped = jax.shard_map(
        local,
        mesh=mesh,
        in_specs=(
            kernel_specs,
            edge_spec,
            (rep_spec,),
            node_spec,
            rep_spec,
            rep_spec,
            rep_spec,
            rep_spec,
        ),
        out_specs=output_specs,
        check_vma=False,
    )
    rep = NamedSharding(mesh, rep_spec)
    node_sharding = NamedSharding(mesh, node_spec)
    edge_sharding = NamedSharding(mesh, edge_spec)
    output_shardings = jax.tree_util.tree_map(lambda _: rep, output_specs)
    return jax.jit(
        mapped,
        in_shardings=(
            jax.tree_util.tree_map(lambda _: rep, kernel_template),
            edge_sharding,
            (rep,),
            node_sharding,
            rep,
            rep,
            rep,
            rep,
        ),
        out_shardings=output_shardings,
    )


__all__ = [
    "build_sequence_pair_permute",
    "build_sequence_parallel_contextualizer",
    "build_sequence_parallel_edge_global_update",
    "build_sequence_parallel_physical_leaf",
    "build_sequence_parallel_physical_reducer",
    "build_sequence_parallel_shared_trunk",
    "permute_pair_rows_and_columns_local",
    "ring_permute_rows_local",
    "sequence_parallel_contextualizer_local",
    "sequence_parallel_edge_global_update_local",
    "sequence_parallel_physical_leaf_local",
    "sequence_parallel_reduce_physical_local",
    "sequence_parallel_shared_trunk_local",
    "transpose_pair_rows_local",
]
