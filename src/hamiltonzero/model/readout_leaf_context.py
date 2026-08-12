# Copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray

from .odd_ops import BiasFreeLinear, MLP, _RMS


def _inline_norm(
    norm,
    x,
    *,
    pathway="even",
    kfac_structural_mask=None,
    kfac_scan_shared: bool = False,
    kfac_repeat_ndim: int = 0,
    kfac_context_primal_reused_over_walkers: bool = False,
):
    from .tree import _tagged_rms_eqx_style

    return _tagged_rms_eqx_style(
        norm.weight,
        x,
        eps=norm.eps,
        tag_id=norm._use_id,
        pathway=pathway,
        kfac_structural_mask=kfac_structural_mask,
        kfac_scan_shared=kfac_scan_shared,
        kfac_repeat_ndim=kfac_repeat_ndim,
        kfac_context_primal_reused_over_walkers=(
            kfac_context_primal_reused_over_walkers
        ),
    )


def _inline_mlp_forward(
    mlp,
    x,
    *,
    pathway="even",
    kfac_structural_mask=None,
    kfac_scan_shared: bool = False,
    kfac_repeat_ndim: int = 0,
    kfac_context_primal_reused_over_walkers: bool = False,
):
    from .tree import _tagged_dense

    arguments = dict(
        pathway=pathway,
        kfac_structural_mask=kfac_structural_mask,
        kfac_scan_shared=kfac_scan_shared,
        kfac_repeat_ndim=kfac_repeat_ndim,
        kfac_context_primal_reused_over_walkers=(
            kfac_context_primal_reused_over_walkers
        ),
    )
    x = _tagged_dense(
        mlp.in_proj.weight,
        mlp.in_proj.bias,
        x,
        tag_id=mlp.in_proj._use_id,
        **arguments,
    )
    for norm, linear_1, linear_2 in zip(
        mlp.block_norms,
        mlp.block_l1s,
        mlp.block_l2s,
        strict=True,
    ):
        normalized = _inline_norm(norm, x, **arguments)
        inner = _tagged_dense(
            linear_1.weight,
            linear_1.bias,
            normalized,
            tag_id=linear_1._use_id,
            **arguments,
        )
        inner = mlp._act(inner)
        inner = _tagged_dense(
            linear_2.weight,
            linear_2.bias,
            inner,
            tag_id=linear_2._use_id,
            **arguments,
        )
        x = x + mlp.inner_gain * inner
    normalized = _inline_norm(mlp.out_norm, x, **arguments)
    return _tagged_dense(
        mlp.out_proj.weight,
        mlp.out_proj.bias,
        normalized,
        tag_id=mlp.out_proj._use_id,
        **arguments,
    )


def _inline_bias_free_linear(
    layer: BiasFreeLinear,
    x,
    *,
    pathway="even",
    kfac_structural_mask=None,
    kfac_scan_shared: bool = False,
    kfac_repeat_ndim: int = 0,
    kfac_context_primal_reused_over_walkers: bool = False,
):
    from .tree import _tagged_dense_no_bias

    return _tagged_dense_no_bias(
        layer.weight,
        x,
        tag_id=layer._use_id,
        pathway=pathway,
        kfac_structural_mask=kfac_structural_mask,
        kfac_scan_shared=kfac_scan_shared,
        kfac_repeat_ndim=kfac_repeat_ndim,
        kfac_context_primal_reused_over_walkers=(
            kfac_context_primal_reused_over_walkers
        ),
    )


_ALLOWED_ATTN_IMPLS = ("einsum", "mhsea_tuned")
_LCA_MAX_LEVELS = 13


def default_tree_depth(max_n: int) -> int:
    max_n = max(1, int(max_n))
    return max(1, int(math.ceil(math.log2(max_n))))


def _relative_positions(n: int) -> tuple[Int[Array, "n n"], Float[Array, "n n 1"]]:
    index = jnp.arange(n, dtype=jnp.int32)
    relative = index[:, None] - index[None, :]
    sign = jnp.where(
        relative < 0,
        1.0,
        jnp.where(relative > 0, -1.0, 0.0),
    )
    return relative, sign[..., None]


def lca_level(pos_q, pos_k):
    xor = jnp.bitwise_xor(
        jnp.asarray(pos_q, jnp.int32)[..., :, None],
        jnp.asarray(pos_k, jnp.int32)[..., None, :],
    )
    thresholds = jnp.exp2(jnp.arange(_LCA_MAX_LEVELS, dtype=jnp.float32)).astype(
        jnp.int32
    )
    return jnp.sum((xor[..., None] >= thresholds).astype(jnp.int32), axis=-1)


def lca_alibi_bias(pos_q, pos_k, slopes):
    level = lca_level(pos_q, pos_k).astype(slopes.dtype)
    return -(slopes[:, None, None] * level[None, :, :])


def lca_fixed_slopes(n_heads, dtype=jnp.float32):
    head = jnp.arange(n_heads, dtype=jnp.float32)
    denominator = jnp.maximum(jnp.asarray(n_heads - 1, jnp.float32), 1.0)
    return (1.5 * jnp.exp2(-(head * 4.0 / denominator))).astype(dtype)


def lca_gaussian_decay(pos_q, pos_k, w_raw, b):
    level = lca_level(pos_q, pos_k).astype(b.dtype)
    weight = jax.nn.softplus(w_raw)
    distance = level[:, :, None] - b[None, None, :]
    return jnp.exp(-(weight[None, None, :] * distance * distance))


def lca_gaussian_decay_row(pos_q_scalar, pos_k, w_raw, b):
    xor = jnp.bitwise_xor(
        jnp.asarray(pos_q_scalar, jnp.int32),
        jnp.asarray(pos_k, jnp.int32),
    )
    thresholds = jnp.exp2(jnp.arange(_LCA_MAX_LEVELS, dtype=jnp.float32)).astype(
        jnp.int32
    )
    level = jnp.sum(
        (xor[:, None] >= thresholds[None, :]).astype(jnp.int32),
        axis=-1,
    ).astype(b.dtype)
    weight = jax.nn.softplus(w_raw)
    distance = level[:, None] - b[None, :]
    return jnp.exp(-(weight[None, :] * distance * distance))


def register_vector_as_dense(w2d, *, tag_id=""):
    from hamiltonzero.optim.spin_blocks import register_small_full

    return register_small_full(w2d, tag_id=tag_id)


def lca_order_init_w_b(d_model, dtype=jnp.float32):
    width = int(d_model)
    centers = (
        (jnp.arange(width, dtype=jnp.float32) % 9.0).astype(dtype).reshape(1, width)
    )
    raw_weight = jnp.full(
        (1, width),
        float(jnp.log(jnp.expm1(jnp.asarray(0.7)))),
        dtype=dtype,
    )
    return raw_weight, centers


def _attention_dimensions(
    *,
    d_e: int,
    n_heads: int,
    attn_dim: int,
    attn_impl: str,
    require_even_model: bool,
) -> tuple[int, int, int]:
    if n_heads < 1:
        raise ValueError("contextualizer n_heads must be positive")
    if attn_dim < 1 or attn_dim % n_heads:
        raise ValueError("contextualizer attention width must divide by n_heads")
    if require_even_model and d_e % 2:
        raise ValueError("physical contextualizer width must be even")
    d_head = attn_dim // n_heads
    if d_head % 2:
        raise ValueError("contextualizer attention head width must be even")
    if attn_impl not in _ALLOWED_ATTN_IMPLS:
        raise ValueError("attention must be 'einsum' or 'mhsea_tuned'")
    return 2 * n_heads, d_head, attn_dim


def _global_modules(
    key, d_g: int, d_e: int, residual_scale: float, global_tap_dim: int
):
    from .global_ladder import GDescriptorPool, ResidualGlobalUpdate

    keys = jax.random.split(jax.random.fold_in(key, 25007), 3)
    pool = GDescriptorPool(d_g, d_e, key=keys[0], tag="gladder.ctx.pool")
    update = ResidualGlobalUpdate(
        d_g,
        pool.d_out,
        key=keys[1],
        tag="gladder.ctx.upd",
        tap_dim=global_tap_dim,
        residual_gain=residual_scale,
    )
    projection = jax.random.normal(keys[2], (d_g, 64)) * d_g ** (-0.5)
    return pool, update, projection


def _run_attention(query, key, value, bias, mask, *, implementation: str, d_head: int):
    from .pallas_attention import mhsea_tuned_edge_attention, reference_edge_attention

    if implementation == "einsum":
        return reference_edge_attention(query, key, value, bias, mask)
    padded_width = max(16, d_head)
    padding = padded_width - d_head
    scale = jnp.sqrt(jnp.asarray(padded_width / d_head, dtype=jnp.float32)).astype(
        query.dtype
    )
    zeros = jnp.zeros(
        (query.shape[0], query.shape[1], padding),
        dtype=query.dtype,
    )
    query = jnp.concatenate([query * scale, zeros], axis=-1)
    key = jnp.concatenate([key, zeros], axis=-1)
    value = jnp.concatenate([value, zeros], axis=-1)
    return mhsea_tuned_edge_attention(query, key, value, bias, mask)[..., :d_head]


def _edge_update_dense(layer, edge, edge_n, edge_context, bmask_f, g):
    n = edge.shape[0]
    structural = bmask_f.astype(bool)
    pair_structural = structural[:, None] & structural[None, :]
    context_i = jnp.broadcast_to(
        edge_context[:, None, :],
        (n, n, edge_context.shape[-1]),
    )
    context_j = jnp.broadcast_to(
        edge_context[None, :, :],
        (n, n, edge_context.shape[-1]),
    )
    pair_mask = (bmask_f[:, None] * bmask_f[None, :])[..., None]
    from .tree import _tagged_dense_no_bias

    global_edge = _tagged_dense_no_bias(
        layer.g_edge_proj_w,
        g,
        tag_id="gladder.ctx.eproj",
        pathway="even",
        kfac_structural_mask=jnp.any(structural),
        kfac_repeat_ndim=0,
        kfac_context_primal_reused_over_walkers=True,
    )
    edge_input = jnp.concatenate(
        [
            pair_mask * edge_n,
            context_i,
            context_j,
            jnp.broadcast_to(
                global_edge[None, None, :],
                (n, n, global_edge.shape[-1]),
            ).astype(edge_n.dtype),
        ],
        axis=-1,
    )
    delta = _inline_mlp_forward(
        layer.edge_ffn,
        edge_input,
        pathway="even",
        kfac_structural_mask=pair_structural,
        kfac_repeat_ndim=2,
        kfac_context_primal_reused_over_walkers=True,
    )
    return jnp.where(
        pair_structural[..., None],
        edge + layer.residual_scale * delta,
        jnp.zeros_like(edge),
    )


def _edge_update_tiled(
    layer,
    edge_rows,
    edge_n_rows,
    edge_context_rows,
    edge_context_all,
    bmask,
    *,
    row_indices=None,
    g,
    tile_size: int,
):
    rows, n, _ = edge_rows.shape
    if edge_n_rows.shape != edge_rows.shape:
        raise ValueError("normalized edge rows must match edge rows")
    if edge_context_all.shape[0] != n or bmask.shape != (n,):
        raise ValueError("context and mask must have global width N")
    if row_indices is None:
        if rows != n:
            raise ValueError("row_indices is required for row-sharded edges")
        row_indices = jnp.arange(n, dtype=jnp.int32)
    row_indices = jnp.asarray(row_indices, dtype=jnp.int32)
    mask = bmask.astype(edge_rows.dtype)
    mask_rows = mask[row_indices]
    from .tree import _tagged_dense_no_bias

    global_edge = _tagged_dense_no_bias(
        layer.g_edge_proj_w,
        g,
        tag_id="gladder.ctx.eproj",
        pathway="even",
        kfac_structural_mask=jnp.any(bmask.astype(bool)),
        kfac_repeat_ndim=0,
        kfac_context_primal_reused_over_walkers=True,
    )
    tile_width = min(n, int(tile_size))
    if tile_width < 1:
        raise ValueError("tile_size must be positive")
    full_tiles = n // tile_width
    tail_start = full_tiles * tile_width

    def update_tile(start, width, output):
        mask_tile = jax.lax.dynamic_slice_in_dim(mask, start, width, axis=0)
        edge_n_tile = jax.lax.dynamic_slice_in_dim(edge_n_rows, start, width, axis=1)
        context_tile = jax.lax.dynamic_slice_in_dim(
            edge_context_all, start, width, axis=0
        )
        edge_tile = jax.lax.dynamic_slice_in_dim(output, start, width, axis=1)
        pair_structural = mask_rows[:, None].astype(bool) & mask_tile[None, :].astype(
            bool
        )
        context_i = jnp.broadcast_to(
            edge_context_rows[:, None, :],
            (rows, width, edge_context_rows.shape[-1]),
        )
        context_j = jnp.broadcast_to(
            context_tile[None, :, :],
            (rows, width, edge_context_all.shape[-1]),
        )
        edge_input = jnp.concatenate(
            [
                (mask_rows[:, None] * mask_tile[None, :])[..., None] * edge_n_tile,
                context_i,
                context_j,
                jnp.broadcast_to(
                    global_edge,
                    (rows, width, global_edge.shape[-1]),
                ).astype(edge_rows.dtype),
            ],
            axis=-1,
        )
        delta = _inline_mlp_forward(
            layer.edge_ffn,
            edge_input,
            pathway="even",
            kfac_structural_mask=pair_structural,
            kfac_repeat_ndim=2,
            kfac_context_primal_reused_over_walkers=True,
        )
        updated = jnp.where(
            pair_structural[..., None],
            edge_tile + layer.residual_scale * delta,
            jnp.zeros_like(edge_tile),
        )
        return jax.lax.dynamic_update_slice_in_dim(output, updated, start, axis=1)

    output = jax.lax.fori_loop(
        0,
        full_tiles,
        lambda tile_index, current: update_tile(
            tile_index * tile_width, tile_width, current
        ),
        edge_rows,
    )
    if tail_start < n:
        output = update_tile(tail_start, n - tail_start, output)
    return output


class PhysicalReadoutContextLayer(eqx.Module):
    ln_edge: _RMS
    ln_edge_attn: _RMS
    ln_c: _RMS
    ln_summary: _RMS
    ln_attn: _RMS
    ln_edge_ctx: _RMS
    summary_mlp: MLP
    ctx_mlp: MLP
    edge_node_ctx_proj: BiasFreeLinear
    edge_ffn: MLP
    g_pool: "GDescriptorPool"
    g_update: "ResidualGlobalUpdate"
    g_edge_proj_w: Float[Array, "d_g 64"]
    W_QKV: BiasFreeLinear
    W_O: BiasFreeLinear
    bias_mlp: MLP
    d_e: int = eqx.field(static=True)
    n_heads: int = eqx.field(static=True)
    n_heads_kernel: int = eqx.field(static=True)
    d_head: int = eqx.field(static=True)
    d_attn: int = eqx.field(static=True)
    attn_impl: str = eqx.field(static=True)
    rope_base: float = eqx.field(static=True)
    rope_scaling: float = eqx.field(static=True)
    residual_scale: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        d_e: int,
        d_edge: int,
        n_heads: int,
        summary_hidden: int,
        mlp_hidden: int,
        bias_hidden: int,
        edge_ffn_hidden: int,
        attn_dim: int,
        edge_node_ctx_dim: int,
        attn_impl: str,
        rope_base: float,
        rope_scaling: float,
        gladder_d_g: int,
        global_tap_dim: int,
        residual_scale: float,
        key: PRNGKeyArray,
    ):
        n_heads_kernel, d_head, d_attn = _attention_dimensions(
            d_e=d_e,
            n_heads=n_heads,
            attn_dim=attn_dim,
            attn_impl=attn_impl,
            require_even_model=True,
        )
        if rope_base <= 0.0 or rope_scaling <= 0.0:
            raise ValueError("physical contextualizer rope values must be positive")
        key_sum, key_ctx, key_edge, key_qkv, key_out, key_bias = jax.random.split(
            key, 6
        )
        self.ln_edge = _RMS(d_edge)
        self.ln_edge_attn = _RMS(d_edge)
        self.ln_c = _RMS(d_e)
        self.ln_summary = _RMS(d_e)
        self.ln_attn = _RMS(d_e)
        self.ln_edge_ctx = _RMS(d_e)
        self.summary_mlp = MLP(
            2 * d_edge + 1,
            int(summary_hidden),
            d_e,
            key=key_sum,
            n_blocks=1,
        )
        self.ctx_mlp = MLP(
            2 * d_e,
            int(mlp_hidden),
            d_e,
            key=key_ctx,
            n_blocks=2,
        )
        self.edge_node_ctx_proj = BiasFreeLinear(
            d_e,
            int(edge_node_ctx_dim),
            key=jax.random.fold_in(key, 3694),
        )
        self.edge_ffn = MLP(
            d_edge + 2 * int(edge_node_ctx_dim) + 64,
            int(edge_ffn_hidden),
            d_edge,
            key=key_edge,
            n_blocks=1,
        )
        self.g_pool, self.g_update, self.g_edge_proj_w = _global_modules(
            key,
            int(gladder_d_g),
            d_e,
            residual_scale,
            global_tap_dim,
        )
        self.W_QKV = BiasFreeLinear(d_e, 3 * n_heads_kernel * d_head, key=key_qkv)
        self.W_O = BiasFreeLinear(d_attn, d_e, key=key_out)
        self.bias_mlp = MLP(
            d_edge + 1,
            int(bias_hidden),
            n_heads_kernel,
            key=key_bias,
            n_blocks=1,
        )
        self.d_e = int(d_e)
        self.n_heads = int(n_heads)
        self.n_heads_kernel = int(n_heads_kernel)
        self.d_head = int(d_head)
        self.d_attn = int(d_attn)
        self.attn_impl = str(attn_impl)
        self.rope_base = float(rope_base)
        self.rope_scaling = float(rope_scaling)
        self.residual_scale = float(residual_scale)

    def _slot_clock(self, n: int, dtype, bmask) -> Float[Array, "n d_e"]:
        position = jnp.arange(n, dtype=jnp.float32)
        n_active = jnp.maximum(jnp.sum(jnp.asarray(bmask, dtype=jnp.int32)), 1)
        depth = jnp.ceil(jnp.log2(n_active.astype(jnp.float32))).astype(jnp.int32)
        span = jnp.power(
            jnp.asarray(2.0, dtype=jnp.float32),
            depth.astype(jnp.float32),
        )
        position = position - (span - jnp.asarray(1.0, jnp.float32)) * 0.5
        position = position / jnp.asarray(self.rope_scaling, jnp.float32)
        half = (self.d_e + 1) // 2
        band = jnp.arange(half, dtype=jnp.float32)
        inverse_frequency = jnp.exp(
            -jnp.log(jnp.asarray(self.rope_base, jnp.float32))
            * band
            / jnp.asarray(max(half, 1), jnp.float32)
        )
        angle = position[:, None] * inverse_frequency[None, :]
        embedding = jnp.concatenate([jnp.sin(angle), jnp.cos(angle)], axis=-1)
        return embedding[:, : self.d_e].astype(dtype)

    def _edge_summary(self, edge_n, direction, bmask_f):
        n = edge_n.shape[0]
        dtype = edge_n.dtype
        structural = bmask_f.astype(bool)
        pair_structural = (
            structural[:, None] & structural[None, :] & ~jnp.eye(n, dtype=bool)
        )
        pair = jnp.concatenate(
            [edge_n, jnp.swapaxes(edge_n, 0, 1), direction.astype(dtype)],
            axis=-1,
        )
        message = _inline_mlp_forward(
            self.summary_mlp,
            pair,
            pathway="even",
            kfac_structural_mask=pair_structural,
            kfac_repeat_ndim=2,
            kfac_context_primal_reused_over_walkers=True,
        )
        off_diagonal = 1.0 - jnp.eye(n, dtype=dtype)
        weight = bmask_f[:, None] * bmask_f[None, :] * off_diagonal
        index = jnp.arange(n, dtype=jnp.int32)
        weight = weight * jnp.exp(-0.5 * lca_level(index, index).astype(dtype))
        denominator = jnp.maximum(jnp.sum(weight, axis=1, keepdims=True), 1.0)
        return jnp.sum(message * weight[..., None], axis=1) / denominator

    def edge_summary_tiled(
        self,
        edge_n_rows,
        summary_mask,
        *,
        edge_reverse_rows=None,
        row_indices=None,
        tile_size: int = 128,
    ):
        rows, n, _ = edge_n_rows.shape
        dtype = edge_n_rows.dtype
        if summary_mask.shape != (n,):
            raise ValueError("summary_mask must have global shape [N]")
        if row_indices is None:
            if rows != n:
                raise ValueError("row_indices is required for sharded summaries")
            row_indices = jnp.arange(n, dtype=jnp.int32)
        row_indices = jnp.asarray(row_indices, dtype=jnp.int32)
        if edge_reverse_rows is None:
            if rows != n:
                raise ValueError("reverse rows are required for sharded summaries")
            edge_reverse_rows = jnp.swapaxes(edge_n_rows, 0, 1)
        mask = summary_mask.astype(dtype)
        mask_rows = mask[row_indices]
        numerator = jnp.zeros((rows, self.d_e), dtype=dtype)
        denominator = jnp.zeros((rows, 1), dtype=dtype)
        tile_width = min(n, int(tile_size))
        if tile_width < 1:
            raise ValueError("tile_size must be positive")
        full_tiles = n // tile_width
        tail_start = full_tiles * tile_width

        def accumulate(start, width, carry):
            numerator, denominator = carry
            column = start + jnp.arange(width, dtype=jnp.int32)
            relative = row_indices[:, None] - column[None, :]
            direction = jnp.where(
                relative < 0,
                1.0,
                jnp.where(relative > 0, -1.0, 0.0),
            ).astype(dtype)[..., None]
            off_diagonal = row_indices[:, None] != column[None, :]
            mask_tile = jax.lax.dynamic_slice_in_dim(mask, start, width, axis=0)
            edge_tile = jax.lax.dynamic_slice_in_dim(edge_n_rows, start, width, axis=1)
            reverse_tile = jax.lax.dynamic_slice_in_dim(
                edge_reverse_rows, start, width, axis=1
            )
            structural = (
                mask_rows[:, None].astype(bool)
                & mask_tile[None, :].astype(bool)
                & off_diagonal
            )
            pair = jnp.concatenate([edge_tile, reverse_tile, direction], axis=-1)
            message = _inline_mlp_forward(
                self.summary_mlp,
                pair,
                pathway="even",
                kfac_structural_mask=structural,
                kfac_repeat_ndim=2,
                kfac_context_primal_reused_over_walkers=True,
            )
            weight = (
                mask_rows[:, None] * mask_tile[None, :] * off_diagonal.astype(dtype)
            )
            weight = weight * jnp.exp(
                -0.5 * lca_level(row_indices, column).astype(dtype)
            )
            numerator = numerator + jnp.sum(message * weight[..., None], axis=1)
            denominator = denominator + jnp.sum(weight, axis=1, keepdims=True)
            return numerator, denominator

        numerator, denominator = jax.lax.fori_loop(
            0,
            full_tiles,
            lambda tile_index, carry: accumulate(
                tile_index * tile_width, tile_width, carry
            ),
            (numerator, denominator),
        )
        if tail_start < n:
            numerator, denominator = accumulate(
                tail_start,
                n - tail_start,
                (numerator, denominator),
            )
        return numerator / jnp.maximum(denominator, 1.0)

    def edge_update_tiled(self, *args, **kwargs):
        return _edge_update_tiled(self, *args, **kwargs)

    def _attend(self, c, edge_n, direction, bmask):
        n = c.shape[0]
        dtype = c.dtype
        structural = bmask.astype(bool)
        pair_structural = structural[:, None] & structural[None, :]
        qkv = _inline_bias_free_linear(
            self.W_QKV,
            c,
            pathway="even",
            kfac_structural_mask=structural,
            kfac_repeat_ndim=1,
            kfac_context_primal_reused_over_walkers=True,
        ).reshape(n, 3, self.n_heads_kernel, self.d_head)
        query, key, value = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        bias_input = jnp.concatenate([edge_n, direction.astype(dtype)], axis=-1)
        bias = _inline_mlp_forward(
            self.bias_mlp,
            bias_input,
            pathway="even",
            kfac_structural_mask=pair_structural,
            kfac_repeat_ndim=2,
            kfac_context_primal_reused_over_walkers=True,
        )
        bias = bias / jnp.sqrt(jnp.asarray(self.d_head, dtype=dtype))
        index = jnp.arange(n, dtype=jnp.int32)
        bias = bias + jnp.transpose(
            lca_alibi_bias(
                index,
                index,
                lca_fixed_slopes(self.n_heads_kernel, dtype=dtype),
            ),
            (1, 2, 0),
        )
        output = _run_attention(
            query,
            key,
            value,
            bias,
            bmask.astype(dtype),
            implementation=self.attn_impl,
            d_head=self.d_head,
        )
        output = jax.nn.sigmoid(output[:, : self.n_heads]) * output[:, self.n_heads :]
        return _inline_bias_free_linear(
            self.W_O,
            output.reshape(n, -1),
            pathway="even",
            kfac_structural_mask=structural,
            kfac_repeat_ndim=1,
            kfac_context_primal_reused_over_walkers=True,
        )

    def __call__(self, c, edge, mask, bmask, g):
        del mask
        n = c.shape[0]
        dtype = c.dtype
        bmask_f = bmask.astype(dtype)
        structural = bmask.astype(bool)
        pair_structural = structural[:, None] & structural[None, :]
        edge_n = _inline_norm(
            self.ln_edge,
            edge.astype(dtype),
            pathway="even",
            kfac_structural_mask=pair_structural,
            kfac_repeat_ndim=2,
            kfac_context_primal_reused_over_walkers=True,
        )
        _relative, direction = _relative_positions(n)
        direction = direction.astype(dtype)
        clock = self._slot_clock(n, dtype, bmask)
        summary = self._edge_summary(edge_n, direction, bmask_f)
        context_input = jnp.concatenate(
            [
                _inline_norm(
                    self.ln_c,
                    c + clock,
                    pathway="even",
                    kfac_structural_mask=structural,
                    kfac_repeat_ndim=1,
                    kfac_context_primal_reused_over_walkers=True,
                ),
                _inline_norm(
                    self.ln_summary,
                    summary,
                    pathway="even",
                    kfac_structural_mask=structural,
                    kfac_repeat_ndim=1,
                    kfac_context_primal_reused_over_walkers=True,
                ),
            ],
            axis=-1,
        )
        delta = self.residual_scale * _inline_mlp_forward(
            self.ctx_mlp,
            context_input,
            pathway="even",
            kfac_structural_mask=structural,
            kfac_repeat_ndim=1,
            kfac_context_primal_reused_over_walkers=True,
        )
        c1 = jnp.where(structural[:, None], c + delta, jnp.zeros_like(c))
        global_active = jnp.any(structural)
        g = self.g_update(
            g,
            self.g_pool(
                g,
                c1,
                bmask_f,
                kfac_structural_mask=structural,
                kfac_update_mask=global_active,
                kfac_repeat_ndim=1,
                kfac_context_primal_reused_over_walkers=True,
            ),
            kfac_structural_mask=global_active,
            kfac_context_primal_reused_over_walkers=True,
        )
        edge_context = _inline_bias_free_linear(
            self.edge_node_ctx_proj,
            _inline_norm(
                self.ln_edge_ctx,
                c1,
                pathway="even",
                kfac_structural_mask=structural,
                kfac_repeat_ndim=1,
                kfac_context_primal_reused_over_walkers=True,
            ),
            pathway="even",
            kfac_structural_mask=structural,
            kfac_repeat_ndim=1,
            kfac_context_primal_reused_over_walkers=True,
        )
        edge1 = _edge_update_dense(self, edge, edge_n, edge_context, bmask_f, g)
        attention_input = _inline_norm(
            self.ln_attn,
            c1 + clock,
            pathway="even",
            kfac_structural_mask=structural,
            kfac_repeat_ndim=1,
            kfac_context_primal_reused_over_walkers=True,
        )
        edge_attention = _inline_norm(
            self.ln_edge_attn,
            edge1.astype(dtype),
            pathway="even",
            kfac_structural_mask=pair_structural,
            kfac_repeat_ndim=2,
            kfac_context_primal_reused_over_walkers=True,
        )
        delta_attention = self.residual_scale * self._attend(
            attention_input, edge_attention, direction, bmask
        )
        output = jnp.where(
            structural[:, None],
            c1 + delta_attention,
            jnp.zeros_like(c1),
        )
        return output, edge1, g


class RouterContextLayer(eqx.Module):
    ln_edge: _RMS
    ln_edge_attn: _RMS
    ln_c: _RMS
    ln_attn: _RMS
    ln_edge_ctx: _RMS
    ctx_mlp: MLP
    edge_node_ctx_proj: BiasFreeLinear
    edge_ffn: MLP
    g_pool: "GDescriptorPool"
    g_update: "ResidualGlobalUpdate"
    g_edge_proj_w: Float[Array, "d_g 64"]
    W_QKV: BiasFreeLinear
    W_O: BiasFreeLinear
    bias_mlp: MLP
    n_heads: int = eqx.field(static=True)
    n_heads_kernel: int = eqx.field(static=True)
    d_head: int = eqx.field(static=True)
    d_attn: int = eqx.field(static=True)
    attn_impl: str = eqx.field(static=True)
    residual_scale: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        d_e: int,
        d_edge: int,
        n_heads: int,
        mlp_hidden: int,
        bias_hidden: int,
        edge_ffn_hidden: int,
        attn_dim: int,
        edge_node_ctx_dim: int,
        attn_impl: str,
        gladder_d_g: int,
        global_tap_dim: int,
        residual_scale: float,
        key: PRNGKeyArray,
    ):
        n_heads_kernel, d_head, d_attn = _attention_dimensions(
            d_e=d_e,
            n_heads=n_heads,
            attn_dim=attn_dim,
            attn_impl=attn_impl,
            require_even_model=False,
        )
        _key_sum, key_ctx, key_edge, key_qkv, key_out, key_bias = jax.random.split(
            key, 6
        )
        self.ln_edge = _RMS(d_edge)
        self.ln_edge_attn = _RMS(d_edge)
        self.ln_c = _RMS(d_e)
        self.ln_attn = _RMS(d_e)
        self.ln_edge_ctx = _RMS(d_e)
        self.ctx_mlp = MLP(
            d_e,
            int(mlp_hidden),
            d_e,
            key=key_ctx,
            n_blocks=2,
        )
        self.edge_node_ctx_proj = BiasFreeLinear(
            d_e,
            int(edge_node_ctx_dim),
            key=jax.random.fold_in(key, 3694),
        )
        self.edge_ffn = MLP(
            d_edge + 2 * int(edge_node_ctx_dim) + 64,
            int(edge_ffn_hidden),
            d_edge,
            key=key_edge,
            n_blocks=1,
        )
        self.g_pool, self.g_update, self.g_edge_proj_w = _global_modules(
            key,
            int(gladder_d_g),
            d_e,
            residual_scale,
            global_tap_dim,
        )
        self.W_QKV = BiasFreeLinear(d_e, 3 * n_heads_kernel * d_head, key=key_qkv)
        self.W_O = BiasFreeLinear(d_attn, d_e, key=key_out)
        self.bias_mlp = MLP(
            d_edge + 1,
            int(bias_hidden),
            n_heads_kernel,
            key=key_bias,
            n_blocks=1,
        )
        self.n_heads = int(n_heads)
        self.n_heads_kernel = int(n_heads_kernel)
        self.d_head = int(d_head)
        self.d_attn = int(d_attn)
        self.attn_impl = str(attn_impl)
        self.residual_scale = float(residual_scale)

    def edge_update_tiled(self, *args, **kwargs):
        return _edge_update_tiled(self, *args, **kwargs)

    def _attend(self, c, edge_n, bmask):
        n = c.shape[0]
        dtype = c.dtype
        structural = bmask.astype(bool)
        pair_structural = structural[:, None] & structural[None, :]
        qkv = _inline_bias_free_linear(
            self.W_QKV,
            c,
            pathway="even",
            kfac_structural_mask=structural,
            kfac_repeat_ndim=1,
            kfac_context_primal_reused_over_walkers=True,
        ).reshape(n, 3, self.n_heads_kernel, self.d_head)
        query, key, value = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        direction = jnp.zeros((n, n, 1), dtype=dtype)
        bias = _inline_mlp_forward(
            self.bias_mlp,
            jnp.concatenate([edge_n, direction], axis=-1),
            pathway="even",
            kfac_structural_mask=pair_structural,
            kfac_repeat_ndim=2,
            kfac_context_primal_reused_over_walkers=True,
        )
        bias = bias / jnp.sqrt(jnp.asarray(self.d_head, dtype=dtype))
        output = _run_attention(
            query,
            key,
            value,
            bias,
            bmask.astype(dtype),
            implementation=self.attn_impl,
            d_head=self.d_head,
        )
        output = jax.nn.sigmoid(output[:, : self.n_heads]) * output[:, self.n_heads :]
        return _inline_bias_free_linear(
            self.W_O,
            output.reshape(n, -1),
            pathway="even",
            kfac_structural_mask=structural,
            kfac_repeat_ndim=1,
            kfac_context_primal_reused_over_walkers=True,
        )

    def __call__(self, c, edge, mask, bmask, g):
        del mask
        dtype = c.dtype
        bmask_f = bmask.astype(dtype)
        structural = bmask.astype(bool)
        pair_structural = structural[:, None] & structural[None, :]
        edge_n = _inline_norm(
            self.ln_edge,
            edge.astype(dtype),
            pathway="even",
            kfac_structural_mask=pair_structural,
            kfac_repeat_ndim=2,
            kfac_context_primal_reused_over_walkers=True,
        )
        context_input = _inline_norm(
            self.ln_c,
            c,
            pathway="even",
            kfac_structural_mask=structural,
            kfac_repeat_ndim=1,
            kfac_context_primal_reused_over_walkers=True,
        )
        delta = self.residual_scale * _inline_mlp_forward(
            self.ctx_mlp,
            context_input,
            pathway="even",
            kfac_structural_mask=structural,
            kfac_repeat_ndim=1,
            kfac_context_primal_reused_over_walkers=True,
        )
        c1 = jnp.where(structural[:, None], c + delta, jnp.zeros_like(c))
        global_active = jnp.any(structural)
        g = self.g_update(
            g,
            self.g_pool(
                g,
                c1,
                bmask_f,
                kfac_structural_mask=structural,
                kfac_update_mask=global_active,
                kfac_repeat_ndim=1,
                kfac_context_primal_reused_over_walkers=True,
            ),
            kfac_structural_mask=global_active,
            kfac_context_primal_reused_over_walkers=True,
        )
        edge_context = _inline_bias_free_linear(
            self.edge_node_ctx_proj,
            _inline_norm(
                self.ln_edge_ctx,
                c1,
                pathway="even",
                kfac_structural_mask=structural,
                kfac_repeat_ndim=1,
                kfac_context_primal_reused_over_walkers=True,
            ),
            pathway="even",
            kfac_structural_mask=structural,
            kfac_repeat_ndim=1,
            kfac_context_primal_reused_over_walkers=True,
        )
        edge1 = _edge_update_dense(self, edge, edge_n, edge_context, bmask_f, g)
        attention_input = _inline_norm(
            self.ln_attn,
            c1,
            pathway="even",
            kfac_structural_mask=structural,
            kfac_repeat_ndim=1,
            kfac_context_primal_reused_over_walkers=True,
        )
        edge_attention = _inline_norm(
            self.ln_edge_attn,
            edge1.astype(dtype),
            pathway="even",
            kfac_structural_mask=pair_structural,
            kfac_repeat_ndim=2,
            kfac_context_primal_reused_over_walkers=True,
        )
        delta_attention = self.residual_scale * self._attend(
            attention_input, edge_attention, bmask
        )
        output = jnp.where(
            structural[:, None],
            c1 + delta_attention,
            jnp.zeros_like(c1),
        )
        return output, edge1, g


def _stack_layers(layers):
    dynamic_static = [eqx.partition(layer, eqx.is_array) for layer in layers]
    dynamic = [item[0] for item in dynamic_static]
    static = dynamic_static[0][1]
    stacked = jax.tree.map(lambda *values: jnp.stack(values, axis=0), *dynamic)
    return eqx.combine(stacked, static)


def _initialize_structural_inputs(contextualizer, c, edge, mask, bmask):
    real = mask.astype(bool)
    active = bmask.astype(bool)
    virtual = active & ~real
    virtual_node = register_vector_as_dense(
        contextualizer.virtual_node,
        tag_id=contextualizer._use_id_virtual_node,
    )[0].astype(c.dtype)
    empty_nonempty = register_vector_as_dense(
        contextualizer.edge_empty_nonempty,
        tag_id=contextualizer._use_id_edge_empty_nonempty,
    )[0].astype(edge.dtype)
    empty_empty = register_vector_as_dense(
        contextualizer.edge_empty_empty,
        tag_id=contextualizer._use_id_edge_empty_empty,
    )[0].astype(edge.dtype)
    c = jnp.where(
        real[:, None],
        c,
        jnp.where(
            virtual[:, None],
            virtual_node[None, :],
            jnp.zeros_like(c),
        ),
    )
    real_pair = real[:, None] & real[None, :]
    mixed_pair = (real[:, None] & virtual[None, :]) | (virtual[:, None] & real[None, :])
    virtual_pair = virtual[:, None] & virtual[None, :]
    edge = jnp.where(
        real_pair[..., None],
        edge,
        jnp.where(
            mixed_pair[..., None],
            empty_nonempty[None, None, :],
            jnp.where(
                virtual_pair[..., None],
                empty_empty[None, None, :],
                jnp.zeros_like(edge),
            ),
        ),
    )
    return c, edge


def _run_context_layers(contextualizer, c, edge, mask, bmask, g):
    dynamic, static = eqx.partition(contextualizer.layers, eqx.is_array)

    def scan_step(carry, layer_dynamic):
        layer = eqx.combine(layer_dynamic, static)
        c_value, edge_value, g_value = carry
        return (
            layer(c_value, edge_value, mask, bmask, g_value),
            None,
        )

    (c, edge, g), _ = jax.lax.scan(
        scan_step,
        (c, edge, g),
        dynamic,
    )
    return c, edge, g


class PhysicalReadoutContext(eqx.Module):
    layers: PhysicalReadoutContextLayer
    virtual_node: Float[Array, "one d_e"]
    edge_empty_nonempty: Float[Array, "one d_edge"]
    edge_empty_empty: Float[Array, "one d_edge"]
    _use_id_virtual_node: str = eqx.field(static=True, default="")
    _use_id_edge_empty_nonempty: str = eqx.field(static=True, default="")
    _use_id_edge_empty_empty: str = eqx.field(static=True, default="")

    def __init__(
        self,
        *,
        d_e: int,
        d_edge: int,
        n_layers: int,
        n_heads: int,
        summary_hidden: int,
        mlp_hidden: int,
        bias_hidden: int,
        edge_ffn_hidden: int,
        attn_dim: int,
        edge_node_ctx_dim: int,
        attn_impl: str,
        rope_base: float,
        rope_scaling: float,
        gladder_d_g: int,
        global_tap_dim: int,
        key: PRNGKeyArray,
    ):
        if n_layers < 1:
            raise ValueError("physical contextualizer layers must be positive")
        residual_scale = float(n_layers) ** (-0.5)
        keys = jax.random.split(key, n_layers)
        self.layers = _stack_layers(
            [
                PhysicalReadoutContextLayer(
                    d_e=d_e,
                    d_edge=d_edge,
                    n_heads=n_heads,
                    summary_hidden=summary_hidden,
                    mlp_hidden=mlp_hidden,
                    bias_hidden=bias_hidden,
                    edge_ffn_hidden=edge_ffn_hidden,
                    attn_dim=attn_dim,
                    edge_node_ctx_dim=edge_node_ctx_dim,
                    attn_impl=attn_impl,
                    rope_base=rope_base,
                    rope_scaling=rope_scaling,
                    gladder_d_g=gladder_d_g,
                    global_tap_dim=global_tap_dim,
                    residual_scale=residual_scale,
                    key=layer_key,
                )
                for layer_key in keys
            ]
        )
        self.virtual_node = jax.random.normal(
            jax.random.fold_in(key, 201793223), (1, d_e)
        ) * d_e ** (-0.5)
        self.edge_empty_nonempty = jax.random.normal(
            jax.random.fold_in(key, 235798529), (1, d_edge)
        ) * d_edge ** (-0.5)
        self.edge_empty_empty = jax.random.normal(
            jax.random.fold_in(key, 235798530), (1, d_edge)
        ) * d_edge ** (-0.5)
        self._use_id_virtual_node = ""
        self._use_id_edge_empty_nonempty = ""
        self._use_id_edge_empty_empty = ""

    def with_edge(self, c, edge, mask, bmask, *, g):
        c = c.astype(jnp.float32)
        edge = edge.astype(jnp.float32)
        c, edge = _initialize_structural_inputs(self, c, edge, mask, bmask)
        return _run_context_layers(self, c, edge, mask, bmask, g)


class RouterContext(eqx.Module):
    layers: RouterContextLayer
    virtual_node: Float[Array, "one d_e"]
    edge_empty_nonempty: Float[Array, "one d_edge"]
    edge_empty_empty: Float[Array, "one d_edge"]
    _use_id_virtual_node: str = eqx.field(static=True, default="")
    _use_id_edge_empty_nonempty: str = eqx.field(static=True, default="")
    _use_id_edge_empty_empty: str = eqx.field(static=True, default="")

    def __init__(
        self,
        *,
        d_e: int,
        d_edge: int,
        n_layers: int,
        n_heads: int,
        mlp_hidden: int,
        bias_hidden: int,
        edge_ffn_hidden: int,
        attn_dim: int,
        edge_node_ctx_dim: int,
        attn_impl: str,
        gladder_d_g: int,
        global_tap_dim: int,
        key: PRNGKeyArray,
    ):
        if n_layers < 1:
            raise ValueError("router contextualizer layers must be positive")
        residual_scale = float(n_layers) ** (-0.5)
        keys = jax.random.split(key, n_layers)
        self.layers = _stack_layers(
            [
                RouterContextLayer(
                    d_e=d_e,
                    d_edge=d_edge,
                    n_heads=n_heads,
                    mlp_hidden=mlp_hidden,
                    bias_hidden=bias_hidden,
                    edge_ffn_hidden=edge_ffn_hidden,
                    attn_dim=attn_dim,
                    edge_node_ctx_dim=edge_node_ctx_dim,
                    attn_impl=attn_impl,
                    gladder_d_g=gladder_d_g,
                    global_tap_dim=global_tap_dim,
                    residual_scale=residual_scale,
                    key=layer_key,
                )
                for layer_key in keys
            ]
        )
        self.virtual_node = jax.random.normal(
            jax.random.fold_in(key, 201793223), (1, d_e)
        ) * d_e ** (-0.5)
        self.edge_empty_nonempty = jax.random.normal(
            jax.random.fold_in(key, 235798529), (1, d_edge)
        ) * d_edge ** (-0.5)
        self.edge_empty_empty = jax.random.normal(
            jax.random.fold_in(key, 235798530), (1, d_edge)
        ) * d_edge ** (-0.5)
        self._use_id_virtual_node = ""
        self._use_id_edge_empty_nonempty = ""
        self._use_id_edge_empty_empty = ""

    def with_edge(self, c, edge, mask, bmask, *, g):
        c = c.astype(jnp.float32)
        edge = edge.astype(jnp.float32)
        c, edge = _initialize_structural_inputs(self, c, edge, mask, bmask)
        return _run_context_layers(self, c, edge, mask, bmask, g)


__all__ = [
    "PhysicalReadoutContext",
    "PhysicalReadoutContextLayer",
    "RouterContext",
    "RouterContextLayer",
    "default_tree_depth",
    "lca_alibi_bias",
    "lca_fixed_slopes",
    "lca_gaussian_decay",
    "lca_gaussian_decay_row",
    "lca_level",
    "lca_order_init_w_b",
    "register_vector_as_dense",
]
