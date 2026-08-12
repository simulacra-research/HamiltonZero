# Copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray

from .fused_silu import fused_silu
from .readout_leaf_context import (
    default_tree_depth,
    lca_alibi_bias,
    lca_fixed_slopes,
    lca_gaussian_decay,
    lca_gaussian_decay_row,
)
from .route_quotient import (
    conditional_orbit_ids_from_keys,
    conditional_orbit_pair_ids_from_keys,
)
from .tree import (
    CausalRouterEdgeFWLUpdate,
    EdgeMergeOp,
    _TREE_NGPT_DEPTH_FEAT_DIM,
    _tree_ngpt_residual,
    _tree_clock_root_center_from_depth,
    _tree_dyadic_segment_clock,
    _tree_ngpt_level_counts,
    _tree_sphere,
)

QuotientCarrier = tuple[Array, Array] | tuple[Array, Array, Array]


def _route_next_pow2(n: int) -> int:
    return 1 << (int(n) - 1).bit_length()


def _dyadic_frontier_add(frontier, value, position):

    carry = jnp.asarray(value, dtype=frontier.dtype)
    active = jnp.asarray(True)
    pos = jnp.asarray(position, dtype=jnp.int32)
    for level in range(frontier.shape[0]):
        old = frontier[level]
        occupied = jnp.bitwise_and(jnp.right_shift(pos, level), 1) == 1
        merge = active & occupied
        place = active & ~occupied
        frontier = frontier.at[level].set(
            jnp.where(place, carry, jnp.where(merge, jnp.zeros_like(old), old))
        )
        carry = jnp.where(merge, old + carry, carry)
        active = merge
    return frontier


def _dyadic_lca_frontier_sum(frontier, position, w_raw, b):

    depth = frontier.shape[0]
    pos = jnp.asarray(position, dtype=jnp.int32)
    levels = jnp.arange(depth, dtype=jnp.int32)
    widths = jnp.left_shift(jnp.ones((depth,), dtype=jnp.int32), levels + 1)
    starts = jnp.bitwise_and(pos, jnp.bitwise_not(widths - 1))
    present = jnp.bitwise_and(jnp.right_shift(pos, levels), 1) == 1
    decay = lca_gaussian_decay_row(pos, starts, w_raw, b)
    scale = jnp.where(present[:, None], decay, jnp.zeros_like(decay))
    while scale.ndim < frontier.ndim:
        scale = scale[:, None, :]
    return jnp.sum(frontier * scale, axis=0)


def _replace_square_row_column(matrix, index, row_value, column_value):

    idx = jnp.arange(matrix.shape[0], dtype=jnp.int32)
    select = idx == jnp.asarray(index, dtype=jnp.int32)
    diagonal = row_value[index]
    column_value = jnp.where(select[:, None], diagonal, column_value)
    matrix = jnp.where(select[:, None, None], row_value[None, :, :], matrix)
    return jnp.where(
        select[None, :, None],
        column_value[:, None, :],
        matrix,
    )


def _square_row_by_reduction(matrix, index):

    idx = jnp.arange(matrix.shape[0], dtype=jnp.int32)
    select = idx == jnp.asarray(index, dtype=jnp.int32)
    return jnp.sum(
        jnp.where(select[:, None, None], matrix, jnp.zeros_like(matrix)),
        axis=0,
    )


def _square_column_local(matrix, index):

    idx = jnp.arange(matrix.shape[1], dtype=jnp.int32)
    select = idx == jnp.asarray(index, dtype=jnp.int32)
    return jnp.sum(
        jnp.where(select[None, :, None], matrix, jnp.zeros_like(matrix)),
        axis=1,
    )


def _route_clock(pos, width: int, dtype, *, base: float | Array = 10000.0, scale=None):
    if int(width) <= 0:
        pos_arr = jnp.asarray(pos)
        return jnp.zeros(pos_arr.shape + (0,), dtype=dtype)
    pos_f = jnp.asarray(pos, dtype=jnp.float32)
    if scale is not None:
        denom = jnp.maximum(
            jnp.asarray(scale, dtype=jnp.float32) - jnp.asarray(1.0, dtype=jnp.float32),
            jnp.asarray(1.0, dtype=jnp.float32),
        )
        pos_f = pos_f / denom
    half = (int(width) + 1) // 2
    band = jnp.arange(half, dtype=jnp.float32)
    base_f = jnp.maximum(
        jnp.asarray(base, dtype=jnp.float32),
        jnp.asarray(2.0, dtype=jnp.float32),
    )
    inv_freq = jnp.exp(
        -jnp.log(base_f) * band / jnp.asarray(max(half, 1), dtype=jnp.float32)
    )
    angle = pos_f[..., None] * inv_freq
    emb = jnp.concatenate([jnp.sin(angle), jnp.cos(angle)], axis=-1)
    return emb[..., : int(width)].astype(dtype)


def _route_merge_clock(
    level_idx,
    pair_idx,
    pair_base,
    width: int,
    max_depth,
    dtype,
    *,
    root_centered: bool = False,
):
    del pair_base
    root_center = (
        _tree_clock_root_center_from_depth(max_depth, dtype) if root_centered else None
    )
    return _tree_dyadic_segment_clock(
        level_idx,
        pair_idx,
        width,
        dtype,
        root_center=root_center,
    )


class _RoutePointerBase(eqx.Module):
    w_global: Float[Array, "d_global d_model"]
    b_global: Float[Array, "d_model"]

    pref_msg_ln_scale: Float[Array, "two_d_edge"]
    pref_msg_w1: Float[Array, "two_d_edge d_msg_hidden"]
    pref_msg_b1: Float[Array, "d_msg_hidden"]
    pref_msg_w2: Float[Array, "d_msg_hidden d_model"]
    pref_msg_b2: Float[Array, "d_model"]
    suff_msg_ln_scale: Float[Array, "two_d_edge"]
    suff_msg_w1: Float[Array, "two_d_edge d_msg_hidden"]
    suff_msg_b1: Float[Array, "d_msg_hidden"]
    suff_msg_w2: Float[Array, "d_msg_hidden d_model"]
    suff_msg_b2: Float[Array, "d_model"]

    virt_emb: Float[Array, "one d_model"]

    order_decay_w: Float[Array, "one d_model"]
    order_decay_b: Float[Array, "one d_model"]

    virt_decay_w: Float[Array, "one d_model"]
    virt_decay_b: Float[Array, "one d_model"]
    cand_node_ln_scale: Float[Array, "d_in"]
    cand_global_ln_scale: Float[Array, "d_model"]

    cand_graw_ln_scale: Float[Array, "d_graw"]

    cand_g_tap_w: Float[Array, "d_global d_graw"]
    cand_pref_ln_scale: Float[Array, "d_model"]
    cand_pref_order_ln_scale: Float[Array, "d_model"]
    cand_suff_ln_scale: Float[Array, "d_model"]
    cand_virt_pref_ln_scale: Float[Array, "d_model"]
    cand_node_w: Float[Array, "d_in d_cand_hidden"]
    cand_global_w: Float[Array, "d_model d_cand_hidden"]
    cand_graw_w: Float[Array, "d_graw d_cand_hidden"]
    cand_pref_w: Float[Array, "d_model d_cand_hidden"]
    cand_pref_order_w: Float[Array, "d_model d_cand_hidden"]
    cand_suff_w: Float[Array, "d_model d_cand_hidden"]
    cand_virt_pref_w: Float[Array, "d_model d_cand_hidden"]
    cand_virt_ratios_w: Float[Array, "three d_cand_hidden"]
    cand_b_in: Float[Array, "d_cand_hidden"]
    cand_block_ln_scale: Float[Array, "b d_cand_hidden"]
    cand_block_w1: Float[Array, "b d_cand_hidden d_cand_hidden"]
    cand_block_b1: Float[Array, "b d_cand_hidden"]
    cand_block_w2: Float[Array, "b d_cand_hidden d_cand_hidden"]
    cand_block_b2: Float[Array, "b d_cand_hidden"]
    cand_out_ln_scale: Float[Array, "d_cand_hidden"]
    cand_w_out: Float[Array, "d_cand_hidden d_model"]
    cand_b_out: Float[Array, "d_model"]

    pointer_q_w: Float[Array, "d_model d_score"]
    pointer_k_w: Float[Array, "d_model d_score"]
    d_in: int = eqx.field(static=True)
    d_global: int = eqx.field(static=True)
    d_edge: int = eqx.field(static=True)
    d_model: int = eqx.field(static=True)
    d_attn: int = eqx.field(static=True)
    pointer_score_dim: int = eqx.field(static=True)
    n_heads: int = eqx.field(static=True)
    n_heads_kernel: int = eqx.field(static=True)
    d_head: int = eqx.field(static=True)
    max_n: int = eqx.field(static=True)
    ffn_hidden: int = eqx.field(static=True)
    msg_hidden: int = eqx.field(static=True)
    cand_hidden: int = eqx.field(static=True)
    rope_base: float = eqx.field(static=True)
    rope_scaling: float = eqx.field(static=True)
    ln_eps: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        d_in: int,
        d_edge: int,
        d_global: int,
        d_model: int,
        n_heads: int,
        max_n: int,
        key: PRNGKeyArray,
        rope_base: float = 10000.0,
        rope_scaling: float = 1.0,
        attention_dim: int,
        pointer_score_dim: int,
        candidate_hidden: int,
        summary_hidden: int,
        ffn_hidden: int,
        global_tap_dim: int,
        score_init_scale: float = 1.0,
    ):
        ln_eps = 1.0e-5
        if d_in != d_model:
            raise ValueError("router requires d_in == d_model")
        if d_global < 1:
            raise ValueError("route pointer d_global must be >= 1")
        d_attn = int(attention_dim)
        if d_attn < 1:
            raise ValueError("route pointer attention_dim must be positive")
        if d_attn % n_heads != 0:
            raise ValueError("route pointer attention_dim must be divisible by n_heads")
        d_head = d_attn // n_heads
        if d_head % 2 != 0:
            raise ValueError("route pointer RoPE requires an even per-head dim")
        score_dim = int(pointer_score_dim)
        if score_dim < 1:
            raise ValueError("route pointer pointer_score_dim must be positive")
        if max_n < 1:
            raise ValueError("route pointer max_n must be >= 1")
        if rope_base <= 0.0 or rope_scaling <= 0.0:
            raise ValueError("route pointer RoPE base/scaling must be positive")
        n_heads_kernel = 2 * n_heads
        ffn_hidden = int(ffn_hidden)
        msg_hidden = int(summary_hidden)
        if ffn_hidden < 1 or msg_hidden < 1:
            raise ValueError("route pointer FFN/summary widths must be positive")
        n_virt_ratios = 3

        _graw_dim = int(global_tap_dim)
        if _graw_dim < 1 or _graw_dim >= int(d_global):
            raise ValueError(
                "global_tap_dim must be positive and smaller than d_global"
            )
        cand_in = d_in + 5 * d_model + n_virt_ratios + _graw_dim
        cand_hidden = int(candidate_hidden)
        if cand_hidden < 1:
            raise ValueError("route pointer candidate_hidden must be positive")
        keys = jax.random.split(key, 22)

        def w(k, shape, fan_in):
            return jax.random.normal(k, shape) * (fan_in**-0.5)

        k_in, k_global = jax.random.split(keys[0], 2)
        del k_in
        self.w_global = w(k_global, (int(d_global), d_model), int(d_global))
        self.b_global = jnp.zeros((d_model,))

        self.cand_graw_ln_scale = jnp.ones((_graw_dim,))
        self.cand_g_tap_w = w(
            jax.random.fold_in(k_global, 0x67AB),
            (int(d_global), _graw_dim),
            int(d_global),
        )

        self.pref_msg_ln_scale = jnp.ones((2 * d_edge,))
        self.pref_msg_w1 = w(keys[7], (2 * d_edge, msg_hidden), 2 * d_edge)
        self.pref_msg_b1 = jnp.zeros((msg_hidden,))
        self.pref_msg_w2 = w(keys[8], (msg_hidden, d_model), msg_hidden)
        self.pref_msg_b2 = jnp.zeros((d_model,))
        self.suff_msg_ln_scale = jnp.ones((2 * d_edge,))
        self.suff_msg_w1 = w(keys[9], (2 * d_edge, msg_hidden), 2 * d_edge)
        self.suff_msg_b1 = jnp.zeros((msg_hidden,))
        self.suff_msg_w2 = w(keys[10], (msg_hidden, d_model), msg_hidden)
        self.suff_msg_b2 = jnp.zeros((d_model,))

        vkey = jax.random.fold_in(key, 0x5710C)
        self.virt_emb = jax.random.normal(vkey, (1, d_model)) * (d_model**-0.5)
        from .readout_leaf_context import lca_order_init_w_b

        self.order_decay_w, self.order_decay_b = lca_order_init_w_b(d_model)
        self.virt_decay_w, self.virt_decay_b = lca_order_init_w_b(d_model)
        self.cand_node_ln_scale = jnp.ones((d_in,))
        self.cand_global_ln_scale = jnp.ones((d_model,))
        self.cand_pref_ln_scale = jnp.ones((d_model,))
        self.cand_pref_order_ln_scale = jnp.ones((d_model,))
        self.cand_suff_ln_scale = jnp.ones((d_model,))
        self.cand_virt_pref_ln_scale = jnp.ones((d_model,))

        compose_key = keys[15]
        self.cand_node_w = w(
            jax.random.fold_in(compose_key, 0),
            (d_in, cand_hidden),
            cand_in,
        )
        self.cand_global_w = w(
            jax.random.fold_in(compose_key, 1),
            (d_model, cand_hidden),
            cand_in,
        )
        self.cand_graw_w = w(
            jax.random.fold_in(compose_key, 2),
            (_graw_dim, cand_hidden),
            cand_in,
        )
        self.cand_pref_w = w(
            jax.random.fold_in(compose_key, 3),
            (d_model, cand_hidden),
            cand_in,
        )
        self.cand_pref_order_w = w(
            jax.random.fold_in(compose_key, 4),
            (d_model, cand_hidden),
            cand_in,
        )
        self.cand_suff_w = w(
            jax.random.fold_in(compose_key, 5),
            (d_model, cand_hidden),
            cand_in,
        )
        self.cand_virt_pref_w = w(
            jax.random.fold_in(compose_key, 6),
            (d_model, cand_hidden),
            cand_in,
        )
        self.cand_virt_ratios_w = w(
            jax.random.fold_in(compose_key, 10),
            (n_virt_ratios, cand_hidden),
            cand_in,
        )
        self.cand_b_in = jnp.zeros((cand_hidden,))
        self.cand_block_ln_scale = jnp.ones((1, cand_hidden))
        self.cand_block_w1 = w(
            keys[16],
            (1, cand_hidden, cand_hidden),
            cand_hidden,
        )
        self.cand_block_b1 = jnp.zeros((1, cand_hidden))
        self.cand_block_w2 = w(
            keys[17],
            (1, cand_hidden, cand_hidden),
            cand_hidden,
        )
        self.cand_block_b2 = jnp.zeros((1, cand_hidden))
        cand_out_key = keys[18]
        self.cand_out_ln_scale = jnp.ones((cand_hidden,))
        self.cand_w_out = w(cand_out_key, (cand_hidden, d_model), cand_hidden)
        self.cand_b_out = jnp.zeros((d_model,))

        pointer_q_key = keys[19]
        pointer_k_key = keys[20]

        self.pointer_q_w = w(pointer_q_key, (d_model, score_dim), d_model) * float(
            score_init_scale
        )
        self.pointer_k_w = w(pointer_k_key, (d_model, score_dim), d_model)
        del keys

        self.d_in = d_in
        self.d_global = int(d_global)
        self.d_edge = d_edge
        self.d_model = d_model
        self.d_attn = d_attn
        self.pointer_score_dim = score_dim
        self.n_heads = n_heads
        self.n_heads_kernel = n_heads_kernel
        self.d_head = d_head
        self.max_n = max_n
        self.ffn_hidden = ffn_hidden
        self.msg_hidden = msg_hidden
        self.cand_hidden = cand_hidden
        self.rope_base = float(rope_base)
        self.rope_scaling = float(rope_scaling)
        self.ln_eps = float(ln_eps)

    def _ln(
        self,
        scale,
        x,
        *,
        tag_id: str,
        kfac_structural_mask=None,
        kfac_scan_shared: bool = False,
        kfac_repeat_ndim: int = 0,
        kfac_context_primal_reused_over_walkers: bool = False,
    ):
        from hamiltonzero.model.tree import _tagged_rms_eqx_style

        return _tagged_rms_eqx_style(
            scale,
            x,
            eps=self.ln_eps,
            tag_id=tag_id,
            pathway="even",
            var_floor=1e-2,
            kfac_structural_mask=kfac_structural_mask,
            kfac_scan_shared=kfac_scan_shared,
            kfac_repeat_ndim=kfac_repeat_ndim,
            kfac_context_primal_reused_over_walkers=(
                kfac_context_primal_reused_over_walkers
            ),
        )

    def _cross_ln(
        self,
        scale,
        shift,
        x,
        *,
        tag_id: str,
        kfac_structural_mask=None,
        kfac_scan_shared: bool = False,
        kfac_repeat_ndim: int = 0,
        kfac_context_primal_reused_over_walkers: bool = False,
    ):
        from hamiltonzero.model.tree import _tagged_ln_eqx_style

        return _tagged_ln_eqx_style(
            scale,
            shift,
            x,
            eps=self.ln_eps,
            tag_id=tag_id,
            pathway="even",
            var_floor=1e-2,
            kfac_structural_mask=kfac_structural_mask,
            kfac_scan_shared=kfac_scan_shared,
            kfac_repeat_ndim=kfac_repeat_ndim,
            kfac_context_primal_reused_over_walkers=(
                kfac_context_primal_reused_over_walkers
            ),
        )

    def _dense(
        self,
        weight,
        bias,
        x,
        *,
        tag_id: str,
        kfac_structural_mask=None,
        kfac_scan_shared: bool = False,
        kfac_repeat_ndim: int = 0,
        kfac_context_primal_reused_over_walkers: bool = False,
    ):
        from hamiltonzero.model.tree import _tagged_dense

        return _tagged_dense(
            weight,
            bias,
            x,
            tag_id=tag_id,
            pathway="even",
            kfac_structural_mask=kfac_structural_mask,
            kfac_scan_shared=kfac_scan_shared,
            kfac_repeat_ndim=kfac_repeat_ndim,
            kfac_context_primal_reused_over_walkers=(
                kfac_context_primal_reused_over_walkers
            ),
        )

    def _dense_no_bias(
        self,
        weight,
        x,
        *,
        tag_id: str,
        kfac_structural_mask=None,
        kfac_scan_shared: bool = False,
        kfac_repeat_ndim: int = 0,
        kfac_context_primal_reused_over_walkers: bool = False,
    ):
        from hamiltonzero.model.tree import _tagged_dense_no_bias

        return _tagged_dense_no_bias(
            weight,
            x,
            tag_id=tag_id,
            pathway="even",
            kfac_structural_mask=kfac_structural_mask,
            kfac_scan_shared=kfac_scan_shared,
            kfac_repeat_ndim=kfac_repeat_ndim,
            kfac_context_primal_reused_over_walkers=(
                kfac_context_primal_reused_over_walkers
            ),
        )

    def _project_nodes(self, h: Float[Array, "n d_in"], structural_mask=None):
        del structural_mask
        return h

    def _prepare_nodes(
        self,
        h: Float[Array, "n d_in"],
        mask: Int[Array, "n"] | Array,
    ):

        projected, node_mean = self._center_nodes(
            self._project_nodes(h, mask.astype(bool)),
            mask,
        )
        return (h, projected), node_mean

    def _project_global(
        self,
        global_feat: Float[Array, "d_global"],
        dtype,
        structural_mask=None,
    ):

        raw = global_feat.astype(dtype)
        g_dm = self._dense(
            self.w_global,
            self.b_global,
            raw,
            tag_id="route.global_input",
            kfac_structural_mask=structural_mask,
            kfac_repeat_ndim=0,
        )
        return (raw, g_dm)

    def _center_nodes(
        self,
        node_state: Float[Array, "n d_model"],
        mask: Int[Array, "n"] | Array,
    ):
        dtype = node_state.dtype
        active = mask.astype(dtype).reshape(node_state.shape[0], 1)
        denom = jnp.maximum(jnp.sum(active), jnp.asarray(1.0, dtype=dtype))
        global_state = jnp.sum(node_state * active, axis=0) / denom
        return node_state, global_state

    def _message_mlp(self, edge_pair, *, prefix: bool, structural_mask=None):
        if prefix:
            ln_s = self.pref_msg_ln_scale
            w1, b1, w2, b2 = (
                self.pref_msg_w1,
                self.pref_msg_b1,
                self.pref_msg_w2,
                self.pref_msg_b2,
            )
            name = "pref"
        else:
            ln_s = self.suff_msg_ln_scale
            w1, b1, w2, b2 = (
                self.suff_msg_w1,
                self.suff_msg_b1,
                self.suff_msg_w2,
                self.suff_msg_b2,
            )
            name = "suff"
        structural_mask = (
            jnp.ones(edge_pair.shape[:-1], dtype=bool)
            if structural_mask is None
            else jnp.broadcast_to(
                jnp.asarray(structural_mask, dtype=bool), edge_pair.shape[:-1]
            )
        )
        kfac_kwargs = dict(
            kfac_structural_mask=structural_mask,
            kfac_repeat_ndim=structural_mask.ndim,
        )
        x = self._ln(
            ln_s,
            edge_pair,
            tag_id=f"route.candidate.{name}_msg_ln",
            **kfac_kwargs,
        )
        x = self._dense(
            w1,
            b1,
            x,
            tag_id=f"route.candidate.{name}_msg1",
            **kfac_kwargs,
        )
        x = fused_silu(x)
        return self._dense(
            w2,
            b2,
            x,
            tag_id=f"route.candidate.{name}_msg2",
            **kfac_kwargs,
        )

    def _edge_pair_for_source(
        self,
        edge: Float[Array, "n n d_edge"],
        source: Int[Array, ""],
    ) -> Float[Array, "n two_d_edge"]:
        return jnp.concatenate(
            [edge[:, source, :], edge[source, :, :]],
            axis=-1,
        )

    def _ordered_edge_messages(
        self,
        edge: Float[Array, "n n d_edge"],
        perm: Int[Array, "n"],
        mask: Int[Array, "n"] | Array,
    ):
        n = edge.shape[0]
        idx = jnp.arange(n, dtype=jnp.int32)
        edge_i_p = edge[idx[None, :], perm[:, None], :]
        edge_p_i = edge[perm[:, None], idx[None, :], :]
        edge_pair = jnp.concatenate([edge_i_p, edge_p_i], axis=-1)
        mask_bool = mask.astype(bool)
        pair_structural_mask = mask_bool[perm][:, None] & mask_bool[None, :]
        return (
            self._message_mlp(
                edge_pair,
                prefix=True,
                structural_mask=pair_structural_mask,
            ),
            self._message_mlp(
                edge_pair,
                prefix=False,
                structural_mask=pair_structural_mask,
            ),
        )

    def _clock_root_center_from_mask(self, mask):
        n_active = jnp.maximum(jnp.sum(jnp.asarray(mask, dtype=jnp.int32)), 1)
        depth = jnp.ceil(jnp.log2(n_active.astype(jnp.float32))).astype(jnp.int32)
        return _tree_clock_root_center_from_depth(depth, jnp.float32)

    def _route_position_embedding(self, pos, dtype, *, mask=None):
        pos_f = jnp.asarray(pos, dtype=jnp.float32)
        if mask is not None:
            pos_f = pos_f - self._clock_root_center_from_mask(mask)
        pos_f = pos_f / jnp.asarray(
            self.rope_scaling,
            dtype=jnp.float32,
        )
        half = (self.d_model + 1) // 2
        band = jnp.arange(half, dtype=jnp.float32)
        inv_freq = jnp.exp(
            -jnp.log(jnp.asarray(self.rope_base, dtype=jnp.float32))
            * band
            / jnp.asarray(max(half, 1), dtype=jnp.float32)
        )
        angle = pos_f[..., None] * inv_freq
        emb = jnp.concatenate([jnp.sin(angle), jnp.cos(angle)], axis=-1)
        return emb[..., : self.d_model].astype(dtype)

    def _first_active_index(self, mask):
        return jnp.argmax(mask.astype(jnp.int32)).astype(jnp.int32)

    def _compose_candidates(
        self,
        node_state,
        global_state: Float[Array, "d_global"],
        prefix_summary: Float[Array, "... n d_model"],
        prefix_order_summary: Float[Array, "... n d_model"],
        suffix_summary: Float[Array, "... n d_model"],
        route_pos,
        virt_pref_order_summary: Float[Array, "... n d_model"],
        virt_ratios: Float[Array, "... n 3"],
        clock_mask=None,
        candidate_mask=None,
    ) -> Float[Array, "... n d_model"]:
        node_input, node_projected = node_state
        g_raw, g_dm = global_state
        candidate_structural_mask = (
            jnp.ones(prefix_summary.shape[:-1], dtype=bool)
            if candidate_mask is None
            else jnp.broadcast_to(
                jnp.asarray(candidate_mask, dtype=bool),
                prefix_summary.shape[:-1],
            )
        )
        kfac_kwargs = dict(
            kfac_structural_mask=candidate_structural_mask,
            kfac_repeat_ndim=candidate_structural_mask.ndim,
        )
        from hamiltonzero.model.tree import _tagged_dense_no_bias

        g_raw = _tagged_dense_no_bias(
            self.cand_g_tap_w,
            g_raw,
            tag_id="route.candidate.gtap",
            pathway="even",
            kfac_structural_mask=jnp.any(candidate_structural_mask),
            kfac_repeat_ndim=0,
        )
        if prefix_summary.ndim == node_projected.ndim:
            nodes = node_projected
            node_inputs = node_input
            global_nodes = jnp.broadcast_to(g_dm[None, :], nodes.shape)
            graw_nodes = jnp.broadcast_to(
                g_raw[None, :], nodes.shape[:-1] + (g_raw.shape[-1],)
            )
        else:
            nodes = jnp.broadcast_to(
                node_projected,
                prefix_summary.shape[:-1] + (self.d_model,),
            )
            node_inputs = jnp.broadcast_to(
                node_input,
                prefix_summary.shape[:-1] + (self.d_in,),
            )
            global_nodes = jnp.broadcast_to(
                g_dm,
                prefix_summary.shape[:-1] + (self.d_model,),
            )
            graw_nodes = jnp.broadcast_to(
                g_raw,
                prefix_summary.shape[:-1] + (g_raw.shape[-1],),
            )
        pos_nodes = self._route_position_embedding(
            route_pos,
            prefix_summary.dtype,
            mask=clock_mask,
        )
        while pos_nodes.ndim < global_nodes.ndim:
            pos_nodes = pos_nodes[..., None, :]
        global_nodes = global_nodes + jnp.broadcast_to(pos_nodes, global_nodes.shape)
        node_in = self._ln(
            self.cand_node_ln_scale,
            node_inputs,
            tag_id="route.candidate.node_ln",
            **kfac_kwargs,
        )
        global_in = self._ln(
            self.cand_global_ln_scale,
            global_nodes,
            tag_id="route.candidate.global_ln",
            **kfac_kwargs,
        )
        graw_in = self._ln(
            self.cand_graw_ln_scale,
            graw_nodes,
            tag_id="route.candidate.graw_ln",
            **kfac_kwargs,
        )
        pref_in = self._ln(
            self.cand_pref_ln_scale,
            prefix_summary,
            tag_id="route.candidate.pref_ln",
            **kfac_kwargs,
        )
        pref_order_in = self._ln(
            self.cand_pref_order_ln_scale,
            prefix_order_summary,
            tag_id="route.candidate.pref_order_ln",
            **kfac_kwargs,
        )
        suff_in = self._ln(
            self.cand_suff_ln_scale,
            suffix_summary,
            tag_id="route.candidate.suff_ln",
            **kfac_kwargs,
        )

        _vp = jnp.broadcast_to(virt_pref_order_summary, prefix_summary.shape)
        virt_pref_in = self._ln(
            self.cand_virt_pref_ln_scale,
            _vp,
            tag_id="route.candidate.virt_pref_ln",
            **kfac_kwargs,
        )
        virt_ratios_in = jnp.broadcast_to(
            virt_ratios,
            prefix_summary.shape[:-1] + (3,),
        ).astype(prefix_summary.dtype)

        x = self._dense(
            self.cand_node_w,
            self.cand_b_in,
            node_in,
            tag_id="route.candidate.compose.node",
            **kfac_kwargs,
        )
        x = x + self._dense_no_bias(
            self.cand_global_w,
            global_in,
            tag_id="route.candidate.compose.global",
            **kfac_kwargs,
        )
        x = x + self._dense_no_bias(
            self.cand_graw_w,
            graw_in,
            tag_id="route.candidate.compose.graw",
            **kfac_kwargs,
        )
        x = x + self._dense_no_bias(
            self.cand_pref_w,
            pref_in,
            tag_id="route.candidate.compose.pref",
            **kfac_kwargs,
        )
        x = x + self._dense_no_bias(
            self.cand_pref_order_w,
            pref_order_in,
            tag_id="route.candidate.compose.pref_order",
            **kfac_kwargs,
        )
        x = x + self._dense_no_bias(
            self.cand_suff_w,
            suff_in,
            tag_id="route.candidate.compose.suff",
            **kfac_kwargs,
        )
        x = x + self._dense_no_bias(
            self.cand_virt_pref_w,
            virt_pref_in,
            tag_id="route.candidate.compose.virt_pref",
            **kfac_kwargs,
        )
        x = x + self._dense_no_bias(
            self.cand_virt_ratios_w,
            virt_ratios_in,
            tag_id="route.candidate.compose.virt_ratios",
            **kfac_kwargs,
        )

        def block_body(x, params):
            ln_s, w1, b1, w2, b2 = params
            y = self._ln(
                ln_s,
                x,
                tag_id="route.candidate.block.ln",
                **kfac_kwargs,
            )
            y = self._dense(
                w1,
                b1,
                y,
                tag_id="route.candidate.block.ffn1",
                **kfac_kwargs,
            )
            y = fused_silu(y)
            y = self._dense(
                w2,
                b2,
                y,
                tag_id="route.candidate.block.ffn2",
                **kfac_kwargs,
            )
            return x + y, None

        x, _ = jax.lax.scan(
            block_body,
            x,
            (
                self.cand_block_ln_scale,
                self.cand_block_w1,
                self.cand_block_b1,
                self.cand_block_w2,
                self.cand_block_b2,
            ),
        )
        x = self._ln(
            self.cand_out_ln_scale,
            x,
            tag_id="route.candidate.compose_out_ln",
            **kfac_kwargs,
        )
        delta = self._dense(
            self.cand_w_out,
            self.cand_b_out,
            x,
            tag_id="route.candidate.compose_out",
            **kfac_kwargs,
        )
        return nodes + delta

    def _teacher_candidate_states(
        self,
        node_state,
        global_state: Float[Array, "d_global"],
        edge: Float[Array, "n n d_edge"],
        perm: Int[Array, "n"],
        mask: Int[Array, "n"] | Array,
        real_mask: Int[Array, "n"] | Array,
    ) -> Float[Array, "n n d_model"]:
        _node_input, node_projected = node_state
        n = node_projected.shape[0]
        dtype = node_projected.dtype
        idx = jnp.arange(n, dtype=jnp.int32)
        mask_bool = mask.astype(bool)

        rm_bool = real_mask.astype(bool)
        virt_slot = mask_bool & (~rm_bool)

        virt_at_pos = virt_slot[perm].astype(dtype)
        pref_msg, suff_msg = self._ordered_edge_messages(edge, perm, mask)
        row_active = mask_bool.astype(dtype).reshape(n, 1, 1)
        pref_msg = pref_msg * row_active
        suff_msg = suff_msg * row_active

        pref_before = jnp.cumsum(pref_msg, axis=0) - pref_msg

        from .readout_leaf_context import lca_gaussian_decay, register_vector_as_dense

        _odw = register_vector_as_dense(
            self.order_decay_w, tag_id="route.order_decay_w"
        )[0]
        _odb = register_vector_as_dense(
            self.order_decay_b, tag_id="route.order_decay_b"
        )[0]
        _tri = (idx[:, None] > idx[None, :]).astype(dtype)
        _odecay = lca_gaussian_decay(idx, idx, _odw, _odb)
        pref_order_before = jnp.einsum("ts,tsd,sid->tid", _tri, _odecay, pref_msg)
        suff_before = jnp.cumsum(suff_msg, axis=0) - suff_msg
        suff_including_self = jnp.sum(suff_msg, axis=0)[None, :, :] - suff_before

        pos_of_node = jnp.zeros((n,), dtype=jnp.int32).at[perm].set(idx)
        self_msg = suff_msg[pos_of_node, idx, :]
        remaining = mask_bool[None, :] & (pos_of_node[None, :] >= idx[:, None])
        candidate_structural_mask = mask_bool[:, None] & remaining
        suff_other = suff_including_self - jnp.where(
            remaining[:, :, None],
            self_msg[None, :, :],
            0.0,
        )

        virt_emb = register_vector_as_dense(
            self.virt_emb,
            tag_id="route.virt_emb",
        )[0]
        virt_msg = virt_at_pos[:, None] * virt_emb[None, :]

        _vdw = register_vector_as_dense(self.virt_decay_w, tag_id="route.virt_decay_w")[
            0
        ]
        _vdb = register_vector_as_dense(self.virt_decay_b, tag_id="route.virt_decay_b")[
            0
        ]
        _vdecay = lca_gaussian_decay(idx, idx, _vdw, _vdb)
        virt_pref_order_before = jnp.einsum(
            "ts,tsd,sd->td",
            _tri,
            _vdecay,
            virt_msg,
        )
        virt_cnt_prefix = jnp.cumsum(virt_at_pos) - virt_at_pos
        total_empty = jnp.sum(virt_at_pos)
        total_leafs = jnp.sum(mask_bool.astype(dtype))
        virt_cnt_suffix = total_empty - virt_cnt_prefix
        virt_norm = jnp.sqrt(jnp.maximum(virt_cnt_prefix, 1.0))[:, None]
        virt_ratios = jnp.stack(
            [
                virt_cnt_suffix / jnp.maximum(total_empty, 1.0),
                virt_cnt_suffix / jnp.maximum(total_leafs, 1.0),
                jnp.log((virt_cnt_prefix + 1.0) / (virt_cnt_suffix + 1.0)),
            ],
            axis=-1,
        )
        virt_pref_order_summary = (virt_pref_order_before / virt_norm)[:, None, :]
        virt_ratios_summary = virt_ratios[:, None, :]

        pref_den = jnp.sqrt(jnp.maximum(idx, 1).astype(dtype)).reshape(n, 1, 1)

        n_active = jnp.sum(mask_bool.astype(jnp.int32))
        suff_den = jnp.sqrt(jnp.maximum(n_active - idx - 1, 1).astype(dtype)).reshape(
            n, 1, 1
        )
        return self._compose_candidates(
            node_state,
            global_state,
            pref_before / pref_den,
            pref_order_before / pref_den,
            suff_other / suff_den,
            idx,
            virt_pref_order_summary=virt_pref_order_summary,
            virt_ratios=virt_ratios_summary,
            clock_mask=mask,
            candidate_mask=candidate_structural_mask,
        )

    def _initial_summaries_and_edge_messages(
        self,
        edge: Float[Array, "n n d_edge"],
        mask: Int[Array, "n"] | Array,
        dtype,
    ):

        n = edge.shape[0]
        idx = jnp.arange(n, dtype=jnp.int32)
        pref_msg, suff_msg = self._ordered_edge_messages(edge, idx, mask)
        source_active = mask.astype(dtype).reshape(n, 1, 1)
        not_self = (idx[:, None] != idx[None, :]).astype(dtype).reshape(n, n, 1)
        suffix_raw = jnp.sum(suff_msg * source_active * not_self, axis=0)
        zeros = jnp.zeros_like(suffix_raw)
        virt_prefix_order0 = jnp.zeros((self.d_model,), dtype=suffix_raw.dtype)
        virt_count0 = jnp.zeros((), dtype=suffix_raw.dtype)
        summaries = (
            zeros,
            zeros,
            suffix_raw,
            virt_prefix_order0,
            virt_count0,
        )
        return summaries, (pref_msg, suff_msg)

    def _initial_summaries(
        self,
        edge: Float[Array, "n n d_edge"],
        mask: Int[Array, "n"] | Array,
        dtype,
    ):

        summaries, _edge_messages = self._initial_summaries_and_edge_messages(
            edge, mask, dtype
        )
        return summaries

    def _initial_summaries_streamed(
        self,
        edge: Float[Array, "n n d_edge"],
        edge_transpose: Float[Array, "n n d_edge"],
        mask: Int[Array, "n"] | Array,
        dtype,
        *,
        pair_tile_size: int | None = None,
        sequence_axis_name: str | None = None,
        sequence_mesh=None,
    ):

        n = edge.shape[0]
        idx = jnp.arange(n, dtype=jnp.int32)

        def _seq_constraint(value, *axes):
            if sequence_axis_name is None:
                return value
            from jax.sharding import NamedSharding, PartitionSpec as P

            spec = P(*axes)
            if sequence_mesh is not None:
                spec = NamedSharding(sequence_mesh, spec)
            return jax.lax.with_sharding_constraint(value, spec)

        tile = n if pair_tile_size is None else min(int(pair_tile_size), n)
        if tile < 1:
            raise ValueError("pair_tile_size must be positive")
        n_tiles = (n + tile - 1) // tile
        padded_n = n_tiles * tile
        source_pad = padded_n - n
        edge_padded = jnp.pad(edge, ((0, 0), (0, source_pad), (0, 0)))
        edge_transpose_padded = jnp.pad(
            edge_transpose,
            ((0, 0), (0, source_pad), (0, 0)),
        )
        source_mask = jnp.pad(mask.astype(bool), ((0, source_pad),))
        candidate_mask = mask.astype(bool)[:, None]
        candidate_ids = idx[:, None]
        suffix0 = _seq_constraint(
            jnp.zeros((n, self.d_model), dtype=dtype),
            sequence_axis_name,
            None,
        )

        def add_source_tile(tile_index, suffix_sum):
            start = tile_index * tile
            edge_tile = jax.lax.dynamic_slice_in_dim(
                edge_padded,
                start,
                tile,
                axis=1,
            )
            edge_transpose_tile = jax.lax.dynamic_slice_in_dim(
                edge_transpose_padded,
                start,
                tile,
                axis=1,
            )
            edge_pair = jnp.concatenate(
                [edge_tile, edge_transpose_tile],
                axis=-1,
            )
            source_mask_tile = jax.lax.dynamic_slice_in_dim(
                source_mask,
                start,
                tile,
                axis=0,
            )
            source_ids = start + jnp.arange(tile, dtype=jnp.int32)
            pair_mask = candidate_mask & source_mask_tile[None, :]
            suff_by_candidate = self._message_mlp(
                edge_pair,
                prefix=False,
                structural_mask=pair_mask,
            )
            source_weight = source_mask_tile.astype(dtype)[None, :, None]
            not_self = (candidate_ids != source_ids[None, :]).astype(dtype)
            suffix_sum = suffix_sum + jnp.sum(
                suff_by_candidate * source_weight * not_self[..., None],
                axis=1,
            )
            return _seq_constraint(
                suffix_sum,
                sequence_axis_name,
                None,
            )

        suffix_raw = jax.lax.fori_loop(0, n_tiles, add_source_tile, suffix0)
        zeros = jnp.zeros_like(suffix_raw)
        return (
            zeros,
            zeros,
            suffix_raw,
            jnp.zeros((self.d_model,), dtype=suffix_raw.dtype),
            jnp.zeros((), dtype=suffix_raw.dtype),
        )

    def _candidate_states_from_summaries(
        self,
        node_state,
        global_state: Float[Array, "d_global"],
        prefix_raw: Float[Array, "n d_model"],
        prefix_order_raw: Float[Array, "n d_model"],
        suffix_raw: Float[Array, "n d_model"],
        route_pos: Int[Array, ""] | Float[Array, ""],
        edge: Float[Array, "n n d_edge"],
        mask: Int[Array, "n"] | Array,
        prefix_ids: Int[Array, "n"],
        virt_prefix_order_raw: Float[Array, "d_model"],
        virt_count: Float[Array, ""],
        real_mask: Int[Array, "n"] | Array,
    ) -> Float[Array, "n d_model"]:
        _node_input, node_projected = node_state
        dtype = node_projected.dtype
        route_pos_i = jnp.asarray(route_pos, dtype=jnp.int32)
        mask_bool = mask.astype(bool)
        pref_den = jnp.sqrt(jnp.maximum(route_pos_i, 1).astype(dtype))
        n_active = jnp.sum(mask_bool.astype(jnp.int32))
        suff_den = jnp.sqrt(jnp.maximum(n_active - route_pos_i - 1, 1).astype(dtype))
        _cand_idx = jnp.arange(node_projected.shape[0], dtype=jnp.int32)
        _placed_pos = jnp.arange(node_projected.shape[0], dtype=jnp.int32)
        _already_picked = jnp.any(
            (_placed_pos < route_pos_i)[:, None]
            & (prefix_ids[:, None] == _cand_idx[None, :]),
            axis=0,
        )
        candidate_structural_mask = mask_bool & ~_already_picked

        _vnorm = jnp.sqrt(jnp.maximum(virt_count, 1.0))
        _vp = virt_prefix_order_raw / _vnorm
        virt_slot = mask_bool & (~real_mask.astype(bool))
        total_empty = jnp.sum(virt_slot.astype(dtype))
        total_leafs = jnp.sum(mask.astype(dtype))
        virt_cnt_suffix = total_empty - virt_count
        _vr = jnp.stack(
            [
                virt_cnt_suffix / jnp.maximum(total_empty, 1.0),
                virt_cnt_suffix / jnp.maximum(total_leafs, 1.0),
                jnp.log((virt_count + 1.0) / (virt_cnt_suffix + 1.0)),
            ],
        )
        return self._compose_candidates(
            node_state,
            global_state,
            prefix_raw / pref_den,
            prefix_order_raw / pref_den,
            suffix_raw / suff_den,
            route_pos_i,
            virt_pref_order_summary=_vp,
            virt_ratios=_vr,
            clock_mask=mask,
            candidate_mask=candidate_structural_mask,
        )

    def _pointer_raw(self, hidden, candidate_state, structural_mask=None):
        candidate_structural_mask = (
            jnp.ones(candidate_state.shape[:-1], dtype=bool)
            if structural_mask is None
            else jnp.broadcast_to(
                jnp.asarray(structural_mask, dtype=bool),
                candidate_state.shape[:-1],
            )
        )
        if hidden.ndim == 1:
            query_structural_mask = jnp.any(candidate_structural_mask)
            q = self._dense_no_bias(
                self.pointer_q_w,
                hidden,
                tag_id="route.pointer.q",
                kfac_structural_mask=query_structural_mask,
                kfac_repeat_ndim=0,
            )
            k = self._dense_no_bias(
                self.pointer_k_w,
                candidate_state,
                tag_id="route.pointer.k",
                kfac_structural_mask=candidate_structural_mask,
                kfac_repeat_ndim=1,
            )
            raw = jnp.einsum("d,nd->n", q, k)
        else:
            query_structural_mask = jnp.any(candidate_structural_mask, axis=-1)
            q = self._dense_no_bias(
                self.pointer_q_w,
                hidden,
                tag_id="route.pointer.q",
                kfac_structural_mask=query_structural_mask,
                kfac_repeat_ndim=1,
            )
            k = self._dense_no_bias(
                self.pointer_k_w,
                candidate_state,
                tag_id="route.pointer.k",
                kfac_structural_mask=candidate_structural_mask,
                kfac_repeat_ndim=2,
            )
            raw = jnp.einsum("td,tnd->tn", q, k)
        scale = jax.lax.rsqrt(jnp.asarray(self.pointer_score_dim, dtype=raw.dtype))
        return raw * scale

    def _pointer_logits(self, hidden, candidate_state, picked, mask, tau):
        active = mask.astype(bool) & (~picked)
        raw = self._pointer_raw(hidden, candidate_state, structural_mask=active)
        neg = jnp.asarray(-1.0e30, dtype=raw.dtype)
        return jnp.where(active, raw / jnp.asarray(tau, dtype=raw.dtype), neg)

    def _learned_first_choice_mask(self, mask, real_mask):
        mask_bool = mask.astype(bool)
        if real_mask is None:
            return mask_bool
        real_bool = real_mask.astype(bool) & mask_bool
        return jnp.where(jnp.any(real_bool), real_bool, mask_bool)

    def _step_choice_mask(self, first_step, mask, real_mask):
        mask_bool = mask.astype(bool)
        first_mask = self._learned_first_choice_mask(mask, real_mask)
        return jnp.where(first_step, first_mask, mask_bool)

    def _step_pointer_hidden(self, first_step, global_state, hidden):
        first_hidden = jnp.broadcast_to(global_state[1], hidden.shape)
        return jnp.where(first_step, first_hidden, hidden)

    def _teacher_hidden_with_first(self, hidden, global_state, first_active_idx):
        return hidden.at[first_active_idx].set(global_state[1])

    def _score_step_for_logp(self, first_step, predict_step):
        return predict_step | first_step

    def _logprob_contribute_mask(self, mask_bool, idx, first_active):
        del idx, first_active
        return mask_bool

    def _collapse_quotient_logits(self, logits, ids, valid):
        n = logits.shape[-1]
        dtype = logits.dtype
        idx = jnp.arange(n, dtype=jnp.int32)
        ids = jnp.asarray(ids, dtype=jnp.int32)
        valid = valid.astype(bool) & (ids >= 0)
        same = (ids[:, None] == ids[None, :]) & valid[:, None] & valid[None, :]
        rep_idx = jnp.min(
            jnp.where(same, idx[None, :], jnp.asarray(n, dtype=jnp.int32)),
            axis=1,
        )
        reps = valid & (idx == rep_idx)
        neg = jnp.asarray(-1.0e30, dtype=dtype)
        member_logits = jnp.where(same, logits[None, :], neg)
        max_l = jnp.max(member_logits, axis=1)
        max_l = jnp.where(jnp.isfinite(max_l), max_l, jnp.asarray(0.0, dtype=dtype))
        class_lse = max_l + jnp.log(
            jnp.sum(jnp.exp(member_logits - max_l[:, None]), axis=1)
        )
        class_size = jnp.maximum(
            jnp.sum(same.astype(dtype), axis=1),
            jnp.asarray(1.0, dtype=dtype),
        )
        quotient_logits = class_lse - jnp.log(class_size)
        return jnp.where(reps, quotient_logits, neg)

    def _apply_quotient_logits(
        self,
        logits,
        first_orbit_ids,
        valid_mask,
        context_mask,
        prefix_ids,
        prefix_len,
    ):
        if len(first_orbit_ids) == 2:
            node_key, edge_key = first_orbit_ids
            ids = conditional_orbit_ids_from_keys(
                node_key,
                edge_key,
                valid_mask,
                context_mask,
                prefix_ids,
                prefix_len,
            )
        elif len(first_orbit_ids) == 3:
            node_key, edge_key, needs_fwl2 = first_orbit_ids
            inputs = (
                node_key,
                edge_key,
                valid_mask,
                context_mask,
                prefix_ids,
                prefix_len,
            )
            ids = jax.lax.cond(
                jnp.asarray(needs_fwl2, dtype=jnp.bool_),
                lambda values: conditional_orbit_pair_ids_from_keys(*values),
                lambda values: conditional_orbit_ids_from_keys(*values),
                inputs,
            )
        else:
            raise ValueError(
                "quotient carrier must contain node key, edge key, and "
                "optionally needs_fwl2"
            )
        return self._collapse_quotient_logits(logits, ids, valid_mask)

    def _stopgrad_logit_scale(self, logits, valid_mask):
        del valid_mask
        return logits

    def _append_token(
        self,
        token: Float[Array, "d_model"],
        chosen: Int[Array, ""],
        t: Int[Array, ""],
        prefix_ids: Int[Array, "n"],
        k_cache: Float[Array, "l n h d_head"],
        v_cache: Float[Array, "l n h d_head"],
        edge: Float[Array, "n n d_edge"],
        mask: Int[Array, "n"] | Array,
    ):
        del chosen, t, prefix_ids, edge, mask
        return token, k_cache, v_cache

    def logprob_perm(
        self,
        h: Float[Array, "n d_in"],
        edge: Float[Array, "n n d_edge"],
        perm: Int[Array, "n"],
        mask: Int[Array, "n"] | Array,
        *,
        global_feat: Float[Array, "d_global"] | None = None,
        tau: float | Float[Array, ""] = 1.0,
        real_mask: Int[Array, "n"] | Array | None = None,
        first_orbit_ids: QuotientCarrier,
    ) -> Float[Array, ""]:
        scores = self._teacher_logits(
            h,
            edge,
            perm,
            mask,
            global_feat=global_feat,
            tau=tau,
            real_mask=real_mask,
            first_orbit_ids=first_orbit_ids,
        )
        n = h.shape[0]
        if n <= 1:
            return jnp.asarray(0.0, dtype=h.dtype)
        mask_bool = mask.astype(bool)
        first_active = self._first_active_index(mask)
        idx = jnp.arange(n, dtype=jnp.int32)
        contribute = self._logprob_contribute_mask(mask_bool, idx, first_active)
        neg = jnp.asarray(-1.0e30, dtype=scores.dtype)
        scores = self._stopgrad_logit_scale(
            scores,
            scores > (neg * jnp.asarray(0.5, dtype=scores.dtype)),
        )
        log_probs = jax.nn.log_softmax(scores.astype(jnp.float32), axis=-1)
        chosen = jnp.take_along_axis(log_probs, perm[:, None], axis=-1)[:, 0]
        return jnp.sum(jnp.where(contribute, chosen, 0.0))

    def logprob_identity(
        self,
        h: Float[Array, "n d_in"],
        edge: Float[Array, "n n d_edge"],
        mask: Int[Array, "n"] | Array,
        *,
        global_feat: Float[Array, "d_global"] | None = None,
        tau: float | Float[Array, ""] = 1.0,
        real_mask: Int[Array, "n"] | Array | None = None,
        first_orbit_ids: QuotientCarrier,
    ) -> Float[Array, ""]:
        n = h.shape[0]
        return self.logprob_perm(
            h,
            edge,
            jnp.arange(n, dtype=jnp.int32),
            mask,
            global_feat=global_feat,
            tau=tau,
            real_mask=real_mask,
            first_orbit_ids=first_orbit_ids,
        )


class _TreePrefixMerge(eqx.Module):
    ln_scale: Float[Array, "d_in"]

    g_proj_w: Float[Array, "d_gstream d_gsec"]
    w1: Float[Array, "d_in d_hidden"]
    b1: Float[Array, "d_hidden"]
    w2: Float[Array, "d_hidden d_model"]
    b2: Float[Array, "d_model"]

    alpha_route: Float[Array, "d_model"]

    d_model: int = eqx.field(static=True)
    d_hidden: int = eqx.field(static=True)
    d_in: int = eqx.field(static=True)
    max_depth: int = eqx.field(static=True)
    ngpt_alpha_max: float = eqx.field(static=True)
    ln_eps: float = eqx.field(static=True)

    def __init__(
        self,
        d_model: int,
        *,
        hidden: int,
        max_depth: int,
        key: PRNGKeyArray,
        gladder_d_g: int,
        alpha_init: float,
        alpha_max: float,
        ln_eps: float = 1e-5,
    ):
        d_hidden = int(hidden)

        d_in = 5 * int(d_model) + 64 + _TREE_NGPT_DEPTH_FEAT_DIM
        k1, k2 = jax.random.split(key, 2)
        self.ln_scale = jnp.ones((d_in,))
        self.w1 = jax.random.normal(k1, (d_in, d_hidden)) * (d_in**-0.5)
        self.b1 = jnp.zeros((d_hidden,))
        self.w2 = jax.random.normal(k2, (d_hidden, d_model)) * (d_hidden**-0.5)
        self.b2 = jnp.zeros((d_model,))
        kg = jax.random.fold_in(k2, 0x61B5)
        self.g_proj_w = jax.random.normal(kg, (int(gladder_d_g), 64)) * (
            int(gladder_d_g) ** -0.5
        )
        self.alpha_route = float(alpha_init) * jnp.ones((int(d_model),))
        self.d_model = int(d_model)
        self.d_hidden = int(d_hidden)
        self.d_in = int(d_in)
        self.max_depth = int(max_depth)
        self.ngpt_alpha_max = float(alpha_max)
        self.ln_eps = float(ln_eps)

    def project_global(self, g, structural_mask):

        from hamiltonzero.model.tree import _tagged_dense_no_bias

        return _tagged_dense_no_bias(
            self.g_proj_w,
            g,
            tag_id="gladder.route.merge_gproj",
            pathway="even",
            kfac_structural_mask=jnp.asarray(structural_mask, dtype=bool),
            kfac_scan_shared=False,
            kfac_repeat_ndim=0,
            kfac_context_primal_reused_over_walkers=True,
        )

    def __call__(
        self,
        left,
        right,
        left_mask,
        right_mask,
        sibling_edge_lr,
        sibling_edge_rl,
        level_idx,
        pair_idx=None,
        pair_base=None,
        clock_depth=None,
        depth_feats=None,
        g=None,
        g_structural_mask=None,
        g_projected=None,
    ):
        from hamiltonzero.model.tree import _tagged_dense, _tagged_rms_eqx_style
        from hamiltonzero.model.tree import _rownorm_cols

        _we = _rownorm_cols

        dtype = left.dtype
        left_mask = left_mask.astype(dtype)
        right_mask = right_mask.astype(dtype)
        out_mask = left_mask + right_mask - left_mask * right_mask
        both = left_mask * right_mask
        merge_structural_mask = both.astype(bool)
        depth_i = (
            jnp.asarray(max(self.max_depth, 1), dtype=jnp.int32)
            if clock_depth is None
            else jnp.maximum(jnp.asarray(clock_depth, dtype=jnp.int32), 1)
        )
        parts = [left, right, sibling_edge_lr, sibling_edge_rl]
        g_active = (
            jnp.any(merge_structural_mask)
            if g_structural_mask is None
            else jnp.asarray(g_structural_mask, dtype=bool)
        )
        gg = (
            g_projected if g_projected is not None else self.project_global(g, g_active)
        )
        parts.append(
            jnp.broadcast_to(gg[None, :], left.shape[:-1] + (gg.shape[-1],)).astype(
                dtype
            )
        )
        if depth_feats is None:
            raise ValueError("tree prefix merge requires depth features")
        parts.append(depth_feats.astype(dtype))
        clock = _route_merge_clock(
            level_idx,
            pair_idx,
            pair_base,
            self.d_model,
            depth_i,
            dtype,
            root_centered=True,
        )
        parts.append(jnp.broadcast_to(clock, left.shape))
        x = jnp.concatenate(parts, axis=-1)
        x = _tagged_rms_eqx_style(
            self.ln_scale,
            x,
            eps=self.ln_eps,
            tag_id="route.tree_prefix.merge.ln",
            pathway="even",
            kfac_structural_mask=merge_structural_mask,
            kfac_scan_shared=True,
            kfac_repeat_ndim=1,
        )
        h = _tagged_dense(
            self.w1,
            self.b1,
            x,
            tag_id="route.tree_prefix.merge.ffn1",
            pathway="even",
            weight_eff=_we(self.w1),
            kfac_structural_mask=merge_structural_mask,
            kfac_scan_shared=True,
            kfac_repeat_ndim=1,
        )
        h = fused_silu(h)
        delta = _tagged_dense(
            self.w2,
            self.b2,
            h,
            tag_id="route.tree_prefix.merge.ffn2",
            pathway="even",
            weight_eff=_we(self.w2),
            kfac_structural_mask=merge_structural_mask,
            kfac_scan_shared=True,
            kfac_repeat_ndim=1,
        )
        raw = _tree_ngpt_residual(
            0.5 * (left + right),
            delta,
            self.alpha_route,
            max_gain=self.ngpt_alpha_max,
            tag_id="route.tree_prefix.merge.alpha_route",
            pathway="even",
            kfac_structural_mask=merge_structural_mask,
            kfac_scan_shared=True,
            kfac_repeat_ndim=1,
        )
        carry = jnp.where(left_mask[:, None] > 0, left, right)
        out = jnp.where(both[:, None] > 0, raw, carry)
        out = jnp.where(
            out_mask[:, None] > 0,
            out,
            jnp.zeros_like(out),
        )
        return out, out_mask, both


class _TreePrefixSelfLayer(eqx.Module):
    ln_scale: Float[Array, "d_model"]
    w_qkv: Float[Array, "d_model three_qv"]
    w_o: Float[Array, "d_o_in d_model"]
    edge_ln_scale: Float[Array, "d_model"]
    edge_w1: Float[Array, "d_model d_hidden"]
    edge_b1: Float[Array, "d_hidden"]
    edge_w2: Float[Array, "d_hidden h_kernel"]
    edge_b2: Float[Array, "h_kernel"]
    ffn_ln_scale: Float[Array, "d_model"]
    ffn_w1: Float[Array, "d_model d_ffn"]
    ffn_b1: Float[Array, "d_ffn"]
    ffn_w2: Float[Array, "d_ffn d_model"]
    ffn_b2: Float[Array, "d_model"]

    def as_tuple(self):
        return (
            self.ln_scale,
            self.w_qkv,
            self.w_o,
            self.edge_ln_scale,
            self.edge_w1,
            self.edge_b1,
            self.edge_w2,
            self.edge_b2,
            self.ffn_ln_scale,
            self.ffn_w1,
            self.ffn_b1,
            self.ffn_w2,
            self.ffn_b2,
        )


class _TreePrefixCandidateLayer(eqx.Module):
    cand_ln_scale: Float[Array, "d_model"]
    prefix_ln_scale: Float[Array, "d_model"]
    cand_w_qv: Float[Array, "d_model two_qv"]
    prefix_w_kv: Float[Array, "d_model two_qv"]
    w_o: Float[Array, "d_o_in d_model"]
    edge_ln_scale: Float[Array, "d_model"]
    edge_w1: Float[Array, "d_model d_hidden"]
    edge_b1: Float[Array, "d_hidden"]
    edge_w2: Float[Array, "d_hidden h_kernel"]
    edge_b2: Float[Array, "h_kernel"]
    ffn_ln_scale: Float[Array, "d_model"]
    ffn_w1: Float[Array, "d_model d_ffn"]
    ffn_b1: Float[Array, "d_ffn"]
    ffn_w2: Float[Array, "d_ffn d_model"]
    ffn_b2: Float[Array, "d_model"]

    def as_tuple(self):
        return (
            self.cand_ln_scale,
            self.prefix_ln_scale,
            self.cand_w_qv,
            self.prefix_w_kv,
            self.w_o,
            self.edge_ln_scale,
            self.edge_w1,
            self.edge_b1,
            self.edge_w2,
            self.edge_b2,
            self.ffn_ln_scale,
            self.ffn_w1,
            self.ffn_b1,
            self.ffn_w2,
            self.ffn_b2,
        )


class _HeavyRouteLayer(eqx.Module):
    cross_ln_scale: Float[Array, "d_model"]
    cross_ln_shift: Float[Array, "d_model"]
    cross_prefix_ln_scale: Float[Array, "d_model"]
    cross_prefix_ln_shift: Float[Array, "d_model"]
    cross_w_qv: Float[Array, "d_model two_qv"]
    cross_w_kv: Float[Array, "d_model two_qv"]
    cross_w_o: Float[Array, "d_o_in d_model"]
    cross_edge_ln_scale: Float[Array, "two_d_edge"]
    cross_edge_ln_shift: Float[Array, "two_d_edge"]
    cross_edge_w1: Float[Array, "two_d_edge d_heavy_edge_hidden"]
    cross_edge_b1: Float[Array, "d_heavy_edge_hidden"]
    cross_edge_w2: Float[Array, "d_heavy_edge_hidden h_kernel"]
    cross_edge_b2: Float[Array, "h_kernel"]

    self_ln_scale: Float[Array, "d_model"]
    self_w_qkv: Float[Array, "d_model three_qv"]
    self_w_o: Float[Array, "d_o_in d_model"]
    self_edge_ln_scale: Float[Array, "two_d_edge"]
    self_edge_w1: Float[Array, "two_d_edge d_heavy_edge_hidden"]
    self_edge_b1: Float[Array, "d_heavy_edge_hidden"]
    self_edge_w2: Float[Array, "d_heavy_edge_hidden h_kernel"]
    self_edge_b2: Float[Array, "h_kernel"]

    ffn_ln_scale: Float[Array, "d_model"]
    ffn_w1: Float[Array, "d_model d_ffn"]
    ffn_b1: Float[Array, "d_ffn"]
    ffn_w2: Float[Array, "d_ffn d_model"]
    ffn_b2: Float[Array, "d_model"]

    def as_tuple(self):
        return (
            self.cross_ln_scale,
            self.cross_ln_shift,
            self.cross_prefix_ln_scale,
            self.cross_prefix_ln_shift,
            self.cross_w_qv,
            self.cross_w_kv,
            self.cross_w_o,
            self.cross_edge_ln_scale,
            self.cross_edge_ln_shift,
            self.cross_edge_w1,
            self.cross_edge_b1,
            self.cross_edge_w2,
            self.cross_edge_b2,
            self.self_ln_scale,
            self.self_w_qkv,
            self.self_w_o,
            self.self_edge_ln_scale,
            self.self_edge_w1,
            self.self_edge_b1,
            self.self_edge_w2,
            self.self_edge_b2,
            self.ffn_ln_scale,
            self.ffn_w1,
            self.ffn_b1,
            self.ffn_w2,
            self.ffn_b2,
        )


class _PrefixSuffixRouteBase(_RoutePointerBase):
    heavy_layers: list[_HeavyRouteLayer]
    route_prefix_suffix_layers: int = eqx.field(static=True)
    route_decoder_attn_impl: str = eqx.field(static=True)
    heavy_edge_hidden: int = eqx.field(static=True)
    heavy_residual_gain: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        d_in: int,
        d_edge: int,
        d_global: int,
        d_model: int,
        n_heads: int,
        max_n: int,
        key: PRNGKeyArray,
        route_prefix_suffix_layers: int = 1,
        route_decoder_attn_impl: str = "mhsea_tuned",
        score_init_scale: float = 1.0,
        rope_base: float = 10000.0,
        rope_scaling: float = 1.0,
        attention_dim: int,
        pointer_score_dim: int,
        candidate_hidden: int,
        summary_hidden: int,
        ffn_hidden: int,
        global_tap_dim: int,
    ):
        if route_prefix_suffix_layers < 0:
            raise ValueError("route_prefix_suffix_layers must be >= 0")
        allowed_impls = {"mhsea_tuned", "einsum"}
        if route_decoder_attn_impl not in allowed_impls:
            raise ValueError(
                f"route_decoder_attn_impl must be one of {sorted(allowed_impls)}"
            )

        key_base, key_heavy = jax.random.split(key)
        super().__init__(
            d_in=d_in,
            d_edge=d_edge,
            d_global=d_global,
            d_model=d_model,
            n_heads=n_heads,
            max_n=max_n,
            key=key_base,
            score_init_scale=score_init_scale,
            rope_base=rope_base,
            rope_scaling=rope_scaling,
            attention_dim=attention_dim,
            pointer_score_dim=pointer_score_dim,
            candidate_hidden=candidate_hidden,
            summary_hidden=summary_hidden,
            ffn_hidden=ffn_hidden,
            global_tap_dim=global_tap_dim,
        )

        layers = int(route_prefix_suffix_layers)
        d_qv = self.n_heads_kernel * self.d_head
        d_o_in = self.n_heads * self.d_head
        pair_dim = 2 * self.d_edge
        heavy_edge_hidden = max(32, 2 * self.n_heads_kernel, 4 * self.d_edge)

        def w(k, shape, fan_in):
            return jax.random.normal(k, shape) * (fan_in**-0.5)

        layer_keys = jax.random.split(key_heavy, layers)
        heavy_layers = []
        for li in range(layers):
            keys = jax.random.split(layer_keys[li], 11)
            heavy_layers.append(
                _HeavyRouteLayer(
                    cross_ln_scale=jnp.ones((self.d_model,)),
                    cross_ln_shift=jnp.zeros((self.d_model,)),
                    cross_prefix_ln_scale=jnp.ones((self.d_model,)),
                    cross_prefix_ln_shift=jnp.zeros((self.d_model,)),
                    cross_w_qv=w(keys[0], (self.d_model, 2 * d_qv), self.d_model),
                    cross_w_kv=w(keys[1], (self.d_model, 2 * d_qv), self.d_model),
                    cross_w_o=w(keys[2], (d_o_in, self.d_model), d_o_in),
                    cross_edge_ln_scale=jnp.ones((pair_dim,)),
                    cross_edge_ln_shift=jnp.zeros((pair_dim,)),
                    cross_edge_w1=w(keys[3], (pair_dim, heavy_edge_hidden), pair_dim),
                    cross_edge_b1=jnp.zeros((heavy_edge_hidden,)),
                    cross_edge_w2=w(
                        keys[4],
                        (heavy_edge_hidden, self.n_heads_kernel),
                        heavy_edge_hidden,
                    ),
                    cross_edge_b2=jnp.zeros((self.n_heads_kernel,)),
                    self_ln_scale=jnp.ones((self.d_model,)),
                    self_w_qkv=w(keys[5], (self.d_model, 3 * d_qv), self.d_model),
                    self_w_o=w(keys[6], (d_o_in, self.d_model), d_o_in),
                    self_edge_ln_scale=jnp.ones((pair_dim,)),
                    self_edge_w1=w(keys[7], (pair_dim, heavy_edge_hidden), pair_dim),
                    self_edge_b1=jnp.zeros((heavy_edge_hidden,)),
                    self_edge_w2=w(
                        keys[8],
                        (heavy_edge_hidden, self.n_heads_kernel),
                        heavy_edge_hidden,
                    ),
                    self_edge_b2=jnp.zeros((self.n_heads_kernel,)),
                    ffn_ln_scale=jnp.ones((self.d_model,)),
                    ffn_w1=w(keys[9], (self.d_model, self.ffn_hidden), self.d_model),
                    ffn_b1=jnp.zeros((self.ffn_hidden,)),
                    ffn_w2=w(
                        keys[10], (self.ffn_hidden, self.d_model), self.ffn_hidden
                    ),
                    ffn_b2=jnp.zeros((self.d_model,)),
                )
            )
        self.heavy_layers = heavy_layers

        self.route_prefix_suffix_layers = layers
        self.route_decoder_attn_impl = str(route_decoder_attn_impl)
        self.heavy_edge_hidden = heavy_edge_hidden
        self.heavy_residual_gain = 0.0 if layers == 0 else float(layers) ** -0.5

    def _heavy_layer_params(self):
        return [layer.as_tuple() for layer in self.heavy_layers]

    def _resolve_heavy_attn_impl(self, n: int) -> str:
        del n
        return self.route_decoder_attn_impl

    def _heavy_edge_bias(
        self,
        edge_pair,
        params,
        *,
        prefix: str,
        structural_mask=None,
        scan_shared: bool = False,
        repeat_ndim: int | None = None,
        context_primal_reused_over_walkers: bool = False,
    ):
        ln_s, w1, b1, w2, b2 = params
        structural_mask = (
            jnp.ones(edge_pair.shape[:-1], dtype=bool)
            if structural_mask is None
            else jnp.broadcast_to(
                jnp.asarray(structural_mask, dtype=bool), edge_pair.shape[:-1]
            )
        )
        repeat_ndim = structural_mask.ndim if repeat_ndim is None else repeat_ndim
        kfac_kwargs = dict(
            kfac_structural_mask=structural_mask,
            kfac_scan_shared=scan_shared,
            kfac_repeat_ndim=repeat_ndim,
            kfac_context_primal_reused_over_walkers=(
                context_primal_reused_over_walkers
            ),
        )
        x = self._ln(
            ln_s,
            edge_pair,
            tag_id=f"route.heavy.{prefix}.edge_ln",
            **kfac_kwargs,
        )
        x = self._dense(
            w1,
            b1,
            x,
            tag_id=f"route.heavy.{prefix}.edge_bias1",
            **kfac_kwargs,
        )
        x = fused_silu(x)
        bias = self._dense(
            w2,
            b2,
            x,
            tag_id=f"route.heavy.{prefix}.edge_bias2",
            **kfac_kwargs,
        )
        return bias

    def _heavy_cross_edge_bias(
        self,
        edge_pair,
        params,
        *,
        structural_mask=None,
        scan_shared: bool = False,
        repeat_ndim: int | None = None,
        context_primal_reused_over_walkers: bool = False,
    ):
        ln_s, ln_b, w1, b1, w2, b2 = params
        structural_mask = (
            jnp.ones(edge_pair.shape[:-1], dtype=bool)
            if structural_mask is None
            else jnp.broadcast_to(
                jnp.asarray(structural_mask, dtype=bool), edge_pair.shape[:-1]
            )
        )
        repeat_ndim = structural_mask.ndim if repeat_ndim is None else repeat_ndim
        kfac_kwargs = dict(
            kfac_structural_mask=structural_mask,
            kfac_scan_shared=scan_shared,
            kfac_repeat_ndim=repeat_ndim,
            kfac_context_primal_reused_over_walkers=(
                context_primal_reused_over_walkers
            ),
        )
        x = self._cross_ln(
            ln_s,
            ln_b,
            edge_pair,
            tag_id="route.heavy.cross.edge_ln",
            **kfac_kwargs,
        )
        x = self._dense(
            w1,
            b1,
            x,
            tag_id="route.heavy.cross.edge_bias1",
            **kfac_kwargs,
        )
        x = fused_silu(x)
        return self._dense(
            w2,
            b2,
            x,
            tag_id="route.heavy.cross.edge_bias2",
            **kfac_kwargs,
        )

    def _route_attention(
        self,
        q: Float[Array, "b n h d_head"],
        k: Float[Array, "b n h d_head"],
        v: Float[Array, "b n h d_head"],
        edge_bias: Float[Array, "b n n h"],
        key_mask: Int[Array, "b n"] | Array,
        *,
        impl: str,
        key_mask_only: bool = False,
        attention_mask: Array | None = None,
        sequence_axis_name=None,
        sequence_mesh=None,
    ) -> Float[Array, "b n h d_head"]:
        dtype = q.dtype
        valid = key_mask.astype(bool)[:, None, :]
        if attention_mask is not None:
            valid = valid & attention_mask.astype(bool)
        has_key = jnp.any(valid, axis=-1)
        if impl == "einsum":
            q_c = q
            k_c = k
            v_c = v
            bias_c = edge_bias
            if sequence_axis_name is not None:
                from jax.sharding import NamedSharding, PartitionSpec as P

                def _sharding(*axes):
                    spec = P(*axes)
                    return (
                        NamedSharding(sequence_mesh, spec)
                        if sequence_mesh is not None
                        else spec
                    )

                q_c = jax.lax.with_sharding_constraint(
                    q_c,
                    _sharding(None, sequence_axis_name, None, None),
                )
                k_c = jax.lax.with_sharding_constraint(
                    k_c,
                    _sharding(None, None, None, None),
                )
                v_c = jax.lax.with_sharding_constraint(
                    v_c,
                    _sharding(None, None, None, None),
                )
                bias_c = jax.lax.with_sharding_constraint(
                    bias_c,
                    _sharding(None, sequence_axis_name, None, None),
                )
            logits = jnp.einsum("bihd,bjhd->bhij", q_c, k_c)
            logits = logits / jnp.sqrt(jnp.asarray(self.d_head, dtype=dtype))
            logits = logits + jnp.transpose(bias_c, (0, 3, 1, 2))
            if sequence_axis_name is not None:
                logits = jax.lax.with_sharding_constraint(
                    logits,
                    _sharding(None, None, sequence_axis_name, None),
                )
            logits = jnp.where(
                valid[:, None, :, :],
                logits,
                jnp.asarray(-1.0e30, dtype=dtype),
            )
            if sequence_axis_name is not None:
                logits = jax.lax.with_sharding_constraint(
                    logits,
                    _sharding(None, None, sequence_axis_name, None),
                )
            alpha = jax.nn.softmax(logits, axis=-1)
            if sequence_axis_name is not None:
                alpha = jax.lax.with_sharding_constraint(
                    alpha,
                    _sharding(None, None, sequence_axis_name, None),
                )
            out = jnp.einsum("bhij,bjhd->bihd", alpha, v_c)
            if sequence_axis_name is not None:
                out = jax.lax.with_sharding_constraint(
                    out,
                    _sharding(None, sequence_axis_name, None, None),
                )
        elif impl == "mhsea_tuned":
            from hamiltonzero.model.pallas_attention import mhsea_tuned_edge_attention

            if key_mask_only or attention_mask is not None:
                edge_bias = jnp.where(
                    valid[..., None],
                    edge_bias,
                    jnp.asarray(-1.0e30, dtype=edge_bias.dtype),
                )
                key_mask = jnp.ones_like(key_mask)
            d_head_padded = max(16, self.d_head)
            pad_amount = d_head_padded - self.d_head
            if pad_amount:
                scale = jnp.sqrt(jnp.asarray(d_head_padded / self.d_head, dtype=dtype))
                q = jnp.concatenate(
                    [
                        q * scale,
                        jnp.zeros(q.shape[:-1] + (pad_amount,), dtype=q.dtype),
                    ],
                    axis=-1,
                )
                k = jnp.concatenate(
                    [
                        k,
                        jnp.zeros(k.shape[:-1] + (pad_amount,), dtype=k.dtype),
                    ],
                    axis=-1,
                )
                v = jnp.concatenate(
                    [
                        v,
                        jnp.zeros(v.shape[:-1] + (pad_amount,), dtype=v.dtype),
                    ],
                    axis=-1,
                )
            out = jax.vmap(
                lambda q_b, k_b, v_b, bias_b, mask_b: mhsea_tuned_edge_attention(
                    q_b, k_b, v_b, bias_b, mask_b.astype(jnp.int32)
                )
            )(q, k, v, edge_bias, key_mask)
            out = out[..., : self.d_head]
        else:
            raise ValueError("route attention must be 'einsum' or 'mhsea_tuned'")
        return jnp.where(has_key[..., None, None], out, jnp.zeros_like(out))

    def _collapse_heavy_heads(self, out):
        gate_heads = out[..., : self.n_heads, :]
        value_heads = out[..., self.n_heads :, :]
        out = jax.nn.sigmoid(gate_heads) * value_heads
        return out.reshape(out.shape[:-2] + (self.n_heads * self.d_head,))

    def _heavy_prefix_pairs_teacher(self, edge, perm):
        n = edge.shape[0]
        edge_i_p = edge[:, perm, :]
        edge_p_i = jnp.transpose(edge[perm, :, :], (1, 0, 2))
        pair = jnp.concatenate([edge_i_p, edge_p_i], axis=-1)

        return pair

    def _heavy_suffix_pairs_teacher(self, edge):
        n = edge.shape[0]
        idx = jnp.arange(n, dtype=jnp.int32)
        edge_i_j = edge[idx[:, None], idx[None, :], :]
        edge_j_i = edge[idx[None, :], idx[:, None], :]
        pair = jnp.concatenate([edge_i_j, edge_j_i], axis=-1)

        return pair

    def _heavy_layer_teacher(
        self,
        cand: Float[Array, "n n d_model"],
        z: Float[Array, "n d_model"],
        edge: Float[Array, "n n d_edge"],
        perm: Int[Array, "n"],
        mask: Int[Array, "n"] | Array,
        pos_of_node: Int[Array, "n"],
        params,
        *,
        impl: str,
    ) -> Float[Array, "n n d_model"]:
        (
            cross_ln_s,
            cross_ln_b,
            cross_prefix_ln_s,
            cross_prefix_ln_b,
            cross_w_qv,
            cross_w_kv,
            cross_w_o,
            cross_edge_ln_s,
            cross_edge_ln_b,
            cross_edge_w1,
            cross_edge_b1,
            cross_edge_w2,
            cross_edge_b2,
            self_ln_s,
            self_w_qkv,
            self_w_o,
            self_edge_ln_s,
            self_edge_w1,
            self_edge_b1,
            self_edge_w2,
            self_edge_b2,
            ffn_ln_s,
            ffn_w1,
            ffn_b1,
            ffn_w2,
            ffn_b2,
        ) = params
        n = cand.shape[0]
        dtype = cand.dtype
        idx = jnp.arange(n, dtype=jnp.int32)
        mask_bool = mask.astype(bool)
        candidate_structural_mask = (
            mask_bool[:, None]
            & mask_bool[None, :]
            & (pos_of_node[None, :] >= idx[:, None])
        )
        prefix_structural_mask = mask_bool
        prefix_pair_structural_mask = (
            mask_bool[:, None]
            & mask_bool[None, :]
            & (idx[None, :] < pos_of_node[:, None])
        )
        suffix_pair_structural_mask = mask_bool[:, None] & mask_bool[None, :]

        context_reuse = True
        candidate_kfac = dict(
            kfac_structural_mask=candidate_structural_mask,
            kfac_repeat_ndim=2,
            kfac_context_primal_reused_over_walkers=context_reuse,
        )
        prefix_kfac = dict(
            kfac_structural_mask=prefix_structural_mask,
            kfac_repeat_ndim=1,
            kfac_context_primal_reused_over_walkers=context_reuse,
        )
        query_mask = mask.astype(dtype)[None, :, None]

        x_ln = self._cross_ln(
            cross_ln_s,
            cross_ln_b,
            cand,
            tag_id="route.heavy.cross.ln",
            **candidate_kfac,
        )
        qv = self._dense_no_bias(
            cross_w_qv,
            x_ln,
            tag_id="route.heavy.cross.qv",
            **candidate_kfac,
        ).reshape(n, n, 2, self.n_heads_kernel, self.d_head)
        q = qv[:, :, 0]
        v_self = qv[:, :, 1]

        z_ln = self._cross_ln(
            cross_prefix_ln_s,
            cross_prefix_ln_b,
            z,
            tag_id="route.heavy.cross.prefix_ln",
            **prefix_kfac,
        )
        kv = self._dense_no_bias(
            cross_w_kv,
            z_ln,
            tag_id="route.heavy.cross.kv",
            **prefix_kfac,
        ).reshape(n, 2, self.n_heads_kernel, self.d_head)
        k = kv[:, 0]
        v = kv[:, 1]
        k_b = jnp.broadcast_to(k[None, :, :, :], q.shape)
        v_b = jnp.broadcast_to(v[None, :, :, :], q.shape)
        prefix_pairs = self._heavy_prefix_pairs_teacher(edge, perm)
        cross_bias = self._heavy_cross_edge_bias(
            prefix_pairs,
            (
                cross_edge_ln_s,
                cross_edge_ln_b,
                cross_edge_w1,
                cross_edge_b1,
                cross_edge_w2,
                cross_edge_b2,
            ),
            structural_mask=prefix_pair_structural_mask,
            repeat_ndim=2,
            context_primal_reused_over_walkers=context_reuse,
        )
        lca_tk = lca_alibi_bias(
            idx,
            idx,
            lca_fixed_slopes(self.n_heads_kernel, dtype=cand.dtype),
        )
        pos_bias = jnp.transpose(lca_tk, (1, 2, 0))
        cross_bias = cross_bias[None, :, :, :] + pos_bias[:, None, :, :]
        key_mask = mask.astype(bool)[None, :] & (idx[None, :] < idx[:, None])
        cross_out = self._route_attention(
            q,
            k_b,
            v_b,
            cross_bias,
            key_mask,
            impl=impl,
            key_mask_only=True,
        )
        cross_flat = self._collapse_heavy_heads(cross_out)
        delta = self._dense_no_bias(
            cross_w_o,
            cross_flat,
            tag_id="route.heavy.cross.o",
            **candidate_kfac,
        )
        cand = cand + query_mask * self.heavy_residual_gain * delta

        x_ln = self._ln(
            self_ln_s,
            cand,
            tag_id="route.heavy.self.ln",
            **candidate_kfac,
        )
        qkv = self._dense_no_bias(
            self_w_qkv,
            x_ln,
            tag_id="route.heavy.self.qkv",
            **candidate_kfac,
        ).reshape(n, n, 3, self.n_heads_kernel, self.d_head)
        q = qkv[:, :, 0]
        k = qkv[:, :, 1]
        v = qkv[:, :, 2]
        suffix_pairs = self._heavy_suffix_pairs_teacher(edge)
        suffix_bias = self._heavy_edge_bias(
            suffix_pairs,
            (
                self_edge_ln_s,
                self_edge_w1,
                self_edge_b1,
                self_edge_w2,
                self_edge_b2,
            ),
            prefix="self",
            structural_mask=suffix_pair_structural_mask,
            repeat_ndim=2,
            context_primal_reused_over_walkers=context_reuse,
        )

        suffix_bias = jnp.broadcast_to(
            suffix_bias[None, :, :, :],
            (n,) + suffix_bias.shape,
        )
        suffix_mask = mask.astype(bool)[None, :] & (
            pos_of_node[None, :] >= idx[:, None]
        )
        self_out = self._route_attention(
            q,
            k,
            v,
            suffix_bias,
            suffix_mask,
            impl=impl,
        )
        self_flat = self._collapse_heavy_heads(self_out)
        delta = self._dense_no_bias(
            self_w_o,
            self_flat,
            tag_id="route.heavy.self.o",
            **candidate_kfac,
        )
        cand = cand + query_mask * self.heavy_residual_gain * delta

        ffn_in = self._ln(
            ffn_ln_s,
            cand,
            tag_id="route.heavy.ffn.ln",
            **candidate_kfac,
        )
        ffn = self._dense(
            ffn_w1,
            ffn_b1,
            ffn_in,
            tag_id="route.heavy.ffn1",
            **candidate_kfac,
        )
        ffn = fused_silu(ffn)
        delta = self._dense(
            ffn_w2,
            ffn_b2,
            ffn,
            tag_id="route.heavy.ffn2",
            **candidate_kfac,
        )
        return cand + query_mask * self.heavy_residual_gain * delta

    def _apply_heavy_teacher(self, base, z, edge, perm, mask):
        n = base.shape[0]
        idx = jnp.arange(n, dtype=jnp.int32)
        pos_of_node = jnp.zeros((n,), dtype=jnp.int32).at[perm].set(idx)
        impl = self._resolve_heavy_attn_impl(n)

        def apply_one(cand, params):
            return self._heavy_layer_teacher(
                cand,
                z,
                edge,
                perm,
                mask,
                pos_of_node,
                params,
                impl=impl,
            )

        params = self._heavy_layer_params()
        cand = base
        for layer in params:
            cand = apply_one(cand, layer)
        return cand

    def _heavy_prefix_pairs_step(
        self,
        edge,
        prefix_ids,
        *,
        edge_transpose=None,
    ):
        edge_i_p = edge[:, prefix_ids, :]
        edge_p_i = (
            jnp.transpose(edge[prefix_ids, :, :], (1, 0, 2))
            if edge_transpose is None
            else edge_transpose[:, prefix_ids, :]
        )
        return jnp.concatenate([edge_i_p, edge_p_i], axis=-1)

    def _heavy_suffix_pairs_step(self, edge, *, edge_transpose=None):
        edge_i_j = edge
        edge_j_i = (
            jnp.swapaxes(edge, 0, 1) if edge_transpose is None else edge_transpose
        )
        return jnp.concatenate([edge_i_j, edge_j_i], axis=-1)

    def _heavy_cross_biases(self, edge):
        all_pairs = self._heavy_suffix_pairs_step(edge)
        params = self._heavy_layer_params()
        return tuple(
            self._heavy_cross_edge_bias(
                all_pairs,
                layer[7:13],
                context_primal_reused_over_walkers=True,
            )
            for layer in params
        )

    def _heavy_suffix_biases(self, edge):
        suffix_pairs = self._heavy_suffix_pairs_step(edge)
        params = self._heavy_layer_params()
        return tuple(
            self._heavy_edge_bias(
                suffix_pairs,
                layer[16:21],
                prefix="self",
                context_primal_reused_over_walkers=True,
            )
            for layer in params
        )

    def _heavy_biases_tiled(
        self,
        edge,
        edge_transpose,
        *,
        pair_tile_size: int,
        sequence_axis_name: str | None = None,
        sequence_mesh=None,
    ):

        n = edge.shape[0]
        tile = min(int(pair_tile_size), n)
        if tile < 1:
            raise ValueError("pair_tile_size must be positive")
        n_tiles = (n + tile - 1) // tile
        padded_n = n_tiles * tile
        source_pad = padded_n - n
        edge_transpose = (
            jnp.swapaxes(edge, 0, 1) if edge_transpose is None else edge_transpose
        )
        edge_padded = jnp.pad(edge, ((0, 0), (0, source_pad), (0, 0)))
        edge_transpose_padded = jnp.pad(
            edge_transpose,
            ((0, 0), (0, source_pad), (0, 0)),
        )

        def _seq_constraint(value, *axes):
            if sequence_axis_name is None:
                return value
            from jax.sharding import NamedSharding, PartitionSpec as P

            spec = P(*axes)
            if sequence_mesh is not None:
                spec = NamedSharding(sequence_mesh, spec)
            return jax.lax.with_sharding_constraint(value, spec)

        params = self._heavy_layer_params()
        layer_params = tuple(params)

        def project_layer(layer, param_slice, project_bias):
            output0 = _seq_constraint(
                jnp.zeros(
                    (n, padded_n, self.n_heads_kernel),
                    dtype=edge.dtype,
                ),
                sequence_axis_name,
                None,
                None,
            )

            def project_tile(tile_index, output):
                start = tile_index * tile
                edge_tile = jax.lax.dynamic_slice_in_dim(
                    edge_padded,
                    start,
                    tile,
                    axis=1,
                )
                edge_transpose_tile = jax.lax.dynamic_slice_in_dim(
                    edge_transpose_padded,
                    start,
                    tile,
                    axis=1,
                )
                edge_pair = jnp.concatenate(
                    [edge_tile, edge_transpose_tile],
                    axis=-1,
                )
                edge_pair = _seq_constraint(
                    edge_pair,
                    sequence_axis_name,
                    None,
                    None,
                )
                bias = project_bias(
                    edge_pair,
                    layer[param_slice],
                    context_primal_reused_over_walkers=True,
                )
                output = jax.lax.dynamic_update_slice_in_dim(
                    output,
                    bias,
                    start,
                    axis=1,
                )
                return _seq_constraint(
                    output,
                    sequence_axis_name,
                    None,
                    None,
                )

            output = jax.lax.fori_loop(
                0,
                n_tiles,
                project_tile,
                output0,
            )
            return output[:, :n, :]

        cross_biases = tuple(
            project_layer(layer, slice(7, 13), self._heavy_cross_edge_bias)
            for layer in layer_params
        )
        suffix_biases = tuple(
            project_layer(
                layer,
                slice(16, 21),
                lambda edge_pair, params, **kwargs: self._heavy_edge_bias(
                    edge_pair, params, prefix="self", **kwargs
                ),
            )
            for layer in layer_params
        )
        return cross_biases, suffix_biases

    def _pack_heavy_static_bias_tables(self, edge):

        return self._heavy_cross_biases(edge) + self._heavy_suffix_biases(edge)

    def _unpack_heavy_static_bias_tables(self, tables):

        layers = int(self.route_prefix_suffix_layers)
        expected = 2 * layers
        if len(tables) != expected:
            raise ValueError(
                "RouterStatic static_bias_tables has "
                f"{len(tables)} leaves; expected {expected} for "
                f"route_prefix_suffix_layers={layers}"
            )
        return tables[:layers], tables[layers:]

    def _heavy_layer_step_candidates(
        self,
        cand: Float[Array, "n d_model"],
        hidden_cache: Float[Array, "n d_model"],
        edge: Float[Array, "n n d_edge"],
        prefix_ids: Int[Array, "n"],
        picked: Array,
        mask: Int[Array, "n"] | Array,
        t: Int[Array, ""],
        params,
        *,
        impl: str,
        cross_bias=None,
        suffix_bias=None,
        edge_transpose=None,
        sequence_axis_name=None,
        sequence_mesh=None,
    ) -> Float[Array, "n d_model"]:
        (
            cross_ln_s,
            cross_ln_b,
            cross_prefix_ln_s,
            cross_prefix_ln_b,
            cross_w_qv,
            cross_w_kv,
            cross_w_o,
            cross_edge_ln_s,
            cross_edge_ln_b,
            cross_edge_w1,
            cross_edge_b1,
            cross_edge_w2,
            cross_edge_b2,
            self_ln_s,
            self_w_qkv,
            self_w_o,
            self_edge_ln_s,
            self_edge_w1,
            self_edge_b1,
            self_edge_w2,
            self_edge_b2,
            ffn_ln_s,
            ffn_w1,
            ffn_b1,
            ffn_w2,
            ffn_b2,
        ) = params
        n = cand.shape[0]
        dtype = cand.dtype
        idx = jnp.arange(n, dtype=jnp.int32)
        mask_bool = mask.astype(bool)
        row_active = mask_bool[t]
        candidate_structural_mask = row_active & mask_bool & (~picked.astype(bool))
        prefix_structural_mask = row_active & mask_bool & (idx < t)
        cross_pair_structural_mask = (
            candidate_structural_mask[:, None] & prefix_structural_mask[None, :]
        )
        self_pair_structural_mask = (
            candidate_structural_mask[:, None] & candidate_structural_mask[None, :]
        )
        context_reuse = True
        candidate_kfac = dict(
            kfac_structural_mask=candidate_structural_mask,
            kfac_scan_shared=True,
            kfac_repeat_ndim=1,
            kfac_context_primal_reused_over_walkers=context_reuse,
        )
        prefix_kfac = dict(
            kfac_structural_mask=prefix_structural_mask,
            kfac_scan_shared=True,
            kfac_repeat_ndim=1,
            kfac_context_primal_reused_over_walkers=context_reuse,
        )
        query_mask = mask.astype(dtype).reshape(n, 1)

        def _seq_constraint(value, *axes):
            if sequence_axis_name is None:
                return value
            from jax.sharding import NamedSharding, PartitionSpec as P

            spec = P(*axes)
            if sequence_mesh is not None:
                spec = NamedSharding(sequence_mesh, spec)
            return jax.lax.with_sharding_constraint(value, spec)

        cand = _seq_constraint(cand, sequence_axis_name, None)

        x_ln = self._cross_ln(
            cross_ln_s,
            cross_ln_b,
            cand,
            tag_id="route.heavy.cross.ln",
            **candidate_kfac,
        )
        qv = self._dense_no_bias(
            cross_w_qv,
            x_ln,
            tag_id="route.heavy.cross.qv",
            **candidate_kfac,
        ).reshape(n, 2, self.n_heads_kernel, self.d_head)
        qv = _seq_constraint(
            qv,
            sequence_axis_name,
            None,
            None,
            None,
        )
        q = qv[:, 0]
        v_self = qv[:, 1]

        z_ln = self._cross_ln(
            cross_prefix_ln_s,
            cross_prefix_ln_b,
            hidden_cache,
            tag_id="route.heavy.cross.prefix_ln",
            **prefix_kfac,
        )
        kv = self._dense_no_bias(
            cross_w_kv,
            z_ln,
            tag_id="route.heavy.cross.kv",
            **prefix_kfac,
        ).reshape(n, 2, self.n_heads_kernel, self.d_head)
        kv = _seq_constraint(kv, None, None, None, None)
        k = kv[:, 0]
        v = kv[:, 1]
        q = _seq_constraint(q, sequence_axis_name, None, None)
        v_self = _seq_constraint(
            v_self,
            sequence_axis_name,
            None,
            None,
        )

        k = _seq_constraint(k, None, None, None)
        v = _seq_constraint(v, None, None, None)
        if cross_bias is None:
            prefix_pairs = self._heavy_prefix_pairs_step(
                edge,
                prefix_ids,
                edge_transpose=edge_transpose,
            )
            cross_bias = self._heavy_cross_edge_bias(
                prefix_pairs,
                (
                    cross_edge_ln_s,
                    cross_edge_ln_b,
                    cross_edge_w1,
                    cross_edge_b1,
                    cross_edge_w2,
                    cross_edge_b2,
                ),
                structural_mask=cross_pair_structural_mask,
                scan_shared=True,
                repeat_ndim=2,
                context_primal_reused_over_walkers=context_reuse,
            )
        else:
            cross_bias = cross_bias[:, prefix_ids, :]
        cross_bias = _seq_constraint(
            cross_bias,
            sequence_axis_name,
            None,
            None,
        )
        pos_bias = lca_alibi_bias(
            jnp.asarray([t], jnp.int32),
            idx,
            lca_fixed_slopes(self.n_heads_kernel, dtype=dtype),
        )[:, 0, :]
        cross_bias = cross_bias + jnp.transpose(pos_bias, (1, 0))[None, :, :]
        key_mask = mask.astype(bool) & (idx < t)
        cross_out = self._route_attention(
            q[None, :, :, :],
            k[None, :, :, :],
            v[None, :, :, :],
            cross_bias[None, :, :, :],
            key_mask[None, :],
            impl=impl,
            key_mask_only=True,
            sequence_axis_name=sequence_axis_name,
            sequence_mesh=sequence_mesh,
        )[0]
        cross_out = _seq_constraint(
            cross_out,
            sequence_axis_name,
            None,
            None,
        )
        cross_flat = self._collapse_heavy_heads(cross_out)
        delta = self._dense_no_bias(
            cross_w_o,
            cross_flat,
            tag_id="route.heavy.cross.o",
            **candidate_kfac,
        )
        cand = cand + query_mask * self.heavy_residual_gain * delta

        x_ln = self._ln(
            self_ln_s,
            cand,
            tag_id="route.heavy.self.ln",
            **candidate_kfac,
        )
        qkv = self._dense_no_bias(
            self_w_qkv,
            x_ln,
            tag_id="route.heavy.self.qkv",
            **candidate_kfac,
        ).reshape(n, 3, self.n_heads_kernel, self.d_head)
        qkv = _seq_constraint(
            qkv,
            sequence_axis_name,
            None,
            None,
            None,
        )
        q = qkv[:, 0]
        k = qkv[:, 1]
        v = qkv[:, 2]
        q = _seq_constraint(q, sequence_axis_name, None, None)
        k = _seq_constraint(k, None, None, None)
        v = _seq_constraint(v, None, None, None)
        if suffix_bias is None:
            suffix_pairs = self._heavy_suffix_pairs_step(
                edge,
                edge_transpose=edge_transpose,
            )
            suffix_bias = self._heavy_edge_bias(
                suffix_pairs,
                (
                    self_edge_ln_s,
                    self_edge_w1,
                    self_edge_b1,
                    self_edge_w2,
                    self_edge_b2,
                ),
                prefix="self",
                structural_mask=self_pair_structural_mask,
                scan_shared=True,
                repeat_ndim=2,
                context_primal_reused_over_walkers=context_reuse,
            )
        suffix_bias = _seq_constraint(
            suffix_bias,
            sequence_axis_name,
            None,
            None,
        )
        suffix_mask = mask.astype(bool) & (~picked)
        self_out = self._route_attention(
            q[None, :, :, :],
            k[None, :, :, :],
            v[None, :, :, :],
            suffix_bias[None, :, :, :],
            suffix_mask[None, :],
            impl=impl,
            sequence_axis_name=sequence_axis_name,
            sequence_mesh=sequence_mesh,
        )[0]
        self_out = _seq_constraint(
            self_out,
            sequence_axis_name,
            None,
            None,
        )
        self_flat = self._collapse_heavy_heads(self_out)
        delta = self._dense_no_bias(
            self_w_o,
            self_flat,
            tag_id="route.heavy.self.o",
            **candidate_kfac,
        )
        cand = cand + query_mask * self.heavy_residual_gain * delta

        ffn_in = self._ln(
            ffn_ln_s,
            cand,
            tag_id="route.heavy.ffn.ln",
            **candidate_kfac,
        )
        ffn = self._dense(
            ffn_w1,
            ffn_b1,
            ffn_in,
            tag_id="route.heavy.ffn1",
            **candidate_kfac,
        )
        ffn = fused_silu(ffn)
        delta = self._dense(
            ffn_w2,
            ffn_b2,
            ffn,
            tag_id="route.heavy.ffn2",
            **candidate_kfac,
        )
        return cand + query_mask * self.heavy_residual_gain * delta

    def _apply_heavy_step(
        self,
        base,
        hidden_cache,
        edge,
        prefix_ids,
        picked,
        mask,
        t,
        *,
        cross_biases=None,
        suffix_biases=None,
        edge_transpose=None,
        sequence_axis_name=None,
        sequence_mesh=None,
    ):
        impl = self._resolve_heavy_attn_impl(base.shape[0])
        layer_params = self._heavy_layer_params()
        n_layers = int(self.route_prefix_suffix_layers)

        if cross_biases is None or len(cross_biases) == 0:
            cross_biases = (None,) * n_layers
        if suffix_biases is None:
            suffix_biases = (None,) * n_layers

        cand = base
        for params, cross_bias, suffix_bias in zip(
            layer_params,
            cross_biases,
            suffix_biases,
        ):
            cand = self._heavy_layer_step_candidates(
                cand,
                hidden_cache,
                edge,
                prefix_ids,
                picked,
                mask,
                t,
                params,
                impl=impl,
                cross_bias=cross_bias,
                suffix_bias=suffix_bias,
                edge_transpose=edge_transpose,
                sequence_axis_name=sequence_axis_name,
                sequence_mesh=sequence_mesh,
            )
        return cand


class TreePrefixPointerMHSEA(_PrefixSuffixRouteBase):
    tree_merge: _TreePrefixMerge
    tree_edge_merge: EdgeMergeOp
    tree_level_layer: _TreePrefixSelfLayer
    tree_level_fwl: CausalRouterEdgeFWLUpdate
    alpha_tree_level_attn: Float[Array, "d_model"]
    alpha_tree_level_ffn: Float[Array, "d_model"]
    tree_prefix_layers: list[_TreePrefixSelfLayer]
    tree_candidate_layers: list[_TreePrefixCandidateLayer]

    g_step_pool: "GDescriptorPool"
    g_step_update: "TreeGlobalUpdate"
    g_prefix_ffn_w: Float[Array, "d_gstream d_model"]
    g_cand_ffn_w: Float[Array, "d_gstream d_model"]

    tree_edge_ln_scale: Float[Array, "two_d_edge"]
    tree_edge_w1: Float[Array, "two_d_edge d_msg_hidden"]
    tree_edge_b1: Float[Array, "d_msg_hidden"]
    tree_edge_w2: Float[Array, "d_msg_hidden d_model"]
    tree_edge_b2: Float[Array, "d_model"]

    route_tree_prefix_layers: int = eqx.field(static=True)
    route_tree_prefix_candidate_layers: int = eqx.field(static=True)
    tree_prefix_merge_hidden: int = eqx.field(static=True)
    tree_prefix_edge_hidden: int = eqx.field(static=True)
    tree_prefix_residual_gain: float = eqx.field(static=True)
    tree_candidate_residual_gain: float = eqx.field(static=True)
    tree_ngpt_alpha_max: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        d_in: int,
        d_edge: int,
        d_global: int,
        d_model: int,
        n_heads: int,
        max_n: int,
        key: PRNGKeyArray,
        route_tree_prefix_layers: int = 1,
        route_tree_prefix_candidate_layers: int = 1,
        route_tree_prefix_merge_hidden: int,
        route_tree_prefix_post_prefix_suffix_layers: int = 0,
        score_init_scale: float = 1.0,
        route_decoder_attn_impl: str = "mhsea_tuned",
        rope_base: float = 10000.0,
        rope_scaling: float = 1.0,
        attention_dim: int,
        pointer_score_dim: int,
        candidate_hidden: int,
        summary_hidden: int,
        ffn_hidden: int,
        global_tap_dim: int,
        alpha_init: float,
        alpha_max: float,
    ):
        if route_tree_prefix_layers < 0:
            raise ValueError("route_tree_prefix_layers must be >= 0")
        if route_tree_prefix_candidate_layers < 0:
            raise ValueError("route_tree_prefix_candidate_layers must be >= 0")
        if route_tree_prefix_post_prefix_suffix_layers < 0:
            raise ValueError("route_tree_prefix_post_prefix_suffix_layers must be >= 0")
        key_base, key_tree = jax.random.split(key)
        super().__init__(
            d_in=d_in,
            d_edge=d_edge,
            d_global=d_global,
            d_model=d_model,
            n_heads=n_heads,
            max_n=max_n,
            key=key_base,
            score_init_scale=score_init_scale,
            route_prefix_suffix_layers=int(route_tree_prefix_post_prefix_suffix_layers),
            route_decoder_attn_impl=route_decoder_attn_impl,
            rope_base=rope_base,
            rope_scaling=rope_scaling,
            attention_dim=attention_dim,
            pointer_score_dim=pointer_score_dim,
            candidate_hidden=candidate_hidden,
            summary_hidden=summary_hidden,
            ffn_hidden=ffn_hidden,
            global_tap_dim=global_tap_dim,
        )

        layers = int(route_tree_prefix_layers)
        cand_layers = int(route_tree_prefix_candidate_layers)
        merge_hidden = int(route_tree_prefix_merge_hidden)
        d_qv = self.n_heads_kernel * self.d_head
        d_o_in = self.n_heads * self.d_head
        edge_hidden = max(32, 2 * self.n_heads_kernel, self.msg_hidden)
        msg_hidden = self.msg_hidden
        k_edge, k_merge, k_level, k_prefix, k_cand = jax.random.split(key_tree, 5)

        def w(k, shape, fan_in):
            return jax.random.normal(k, shape) * (fan_in**-0.5)

        self.tree_edge_ln_scale = jnp.ones((2 * self.d_edge,))
        ek1, ek2 = jax.random.split(k_edge)
        self.tree_edge_w1 = w(ek1, (2 * self.d_edge, msg_hidden), 2 * self.d_edge)
        self.tree_edge_b1 = jnp.zeros((msg_hidden,))
        self.tree_edge_w2 = w(ek2, (msg_hidden, self.d_model), msg_hidden)
        self.tree_edge_b2 = jnp.zeros((self.d_model,))

        self.tree_merge = _TreePrefixMerge(
            self.d_model,
            hidden=merge_hidden,
            max_depth=max(1, default_tree_depth(max_n)),
            key=k_merge,
            ln_eps=1.0e-5,
            gladder_d_g=int(self.d_global),
            alpha_init=alpha_init,
            alpha_max=alpha_max,
        )
        self.tree_edge_merge = EdgeMergeOp(
            d_edge=self.d_model,
            d_c=self.d_model,
            key=jax.random.fold_in(k_level, 0xE06E),
            alpha_init=alpha_init,
            alpha_max=alpha_max,
            d_hidden=None,
            n_blocks=2,
            edge_node_ctx_dim=None,
        )
        from .global_ladder import GDescriptorPool, TreeGlobalUpdate

        _k_rg = jax.random.split(jax.random.fold_in(k_merge, 0x61B6), 4)
        self.g_step_pool = GDescriptorPool(
            int(self.d_global),
            self.d_model,
            key=_k_rg[0],
            tag="gladder.route.step.pool",
        )
        self.g_step_update = TreeGlobalUpdate(
            int(self.d_global),
            self.g_step_pool.d_out,
            key=_k_rg[1],
            tag="gladder.route.step.upd",
            tap_dim=global_tap_dim,
            alpha_init=alpha_init,
            alpha_max=alpha_max,
        )
        self.g_prefix_ffn_w = jax.random.normal(
            _k_rg[2], (int(self.d_global), self.d_model)
        ) * (int(self.d_global) ** -0.5)
        self.g_cand_ffn_w = jax.random.normal(
            _k_rg[3], (int(self.d_global), self.d_model)
        ) * (int(self.d_global) ** -0.5)

        ks = jax.random.split(k_level, 7)
        self.tree_level_layer = _TreePrefixSelfLayer(
            ln_scale=jnp.ones((self.d_model,)),
            w_qkv=w(ks[0], (self.d_model, 3 * d_qv), self.d_model),
            w_o=w(ks[1], (d_o_in, self.d_model), d_o_in),
            edge_ln_scale=jnp.ones((self.d_model,)),
            edge_w1=w(ks[2], (self.d_model, edge_hidden), self.d_model),
            edge_b1=jnp.zeros((edge_hidden,)),
            edge_w2=w(ks[3], (edge_hidden, self.n_heads_kernel), edge_hidden),
            edge_b2=jnp.zeros((self.n_heads_kernel,)),
            ffn_ln_scale=jnp.ones((self.d_model,)),
            ffn_w1=w(ks[4], (self.d_model, self.ffn_hidden), self.d_model),
            ffn_b1=jnp.zeros((self.ffn_hidden,)),
            ffn_w2=w(ks[5], (self.ffn_hidden, self.d_model), self.ffn_hidden),
            ffn_b2=jnp.zeros((self.d_model,)),
        )
        self.tree_level_fwl = CausalRouterEdgeFWLUpdate(
            d_c=self.d_model,
            d_edge=self.d_model,
            channels=max(32, self.d_model // 2),
            alpha_init=alpha_init,
            alpha_max=alpha_max,
            key=ks[6],
        )
        self.alpha_tree_level_attn = float(alpha_init) * jnp.ones((self.d_model,))
        self.alpha_tree_level_ffn = float(alpha_init) * jnp.ones((self.d_model,))
        self.tree_ngpt_alpha_max = float(alpha_max)

        prefix_keys = jax.random.split(k_prefix, max(layers, 1))
        prefix_layers = []
        for li in range(layers):
            ks = jax.random.split(prefix_keys[li], 7)
            prefix_layers.append(
                _TreePrefixSelfLayer(
                    ln_scale=jnp.ones((self.d_model,)),
                    w_qkv=w(ks[0], (self.d_model, 3 * d_qv), self.d_model),
                    w_o=w(ks[1], (d_o_in, self.d_model), d_o_in),
                    edge_ln_scale=jnp.ones((self.d_model,)),
                    edge_w1=w(ks[2], (self.d_model, edge_hidden), self.d_model),
                    edge_b1=jnp.zeros((edge_hidden,)),
                    edge_w2=w(ks[3], (edge_hidden, self.n_heads_kernel), edge_hidden),
                    edge_b2=jnp.zeros((self.n_heads_kernel,)),
                    ffn_ln_scale=jnp.ones((self.d_model,)),
                    ffn_w1=w(ks[4], (self.d_model, self.ffn_hidden), self.d_model),
                    ffn_b1=jnp.zeros((self.ffn_hidden,)),
                    ffn_w2=w(ks[5], (self.ffn_hidden, self.d_model), self.ffn_hidden),
                    ffn_b2=jnp.zeros((self.d_model,)),
                )
            )
        self.tree_prefix_layers = prefix_layers

        cand_keys = jax.random.split(k_cand, max(cand_layers, 1))
        tree_candidate_layers = []
        for li in range(cand_layers):
            ks = jax.random.split(cand_keys[li], 8)
            tree_candidate_layers.append(
                _TreePrefixCandidateLayer(
                    cand_ln_scale=jnp.ones((self.d_model,)),
                    prefix_ln_scale=jnp.ones((self.d_model,)),
                    cand_w_qv=w(ks[0], (self.d_model, 2 * d_qv), self.d_model),
                    prefix_w_kv=w(ks[1], (self.d_model, 2 * d_qv), self.d_model),
                    w_o=w(ks[2], (d_o_in, self.d_model), d_o_in),
                    edge_ln_scale=jnp.ones((self.d_model,)),
                    edge_w1=w(ks[3], (self.d_model, edge_hidden), self.d_model),
                    edge_b1=jnp.zeros((edge_hidden,)),
                    edge_w2=w(ks[4], (edge_hidden, self.n_heads_kernel), edge_hidden),
                    edge_b2=jnp.zeros((self.n_heads_kernel,)),
                    ffn_ln_scale=jnp.ones((self.d_model,)),
                    ffn_w1=w(ks[5], (self.d_model, self.ffn_hidden), self.d_model),
                    ffn_b1=jnp.zeros((self.ffn_hidden,)),
                    ffn_w2=w(ks[6], (self.ffn_hidden, self.d_model), self.ffn_hidden),
                    ffn_b2=jnp.zeros((self.d_model,)),
                )
            )
        self.tree_candidate_layers = tree_candidate_layers

        self.route_tree_prefix_layers = layers
        self.route_tree_prefix_candidate_layers = cand_layers
        self.tree_prefix_merge_hidden = int(merge_hidden)
        self.tree_prefix_edge_hidden = int(edge_hidden)
        self.tree_prefix_residual_gain = 0.0 if layers == 0 else float(layers) ** -0.5
        self.tree_candidate_residual_gain = (
            0.0 if cand_layers == 0 else float(cand_layers) ** -0.5
        )

    def _tree_prefix_layer_params(self):
        return [layer.as_tuple() for layer in self.tree_prefix_layers]

    def _tree_candidate_layer_params(self):
        return [layer.as_tuple() for layer in self.tree_candidate_layers]

    def _resolve_tree_attn_impl(self) -> str:
        return self.route_decoder_attn_impl

    def _tree_edge_message_mlp(self, edge_pair, structural_mask):
        x = self._ln(
            self.tree_edge_ln_scale,
            edge_pair,
            tag_id="route.tree_prefix.edge_msg_ln",
            kfac_structural_mask=structural_mask,
            kfac_repeat_ndim=2,
        )
        x = self._dense(
            self.tree_edge_w1,
            self.tree_edge_b1,
            x,
            tag_id="route.tree_prefix.edge_msg1",
            kfac_structural_mask=structural_mask,
            kfac_repeat_ndim=2,
        )
        x = fused_silu(x)
        return self._dense(
            self.tree_edge_w2,
            self.tree_edge_b2,
            x,
            tag_id="route.tree_prefix.edge_msg2",
            kfac_structural_mask=structural_mask,
            kfac_repeat_ndim=2,
        )

    def _tree_pair_messages(self, edge, mask):
        edge_pair = jnp.concatenate([jnp.swapaxes(edge, 0, 1), edge], axis=-1)
        mask_bool = mask.astype(bool)
        structural_mask = mask_bool[:, None] & mask_bool[None, :]
        return self._tree_edge_message_mlp(edge_pair, structural_mask)

    def _tree_pair_messages_for_route(
        self,
        edge,
        route_ids,
        mask,
        *,
        edge_transpose=None,
        sequence_axis_name=None,
        sequence_mesh=None,
        row_permute_fn=None,
    ):

        edge_transpose = (
            jnp.swapaxes(edge, 0, 1) if edge_transpose is None else edge_transpose
        )
        edge_rows = (
            jnp.take(edge, route_ids, axis=0)
            if row_permute_fn is None
            else row_permute_fn(edge, route_ids)
        )
        edge_transpose_rows = (
            jnp.take(edge_transpose, route_ids, axis=0)
            if row_permute_fn is None
            else row_permute_fn(edge_transpose, route_ids)
        )
        edge_fwd = jnp.take(edge_rows, route_ids, axis=1)
        edge_rev = jnp.take(edge_transpose_rows, route_ids, axis=1)
        if sequence_axis_name is not None:
            from jax.sharding import NamedSharding, PartitionSpec as P

            spec = P(sequence_axis_name, None, None)
            if sequence_mesh is not None:
                spec = NamedSharding(sequence_mesh, spec)
            edge_fwd = jax.lax.with_sharding_constraint(edge_fwd, spec)
            edge_rev = jax.lax.with_sharding_constraint(edge_rev, spec)
        edge_pair = jnp.concatenate([edge_rev, edge_fwd], axis=-1)
        route_mask = mask.astype(bool)[route_ids]
        structural_mask = route_mask[:, None] & route_mask[None, :]
        return self._tree_edge_message_mlp(edge_pair, structural_mask)

    def _tree_pair_message_row(
        self,
        edge,
        source,
        mask,
        *,
        edge_transpose=None,
    ):

        edge_transpose = (
            jnp.swapaxes(edge, 0, 1) if edge_transpose is None else edge_transpose
        )
        edge_pair = jnp.concatenate(
            [edge[:, source, :], edge_transpose[:, source, :]],
            axis=-1,
        )
        structural_mask = mask.astype(bool)[source] & mask.astype(bool)
        return self._tree_edge_message_mlp(edge_pair, structural_mask)

    def _tree_pair_message_column(
        self,
        edge,
        destination,
        mask,
        *,
        edge_transpose=None,
    ):

        edge_transpose = (
            jnp.swapaxes(edge, 0, 1) if edge_transpose is None else edge_transpose
        )
        edge_pair = jnp.concatenate(
            [edge_transpose[:, destination, :], edge[:, destination, :]],
            axis=-1,
        )
        structural_mask = mask.astype(bool)[destination] & mask.astype(bool)
        return self._tree_edge_message_mlp(edge_pair, structural_mask)

    def _tree_clock_depth_from_mask(self, mask):
        n_active = jnp.maximum(jnp.sum(mask.astype(jnp.int32)), 1)
        depth = jnp.ceil(jnp.log2(n_active.astype(jnp.float32))).astype(jnp.int32)
        return jnp.maximum(depth, 1)

    def _apply_tree_level_attention(self, nodes, edges, mask, level_idx):
        edges = self.tree_level_fwl.apply_residual(
            edges,
            nodes,
            mask,
            kfac_scan_shared=True,
        )

        (
            ln_s,
            w_qkv,
            w_o,
            edge_ln_s,
            edge_w1,
            edge_b1,
            edge_w2,
            edge_b2,
            ffn_ln_s,
            ffn_w1,
            ffn_b1,
            ffn_w2,
            ffn_b2,
        ) = self.tree_level_layer.as_tuple()
        del level_idx

        n = nodes.shape[0]
        dtype = nodes.dtype
        mask_bool = mask.astype(bool)
        idx = jnp.arange(n, dtype=jnp.int32)
        node_structural_mask = mask_bool
        pair_structural_mask = (
            mask_bool[:, None] & mask_bool[None, :] & (idx[None, :] <= idx[:, None])
        )
        query_mask = mask.astype(dtype)[:, None]
        x_ln = self._ln(
            ln_s,
            nodes,
            tag_id="route.tree_prefix.level.ln",
            kfac_structural_mask=node_structural_mask,
            kfac_scan_shared=True,
            kfac_repeat_ndim=1,
        )
        qkv = self._dense_no_bias(
            w_qkv,
            x_ln,
            tag_id="route.tree_prefix.level.qkv",
            kfac_structural_mask=node_structural_mask,
            kfac_scan_shared=True,
            kfac_repeat_ndim=1,
        ).reshape(n, 3, self.n_heads_kernel, self.d_head)
        q = qkv[:, 0]
        k = qkv[:, 1]
        v = qkv[:, 2]
        edge_bias = self._tree_prefix_edge_bias(
            edges,
            (edge_ln_s, edge_w1, edge_b1, edge_w2, edge_b2),
            prefix="level",
            kfac_structural_mask=pair_structural_mask,
            kfac_scan_shared=True,
        )
        edge_bias = edge_bias + jnp.transpose(
            lca_alibi_bias(
                idx,
                idx,
                lca_fixed_slopes(self.n_heads_kernel, dtype=dtype),
            ),
            (1, 2, 0),
        )

        compute_dtype = dtype
        q_c = q.astype(compute_dtype)
        k_c = k.astype(compute_dtype)
        v_c = v.astype(compute_dtype)
        logits = jnp.einsum("ihd,jhd->hij", q_c, k_c)
        logits = logits / jnp.sqrt(jnp.asarray(self.d_head, dtype=compute_dtype))
        logits = logits + jnp.transpose(edge_bias.astype(compute_dtype), (2, 0, 1))
        valid = (idx[None, :] <= idx[:, None]) & mask_bool[None, :]
        logits = jnp.where(
            valid[None, :, :],
            logits,
            jnp.asarray(-1.0e30, dtype=compute_dtype),
        )
        alpha = jax.nn.softmax(logits, axis=-1)
        out = jnp.einsum("hij,jhd->ihd", alpha, v_c).astype(dtype)
        delta = self._dense_no_bias(
            w_o,
            self._collapse_heavy_heads(out),
            tag_id="route.tree_prefix.level.o",
            kfac_structural_mask=node_structural_mask,
            kfac_scan_shared=True,
            kfac_repeat_ndim=1,
        )
        proposal_attn = query_mask * delta
        x = _tree_ngpt_residual(
            nodes,
            proposal_attn,
            self.alpha_tree_level_attn,
            max_gain=self.tree_ngpt_alpha_max,
            tag_id="",
            update_mask=mask,
            kfac_structural_mask=node_structural_mask,
            kfac_scan_shared=True,
            kfac_repeat_ndim=1,
        )
        y = self._ln(
            ffn_ln_s,
            x,
            tag_id="route.tree_prefix.level.ffn_ln",
            kfac_structural_mask=node_structural_mask,
            kfac_scan_shared=True,
            kfac_repeat_ndim=1,
        )
        y = self._dense(
            ffn_w1,
            ffn_b1,
            y,
            tag_id="route.tree_prefix.level.ffn1",
            kfac_structural_mask=node_structural_mask,
            kfac_scan_shared=True,
            kfac_repeat_ndim=1,
        )
        y = fused_silu(y)
        y = self._dense(
            ffn_w2,
            ffn_b2,
            y,
            tag_id="route.tree_prefix.level.ffn2",
            kfac_structural_mask=node_structural_mask,
            kfac_scan_shared=True,
            kfac_repeat_ndim=1,
        )
        proposal_ffn = query_mask * y
        x = _tree_ngpt_residual(
            x,
            proposal_ffn,
            self.alpha_tree_level_ffn,
            max_gain=self.tree_ngpt_alpha_max,
            tag_id="",
            update_mask=mask,
            kfac_structural_mask=node_structural_mask,
            kfac_scan_shared=True,
            kfac_repeat_ndim=1,
        )
        return x, edges

    def _incremental_edge_parent_vector(
        self,
        e00,
        e01,
        e10,
        e11,
        c0,
        c1,
        d0,
        d1,
        active,
    ):

        skip = jnp.asarray(0.25, dtype=e00.dtype) * (e00 + e01 + e10 + e11)
        proposal = jax.vmap(
            lambda x00, x01, x10, x11, xa, xb, ya, yb, keep: self.tree_edge_merge(
                x00,
                x01,
                x10,
                x11,
                xa,
                xb,
                ya,
                yb,
                kfac_structural_mask=keep,
                kfac_scan_shared=True,
            )
        )(e00, e01, e10, e11, c0, c1, d0, d1, active)
        merged = self.tree_edge_merge.apply_skip(
            skip,
            proposal,
            kfac_structural_mask=active,
            kfac_scan_shared=True,
        )
        merged = _tree_sphere(merged)
        return jnp.where(active[:, None], merged, jnp.zeros_like(merged))

    def _apply_tree_level_attention_append(
        self,
        raw_nodes,
        edge_pre,
        active,
        row,
        b_cache,
        *,
        edge_row,
        edge_col,
        sequence_axis_name=None,
        sequence_mesh=None,
    ):

        edge_row, b_cache = self.tree_level_fwl.append_causal_row(
            edge_pre,
            raw_nodes,
            active,
            row,
            b_cache,
            edge_row=edge_row,
            edge_col=edge_col,
            sequence_axis_name=sequence_axis_name,
            sequence_mesh=sequence_mesh,
        )
        edge_row = jnp.where(
            active[:, None],
            _tree_sphere(edge_row),
            edge_row,
        )

        (
            ln_s,
            w_qkv,
            w_o,
            edge_ln_s,
            edge_w1,
            edge_b1,
            edge_w2,
            edge_b2,
            ffn_ln_s,
            ffn_w1,
            ffn_b1,
            ffn_w2,
            ffn_b2,
        ) = self.tree_level_layer.as_tuple()
        n = raw_nodes.shape[0]
        dtype = raw_nodes.dtype
        idx = jnp.arange(n, dtype=jnp.int32)
        x_ln = self._ln(
            ln_s,
            raw_nodes,
            tag_id="route.tree_prefix.level.ln",
            kfac_structural_mask=active,
            kfac_scan_shared=True,
            kfac_repeat_ndim=1,
        )
        qkv = self._dense_no_bias(
            w_qkv,
            x_ln,
            tag_id="route.tree_prefix.level.qkv",
            kfac_structural_mask=active,
            kfac_scan_shared=True,
            kfac_repeat_ndim=1,
        ).reshape(n, 3, self.n_heads_kernel, self.d_head)
        q = qkv[row, 0]
        k = qkv[:, 1]
        v = qkv[:, 2]
        edge_bias = self._tree_prefix_edge_bias(
            edge_row,
            (edge_ln_s, edge_w1, edge_b1, edge_w2, edge_b2),
            prefix="level",
            kfac_structural_mask=active,
            kfac_scan_shared=True,
            kfac_repeat_ndim=1,
        )
        pos_bias = lca_alibi_bias(
            jnp.asarray([row], dtype=jnp.int32),
            idx,
            lca_fixed_slopes(self.n_heads_kernel, dtype=dtype),
        )[:, 0, :]
        edge_bias = edge_bias + jnp.transpose(pos_bias, (1, 0))
        logits = jnp.einsum("hd,jhd->hj", q, k)
        logits = logits / jnp.sqrt(jnp.asarray(self.d_head, dtype=dtype))
        logits = logits + jnp.transpose(edge_bias, (1, 0))
        logits = jnp.where(
            active[None, :],
            logits,
            jnp.asarray(-1.0e30, dtype=dtype),
        )
        alpha = jax.nn.softmax(logits, axis=-1)
        out = jnp.einsum("hj,jhd->hd", alpha, v)
        delta = self._dense_no_bias(
            w_o,
            self._collapse_heavy_heads(out),
            tag_id="route.tree_prefix.level.o",
            kfac_structural_mask=jnp.asarray(True),
            kfac_scan_shared=True,
            kfac_repeat_ndim=0,
        )
        raw_row = raw_nodes[row]
        x = _tree_ngpt_residual(
            raw_row,
            delta,
            self.alpha_tree_level_attn,
            max_gain=self.tree_ngpt_alpha_max,
            tag_id="",
            update_mask=jnp.asarray(True),
            kfac_structural_mask=jnp.asarray(True),
            kfac_scan_shared=True,
            kfac_repeat_ndim=0,
        )
        y = self._ln(
            ffn_ln_s,
            x,
            tag_id="route.tree_prefix.level.ffn_ln",
            kfac_structural_mask=jnp.asarray(True),
            kfac_scan_shared=True,
            kfac_repeat_ndim=0,
        )
        y = self._dense(
            ffn_w1,
            ffn_b1,
            y,
            tag_id="route.tree_prefix.level.ffn1",
            kfac_structural_mask=jnp.asarray(True),
            kfac_scan_shared=True,
            kfac_repeat_ndim=0,
        )
        y = fused_silu(y)
        y = self._dense(
            ffn_w2,
            ffn_b2,
            y,
            tag_id="route.tree_prefix.level.ffn2",
            kfac_structural_mask=jnp.asarray(True),
            kfac_scan_shared=True,
            kfac_repeat_ndim=0,
        )
        x = _tree_ngpt_residual(
            x,
            y,
            self.alpha_tree_level_ffn,
            max_gain=self.tree_ngpt_alpha_max,
            tag_id="",
            update_mask=jnp.asarray(True),
            kfac_structural_mask=jnp.asarray(True),
            kfac_scan_shared=True,
            kfac_repeat_ndim=0,
        )
        x = _tree_sphere(x)
        return x, edge_row, edge_col, b_cache

    def _incremental_tree_append(
        self,
        state,
        leaf,
        chosen,
        t,
        prefix_ids,
        edge,
        mask,
        *,
        edge_transpose=None,
        g=None,
        sequence_axis_name=None,
        sequence_mesh=None,
    ):

        nodes_raw, nodes_post, level_states = state
        n = mask.shape[0]
        n_pad = nodes_raw.shape[1]
        depth = nodes_raw.shape[0] - 1
        dtype = leaf.dtype
        append = mask[t].astype(bool)
        leaf_node = _tree_sphere(leaf)
        nodes_raw = nodes_raw.at[0, t].set(
            jnp.where(append, leaf_node, jnp.zeros_like(leaf_node))
        )
        nodes_post = nodes_post.at[0, t].set(
            jnp.where(append, leaf_node, jnp.zeros_like(leaf_node))
        )
        if depth == 0:
            return nodes_raw, nodes_post, level_states

        mask_pad = jnp.pad(mask.astype(dtype), (0, n_pad - n))
        route_pad = jnp.pad(
            prefix_ids.astype(jnp.int32),
            (0, n_pad - n),
        )
        clock_depth = self._tree_clock_depth_from_mask(mask)
        depth_features = _tree_ngpt_level_counts(
            mask_pad,
            n_pad // 2,
            depth,
            dtype,
            feature_n_levels=clock_depth,
        )
        clock_state = mask_pad
        clock_pair_bases = []
        fixed_pairs = n_pad // 2
        for _level in range(depth):
            clock_pairs = clock_state.reshape(fixed_pairs, 2)
            clock_parent = (
                clock_pairs[:, 0]
                + clock_pairs[:, 1]
                - clock_pairs[:, 0] * clock_pairs[:, 1]
            )
            clock_pair_bases.append(
                jnp.maximum(
                    jnp.sum(clock_parent.astype(jnp.int32)),
                    jnp.asarray(2, dtype=jnp.int32),
                )
            )
            clock_state = jnp.concatenate(
                [clock_parent, jnp.zeros_like(clock_parent)],
                axis=0,
            )
        g_projected = self.tree_merge.project_global(
            g,
            append & (t > 0),
        )

        if sequence_axis_name is not None:
            from jax.sharding import NamedSharding, PartitionSpec as P

            lanes = (
                int(sequence_mesh.shape[sequence_axis_name])
                if sequence_mesh is not None
                else 1
            )

            def constrain_nodes(value):
                spec = P(None, sequence_axis_name, None)
                if sequence_mesh is not None:
                    spec = NamedSharding(sequence_mesh, spec)
                return jax.lax.with_sharding_constraint(value, spec)

            def constrain_level(level_state):
                width_local = level_state[0].shape[0]
                row_axis = (
                    sequence_axis_name
                    if width_local >= lanes and width_local % lanes == 0
                    else None
                )
                spec = P(row_axis, None, None)
                if sequence_mesh is not None:
                    spec = NamedSharding(sequence_mesh, spec)
                return tuple(
                    jax.lax.with_sharding_constraint(value, spec)
                    for value in level_state
                )

            def constrain_row_value(value):
                spec = P(None, None)
                if sequence_mesh is not None:
                    spec = NamedSharding(sequence_mesh, spec)
                return jax.lax.with_sharding_constraint(value, spec)

            def constrain_column_value(value):
                row_axis = (
                    sequence_axis_name
                    if value.shape[0] >= lanes and value.shape[0] % lanes == 0
                    else None
                )
                spec = P(row_axis, None)
                if sequence_mesh is not None:
                    spec = NamedSharding(sequence_mesh, spec)
                return jax.lax.with_sharding_constraint(value, spec)
        else:
            constrain_nodes = lambda value: value
            constrain_level = lambda level_state: level_state
            constrain_row_value = lambda value: value
            constrain_column_value = lambda value: value

        levels_mut = list(level_states)
        for level in range(depth):
            width = n_pad >> (level + 1)
            block = 1 << (level + 1)
            create = append & (((t + 1) % block) == 0)
            parent = (t + 1) // block - 1
            lower_post_edges = None if level == 0 else levels_mut[level - 1][1]

            def do_create(operand):
                raw_all, post_all, level_state = operand
                edge_pre, edge_post, b_cache = level_state
                q = jnp.arange(width, dtype=jnp.int32)
                q0 = 2 * q
                q1 = q0 + 1
                p0 = 2 * parent
                p1 = p0 + 1
                children = post_all[level]
                left = children[p0]
                right = children[p1]

                if level == 0:
                    source0 = route_pad[p0]
                    source1 = route_pad[p1]
                    row0 = self._tree_pair_message_row(
                        edge,
                        source0,
                        mask,
                        edge_transpose=edge_transpose,
                    )[route_pad]
                    row1 = self._tree_pair_message_row(
                        edge,
                        source1,
                        mask,
                        edge_transpose=edge_transpose,
                    )[route_pad]
                    col0 = self._tree_pair_message_column(
                        edge,
                        source0,
                        mask,
                        edge_transpose=edge_transpose,
                    )[route_pad]
                    col1 = self._tree_pair_message_column(
                        edge,
                        source1,
                        mask,
                        edge_transpose=edge_transpose,
                    )[route_pad]
                    row0 = _tree_sphere(row0)
                    row1 = _tree_sphere(row1)
                    col0 = _tree_sphere(col0)
                    col1 = _tree_sphere(col1)
                    row_cells = (
                        row0[q0],
                        row0[q1],
                        row1[q0],
                        row1[q1],
                    )
                    col_cells = (
                        col0[q0],
                        col1[q0],
                        col0[q1],
                        col1[q1],
                    )
                    sibling_lr = row0[p1]
                    sibling_rl = row1[p0]
                else:
                    assert lower_post_edges is not None
                    lower_row0 = _square_row_by_reduction(
                        lower_post_edges,
                        p0,
                    )
                    lower_row1 = _square_row_by_reduction(
                        lower_post_edges,
                        p1,
                    )
                    lower_col0 = _square_column_local(
                        lower_post_edges,
                        p0,
                    )
                    lower_col1 = _square_column_local(
                        lower_post_edges,
                        p1,
                    )
                    row_cells = (
                        lower_row0[q0],
                        lower_row0[q1],
                        lower_row1[q0],
                        lower_row1[q1],
                    )
                    col_cells = (
                        lower_col0[q0],
                        lower_col1[q0],
                        lower_col0[q1],
                        lower_col1[q1],
                    )
                    sibling_lr = lower_row0[p1]
                    sibling_rl = lower_row1[p0]

                depth_row = (
                    None
                    if depth_features is None
                    else depth_features[level, parent][None, :]
                )
                merged, _valid, _genuine = self.tree_merge(
                    left[None, :],
                    right[None, :],
                    jnp.ones((1,), dtype=dtype),
                    jnp.ones((1,), dtype=dtype),
                    sibling_lr[None, :],
                    sibling_rl[None, :],
                    jnp.asarray(level, dtype=jnp.int32),
                    jnp.asarray([parent], dtype=jnp.int32),
                    clock_pair_bases[level],
                    clock_depth,
                    depth_feats=depth_row,
                    g=g,
                    g_structural_mask=jnp.asarray(True),
                    g_projected=g_projected,
                )
                raw_parent = merged[0]
                raw_all = raw_all.at[level + 1, parent].set(raw_parent)

                active = q <= parent
                new_left = jnp.broadcast_to(left, (width, self.d_model))
                new_right = jnp.broadcast_to(right, (width, self.d_model))
                other_left = children[q0]
                other_right = children[q1]
                parent_row = self._incremental_edge_parent_vector(
                    *row_cells,
                    new_left,
                    new_right,
                    other_left,
                    other_right,
                    active,
                )
                parent_col = self._incremental_edge_parent_vector(
                    *col_cells,
                    other_left,
                    other_right,
                    new_left,
                    new_right,
                    active,
                )

                parent_row = constrain_row_value(parent_row)
                parent_col = constrain_column_value(parent_col)
                edge_pre = _replace_square_row_column(
                    edge_pre,
                    parent,
                    parent_row,
                    parent_col,
                )

                post_parent, post_row, post_col, b_cache = (
                    self._apply_tree_level_attention_append(
                        raw_all[level + 1, :width],
                        edge_pre,
                        active,
                        parent,
                        b_cache,
                        edge_row=parent_row,
                        edge_col=parent_col,
                        sequence_axis_name=sequence_axis_name,
                        sequence_mesh=sequence_mesh,
                    )
                )
                post_all = post_all.at[level + 1, parent].set(post_parent)
                post_row = constrain_row_value(post_row)
                post_col = constrain_column_value(post_col)
                edge_post = _replace_square_row_column(
                    edge_post,
                    parent,
                    post_row,
                    post_col,
                )
                return raw_all, post_all, (edge_pre, edge_post, b_cache)

            nodes_raw, nodes_post, levels_mut[level] = jax.lax.cond(
                create,
                do_create,
                lambda operand: operand,
                (nodes_raw, nodes_post, levels_mut[level]),
            )

            nodes_raw = constrain_nodes(nodes_raw)
            nodes_post = constrain_nodes(nodes_post)
            levels_mut[level] = constrain_level(levels_mut[level])
        return nodes_raw, nodes_post, tuple(levels_mut)

    def _tree_prefix_scan(self, seq, mask, pair_route, *, clock_mask=None, g=None):
        n = seq.shape[0]
        n_pad = _route_next_pow2(n)
        depth = n_pad.bit_length() - 1
        dtype = seq.dtype
        pad = n_pad - n
        clock_mask = mask if clock_mask is None else clock_mask
        clock_depth = self._tree_clock_depth_from_mask(clock_mask)
        nodes0 = jnp.pad(seq, ((0, pad), (0, 0)))
        valid0 = jnp.pad(mask.astype(dtype), (0, pad))
        nodes0 = jnp.where(
            valid0.astype(bool)[:, None],
            _tree_sphere(nodes0),
            jnp.zeros_like(nodes0),
        )
        clock_valid0 = jnp.pad(clock_mask.astype(dtype), (0, pad))
        tree_has_merge = jnp.sum(valid0.astype(jnp.int32)) > 1
        edge0 = jnp.pad(pair_route, ((0, pad), (0, pad), (0, 0)))
        edge0_active = valid0.astype(bool)[:, None] & valid0.astype(bool)[None, :]
        edge0 = jnp.where(
            edge0_active[..., None],
            _tree_sphere(edge0),
            jnp.zeros_like(edge0),
        )
        if depth == 0:
            levels = nodes0[None, :, :]
            valids = valid0[None, :]
            edges = edge0[None, :, :, :]
            scan_nodes = jnp.zeros((0,) + nodes0.shape, dtype=dtype)
            return levels, valids, valids, edges, scan_nodes
        n_pairs = n_pad // 2
        g_projected = self.tree_merge.project_global(g, tree_has_merge)

        def _split(x):
            xr = x.reshape((n_pairs, 2) + x.shape[1:])
            return xr[:, 0], xr[:, 1]

        def _zpad(x):
            return jnp.concatenate([x, jnp.zeros_like(x)], axis=0)

        def _zpad_edge(x):
            pad_n = n_pad - n_pairs
            return jnp.pad(x, ((0, pad_n), (0, pad_n), (0, 0)))

        pidx = jnp.arange(n_pairs, dtype=jnp.int32)

        def body(state, xs_lv):
            level_idx, depth_feats_lv = xs_lv
            nodes, valid, edge_state, clock_valid = state
            left, right = _split(nodes)
            left_m, right_m = _split(valid)
            clock_left_m, clock_right_m = _split(clock_valid)
            clock_pair_active = (
                clock_left_m + clock_right_m - clock_left_m * clock_right_m
            )
            pair_base = jnp.maximum(
                jnp.sum(clock_pair_active.astype(jnp.int32)),
                jnp.asarray(2, dtype=jnp.int32),
            )
            e_rs = edge_state.reshape(n_pairs, 2, n_pairs, 2, self.d_model)
            merged, out_mask, genuine = self.tree_merge(
                left,
                right,
                left_m,
                right_m,
                e_rs[pidx, 0, pidx, 1, :],
                e_rs[pidx, 1, pidx, 0, :],
                level_idx,
                pidx,
                pair_base,
                clock_depth,
                depth_feats=depth_feats_lv,
                g=g,
                g_structural_mask=tree_has_merge,
                g_projected=g_projected,
            )
            valid_pair = valid.reshape(n_pairs, 2).astype(dtype)
            weights = valid_pair[:, :, None, None] * valid_pair[None, None, :, :]
            cell_count = jnp.sum(weights, axis=(1, 3))
            denom = jnp.maximum(
                cell_count,
                jnp.asarray(1.0, dtype=dtype),
            )
            edge_parent = (
                jnp.sum(e_rs * weights[..., None], axis=(1, 3)) / denom[..., None]
            )
            e00 = e_rs[:, 0, :, 0, :]
            e01 = e_rs[:, 0, :, 1, :]
            e10 = e_rs[:, 1, :, 0, :]
            e11 = e_rs[:, 1, :, 1, :]
            edge_keep = (genuine[:, None] * genuine[None, :]).astype(bool)

            def _edge_row(e0, e1, e2, e3, c0, c1, keep_row):
                return jax.vmap(
                    lambda x0, x1, x2, x3, d0, d1, keep: self.tree_edge_merge(
                        x0,
                        x1,
                        x2,
                        x3,
                        c0,
                        c1,
                        d0,
                        d1,
                        kfac_structural_mask=keep,
                        kfac_scan_shared=True,
                    )
                )(e0, e1, e2, e3, left, right, keep_row)

            edge_proposal = jax.vmap(_edge_row)(
                e00,
                e01,
                e10,
                e11,
                left,
                right,
                edge_keep,
            )
            edge_updated = self.tree_edge_merge.apply_skip(
                edge_parent,
                edge_proposal,
                kfac_structural_mask=edge_keep,
                kfac_scan_shared=True,
            )
            edge_parent = jnp.where(
                edge_keep[..., None],
                edge_updated,
                edge_parent,
            )
            edge_parent = jnp.where(
                (cell_count > 1)[..., None],
                _tree_sphere(edge_parent),
                edge_parent,
            )
            merged_skip = merged
            edge_skip = edge_parent
            merged, edge_parent = self._apply_tree_level_attention(
                merged,
                edge_parent,
                genuine,
                level_idx,
            )
            merged = jnp.where(
                genuine.astype(bool)[:, None],
                _tree_sphere(merged),
                merged_skip,
            )
            edge_update_mask = (
                genuine.astype(bool)[:, None] & genuine.astype(bool)[None, :]
            )
            idx = jnp.arange(n_pairs, dtype=jnp.int32)
            edge_update_mask = edge_update_mask & (idx[None, :] <= idx[:, None])
            edge_parent = jnp.where(
                edge_update_mask[..., None],
                _tree_sphere(edge_parent),
                edge_skip,
            )
            next_state = (
                _zpad(merged),
                _zpad(out_mask),
                _zpad_edge(edge_parent),
                _zpad(clock_pair_active),
            )
            ys = (next_state[0], next_state[1], _zpad(genuine), next_state[2])
            return next_state, ys

        depth_feat_levels = _tree_ngpt_level_counts(
            clock_valid0,
            n_pairs,
            depth,
            dtype,
        )
        (_nodes, _valid, _edge, _clock_valid), ys = jax.lax.scan(
            body,
            (nodes0, valid0, edge0, clock_valid0),
            (jnp.arange(depth, dtype=jnp.int32), depth_feat_levels),
        )
        nodes_y, valid_y, genuine_y, edge_y = ys
        tree_levels = jnp.concatenate([nodes0[None, :, :], nodes_y], axis=0)
        valid_levels = jnp.concatenate([valid0[None, :], valid_y], axis=0)
        genuine_levels = jnp.concatenate([valid0[None, :], genuine_y], axis=0)
        edge_levels = jnp.concatenate([edge0[None, :, :, :], edge_y], axis=0)
        return tree_levels, valid_levels, genuine_levels, edge_levels, nodes_y

    def _source_edge_levels(self, pair_msg, mask):
        n = pair_msg.shape[0]
        n_dst = pair_msg.shape[1]
        n_pad = _route_next_pow2(n)
        depth = n_pad.bit_length() - 1
        dtype = pair_msg.dtype
        pad = n_pad - n
        edge_state = jnp.pad(pair_msg, ((0, pad), (0, 0), (0, 0)))
        valid = jnp.pad(mask.astype(dtype), (0, pad))
        levels = [edge_state]
        n_pairs = n_pad // 2
        for _level in range(depth):
            e_rs = edge_state.reshape(n_pairs, 2, n_dst, self.d_model)
            v_rs = valid.reshape(n_pairs, 2)
            weights = v_rs[:, :, None, None]
            denom = jnp.maximum(
                jnp.sum(v_rs, axis=1),
                jnp.asarray(1.0, dtype=dtype),
            )
            parent = jnp.sum(e_rs * weights, axis=1) / denom[:, None, None]
            valid_parent = v_rs[:, 0] + v_rs[:, 1] - v_rs[:, 0] * v_rs[:, 1]
            edge_state = jnp.concatenate([parent, jnp.zeros_like(parent)], axis=0)
            valid = jnp.concatenate(
                [valid_parent, jnp.zeros_like(valid_parent)], axis=0
            )
            levels.append(edge_state)
        return jnp.stack(levels, axis=0)

    def _prefix_cover(self, n: int):
        n_pad = _route_next_pow2(n)
        depth = n_pad.bit_length() - 1
        if depth == 0:
            return (
                jnp.zeros((n, 0), dtype=jnp.int32),
                jnp.zeros((n, 0), dtype=jnp.int32),
                jnp.zeros((n, 0), dtype=bool),
            )
        t = jnp.arange(n, dtype=jnp.int32)
        start = jnp.zeros((n,), dtype=jnp.int32)
        levels = []
        nodes = []
        valids = []
        for bit in range(depth - 1, -1, -1):
            take = (jnp.right_shift(t, bit) & 1) == 1
            node = jnp.right_shift(start, bit)
            levels.append(jnp.where(take, jnp.asarray(bit, jnp.int32), 0))
            nodes.append(jnp.where(take, node, 0))
            valids.append(take)
            start = start + jnp.where(take, jnp.asarray(1 << bit, jnp.int32), 0)
        level_arr = jnp.stack(levels, axis=1)
        node_arr = jnp.stack(nodes, axis=1)
        valid_arr = jnp.stack(valids, axis=1)
        prefix_width = _route_next_pow2(depth)
        pad = prefix_width - depth
        if pad:
            level_arr = jnp.pad(level_arr, ((0, 0), (0, pad)))
            node_arr = jnp.pad(node_arr, ((0, 0), (0, pad)))
            valid_arr = jnp.pad(valid_arr, ((0, 0), (0, pad)), constant_values=False)
        return level_arr, node_arr, valid_arr

    def _segment_weights(self, cover_level, cover_node, cover_valid, mask, dtype):
        n = mask.shape[0]
        idx = jnp.arange(n, dtype=jnp.int32)
        leaf_node = jnp.right_shift(idx[None, None, :], cover_level[..., None])
        member = (
            (leaf_node == cover_node[..., None])
            & cover_valid[..., None]
            & mask.astype(bool)[None, None, :]
        )
        weights = member.astype(dtype)
        denom = jnp.maximum(
            jnp.sum(weights, axis=-1, keepdims=True),
            jnp.asarray(1.0, dtype=dtype),
        )
        return weights / denom

    def _tree_prefix_context_all(
        self,
        seq,
        mask,
        pair_route,
        pair_to_nodes,
        route_ids,
        g=None,
    ):
        n = seq.shape[0]
        dtype = seq.dtype
        cover_level, cover_node, cover_valid = self._prefix_cover(n)
        depth = cover_level.shape[1]
        if depth == 0:
            return (
                jnp.zeros((n, 0, self.d_model), dtype=dtype),
                jnp.zeros((n, n, 0, self.d_model), dtype=dtype),
                jnp.zeros((n, 0, 0, self.d_model), dtype=dtype),
                jnp.zeros((n, 0), dtype=bool),
                None,
            )
        tree_levels, _valid_levels, genuine_levels, _edge_levels, _ys = (
            self._tree_prefix_scan(seq, mask, pair_route, g=g)
        )
        source_to_nodes = self._source_edge_levels(pair_to_nodes, mask)
        prefix_nodes = tree_levels[cover_level, cover_node]

        prefix_mask = (
            cover_valid
            & (genuine_levels[cover_level, cover_node] > 0)
            & mask.astype(bool)[:, None]
        )
        source_nodes = source_to_nodes[cover_level, cover_node]
        cand_prefix_edge = jnp.transpose(source_nodes, (0, 2, 1, 3))

        source_route = jnp.take(source_nodes, route_ids, axis=-2)
        dst_weights = self._segment_weights(
            cover_level,
            cover_node,
            cover_valid,
            mask,
            dtype,
        )
        prefix_prefix_edge = jnp.einsum("tlsd,tms->tlmd", source_route, dst_weights)
        g_rows = self._causal_prefix_g(g, prefix_nodes, prefix_mask)
        return (prefix_nodes, cand_prefix_edge, prefix_prefix_edge, prefix_mask, g_rows)

    def _causal_prefix_g(self, g, prefix_nodes, prefix_mask):

        if prefix_nodes.ndim == 2:
            update_active = jnp.any(prefix_mask.astype(bool))
            return self.g_step_update(
                g,
                self.g_step_pool(
                    g,
                    prefix_nodes,
                    prefix_mask.astype(prefix_nodes.dtype),
                    kfac_structural_mask=prefix_mask.astype(bool),
                    kfac_update_mask=update_active,
                    kfac_repeat_ndim=1,
                ),
                update_mask=update_active,
                kfac_structural_mask=update_active,
                kfac_repeat_ndim=0,
            )

        pool_query_active = jnp.any(prefix_mask.astype(bool))
        return jax.vmap(
            lambda pn, pm: self.g_step_update(
                g,
                self.g_step_pool(
                    g,
                    pn,
                    pm,
                    kfac_structural_mask=pm.astype(bool),
                    kfac_update_mask=pool_query_active,
                    kfac_repeat_ndim=2,
                ),
                update_mask=jnp.any(pm.astype(bool)),
                kfac_structural_mask=jnp.any(pm.astype(bool)),
                kfac_g_structural_mask=pool_query_active,
                kfac_repeat_ndim=1,
            )
        )(prefix_nodes, prefix_mask.astype(prefix_nodes.dtype))

    def _tree_prefix_context_row(
        self,
        seq,
        mask,
        pair_route,
        pair_to_nodes,
        route_ids,
        t,
        *,
        clock_mask=None,
        g=None,
        source_edge_frontier=None,
        source_edge_counts=None,
        sequence_axis_name=None,
        sequence_mesh=None,
    ):
        n = seq.shape[0]
        dtype = seq.dtype
        cover_level, cover_node, cover_valid = self._prefix_cover(n)
        depth = cover_level.shape[1]
        if depth == 0:
            return (
                jnp.zeros((0, self.d_model), dtype=dtype),
                jnp.zeros((n, 0, self.d_model), dtype=dtype),
                jnp.zeros((0, 0, self.d_model), dtype=dtype),
                jnp.zeros((0,), dtype=bool),
                None,
            )
        tree_levels, _valid_levels, genuine_levels, _edge_levels, _ys = (
            self._tree_prefix_scan(seq, mask, pair_route, clock_mask=clock_mask, g=g)
        )
        cl = cover_level[t]
        cn = cover_node[t]
        cv = cover_valid[t]
        prefix_nodes = tree_levels[cl, cn]
        row_active = (
            mask[t].astype(bool) if clock_mask is None else clock_mask[t].astype(bool)
        )
        prefix_mask = cv & (genuine_levels[cl, cn] > 0) & row_active
        if source_edge_frontier is None:
            source_to_nodes = self._source_edge_levels(pair_to_nodes, mask)
            source_nodes = source_to_nodes[cl, cn]
        else:
            if source_edge_counts is None:
                raise ValueError(
                    "source_edge_counts is required with source_edge_frontier"
                )
            source_sums = source_edge_frontier[cl]
            source_counts = source_edge_counts[cl]
            source_nodes = (
                source_sums
                / jnp.maximum(
                    source_counts,
                    jnp.asarray(1.0, dtype=dtype),
                )[:, None, None]
            )
            source_nodes = jnp.where(
                (cv & (source_counts > 0))[:, None, None],
                source_nodes,
                jnp.zeros_like(source_nodes),
            )
        dst_weights = self._segment_weights(
            cl[None, :],
            cn[None, :],
            cv[None, :],
            mask,
            dtype,
        )[0]

        weights_by_candidate = (
            jnp.zeros_like(dst_weights).at[:, route_ids].add(dst_weights)
        )
        if sequence_axis_name is not None:
            from jax.sharding import NamedSharding, PartitionSpec as P

            def _sharding(*axes):
                spec = P(*axes)
                return (
                    NamedSharding(sequence_mesh, spec)
                    if sequence_mesh is not None
                    else spec
                )

            source_nodes = jax.lax.with_sharding_constraint(
                source_nodes,
                _sharding(None, sequence_axis_name, None),
            )
            weights_by_candidate = jax.lax.with_sharding_constraint(
                weights_by_candidate,
                _sharding(None, None),
            )
        cand_prefix_edge = jnp.transpose(source_nodes, (1, 0, 2))
        prefix_prefix_edge = jnp.einsum(
            "lcd,mc->lmd",
            source_nodes,
            weights_by_candidate,
        )
        if sequence_axis_name is not None:
            cand_prefix_edge = jax.lax.with_sharding_constraint(
                cand_prefix_edge,
                _sharding(sequence_axis_name, None, None),
            )
            prefix_prefix_edge = jax.lax.with_sharding_constraint(
                prefix_prefix_edge,
                _sharding(None, None, None),
            )
        g_row = self._causal_prefix_g(g, prefix_nodes, prefix_mask)
        return (prefix_nodes, cand_prefix_edge, prefix_prefix_edge, prefix_mask, g_row)

    def _incremental_tree_state(self, n: int, dtype):

        n_pad = _route_next_pow2(n)
        depth = n_pad.bit_length() - 1
        nodes_raw = jnp.zeros((depth + 1, n_pad, self.d_model), dtype=dtype)
        nodes_post = jnp.zeros_like(nodes_raw)
        channels = self.tree_level_fwl.two_hop_channels
        levels = []
        for level in range(depth):
            width = n_pad >> (level + 1)
            edge_pre = jnp.zeros(
                (width, width, self.d_model),
                dtype=dtype,
            )
            edge_post = jnp.zeros_like(edge_pre)
            b_cache = jnp.zeros((width, width, channels), dtype=dtype)
            levels.append((edge_pre, edge_post, b_cache))
        return nodes_raw, nodes_post, tuple(levels)

    def _tree_prefix_context_row_incremental(
        self,
        nodes_post,
        mask,
        route_ids,
        t,
        *,
        g=None,
        source_edge_frontier,
        source_edge_counts,
        sequence_axis_name=None,
        sequence_mesh=None,
    ):

        n = mask.shape[0]
        dtype = nodes_post.dtype
        cover_level, cover_node, cover_valid = self._prefix_cover(n)
        cl = cover_level[t]
        cn = cover_node[t]
        cv = cover_valid[t]
        prefix_nodes = nodes_post[cl, cn]
        prefix_mask = cv & mask[t].astype(bool)
        source_sums = source_edge_frontier[cl]
        source_counts = source_edge_counts[cl]
        source_nodes = (
            source_sums
            / jnp.maximum(
                source_counts,
                jnp.asarray(1.0, dtype=dtype),
            )[:, None, None]
        )
        source_nodes = jnp.where(
            (cv & (source_counts > 0))[:, None, None],
            source_nodes,
            jnp.zeros_like(source_nodes),
        )
        dst_weights = self._segment_weights(
            cl[None, :],
            cn[None, :],
            cv[None, :],
            mask,
            dtype,
        )[0]
        weights_by_candidate = (
            jnp.zeros_like(dst_weights).at[:, route_ids].add(dst_weights)
        )
        if sequence_axis_name is not None:
            from jax.sharding import NamedSharding, PartitionSpec as P

            def _sharding(*axes):
                spec = P(*axes)
                return (
                    NamedSharding(sequence_mesh, spec)
                    if sequence_mesh is not None
                    else spec
                )

            source_nodes = jax.lax.with_sharding_constraint(
                source_nodes,
                _sharding(None, sequence_axis_name, None),
            )
            weights_by_candidate = jax.lax.with_sharding_constraint(
                weights_by_candidate,
                _sharding(None, None),
            )
        cand_prefix_edge = jnp.transpose(source_nodes, (1, 0, 2))
        prefix_prefix_edge = jnp.einsum(
            "lcd,mc->lmd",
            source_nodes,
            weights_by_candidate,
        )
        if sequence_axis_name is not None:
            cand_prefix_edge = jax.lax.with_sharding_constraint(
                cand_prefix_edge,
                _sharding(sequence_axis_name, None, None),
            )
            prefix_prefix_edge = jax.lax.with_sharding_constraint(
                prefix_prefix_edge,
                _sharding(None, None, None),
            )
        g_row = self._causal_prefix_g(g, prefix_nodes, prefix_mask)
        return (
            prefix_nodes,
            cand_prefix_edge,
            prefix_prefix_edge,
            prefix_mask,
            g_row,
        )

    def _tree_prefix_edge_bias(
        self,
        edge_msg,
        params,
        *,
        prefix: str,
        kfac_structural_mask,
        kfac_scan_shared: bool = False,
        kfac_repeat_ndim: int = 2,
        kfac_context_primal_reused_over_walkers: bool = False,
    ):
        ln_s, w1, b1, w2, b2 = params
        x = self._ln(
            ln_s,
            edge_msg,
            tag_id=f"route.tree_prefix.{prefix}.edge_ln",
            kfac_structural_mask=kfac_structural_mask,
            kfac_scan_shared=kfac_scan_shared,
            kfac_repeat_ndim=kfac_repeat_ndim,
            kfac_context_primal_reused_over_walkers=(
                kfac_context_primal_reused_over_walkers
            ),
        )
        x = self._dense(
            w1,
            b1,
            x,
            tag_id=f"route.tree_prefix.{prefix}.edge_bias1",
            kfac_structural_mask=kfac_structural_mask,
            kfac_scan_shared=kfac_scan_shared,
            kfac_repeat_ndim=kfac_repeat_ndim,
            kfac_context_primal_reused_over_walkers=(
                kfac_context_primal_reused_over_walkers
            ),
        )
        x = fused_silu(x)
        return self._dense(
            w2,
            b2,
            x,
            tag_id=f"route.tree_prefix.{prefix}.edge_bias2",
            kfac_structural_mask=kfac_structural_mask,
            kfac_scan_shared=kfac_scan_shared,
            kfac_repeat_ndim=kfac_repeat_ndim,
            kfac_context_primal_reused_over_walkers=(
                kfac_context_primal_reused_over_walkers
            ),
        )

    def _tree_query_seed(self, query_global, route_pos, mask, dtype):

        pos = jnp.asarray(route_pos, dtype=jnp.int32)
        target_shape = pos.shape + (self.d_model,)
        if query_global is None:
            seed = jnp.zeros(target_shape, dtype=dtype)
        else:
            seed = jnp.broadcast_to(
                jnp.asarray(query_global, dtype=dtype),
                target_shape,
            )
        seed = seed + self._route_position_embedding(
            pos,
            dtype,
            mask=mask,
        )
        return seed

    def _tree_prefix_layer(
        self,
        x,
        prefix_edges,
        token_mask,
        attention_mask,
        token_structural_mask,
        pair_structural_mask,
        params,
        *,
        impl,
        g_projection,
    ):
        (
            ln_s,
            w_qkv,
            w_o,
            edge_ln_s,
            edge_w1,
            edge_b1,
            edge_w2,
            edge_b2,
            ffn_ln_s,
            ffn_w1,
            ffn_b1,
            ffn_w2,
            ffn_b2,
        ) = params
        context_reuse = True
        token_kfac = dict(
            kfac_structural_mask=token_structural_mask,
            kfac_repeat_ndim=2,
            kfac_context_primal_reused_over_walkers=context_reuse,
        )
        bsz, n_tokens = x.shape[:2]
        residual_mask = token_mask.astype(x.dtype)[..., None]
        x_ln = self._ln(
            ln_s,
            x,
            tag_id="route.tree_prefix.graph.ln",
            **token_kfac,
        )
        qkv = self._dense_no_bias(
            w_qkv,
            x_ln,
            tag_id="route.tree_prefix.graph.qkv",
            **token_kfac,
        ).reshape(bsz, n_tokens, 3, self.n_heads_kernel, self.d_head)
        q = qkv[:, :, 0]
        k = qkv[:, :, 1]
        v = qkv[:, :, 2]
        edge_bias = self._tree_prefix_edge_bias(
            prefix_edges,
            (edge_ln_s, edge_w1, edge_b1, edge_w2, edge_b2),
            prefix="graph",
            kfac_structural_mask=pair_structural_mask,
            kfac_repeat_ndim=3,
            kfac_context_primal_reused_over_walkers=context_reuse,
        )
        out = self._route_attention(
            q,
            k,
            v,
            edge_bias,
            token_mask,
            impl=impl,
            attention_mask=attention_mask,
        )
        delta = self._dense_no_bias(
            w_o,
            self._collapse_heavy_heads(out),
            tag_id="route.tree_prefix.graph.o",
            **token_kfac,
        )
        x = x + residual_mask * self.tree_prefix_residual_gain * delta
        y = self._ln(
            ffn_ln_s,
            x,
            tag_id="route.tree_prefix.graph.ffn_ln",
            **token_kfac,
        )
        if g_projection is not None:
            y = y + g_projection
        y = self._dense(
            ffn_w1,
            ffn_b1,
            y,
            tag_id="route.tree_prefix.graph.ffn1",
            **token_kfac,
        )
        y = fused_silu(y)
        y = self._dense(
            ffn_w2,
            ffn_b2,
            y,
            tag_id="route.tree_prefix.graph.ffn2",
            **token_kfac,
        )
        return x + residual_mask * self.tree_prefix_residual_gain * y

    def _tree_candidate_layer(
        self,
        cand,
        prefix_nodes,
        cand_prefix_edge,
        prefix_mask,
        cand_mask,
        candidate_structural_mask,
        prefix_structural_mask,
        cross_pair_structural_mask,
        params,
        *,
        impl,
        g_projection,
        sequence_axis_name=None,
        sequence_mesh=None,
    ):
        (
            cand_ln_s,
            pref_ln_s,
            cand_w_qv,
            pref_w_kv,
            w_o,
            edge_ln_s,
            edge_w1,
            edge_b1,
            edge_w2,
            edge_b2,
            ffn_ln_s,
            ffn_w1,
            ffn_b1,
            ffn_w2,
            ffn_b2,
        ) = params
        context_reuse = True
        candidate_kfac = dict(
            kfac_structural_mask=candidate_structural_mask,
            kfac_repeat_ndim=2,
            kfac_context_primal_reused_over_walkers=context_reuse,
        )
        prefix_kfac = dict(
            kfac_structural_mask=prefix_structural_mask,
            kfac_repeat_ndim=2,
            kfac_context_primal_reused_over_walkers=context_reuse,
        )
        bsz, n_cand = cand.shape[:2]
        n_pref = prefix_nodes.shape[1]

        def _seq_constraint(value, *axes):
            if sequence_axis_name is None:
                return value
            from jax.sharding import NamedSharding, PartitionSpec as P

            spec = P(*axes)
            if sequence_mesh is not None:
                spec = NamedSharding(sequence_mesh, spec)
            return jax.lax.with_sharding_constraint(value, spec)

        cand = _seq_constraint(cand, None, sequence_axis_name, None)
        cand_ln = self._ln(
            cand_ln_s,
            cand,
            tag_id="route.tree_prefix.candidate.ln",
            **candidate_kfac,
        )
        if sequence_axis_name is None:
            qv = self._dense_no_bias(
                cand_w_qv,
                cand_ln,
                tag_id="route.tree_prefix.candidate.qv",
                **candidate_kfac,
            ).reshape(bsz, n_cand, 2, self.n_heads_kernel, self.d_head)
            q = qv[:, :, 0]
            v_self = qv[:, :, 1]
        else:
            d_qv = self.n_heads_kernel * self.d_head
            q = jnp.matmul(cand_ln, cand_w_qv[:, :d_qv]).reshape(
                bsz,
                n_cand,
                self.n_heads_kernel,
                self.d_head,
            )
            v_self = jnp.matmul(cand_ln, cand_w_qv[:, d_qv:]).reshape(
                bsz,
                n_cand,
                self.n_heads_kernel,
                self.d_head,
            )
        q = _seq_constraint(q, None, sequence_axis_name, None, None)
        v_self = _seq_constraint(
            v_self,
            None,
            sequence_axis_name,
            None,
            None,
        )
        pref_ln = self._ln(
            pref_ln_s,
            prefix_nodes,
            tag_id="route.tree_prefix.candidate.prefix_ln",
            **prefix_kfac,
        )
        kv = self._dense_no_bias(
            pref_w_kv,
            pref_ln,
            tag_id="route.tree_prefix.candidate.kv",
            **prefix_kfac,
        ).reshape(bsz, n_pref, 2, self.n_heads_kernel, self.d_head)
        kv = _seq_constraint(kv, None, None, None, None, None)
        k = kv[:, :, 0]
        v = kv[:, :, 1]
        edge_bias = self._tree_prefix_edge_bias(
            cand_prefix_edge,
            (edge_ln_s, edge_w1, edge_b1, edge_w2, edge_b2),
            prefix="candidate",
            kfac_structural_mask=cross_pair_structural_mask,
            kfac_repeat_ndim=3,
            kfac_context_primal_reused_over_walkers=context_reuse,
        )
        edge_bias = _seq_constraint(
            edge_bias,
            None,
            sequence_axis_name,
            None,
            None,
        )
        out = self._route_attention(
            q,
            k,
            v,
            edge_bias,
            prefix_mask,
            impl=impl,
            sequence_axis_name=sequence_axis_name,
            sequence_mesh=sequence_mesh,
        )
        out = _seq_constraint(out, None, sequence_axis_name, None, None)
        delta = self._dense_no_bias(
            w_o,
            self._collapse_heavy_heads(out),
            tag_id="route.tree_prefix.candidate.o",
            **candidate_kfac,
        )
        cand = cand + cand_mask * self.tree_candidate_residual_gain * delta
        y = self._ln(
            ffn_ln_s,
            cand,
            tag_id="route.tree_prefix.candidate.ffn_ln",
            **candidate_kfac,
        )
        if g_projection is not None:
            y = y + g_projection
        y = self._dense(
            ffn_w1,
            ffn_b1,
            y,
            tag_id="route.tree_prefix.candidate.ffn1",
            **candidate_kfac,
        )
        y = fused_silu(y)
        y = self._dense(
            ffn_w2,
            ffn_b2,
            y,
            tag_id="route.tree_prefix.candidate.ffn2",
            **candidate_kfac,
        )
        return cand + cand_mask * self.tree_candidate_residual_gain * y

    def _apply_tree_prefix_layers(
        self,
        prefix_nodes,
        prefix_edges,
        prefix_mask,
        g=None,
        query_seed=None,
        row_mask=None,
    ):

        wants_query = query_seed is not None
        single = prefix_nodes.ndim == 2
        if single:
            prefix_nodes = prefix_nodes[None, :, :]
            prefix_edges = prefix_edges[None, :, :, :]
            prefix_mask = prefix_mask[None, :]
            if wants_query:
                query_seed = query_seed[None, :]
            if row_mask is not None:
                row_mask = jnp.asarray(row_mask, dtype=bool).reshape(1)
        if wants_query:
            assert query_seed is not None
            query_seed = jnp.broadcast_to(
                query_seed,
                prefix_nodes.shape[:-2] + (self.d_model,),
            )
            n_pref = prefix_nodes.shape[1]
            x = jnp.concatenate([prefix_nodes, query_seed[:, None, :]], axis=1)
            prefix_edges = jnp.pad(
                prefix_edges,
                ((0, 0), (0, 1), (0, 1), (0, 0)),
            )
            token_mask = jnp.concatenate(
                [
                    prefix_mask.astype(bool),
                    jnp.ones((prefix_mask.shape[0], 1), dtype=bool),
                ],
                axis=1,
            )
            token_idx = jnp.arange(n_pref + 1, dtype=jnp.int32)
            query_row = token_idx == n_pref
            cover_key = token_idx < n_pref
            attention_mask = token_mask[:, None, :] & (
                query_row[None, :, None] | cover_key[None, None, :]
            )
        else:
            n_pref = prefix_nodes.shape[1]
            x = prefix_nodes
            token_mask = prefix_mask.astype(bool)
            attention_mask = None
        row_structural_mask = (
            jnp.ones((x.shape[0],), dtype=bool)
            if row_mask is None
            else jnp.broadcast_to(jnp.asarray(row_mask, dtype=bool), (x.shape[0],))
        )
        token_structural_mask = token_mask.astype(bool) & row_structural_mask[:, None]
        if attention_mask is None:
            pair_structural_mask = (
                token_structural_mask[:, :, None] & token_structural_mask[:, None, :]
            )
        else:
            pair_structural_mask = (
                token_structural_mask[:, :, None]
                & token_structural_mask[:, None, :]
                & attention_mask.astype(bool)
            )
        if self.route_tree_prefix_layers == 0:
            cover = x[:, :n_pref]
            if not wants_query:
                return cover[0] if single else cover
            query = x[:, n_pref]
            return (
                cover[0] if single else cover,
                query[0] if single else query,
            )
        impl = self._resolve_tree_attn_impl()
        from hamiltonzero.model.tree import _tagged_dense_no_bias as _tdnb

        _gg_pref = _tdnb(
            self.g_prefix_ffn_w,
            g,
            tag_id="gladder.route.prefix_fproj",
            pathway="even",
            kfac_structural_mask=jnp.any(row_structural_mask),
            kfac_repeat_ndim=0,
            kfac_context_primal_reused_over_walkers=True,
        ).astype(prefix_nodes.dtype)

        params = self._tree_prefix_layer_params()

        def apply_one(state, layer):
            return self._tree_prefix_layer(
                state,
                prefix_edges,
                token_mask,
                attention_mask,
                token_structural_mask,
                pair_structural_mask,
                layer,
                impl=impl,
                g_projection=_gg_pref,
            )

        for layer in params:
            x = apply_one(x, layer)
        cover = x[:, :n_pref]
        if not wants_query:
            return cover[0] if single else cover
        query = x[:, n_pref]
        return (
            cover[0] if single else cover,
            query[0] if single else query,
        )

    def _apply_tree_candidate_layers(
        self,
        base,
        prefix_nodes,
        cand_prefix_edge,
        prefix_mask,
        mask,
        g_rows=None,
        candidate_mask=None,
        sequence_axis_name=None,
        sequence_mesh=None,
    ):
        if self.route_tree_prefix_candidate_layers == 0 or prefix_nodes.shape[-2] == 0:
            return base
        single = base.ndim == 2
        if single:
            base = base[None, :, :]
            prefix_nodes = prefix_nodes[None, :, :]
            cand_prefix_edge = cand_prefix_edge[None, :, :, :]
            prefix_mask = prefix_mask[None, :]
            if candidate_mask is not None:
                candidate_mask = jnp.asarray(candidate_mask, dtype=bool)[None, :]
        cand = base
        impl = self._resolve_tree_attn_impl()
        cand_mask = mask.astype(cand.dtype)[None, :, None]
        candidate_structural_mask = (
            jnp.broadcast_to(mask.astype(bool), cand.shape[:2])
            if candidate_mask is None
            else jnp.broadcast_to(
                jnp.asarray(candidate_mask, dtype=bool), cand.shape[:2]
            )
        )
        row_structural_mask = jnp.any(candidate_structural_mask, axis=-1)
        prefix_structural_mask = prefix_mask.astype(bool) & row_structural_mask[:, None]
        cross_pair_structural_mask = (
            candidate_structural_mask[:, :, None] & prefix_structural_mask[:, None, :]
        )
        from hamiltonzero.model.tree import _tagged_dense_no_bias as _tdnb

        _gg = _tdnb(
            self.g_cand_ffn_w,
            g_rows,
            tag_id="gladder.route.cand_fproj",
            pathway="even",
            kfac_structural_mask=(
                row_structural_mask
                if jnp.ndim(g_rows) > 1
                else jnp.any(row_structural_mask)
            ),
            kfac_repeat_ndim=(1 if jnp.ndim(g_rows) > 1 else 0),
            kfac_context_primal_reused_over_walkers=True,
        ).astype(cand.dtype)
        _gg_cand = _gg[..., None, :] if _gg.ndim == cand.ndim - 1 else _gg
        params = self._tree_candidate_layer_params()

        def apply_one(state, layer):
            return self._tree_candidate_layer(
                state,
                prefix_nodes,
                cand_prefix_edge,
                prefix_mask,
                cand_mask,
                candidate_structural_mask,
                prefix_structural_mask,
                cross_pair_structural_mask,
                layer,
                impl=impl,
                g_projection=_gg_cand,
                sequence_axis_name=sequence_axis_name,
                sequence_mesh=sequence_mesh,
            )

        for layer in params:
            cand = apply_one(cand, layer)
        return cand[0] if single else cand

    def _tree_enrich_teacher(
        self,
        base,
        seq,
        edge,
        perm,
        mask,
        g=None,
        query_global=None,
    ):
        pair_msg = self._tree_pair_messages(edge, mask)
        pair_route = pair_msg[perm[:, None], perm[None, :]]
        pair_to_nodes = pair_msg[perm, :]
        (prefix_nodes, cand_prefix_edge, prefix_prefix_edge, prefix_mask, g_rows) = (
            self._tree_prefix_context_all(
                seq, mask, pair_route, pair_to_nodes, perm, g=g
            )
        )
        query_seed = self._tree_query_seed(
            query_global,
            jnp.arange(base.shape[0], dtype=jnp.int32),
            mask,
            base.dtype,
        )
        idx = jnp.arange(base.shape[0], dtype=jnp.int32)
        mask_bool = mask.astype(bool)
        pos_of_node = jnp.zeros((base.shape[0],), dtype=jnp.int32).at[perm].set(idx)
        candidate_structural_mask = (
            mask_bool[:, None]
            & mask_bool[None, :]
            & (pos_of_node[None, :] >= idx[:, None])
        )
        prefix_nodes, query = self._apply_tree_prefix_layers(
            prefix_nodes,
            prefix_prefix_edge,
            prefix_mask,
            g=g,
            query_seed=query_seed,
            row_mask=mask,
        )
        candidate = self._apply_tree_candidate_layers(
            base,
            prefix_nodes,
            cand_prefix_edge,
            prefix_mask,
            mask,
            g_rows=g_rows,
            candidate_mask=candidate_structural_mask,
        )
        return candidate, query

    def _apply_tree_prefix_step(
        self,
        base,
        base_cache,
        prefix_ids,
        mask,
        t,
        pair_msg,
        g=None,
        query_global=None,
        picked=None,
        source_edge_frontier=None,
        source_edge_counts=None,
        raw_edge_for_pair_messages=None,
        raw_edge_transpose=None,
        sequence_axis_name=None,
        sequence_mesh=None,
        row_permute_fn=None,
        incremental_tree_state=None,
    ):
        idx = jnp.arange(base.shape[0], dtype=jnp.int32)
        prefix_mask_positions = mask.astype(bool) & (idx < t)
        if incremental_tree_state is None:
            pair_route = (
                pair_msg[prefix_ids[:, None], prefix_ids[None, :]]
                if raw_edge_for_pair_messages is None
                else self._tree_pair_messages_for_route(
                    raw_edge_for_pair_messages,
                    prefix_ids,
                    mask,
                    edge_transpose=raw_edge_transpose,
                    sequence_axis_name=sequence_axis_name,
                    sequence_mesh=sequence_mesh,
                    row_permute_fn=row_permute_fn,
                )
            )
            pair_to_nodes = (
                pair_msg[prefix_ids, :] if source_edge_frontier is None else None
            )
            (prefix_nodes, cand_prefix_edge, prefix_prefix_edge, prefix_mask, g_row) = (
                self._tree_prefix_context_row(
                    base_cache,
                    prefix_mask_positions,
                    pair_route,
                    pair_to_nodes,
                    prefix_ids,
                    t,
                    clock_mask=mask,
                    g=g,
                    source_edge_frontier=source_edge_frontier,
                    source_edge_counts=source_edge_counts,
                    sequence_axis_name=sequence_axis_name,
                    sequence_mesh=sequence_mesh,
                )
            )
        else:
            if source_edge_frontier is None or source_edge_counts is None:
                raise ValueError(
                    "incremental tree context requires source-edge frontiers"
                )
            (prefix_nodes, cand_prefix_edge, prefix_prefix_edge, prefix_mask, g_row) = (
                self._tree_prefix_context_row_incremental(
                    incremental_tree_state[1],
                    mask,
                    prefix_ids,
                    t,
                    g=g,
                    source_edge_frontier=source_edge_frontier,
                    source_edge_counts=source_edge_counts,
                    sequence_axis_name=sequence_axis_name,
                    sequence_mesh=sequence_mesh,
                )
            )
        query_seed = self._tree_query_seed(
            query_global,
            t,
            mask,
            base.dtype,
        )
        prefix_nodes, query = self._apply_tree_prefix_layers(
            prefix_nodes,
            prefix_prefix_edge,
            prefix_mask,
            g=g,
            query_seed=query_seed,
            row_mask=mask[t],
        )
        candidate = self._apply_tree_candidate_layers(
            base,
            prefix_nodes,
            cand_prefix_edge,
            prefix_mask,
            mask,
            g_rows=g_row,
            candidate_mask=(
                mask.astype(bool)
                & mask[t].astype(bool)
                & (
                    jnp.ones_like(mask, dtype=bool)
                    if picked is None
                    else ~picked.astype(bool)
                )
            ),
            sequence_axis_name=sequence_axis_name,
            sequence_mesh=sequence_mesh,
        )
        return candidate, query

    def _teacher_logits(
        self,
        h: Float[Array, "n d_in"],
        edge: Float[Array, "n n d_edge"],
        perm: Int[Array, "n"],
        mask: Int[Array, "n"] | Array,
        *,
        global_feat: Float[Array, "d_global"] | None = None,
        tau: float | Float[Array, ""] = 1.0,
        real_mask: Int[Array, "n"] | Array | None = None,
        first_orbit_ids: QuotientCarrier,
    ) -> Float[Array, "n n"]:
        n = h.shape[0]
        if n > self.max_n:
            raise ValueError(f"route pointer saw N={n} > max_n={self.max_n}")
        dtype = h.dtype
        idx = jnp.arange(n, dtype=jnp.int32)
        mask_bool = mask.astype(bool)
        first_active_idx = self._first_active_index(mask)
        node_state, _node_mean = self._prepare_nodes(h.astype(dtype), mask)
        global_state = self._project_global(
            global_feat,
            dtype,
            structural_mask=jnp.any(mask.astype(bool)),
        )
        base = self._teacher_candidate_states(
            node_state,
            global_state,
            edge,
            perm,
            mask,
            real_mask=real_mask,
        )
        seq = base[idx, perm, :]
        tree_candidate_state, hidden = self._tree_enrich_teacher(
            base,
            seq,
            edge,
            perm,
            mask,
            g=global_state[0],
            query_global=global_state[1],
        )
        candidate_state = self._apply_heavy_teacher(
            tree_candidate_state,
            hidden,
            edge,
            perm,
            mask,
        )
        neg = jnp.asarray(-1.0e30, dtype=dtype)

        pos_of_node = jnp.zeros((n,), dtype=jnp.int32).at[perm].set(idx)
        valid = mask_bool[None, :] & (pos_of_node[None, :] >= idx[:, None])
        first_choice_mask = self._learned_first_choice_mask(mask, real_mask)
        valid = jnp.where(
            (idx == first_active_idx)[:, None],
            first_choice_mask[None, :],
            valid,
        )
        pointer_structural_mask = mask_bool[:, None] & valid
        raw = self._pointer_raw(
            hidden,
            candidate_state,
            structural_mask=pointer_structural_mask,
        )
        raw = raw / jnp.asarray(tau, dtype=dtype)

        identity = jnp.where(
            idx[None, :] == idx[:, None],
            jnp.asarray(0.0, dtype=dtype),
            neg,
        )
        pointer = jnp.where(valid, raw, neg)
        active_scores = pointer
        active_scores = jax.vmap(
            lambda row_i, row: self._apply_quotient_logits(
                row,
                first_orbit_ids,
                row > (neg * jnp.asarray(0.5, dtype=dtype)),
                mask,
                perm,
                row_i,
            )
        )(idx, active_scores)
        return jnp.where(mask_bool[:, None], active_scores, identity)

    def _decode(
        self,
        h: Float[Array, "n d_in"],
        edge: Float[Array, "n n d_edge"],
        mask: Int[Array, "n"] | Array,
        *,
        tau: float | Float[Array, ""],
        key: PRNGKeyArray,
        real_mask: Int[Array, "n"] | Array | None = None,
        first_orbit_ids: QuotientCarrier,
        router_static,
    ):
        n = h.shape[0]
        if n > self.max_n:
            raise ValueError(f"route pointer saw N={n} > max_n={self.max_n}")
        dtype = h.dtype
        idx = jnp.arange(n, dtype=jnp.int32)
        mask_bool = mask.astype(bool)
        rm_bool = (real_mask if real_mask is not None else mask).astype(bool)
        first_active = self._first_active_index(mask)
        neg = jnp.asarray(-1.0e30, dtype=dtype)
        node_state = (router_static.node_input, router_static.node_projected)
        global_state = (router_static.global_input, router_static.global_projected)
        suffix_raw0 = router_static.initial_suffix
        prefix_raw0 = jnp.zeros_like(suffix_raw0)
        virt_count0 = jnp.zeros((), dtype=dtype)
        prefix_order_raw0 = jnp.zeros((n, n, self.d_model), dtype=dtype)
        virt_prefix_order_raw0 = jnp.zeros((n, self.d_model), dtype=dtype)
        order_decay = router_static.order_decay
        virt_decay = router_static.virtual_decay
        pair_msg = router_static.tree_pair_messages
        cross_biases, suffix_biases = self._unpack_heavy_static_bias_tables(
            router_static.static_bias_tables
        )
        noise = jax.random.gumbel(key, (n, n), dtype=dtype)

        perm0 = idx
        picked0 = jnp.zeros((n,), dtype=bool)
        prefix_ids0 = jnp.zeros((n,), dtype=jnp.int32)
        k_cache0 = jnp.zeros(
            (0, n, self.n_heads_kernel, self.d_head),
            dtype=dtype,
        )
        v_cache0 = jnp.zeros_like(k_cache0)
        hidden_cache0 = jnp.zeros((n, self.d_model), dtype=dtype)
        base_cache0 = jnp.zeros((n, self.d_model), dtype=dtype)
        last_hidden0 = jnp.zeros((self.d_model,), dtype=dtype)

        def body(carry, xs):
            (
                perm,
                picked,
                prefix_raw,
                prefix_order_raw,
                suffix_raw,
                virt_prefix_order_raw,
                virt_count,
                prefix_ids,
                k_cache,
                v_cache,
                hidden_cache,
                base_cache,
                last_hidden,
            ) = carry
            t, noise_t = xs
            pref_msg_buf = prefix_order_raw
            virt_msg_buf = virt_prefix_order_raw
            append_step = mask_bool[t]
            first_step = append_step & (t == first_active)
            tri_t = ((idx < t) & mask_bool).astype(dtype)
            decay_t = order_decay[t]
            prefix_order_row = jnp.einsum("s,sd,sid->id", tri_t, decay_t, pref_msg_buf)
            vdecay_t = virt_decay[t]
            virt_order_row = jnp.einsum("s,sd,sd->d", tri_t, vdecay_t, virt_msg_buf)
            base = self._candidate_states_from_summaries(
                node_state,
                global_state,
                prefix_raw,
                prefix_order_row,
                suffix_raw,
                t,
                edge,
                mask,
                prefix_ids,
                virt_prefix_order_raw=virt_order_row,
                virt_count=virt_count,
                real_mask=real_mask,
            )
            candidate_state, pointer_hidden = self._apply_tree_prefix_step(
                base,
                base_cache,
                prefix_ids,
                mask,
                t,
                pair_msg,
                g=global_state[0],
                query_global=global_state[1],
                picked=picked,
            )
            candidate_state = self._apply_heavy_step(
                candidate_state,
                hidden_cache,
                edge,
                prefix_ids,
                picked,
                mask,
                t,
                cross_biases=cross_biases,
                suffix_biases=suffix_biases,
            )
            active_logits = self._pointer_logits(
                pointer_hidden,
                candidate_state,
                picked,
                self._step_choice_mask(first_step, mask, real_mask),
                tau,
            )
            active_logits = self._apply_quotient_logits(
                active_logits,
                first_orbit_ids,
                active_logits > (neg * jnp.asarray(0.5, dtype=dtype)),
                mask,
                prefix_ids,
                t,
            )
            identity_logits = jnp.where(idx == t, jnp.asarray(0.0, dtype=dtype), neg)
            logits = jnp.where(append_step, active_logits, identity_logits)
            select_scores = logits + noise_t
            sampled = jnp.argmax(select_scores).astype(jnp.int32)
            chosen = jnp.where(append_step, sampled, t)

            base_chosen = base[chosen]
            token_in = jnp.where(
                append_step,
                base_chosen,
                jnp.zeros((self.d_model,), dtype=dtype),
            )
            token, k_new, v_new = self._append_token(
                token_in,
                chosen,
                t,
                prefix_ids,
                k_cache,
                v_cache,
                edge,
                mask,
            )
            k_cache = k_new
            v_cache = v_new
            hidden_cache = hidden_cache.at[t].set(pointer_hidden)
            base_cache = base_cache.at[t].set(
                jnp.where(append_step, base_chosen, jnp.zeros_like(base_chosen))
            )
            last_hidden = pointer_hidden
            pref_update = router_static.prefix_edge_messages[chosen]
            suff_update = router_static.suffix_edge_messages[chosen]
            update_mask = append_step.astype(dtype)
            prefix_raw = prefix_raw + update_mask * pref_update
            prefix_order_raw = pref_msg_buf.at[t].set(update_mask * pref_update)
            suffix_raw = suffix_raw - update_mask * suff_update
            virt_update = update_mask * (~rm_bool[chosen]).astype(dtype)
            virt_prefix_order_raw = virt_msg_buf.at[t].set(
                virt_update * self.virt_emb[0],
            )
            virt_count = virt_count + virt_update
            prefix_ids = prefix_ids.at[t].set(chosen)
            picked = picked.at[chosen].set(jnp.where(append_step, True, picked[chosen]))
            perm = perm.at[t].set(chosen)
            return (
                perm,
                picked,
                prefix_raw,
                prefix_order_raw,
                suffix_raw,
                virt_prefix_order_raw,
                virt_count,
                prefix_ids,
                k_cache,
                v_cache,
                hidden_cache,
                base_cache,
                last_hidden,
            ), None

        init = (
            perm0,
            picked0,
            prefix_raw0,
            prefix_order_raw0,
            suffix_raw0,
            virt_prefix_order_raw0,
            virt_count0,
            prefix_ids0,
            k_cache0,
            v_cache0,
            hidden_cache0,
            base_cache0,
            last_hidden0,
        )
        final, _ = jax.lax.scan(body, init, (idx, noise))
        (
            perm,
            _picked,
            _prefix_raw,
            _prefix_order_raw,
            _suffix_raw,
            _virt_po,
            _virt_cnt,
            _prefix_ids,
            _k,
            _v,
            _hidden_cache,
            _base_cache,
            _hidden,
        ) = final
        return perm

    def _decode_greedy_compact(
        self,
        h: Float[Array, "n d_in"],
        edge: Float[Array, "n n d_edge"],
        mask: Int[Array, "n"] | Array,
        *,
        global_feat: Float[Array, "d_global"],
        tau: float | Float[Array, ""],
        real_mask: Int[Array, "n"] | Array,
        sequence_mesh,
        pair_tile_size: int,
        row_permute_fn,
    ):

        n = h.shape[0]
        if n > self.max_n:
            raise ValueError(f"route pointer saw N={n} > max_n={self.max_n}")
        if int(pair_tile_size) < 1:
            raise ValueError("pair_tile_size must be positive")
        sequence_axis_name = "seq"
        dtype = h.dtype
        idx = jnp.arange(n, dtype=jnp.int32)
        mask_bool = mask.astype(bool)
        rm_bool = real_mask.astype(bool)
        first_active = self._first_active_index(mask)
        neg = jnp.asarray(-1.0e30, dtype=dtype)
        edge_transpose = jnp.swapaxes(edge, 0, 1)
        from jax.sharding import NamedSharding, PartitionSpec as P

        edge_transpose = jax.lax.with_sharding_constraint(
            edge_transpose,
            NamedSharding(sequence_mesh, P(sequence_axis_name, None, None)),
        )
        node_state, _node_mean = self._prepare_nodes(h.astype(dtype), mask)
        global_state = self._project_global(
            global_feat,
            dtype,
            structural_mask=jnp.any(mask_bool),
        )
        (
            prefix_raw0,
            _prefix_order_raw0_unused,
            suffix_raw0,
            _virt_po_unused,
            virt_count0,
        ) = self._initial_summaries_streamed(
            edge,
            edge_transpose,
            mask,
            dtype,
            pair_tile_size=pair_tile_size,
            sequence_axis_name=sequence_axis_name,
            sequence_mesh=sequence_mesh,
        )
        pair_msg = None
        cross_biases, suffix_biases = self._heavy_biases_tiled(
            edge,
            edge_transpose,
            pair_tile_size=int(pair_tile_size),
            sequence_axis_name=sequence_axis_name,
            sequence_mesh=sequence_mesh,
        )

        frontier_depth = max(1, _route_next_pow2(n).bit_length() - 1)
        prefix_order_frontier0 = jnp.zeros(
            (frontier_depth, n, self.d_model),
            dtype=dtype,
        )
        virt_order_frontier0 = jnp.zeros(
            (frontier_depth, self.d_model),
            dtype=dtype,
        )
        source_edge_frontier0 = jnp.zeros(
            (frontier_depth, n, self.d_model),
            dtype=dtype,
        )
        source_edge_counts0 = jnp.zeros((frontier_depth,), dtype=dtype)
        tree_state0 = self._incremental_tree_state(n, dtype)

        perm0 = idx
        picked0 = jnp.zeros((n,), dtype=bool)
        prefix_ids0 = jnp.zeros((n,), dtype=jnp.int32)
        k_cache0 = jnp.zeros(
            (0, n, self.n_heads_kernel, self.d_head),
            dtype=dtype,
        )
        v_cache0 = jnp.zeros_like(k_cache0)
        hidden_cache0 = jnp.zeros((n, self.d_model), dtype=dtype)
        base_cache0 = jnp.zeros((n, self.d_model), dtype=dtype)
        logp0 = jnp.asarray(0.0, dtype=jnp.float32)
        seq = sequence_axis_name

        def _seq_sharding(*axes):
            return NamedSharding(sequence_mesh, P(*axes))

        node_state = tuple(
            jax.lax.with_sharding_constraint(x, _seq_sharding(seq, None))
            for x in node_state
        )
        prefix_raw0 = jax.lax.with_sharding_constraint(
            prefix_raw0,
            _seq_sharding(seq, None),
        )
        suffix_raw0 = jax.lax.with_sharding_constraint(
            suffix_raw0,
            _seq_sharding(seq, None),
        )
        prefix_order_frontier0 = jax.lax.with_sharding_constraint(
            prefix_order_frontier0,
            _seq_sharding(None, seq, None),
        )
        source_edge_frontier0 = jax.lax.with_sharding_constraint(
            source_edge_frontier0,
            _seq_sharding(None, seq, None),
        )
        hidden_cache0 = jax.lax.with_sharding_constraint(
            hidden_cache0,
            _seq_sharding(seq, None),
        )
        base_cache0 = jax.lax.with_sharding_constraint(
            base_cache0,
            _seq_sharding(seq, None),
        )
        k_cache0 = jax.lax.with_sharding_constraint(
            k_cache0,
            _seq_sharding(None, seq, None, None),
        )
        v_cache0 = jax.lax.with_sharding_constraint(
            v_cache0,
            _seq_sharding(None, seq, None, None),
        )
        edge_transpose = jax.lax.with_sharding_constraint(
            edge_transpose,
            _seq_sharding(seq, None, None),
        )
        tree_nodes_raw0, tree_nodes_post0, tree_levels0 = tree_state0
        tree_nodes_raw0 = jax.lax.with_sharding_constraint(
            tree_nodes_raw0,
            _seq_sharding(None, seq, None),
        )
        tree_nodes_post0 = jax.lax.with_sharding_constraint(
            tree_nodes_post0,
            _seq_sharding(None, seq, None),
        )
        lanes = int(sequence_mesh.shape[seq])
        constrained_levels = []
        for edge_pre0, edge_post0, b_cache0 in tree_levels0:
            shard_rows = (
                seq
                if edge_pre0.shape[0] >= lanes and edge_pre0.shape[0] % lanes == 0
                else None
            )
            constrained_levels.append(
                (
                    jax.lax.with_sharding_constraint(
                        edge_pre0,
                        _seq_sharding(shard_rows, None, None),
                    ),
                    jax.lax.with_sharding_constraint(
                        edge_post0,
                        _seq_sharding(shard_rows, None, None),
                    ),
                    jax.lax.with_sharding_constraint(
                        b_cache0,
                        _seq_sharding(shard_rows, None, None),
                    ),
                )
            )
        tree_state0 = (
            tree_nodes_raw0,
            tree_nodes_post0,
            tuple(constrained_levels),
        )

        def body(carry, t):
            (
                perm,
                picked,
                prefix_raw,
                prefix_order_frontier,
                suffix_raw,
                virt_order_frontier,
                virt_count,
                source_edge_frontier,
                source_edge_counts,
                tree_state,
                prefix_ids,
                k_cache,
                v_cache,
                hidden_cache,
                base_cache,
                total_logp,
            ) = carry
            append_step = mask_bool[t]
            first_step = append_step & (t == first_active)
            predict_step = append_step & (t != first_active)

            prefix_order_row = _dyadic_lca_frontier_sum(
                prefix_order_frontier,
                t,
                self.order_decay_w[0],
                self.order_decay_b[0],
            )
            virt_order_row = _dyadic_lca_frontier_sum(
                virt_order_frontier,
                t,
                self.virt_decay_w[0],
                self.virt_decay_b[0],
            )
            base = self._candidate_states_from_summaries(
                node_state,
                global_state,
                prefix_raw,
                prefix_order_row,
                suffix_raw,
                t,
                edge,
                mask,
                prefix_ids,
                virt_prefix_order_raw=virt_order_row,
                virt_count=virt_count,
                real_mask=real_mask,
            )
            candidate_state, pointer_hidden = self._apply_tree_prefix_step(
                base,
                base_cache,
                prefix_ids,
                mask,
                t,
                pair_msg,
                g=global_state[0],
                query_global=global_state[1],
                picked=picked,
                source_edge_frontier=source_edge_frontier,
                source_edge_counts=source_edge_counts,
                raw_edge_for_pair_messages=edge,
                raw_edge_transpose=edge_transpose,
                sequence_axis_name=sequence_axis_name,
                sequence_mesh=sequence_mesh,
                row_permute_fn=row_permute_fn,
                incremental_tree_state=tree_state,
            )
            candidate_state = self._apply_heavy_step(
                candidate_state,
                hidden_cache,
                edge,
                prefix_ids,
                picked,
                mask,
                t,
                cross_biases=cross_biases,
                suffix_biases=suffix_biases,
                edge_transpose=edge_transpose,
                sequence_axis_name=sequence_axis_name,
                sequence_mesh=sequence_mesh,
            )
            active_logits = self._pointer_logits(
                pointer_hidden,
                candidate_state,
                picked,
                self._step_choice_mask(first_step, mask, real_mask),
                tau,
            )
            identity_logits = jnp.where(
                idx == t,
                jnp.asarray(0.0, dtype=dtype),
                neg,
            )
            logits = jnp.where(append_step, active_logits, identity_logits)
            sampled = jnp.argmax(logits).astype(jnp.int32)
            chosen = jnp.where(append_step, sampled, t)
            log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
            score_step = self._score_step_for_logp(first_step, predict_step)
            total_logp = total_logp + jnp.where(
                score_step,
                log_probs[chosen],
                0.0,
            )

            base_chosen = base[chosen]
            token_in = jnp.where(
                append_step,
                base_chosen,
                jnp.zeros((self.d_model,), dtype=dtype),
            )
            _token, k_cache, v_cache = self._append_token(
                token_in,
                chosen,
                t,
                prefix_ids,
                k_cache,
                v_cache,
                edge,
                mask,
            )
            hidden_cache = hidden_cache.at[t].set(pointer_hidden)
            base_cache = base_cache.at[t].set(
                jnp.where(append_step, base_chosen, jnp.zeros_like(base_chosen))
            )
            chosen_edge_pair = jnp.concatenate(
                [edge[:, chosen, :], edge_transpose[:, chosen, :]],
                axis=-1,
            )
            pref_update = self._message_mlp(chosen_edge_pair, prefix=True)
            suff_update = self._message_mlp(chosen_edge_pair, prefix=False)

            update_mask = append_step.astype(dtype)
            prefix_raw = prefix_raw + update_mask * pref_update
            prefix_order_frontier = _dyadic_frontier_add(
                prefix_order_frontier,
                update_mask * pref_update,
                t,
            )
            suffix_raw = suffix_raw - update_mask * suff_update
            virt_update = update_mask * (~rm_bool[chosen]).astype(dtype)
            virt_order_frontier = _dyadic_frontier_add(
                virt_order_frontier,
                virt_update * self.virt_emb[0],
                t,
            )
            virt_count = virt_count + virt_update
            source_edge_update = self._tree_pair_message_row(
                edge,
                chosen,
                mask,
                edge_transpose=edge_transpose,
            )
            source_edge_frontier = _dyadic_frontier_add(
                source_edge_frontier,
                update_mask * source_edge_update,
                t,
            )
            source_edge_counts = _dyadic_frontier_add(
                source_edge_counts,
                update_mask,
                t,
            )
            prefix_ids = prefix_ids.at[t].set(chosen)
            tree_state = self._incremental_tree_append(
                tree_state,
                jnp.where(
                    append_step,
                    base_chosen,
                    jnp.zeros_like(base_chosen),
                ),
                chosen,
                t,
                prefix_ids,
                edge,
                mask,
                edge_transpose=edge_transpose,
                g=global_state[0],
                sequence_axis_name=sequence_axis_name,
                sequence_mesh=sequence_mesh,
            )
            picked = picked.at[chosen].set(jnp.where(append_step, True, picked[chosen]))
            perm = perm.at[t].set(chosen)
            next_carry = (
                perm,
                picked,
                prefix_raw,
                prefix_order_frontier,
                suffix_raw,
                virt_order_frontier,
                virt_count,
                source_edge_frontier,
                source_edge_counts,
                tree_state,
                prefix_ids,
                k_cache,
                v_cache,
                hidden_cache,
                base_cache,
                total_logp,
            )
            return next_carry, None

        init = (
            perm0,
            picked0,
            prefix_raw0,
            prefix_order_frontier0,
            suffix_raw0,
            virt_order_frontier0,
            virt_count0,
            source_edge_frontier0,
            source_edge_counts0,
            tree_state0,
            prefix_ids0,
            k_cache0,
            v_cache0,
            hidden_cache0,
            base_cache0,
            logp0,
        )
        final, _ = jax.lax.scan(body, init, idx)
        perm = final[0]
        logp = final[-1]
        return perm, logp

    def beam_search(
        self,
        h: Float[Array, "n d_in"],
        edge: Float[Array, "n n d_edge"],
        mask: Int[Array, "n"] | Array,
        *,
        global_feat: Float[Array, "d_global"] | None = None,
        tau: float | Float[Array, ""] = 1.0,
        beam_width: int = 4,
        real_mask: Int[Array, "n"] | Array | None = None,
        first_orbit_ids: QuotientCarrier,
        router_static=None,
        distributed_axis_name: str | None = None,
        distributed_lanes: int | None = None,
    ):
        B = int(beam_width)
        if B < 1:
            raise ValueError("beam_width must be >= 1")
        if distributed_axis_name is not None:
            lanes = int(distributed_lanes if distributed_lanes is not None else 8)
            if lanes < 1 or B % lanes:
                raise ValueError(
                    f"distributed beam requires beam_width divisible by the "
                    f"lane count; got beam_width={B}, lanes={lanes}"
                )
            if router_static is None:
                raise ValueError("distributed audit beam requires RouterStatic")
        n = h.shape[0]
        if n > self.max_n:
            raise ValueError(f"route pointer saw N={n} > max_n={self.max_n}")
        dtype = h.dtype
        idx = jnp.arange(n, dtype=jnp.int32)
        mask_bool = mask.astype(bool)
        rm_bool = (real_mask if real_mask is not None else mask).astype(bool)
        first_active = self._first_active_index(mask)
        neg = jnp.asarray(-1.0e30, dtype=dtype)
        if router_static is None:
            node_state, _node_mean = self._prepare_nodes(h.astype(dtype), mask)
            global_state = self._project_global(
                global_feat,
                dtype,
                structural_mask=jnp.any(mask.astype(bool)),
            )
            (
                prefix_raw0,
                _prefix_order_raw0_unused,
                suffix_raw0,
                _virt_po_unused,
                virt_count0,
            ) = self._initial_summaries(edge, mask, dtype)
        else:
            node_state = (router_static.node_input, router_static.node_projected)
            global_state = (router_static.global_input, router_static.global_projected)
            suffix_raw0 = router_static.initial_suffix
            prefix_raw0 = jnp.zeros_like(suffix_raw0)
            virt_count0 = jnp.zeros((), dtype=dtype)
        prefix_order_raw0 = jnp.zeros((n, n, self.d_model), dtype=dtype)
        virt_prefix_order_raw0 = jnp.zeros((n, self.d_model), dtype=dtype)
        if router_static is None:
            order_decay = lca_gaussian_decay(
                idx,
                idx,
                self.order_decay_w[0],
                self.order_decay_b[0],
            )
            virt_decay = lca_gaussian_decay(
                idx,
                idx,
                self.virt_decay_w[0],
                self.virt_decay_b[0],
            )
            pair_msg = self._tree_pair_messages(edge, mask)
        else:
            order_decay = router_static.order_decay
            virt_decay = router_static.virtual_decay
            pair_msg = router_static.tree_pair_messages
        if router_static is None:
            cross_biases = self._heavy_cross_biases(edge)
            suffix_biases = self._heavy_suffix_biases(edge)
        else:
            cross_biases, suffix_biases = self._unpack_heavy_static_bias_tables(
                router_static.static_bias_tables
            )

        def repeat(x):
            return jnp.broadcast_to(x, (B,) + x.shape)

        perm0 = repeat(idx)
        picked0 = jnp.zeros((B, n), dtype=bool)
        prefix_ids0 = jnp.zeros((B, n), dtype=jnp.int32)
        k_cache0 = jnp.zeros(
            (B, 0, n, self.n_heads_kernel, self.d_head),
            dtype=dtype,
        )
        v_cache0 = jnp.zeros_like(k_cache0)
        hidden_cache0 = jnp.zeros((B, n, self.d_model), dtype=dtype)
        base_cache0 = jnp.zeros((B, n, self.d_model), dtype=dtype)
        last_hidden0 = jnp.zeros((B, self.d_model), dtype=dtype)
        logp0 = (
            jnp.full((B,), jnp.asarray(-1e9, dtype=jnp.float32), dtype=jnp.float32)
            .at[0]
            .set(0.0)
        )
        beam_ids = jnp.arange(B, dtype=jnp.int32)
        rows = jnp.arange(B, dtype=jnp.int32)

        def body(carry, t):
            (
                perm,
                picked,
                prefix_raw,
                prefix_order_raw,
                suffix_raw,
                virt_prefix_order_raw,
                virt_count,
                prefix_ids,
                k_cache,
                v_cache,
                hidden_cache,
                base_cache,
                last_hidden,
                total_logp,
            ) = carry
            append_step = mask_bool[t]
            first_step = append_step & (t == first_active)
            predict_step = append_step & (t != first_active)

            def states_one(
                pr, por_buf, sr, vpo_buf, vcnt, pids, pk, bc, hcache, hidden
            ):
                tri_t = ((idx < t) & mask_bool).astype(dtype)
                decay_t = order_decay[t]
                por = jnp.einsum("s,sd,sid->id", tri_t, decay_t, por_buf)
                vdecay_t = virt_decay[t]
                virt_order_row = jnp.einsum("s,sd,sd->d", tri_t, vdecay_t, vpo_buf)
                base = self._candidate_states_from_summaries(
                    node_state,
                    global_state,
                    pr,
                    por,
                    sr,
                    t,
                    edge,
                    mask,
                    pids,
                    virt_prefix_order_raw=virt_order_row,
                    virt_count=vcnt,
                    real_mask=real_mask,
                )
                candidate_state, pointer_hidden = self._apply_tree_prefix_step(
                    base,
                    bc,
                    pids,
                    mask,
                    t,
                    pair_msg,
                    g=global_state[0],
                    query_global=global_state[1],
                    picked=pk,
                )
                candidate_state = self._apply_heavy_step(
                    candidate_state,
                    hcache,
                    edge,
                    pids,
                    pk,
                    mask,
                    t,
                    cross_biases=cross_biases,
                    suffix_biases=suffix_biases,
                )
                active_logits = self._pointer_logits(
                    pointer_hidden,
                    candidate_state,
                    pk,
                    self._step_choice_mask(first_step, mask, real_mask),
                    tau,
                )
                active_logits = self._apply_quotient_logits(
                    active_logits,
                    first_orbit_ids,
                    active_logits > (neg * jnp.asarray(0.5, dtype=dtype)),
                    mask,
                    pids,
                    t,
                )
                identity_logits = jnp.where(
                    idx == t, jnp.asarray(0.0, dtype=dtype), neg
                )
                logits = jnp.where(append_step, active_logits, identity_logits)
                return logits, candidate_state, base, pointer_hidden

            if distributed_axis_name is None:
                parent_rows = rows
            else:
                lane = jax.lax.axis_index(distributed_axis_name)
                _per_lane = B // lanes
                parent_rows = lane * _per_lane + jnp.arange(_per_lane, dtype=jnp.int32)
            (
                logits_local,
                _candidate_state_local,
                base_state_local,
                query_state_local,
            ) = jax.vmap(states_one)(
                prefix_raw[parent_rows],
                prefix_order_raw[parent_rows],
                suffix_raw[parent_rows],
                virt_prefix_order_raw[parent_rows],
                virt_count[parent_rows],
                prefix_ids[parent_rows],
                picked[parent_rows],
                base_cache[parent_rows],
                hidden_cache[parent_rows],
                last_hidden[parent_rows],
            )
            score_step = self._score_step_for_logp(first_step, predict_step)
            neg_f32 = jnp.asarray(-1e9, dtype=jnp.float32)

            def expansion_for(logits_arg, parent_total):
                log_probs_arg = jax.nn.log_softmax(
                    logits_arg.astype(jnp.float32), axis=-1
                )
                step_logp_arg = jnp.where(
                    score_step, log_probs_arg, jnp.zeros_like(log_probs_arg)
                )
                expansion_arg = parent_total[:, None] + step_logp_arg
                forced_scores = jnp.where(
                    idx[None, :] == t.astype(jnp.int32),
                    expansion_arg,
                    neg_f32,
                )
                return jnp.where(~append_step, forced_scores, expansion_arg)

            expansion_local = expansion_for(logits_local, total_logp[parent_rows])
            if distributed_axis_name is None:
                expansion_scores = expansion_local
                base_state = base_state_local
                query_state = query_state_local
            else:
                _pl = B // lanes
                base_shape = base_state_local.shape
                query_shape = query_state_local.shape
                payload_parts = [
                    expansion_local.reshape((_pl, -1)),
                ]
                payload_parts.extend(
                    [
                        base_state_local.astype(jnp.float32).reshape((_pl, -1)),
                        query_state_local.astype(jnp.float32).reshape((_pl, -1)),
                    ]
                )
                payload = jnp.concatenate(payload_parts, axis=-1)
                payload = jax.lax.all_gather(
                    payload,
                    distributed_axis_name,
                    axis=0,
                    tiled=True,
                )
                cursor = 0
                expansion_scores = payload[:, cursor : cursor + n]
                cursor += n
                base_size = n * base_shape[-1]
                base_state = (
                    payload[:, cursor : cursor + base_size]
                    .reshape((B, n, base_shape[-1]))
                    .astype(dtype)
                )
                cursor += base_size
                query_state = payload[:, cursor : cursor + query_shape[-1]].astype(
                    dtype
                )

            rank_scores = expansion_scores
            identity_distance = jnp.abs(idx - t.astype(jnp.int32)).astype(jnp.float32)
            rank_scores = rank_scores - identity_distance[None, :] * 1.0e-6
            rank_scores = rank_scores - beam_ids[:, None].astype(jnp.float32) * 1.0e-9
            _rank_top, flat = jax.lax.top_k(rank_scores.reshape((-1,)), B)
            parent = (flat // n).astype(jnp.int32)
            chosen = (flat % n).astype(jnp.int32)
            total_logp = expansion_scores.reshape((-1,))[flat]

            perm = perm[parent]
            picked = picked[parent]
            prefix_raw = prefix_raw[parent]
            prefix_order_raw = prefix_order_raw[parent]
            suffix_raw = suffix_raw[parent]
            virt_prefix_order_raw = virt_prefix_order_raw[parent]
            virt_count = virt_count[parent]
            prefix_ids = prefix_ids[parent]
            k_cache = k_cache[parent]
            v_cache = v_cache[parent]
            hidden_cache = hidden_cache[parent]
            base_cache = base_cache[parent]

            base_chosen = base_state[parent, chosen]
            query_chosen = query_state[parent]
            token_in = jnp.where(
                append_step,
                base_chosen,
                jnp.zeros_like(base_chosen),
            )

            token, k_cache, v_cache = jax.vmap(
                lambda token_b, chosen_b, prefix_ids_b, k_b, v_b: self._append_token(
                    token_b,
                    chosen_b,
                    t,
                    prefix_ids_b,
                    k_b,
                    v_b,
                    edge,
                    mask,
                )
            )(token_in, chosen, prefix_ids, k_cache, v_cache)
            hidden_cache = hidden_cache.at[:, t, :].set(query_chosen)
            base_cache = base_cache.at[:, t, :].set(
                append_step.astype(dtype) * base_chosen
            )
            last_hidden = query_chosen

            if router_static is None:
                chosen_edge_pair = jax.vmap(
                    lambda chosen_b: self._edge_pair_for_source(edge, chosen_b)
                )(chosen)
                pref_update = jax.vmap(
                    lambda pair: self._message_mlp(pair, prefix=True)
                )(chosen_edge_pair)
                suff_update = jax.vmap(
                    lambda pair: self._message_mlp(pair, prefix=False)
                )(chosen_edge_pair)
            else:
                pref_update = router_static.prefix_edge_messages[chosen]
                suff_update = router_static.suffix_edge_messages[chosen]
            update_mask = append_step.astype(dtype)
            prefix_raw = prefix_raw + update_mask * pref_update
            prefix_order_raw = prefix_order_raw.at[:, t].set(update_mask * pref_update)
            suffix_raw = suffix_raw - update_mask * suff_update
            virt_update = update_mask * (~rm_bool[chosen]).astype(dtype)
            virt_prefix_order_raw = virt_prefix_order_raw.at[:, t].set(
                virt_update[:, None] * self.virt_emb[0][None, :],
            )
            virt_count = virt_count + virt_update
            prefix_ids = prefix_ids.at[:, t].set(chosen)
            old_picked = picked[rows, chosen]
            picked = picked.at[rows, chosen].set(
                jnp.where(append_step, True, old_picked)
            )
            perm = perm.at[:, t].set(chosen)
            return (
                perm,
                picked,
                prefix_raw,
                prefix_order_raw,
                suffix_raw,
                virt_prefix_order_raw,
                virt_count,
                prefix_ids,
                k_cache,
                v_cache,
                hidden_cache,
                base_cache,
                last_hidden,
                total_logp,
            ), None

        init = (
            perm0,
            picked0,
            repeat(prefix_raw0),
            repeat(prefix_order_raw0),
            repeat(suffix_raw0),
            repeat(virt_prefix_order_raw0),
            repeat(virt_count0),
            prefix_ids0,
            k_cache0,
            v_cache0,
            hidden_cache0,
            base_cache0,
            last_hidden0,
            logp0,
        )
        final, _ = jax.lax.scan(body, init, idx)
        (
            perm,
            _picked,
            _prefix_raw,
            _prefix_order_raw,
            _suffix_raw,
            _virt_po,
            _virt_cnt,
            _prefix_ids,
            _k,
            _v,
            _hidden_cache,
            _base_cache,
            _hidden,
            logp,
        ) = final
        return perm, logp


__all__ = ["TreePrefixPointerMHSEA"]
