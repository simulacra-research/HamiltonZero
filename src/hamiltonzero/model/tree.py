# Copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from functools import partial
import equinox as eqx
import jax
import jax.numpy as jnp
import kfac_jax
from jaxtyping import Array, Float, Int, PRNGKeyArray


def _kfac_name_kw(tag_id: str) -> dict:
    return {"name": tag_id} if tag_id else {}


_TREE_DYADIC_CLOCK_BASE = 10000.0


def _tree_coord_clock(
    pos, width: int, dtype, *, base: float = _TREE_DYADIC_CLOCK_BASE, scale=None
):
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
        jnp.asarray(base, dtype=jnp.float32), jnp.asarray(2.0, dtype=jnp.float32)
    )
    inv_freq = jnp.exp(
        -jnp.log(base_f) * band / jnp.asarray(max(half, 1), dtype=jnp.float32)
    )
    phase = pos_f[..., None] * inv_freq
    enc = jnp.concatenate([jnp.sin(phase), jnp.cos(phase)], axis=-1)
    return enc[..., : int(width)].astype(dtype)


def _tree_level_clock(pos, width: int, max_depth, dtype):
    del max_depth
    return _tree_coord_clock(pos, width, dtype, base=_TREE_DYADIC_CLOCK_BASE)


def _tree_clock_root_center_from_depth(depth, dtype):
    depth_f = jnp.maximum(jnp.asarray(depth, dtype=jnp.float32), 0.0)
    span = jnp.power(jnp.asarray(2.0, dtype=jnp.float32), depth_f)
    return (span - jnp.asarray(1.0, dtype=jnp.float32)) * 0.5


def _tree_dyadic_segment_clock(
    level_idx, pair_idx, width: int, dtype, *, root_center=None
):
    if int(width) <= 0:
        level_arr = jnp.asarray(level_idx)
        if pair_idx is None:
            return jnp.zeros(level_arr.shape + (0,), dtype=dtype)
        pair_arr = jnp.asarray(pair_idx)
        return jnp.zeros(pair_arr.shape + (0,), dtype=dtype)
    center_width = max(1, int(width) - 2)
    scale_width = int(width) - center_width
    level_f = jnp.asarray(level_idx, dtype=jnp.float32)
    span = jnp.power(
        jnp.asarray(2.0, dtype=jnp.float32),
        level_f + jnp.asarray(1.0, dtype=jnp.float32),
    )
    scale_pos = span
    if pair_idx is None:
        center_pos = (span - jnp.asarray(1.0, dtype=jnp.float32)) * 0.5
    else:
        pair_f = jnp.asarray(pair_idx, dtype=jnp.float32)
        center_pos = pair_f * span + jnp.asarray(0.5, dtype=jnp.float32) * (
            span - jnp.asarray(1.0, dtype=jnp.float32)
        )
    if root_center is not None:
        center_pos = center_pos - jnp.asarray(root_center, dtype=jnp.float32)
    scale_pos = jnp.broadcast_to(scale_pos, center_pos.shape)
    center_clock = _tree_coord_clock(
        center_pos, center_width, dtype, base=_TREE_DYADIC_CLOCK_BASE
    )
    scale_clock = _tree_coord_clock(
        scale_pos, scale_width, dtype, base=_TREE_DYADIC_CLOCK_BASE
    )
    return jnp.concatenate([center_clock, scale_clock], axis=-1)


def _tree_merge_clock(level_idx, pair_idx, pair_base, width: int, max_depth, dtype):
    del pair_base
    root_center = _tree_clock_root_center_from_depth(max_depth, dtype)
    return _tree_dyadic_segment_clock(
        level_idx, pair_idx, width, dtype, root_center=root_center
    )


_TREE_NGPT_DEPTH_FEAT_DIM = 32


def _tree_sphere(x, axis=-1):
    ms = jnp.mean(jnp.square(x), axis=axis, keepdims=True)
    return x * jax.lax.rsqrt(jnp.maximum(ms, 0.0001))


def _tree_depth_count_features(cnt_a, cnt_b, n_total, level, n_levels, dtype):
    a = cnt_a.astype(jnp.float32)
    b = cnt_b.astype(jnp.float32)
    nt = jnp.asarray(n_total, jnp.float32)
    la = jnp.log2(1.0 + a)
    lb = jnp.log2(1.0 + b)
    rem = jnp.log2(1.0 + jnp.maximum(nt - a - b, 0.0))
    counts = jnp.stack([la, lb, rem], axis=-1)
    omg_c = jnp.pi / 2.0 ** jnp.arange(4, dtype=jnp.float32)
    ang_c = counts[..., :, None] * omg_c
    f_c = jnp.concatenate([jnp.sin(ang_c), jnp.cos(ang_c)], axis=-1)
    f_c = f_c.reshape(f_c.shape[:-2] + (24,))
    lv = jnp.asarray(level, jnp.float32)
    lv_rem = jnp.asarray(n_levels, jnp.float32) - 1.0 - lv
    levels = jnp.stack(
        [jnp.broadcast_to(lv, a.shape), jnp.broadcast_to(lv_rem, a.shape)], axis=-1
    )
    omg_l = jnp.pi / 2.0 ** jnp.arange(2, dtype=jnp.float32)
    ang_l = levels[..., :, None] * omg_l
    f_l = jnp.concatenate([jnp.sin(ang_l), jnp.cos(ang_l)], axis=-1)
    f_l = f_l.reshape(f_l.shape[:-2] + (8,))
    return jnp.concatenate([f_c, f_l], axis=-1).astype(dtype)


def _tree_ngpt_level_counts(m0, n_pairs, n_levels, dtype, *, feature_n_levels=None):
    cnt = m0.astype(jnp.float32)
    n_total = jnp.sum(cnt)
    if feature_n_levels is None:
        feature_n_levels = _tree_active_clock_depth(m0)
    feats = []
    for lv in range(n_levels):
        pairs = cnt.reshape(n_pairs, 2)
        feats.append(
            _tree_depth_count_features(
                pairs[:, 0], pairs[:, 1], n_total, lv, feature_n_levels, dtype
            )
        )
        parents = pairs.sum(axis=1)
        cnt = jnp.concatenate([parents, jnp.zeros_like(parents)], axis=0)
    return jnp.stack(feats, axis=0)


from .odd_ops import BiasFreeLinear, HypernetMatrix, Linear, MLP, _RMS
from .readout_leaf_context import lca_alibi_bias, lca_fixed_slopes
from .fused_silu import fused_silu


def _replace_square_row_column(matrix, index, row_value, column_value):
    idx = jnp.arange(matrix.shape[0], dtype=jnp.int32)
    select = idx == jnp.asarray(index, dtype=jnp.int32)
    diagonal = row_value[index]
    column_value = jnp.where(select[:, None], diagonal, column_value)
    matrix = jnp.where(select[:, None, None], row_value[None, :, :], matrix)
    return jnp.where(select[None, :, None], column_value[:, None, :], matrix)


def _quadrilinear_merge(
    T,
    u_a,
    u_b,
    *,
    tag_id: str = "",
    pathway: str | None = None,
    kfac_structural_mask=None,
    kfac_scan_shared: bool = False,
    kfac_repeat_ndim: int = 0,
):
    from ._custom_lap_primitives import custom_lap_active, quadrilinear_merge_p

    if custom_lap_active():
        return quadrilinear_merge_p.bind(T, u_a, u_b)
    G, d_r = (T.shape[0], T.shape[1])
    _odt = jnp.float32
    T_param = T
    T = T if T.dtype == _odt else T.astype(_odt)
    u_a = u_a if u_a.dtype == _odt else u_a.astype(_odt)
    u_b = u_b if u_b.dtype == _odt else u_b.astype(_odt)
    leading = u_a.shape[:-1]
    u_a_2d = u_a.reshape(*leading, G, d_r)
    u_b_2d = u_b.reshape(*leading, G, d_r)
    Tu_a = jnp.einsum("ijkl,...ik->...ijl", T, u_a_2d)
    y_2d = jnp.einsum("...ijl,...il->...ij", Tu_a, u_b_2d)
    y = y_2d.reshape(*leading, G * d_r)
    if kfac_structural_mask is None:
        return y
    from hamiltonzero.optim.spin_blocks import register_structural_quadrilinear_merge

    return register_structural_quadrilinear_merge(
        y,
        u_a,
        u_b,
        T_param,
        kfac_structural_mask,
        scan_shared=kfac_scan_shared,
        repeat_ndim=kfac_repeat_ndim,
        **_kfac_name_kw(tag_id),
    )


def _rownorm_cols(weight):
    nsq = jnp.sum(jnp.square(weight), axis=0, keepdims=True)
    return weight * jax.lax.rsqrt(jnp.maximum(nsq, 0.0001))


def _tagged_dense(
    weight,
    bias,
    x,
    *,
    tag_id: str = "",
    pathway: str,
    weight_eff=None,
    kfac_structural_mask=None,
    kfac_scan_shared: bool = False,
    kfac_repeat_ndim: int = 0,
    kfac_context_primal_reused_over_walkers: bool = False,
):
    from hamiltonzero.model.fp32 import compute_dtype as _compute_dtype

    cdtype = _compute_dtype() if pathway in ("even", "hypernet_eside") else jnp.float32
    w_src = weight if weight_eff is None else weight_eff
    w_compute = w_src.astype(cdtype) if w_src.dtype != cdtype else w_src
    b_compute = bias.astype(cdtype) if bias.dtype != cdtype else bias
    x_compute = x.astype(cdtype) if x.dtype != cdtype else x
    y = x_compute @ w_compute + b_compute
    if kfac_structural_mask is not None:
        from hamiltonzero.optim.blocks import register_structural_dense

        return register_structural_dense(
            y,
            x,
            kfac_structural_mask,
            weight,
            bias,
            scan_shared=kfac_scan_shared,
            repeat_ndim=kfac_repeat_ndim,
            context_primal_reused_over_walkers=kfac_context_primal_reused_over_walkers,
            **_kfac_name_kw(tag_id),
        )
    return kfac_jax.register_dense(y, x, weight, bias, **_kfac_name_kw(tag_id))


def _tagged_dense_no_bias(
    weight,
    x,
    *,
    tag_id: str = "",
    pathway: str,
    kfac_structural_mask=None,
    kfac_scan_shared: bool = False,
    kfac_repeat_ndim: int = 0,
    kfac_context_primal_reused_over_walkers: bool = False,
):
    from hamiltonzero.model.fp32 import compute_dtype as _compute_dtype

    cdtype = _compute_dtype() if pathway in ("even", "hypernet_eside") else jnp.float32
    w_compute = weight.astype(cdtype) if weight.dtype != cdtype else weight
    x_compute = x.astype(cdtype) if x.dtype != cdtype else x
    y = x_compute @ w_compute
    if kfac_structural_mask is not None:
        from hamiltonzero.optim.blocks import register_structural_dense

        return register_structural_dense(
            y,
            x,
            kfac_structural_mask,
            weight,
            scan_shared=kfac_scan_shared,
            repeat_ndim=kfac_repeat_ndim,
            context_primal_reused_over_walkers=kfac_context_primal_reused_over_walkers,
            **_kfac_name_kw(tag_id),
        )
    return kfac_jax.register_dense(y, x, weight, **_kfac_name_kw(tag_id))


def _tagged_ln_eqx_style(
    scale,
    shift,
    x,
    eps: float = 1e-05,
    *,
    tag_id: str = "",
    pathway: str,
    var_floor: float | None = None,
    kfac_structural_mask=None,
    kfac_scan_shared: bool = False,
    kfac_repeat_ndim: int = 0,
    kfac_context_primal_reused_over_walkers: bool = False,
):
    from hamiltonzero.model.fp32 import compute_dtype as _compute_dtype

    out_cdtype = (
        _compute_dtype() if pathway in ("even", "hypernet_eside") else jnp.float32
    )
    in_dtype = x.dtype
    stats_dtype = jnp.promote_types(jnp.float32, in_dtype)
    x_hi = x.astype(stats_dtype) if in_dtype != stats_dtype else x
    mean = jnp.mean(x_hi, axis=-1, keepdims=True)
    centered_hi = x_hi - mean
    var = jnp.mean(centered_hi * centered_hi, axis=-1, keepdims=True)
    if var_floor is not None:
        var = jnp.maximum(var, var_floor)
    normalized_hi = centered_hi * jax.lax.rsqrt(var + eps)
    normalized = (
        normalized_hi.astype(out_cdtype)
        if normalized_hi.dtype != out_cdtype
        else normalized_hi
    )
    scale_compute = scale.astype(out_cdtype) if scale.dtype != out_cdtype else scale
    shift_compute = shift.astype(out_cdtype) if shift.dtype != out_cdtype else shift
    y = normalized * scale_compute + shift_compute
    if kfac_structural_mask is not None:
        from hamiltonzero.optim.blocks import register_structural_scale_and_shift

        return register_structural_scale_and_shift(
            y,
            normalized,
            kfac_structural_mask,
            scale=scale,
            shift=shift,
            scan_shared=kfac_scan_shared,
            repeat_ndim=kfac_repeat_ndim,
            context_primal_reused_over_walkers=kfac_context_primal_reused_over_walkers,
            **_kfac_name_kw(tag_id),
        )
    return kfac_jax.register_scale_and_shift(
        y, normalized, scale, shift, **_kfac_name_kw(tag_id)
    )


def _tagged_rms_eqx_style(
    scale,
    x,
    eps: float = 1e-05,
    *,
    tag_id: str = "",
    pathway: str,
    var_floor: float | None = None,
    kfac_structural_mask=None,
    kfac_scan_shared: bool = False,
    kfac_repeat_ndim: int = 0,
    kfac_context_primal_reused_over_walkers: bool = False,
):
    from hamiltonzero.model.fp32 import compute_dtype as _compute_dtype

    out_cdtype = (
        _compute_dtype() if pathway in ("even", "hypernet_eside") else jnp.float32
    )
    in_dtype = x.dtype
    _stats_dtype = jnp.promote_types(jnp.float32, in_dtype)
    x_hi = x.astype(_stats_dtype) if in_dtype != _stats_dtype else x
    mean_sq = jnp.mean(x_hi * x_hi, axis=-1, keepdims=True)
    if var_floor is not None:
        rsqrt_hi = jax.lax.rsqrt(jnp.maximum(mean_sq, var_floor))
    else:
        rsqrt_hi = jax.lax.rsqrt(mean_sq + eps)
    normalized_hi = x_hi * rsqrt_hi
    normalized = (
        normalized_hi.astype(out_cdtype)
        if normalized_hi.dtype != out_cdtype
        else normalized_hi
    )
    rsqrt = rsqrt_hi.astype(out_cdtype) if rsqrt_hi.dtype != out_cdtype else rsqrt_hi
    s_compute = scale.astype(out_cdtype) if scale.dtype != out_cdtype else scale
    x_compute = x.astype(out_cdtype) if x.dtype != out_cdtype else x
    inv = s_compute * rsqrt
    y = x_compute * inv
    if kfac_structural_mask is not None:
        from hamiltonzero.optim.blocks import register_structural_scale_and_shift

        return register_structural_scale_and_shift(
            y,
            normalized,
            kfac_structural_mask,
            scale=scale,
            scan_shared=kfac_scan_shared,
            repeat_ndim=kfac_repeat_ndim,
            context_primal_reused_over_walkers=kfac_context_primal_reused_over_walkers,
            **_kfac_name_kw(tag_id),
        )
    return kfac_jax.register_scale_and_shift(
        y, normalized, scale=s_compute, shift=None, **_kfac_name_kw(tag_id)
    )


def _tagged_lerp_alpha(
    alpha,
    d,
    *,
    tag_id: str = "",
    pathway: str,
    kfac_structural_mask=None,
    kfac_scan_shared: bool = False,
    kfac_repeat_ndim: int = 0,
    kfac_context_primal_reused_over_walkers: bool = False,
):
    from hamiltonzero.model.fp32 import compute_dtype as _compute_dtype

    out_cdtype = (
        _compute_dtype() if pathway in ("even", "hypernet_eside") else jnp.float32
    )
    d_compute = d.astype(out_cdtype) if d.dtype != out_cdtype else d
    a_compute = alpha.astype(out_cdtype) if alpha.dtype != out_cdtype else alpha
    y = d_compute * a_compute
    if kfac_structural_mask is not None:
        from hamiltonzero.optim.blocks import register_structural_scale_and_shift

        return register_structural_scale_and_shift(
            y,
            d_compute,
            kfac_structural_mask,
            scale=alpha,
            scan_shared=kfac_scan_shared,
            repeat_ndim=kfac_repeat_ndim,
            context_primal_reused_over_walkers=kfac_context_primal_reused_over_walkers,
            **_kfac_name_kw(tag_id),
        )
    return kfac_jax.register_scale_and_shift(
        y, d_compute, scale=alpha, shift=None, **_kfac_name_kw(tag_id)
    )


def _tagged_bounded_ngpt_gain(
    alpha,
    like,
    *,
    max_gain: float = 0.5,
    tag_id: str = "",
    pathway: str,
    kfac_structural_mask=None,
    kfac_scan_shared: bool = False,
    kfac_repeat_ndim: int = 0,
    kfac_context_primal_reused_over_walkers: bool = False,
):
    ones = jax.lax.stop_gradient(like) * jnp.asarray(
        0.0, dtype=like.dtype
    ) + jnp.asarray(1.0, dtype=like.dtype)
    tagged_alpha = _tagged_lerp_alpha(
        alpha,
        ones,
        tag_id=tag_id,
        pathway=pathway,
        kfac_structural_mask=kfac_structural_mask,
        kfac_scan_shared=kfac_scan_shared,
        kfac_repeat_ndim=kfac_repeat_ndim,
        kfac_context_primal_reused_over_walkers=kfac_context_primal_reused_over_walkers,
    )
    return jnp.asarray(max_gain, dtype=tagged_alpha.dtype) * jax.nn.sigmoid(
        tagged_alpha
    )


def _tree_ngpt_residual(
    skip,
    proposal,
    alpha,
    *,
    max_gain: float,
    tag_id: str,
    pathway: str = "even",
    update_mask=None,
    kfac_structural_mask=None,
    kfac_scan_shared: bool = False,
    kfac_repeat_ndim: int = 0,
    kfac_context_primal_reused_over_walkers: bool = False,
):
    skip_n = _tree_sphere(skip)
    proposal_n = _tree_sphere(proposal)
    direction = proposal_n - skip_n
    gain = _tagged_bounded_ngpt_gain(
        alpha,
        direction,
        max_gain=max_gain,
        tag_id=tag_id,
        pathway=pathway,
        kfac_structural_mask=kfac_structural_mask,
        kfac_scan_shared=kfac_scan_shared,
        kfac_repeat_ndim=kfac_repeat_ndim,
        kfac_context_primal_reused_over_walkers=kfac_context_primal_reused_over_walkers,
    )
    updated = _tree_sphere(skip_n + gain * direction)
    if update_mask is None:
        return updated
    active = update_mask.astype(bool)
    while active.ndim < updated.ndim:
        active = active[..., None]
    return jnp.where(active, updated, skip)


def _inline_norm_forward(
    nrm,
    x,
    *,
    pathway: str,
    tag_id=None,
    kfac_structural_mask=None,
    kfac_scan_shared: bool = False,
    kfac_repeat_ndim: int = 0,
    kfac_context_primal_reused_over_walkers: bool = False,
):
    tid = nrm._use_id if tag_id is None else tag_id
    return _tagged_rms_eqx_style(
        nrm.weight,
        x,
        eps=nrm.eps,
        tag_id=tid,
        pathway=pathway,
        kfac_structural_mask=kfac_structural_mask,
        kfac_scan_shared=kfac_scan_shared,
        kfac_repeat_ndim=kfac_repeat_ndim,
        kfac_context_primal_reused_over_walkers=kfac_context_primal_reused_over_walkers,
    )


class LeafBuilder(eqx.Module):
    P_c: Linear
    P_u: HypernetMatrix

    def __init__(
        self,
        d_e: int,
        d_o: int,
        d_c: int,
        d_r: int,
        rank: int,
        *,
        key: PRNGKeyArray,
        d_g: int,
        leaf_hypernet_rank: int | None = None,
        d_m_merge: int | None = None,
    ):
        keys = jax.random.split(key, 5)
        ctx_dim = d_e + d_g
        p_u_rank = leaf_hypernet_rank if leaf_hypernet_rank is not None else rank
        d_m_eff = d_m_merge if d_m_merge is not None else d_r
        if d_m_eff % d_r != 0:
            raise ValueError(
                f"LeafBuilder: d_m_merge={d_m_merge} must be divisible by d_r={d_r} (HT carrier reshape requires G = d_m_eff // d_r)."
            )
        self.P_c = Linear(d_e, d_c, key=keys[0])
        self.P_u = HypernetMatrix(d_o, d_m_eff, ctx_dim, p_u_rank, key=keys[1])

    def conditioner_context(
        self, e: Float[Array, "n d_e"], g_emb: Float[Array, "d_g"]
    ) -> Float[Array, "n d_ctx"]:
        n = e.shape[0]
        g_emb_b = jnp.broadcast_to(g_emb[None, :], (n, g_emb.shape[0]))
        return jnp.concatenate([e, g_emb_b], axis=-1)

    def __call__(self, e, z, *, g_emb, kfac_structural_mask, kfac_odd_structural_mask):
        n = e.shape[0]
        ctx = self.conditioner_context(e, g_emb)
        c = _tagged_dense(
            self.P_c.weight,
            self.P_c.bias,
            e,
            tag_id=self.P_c._use_id,
            pathway="even",
            kfac_structural_mask=kfac_structural_mask,
            kfac_scan_shared=False,
            kfac_repeat_ndim=1,
        )
        u = self.P_u.apply(
            ctx,
            z,
            e_pathway="even",
            kfac_structural_mask=kfac_odd_structural_mask,
            kfac_scan_shared=False,
            kfac_repeat_ndim=1,
            kfac_context_primal_reused_over_walkers=True,
        )
        return (_tree_sphere(c), u, jnp.zeros((n,), dtype=jnp.float32))


class EdgeMergeOp(eqx.Module):
    mlp: MLP
    node_ctx_proj: BiasFreeLinear | None
    alpha: Float[Array, "d_edge"]
    ngpt_alpha_max: float = eqx.field(static=True, default=0.5)

    def __init__(
        self,
        d_edge: int,
        d_c: int,
        *,
        key: PRNGKeyArray,
        alpha_init: float,
        alpha_max: float,
        d_hidden: int | None = None,
        n_blocks: int = 2,
        edge_node_ctx_dim: int | None = None,
    ):
        node_ctx_dim = int(d_c) if edge_node_ctx_dim is None else int(edge_node_ctx_dim)
        if node_ctx_dim < 1:
            raise ValueError(
                f"tree edge_node_ctx_dim must be positive or None, got {edge_node_ctx_dim}"
            )
        self.node_ctx_proj = (
            None
            if node_ctx_dim == int(d_c)
            else BiasFreeLinear(d_c, node_ctx_dim, key=jax.random.fold_in(key, 60782))
        )
        d_in = 4 * d_edge + 4 * node_ctx_dim
        d_hidden_eff = d_hidden if d_hidden is not None else max(d_edge * 2, 64)
        self.mlp = MLP(d_in, d_hidden_eff, d_edge, key=key, n_blocks=n_blocks)
        self.ngpt_alpha_max = float(alpha_max)
        self.alpha = float(alpha_init) * jnp.ones((int(d_edge),))

    def apply_skip(
        self,
        skip,
        proposal,
        *,
        kfac_structural_mask=None,
        kfac_scan_shared: bool = False,
    ):
        return _tree_ngpt_residual(
            skip,
            proposal,
            self.alpha,
            max_gain=self.ngpt_alpha_max,
            tag_id="",
            kfac_structural_mask=kfac_structural_mask,
            kfac_scan_shared=kfac_scan_shared,
            kfac_repeat_ndim=2,
        )

    def __call__(
        self,
        e_2i_2j: Float[Array, "d_edge"],
        e_2i_2j1: Float[Array, "d_edge"],
        e_2i1_2j: Float[Array, "d_edge"],
        e_2i1_2j1: Float[Array, "d_edge"],
        c_2i: Float[Array, "d_c"],
        c_2i1: Float[Array, "d_c"],
        c_2j: Float[Array, "d_c"],
        c_2j1: Float[Array, "d_c"],
        *,
        kfac_structural_mask=None,
        kfac_scan_shared: bool = False,
    ) -> Float[Array, "d_edge"]:
        child_ctx = jnp.stack([c_2i, c_2i1, c_2j, c_2j1], axis=0)
        if self.node_ctx_proj is not None:
            child_structural_mask = (
                None
                if kfac_structural_mask is None
                else jnp.broadcast_to(kfac_structural_mask, (4,))
            )
            child_ctx = _tagged_dense_no_bias(
                self.node_ctx_proj.weight,
                child_ctx,
                tag_id=self.node_ctx_proj._use_id,
                pathway="even",
                kfac_structural_mask=child_structural_mask,
                kfac_scan_shared=kfac_scan_shared,
                kfac_repeat_ndim=3,
            )
        c_2i, c_2i1, c_2j, c_2j1 = child_ctx
        mlp_in = jnp.concatenate(
            [e_2i_2j, e_2i_2j1, e_2i1_2j, e_2i1_2j1, c_2i, c_2i1, c_2j, c_2j1]
        )
        mlp = self.mlp
        x = _tagged_dense(
            mlp.in_proj.weight,
            mlp.in_proj.bias,
            mlp_in,
            tag_id=mlp.in_proj._use_id,
            pathway="even",
            kfac_structural_mask=kfac_structural_mask,
            kfac_scan_shared=kfac_scan_shared,
            kfac_repeat_ndim=2,
        )
        for nrm, l1, l2 in zip(mlp.block_norms, mlp.block_l1s, mlp.block_l2s):
            normed = _inline_norm_forward(
                nrm,
                x,
                pathway="even",
                kfac_structural_mask=kfac_structural_mask,
                kfac_scan_shared=kfac_scan_shared,
                kfac_repeat_ndim=2,
            )
            inner = _tagged_dense(
                l1.weight,
                l1.bias,
                normed,
                tag_id=l1._use_id,
                pathway="even",
                kfac_structural_mask=kfac_structural_mask,
                kfac_scan_shared=kfac_scan_shared,
                kfac_repeat_ndim=2,
            )
            inner_act = mlp._act(inner)
            inner_out = _tagged_dense(
                l2.weight,
                l2.bias,
                inner_act,
                tag_id=l2._use_id,
                pathway="even",
                kfac_structural_mask=kfac_structural_mask,
                kfac_scan_shared=kfac_scan_shared,
                kfac_repeat_ndim=2,
            )
            x = x + mlp.inner_gain * inner_out
        out_normed = _inline_norm_forward(
            mlp.out_norm,
            x,
            pathway="even",
            kfac_structural_mask=kfac_structural_mask,
            kfac_scan_shared=kfac_scan_shared,
            kfac_repeat_ndim=2,
        )
        return _tagged_dense(
            mlp.out_proj.weight,
            mlp.out_proj.bias,
            out_normed,
            tag_id=mlp.out_proj._use_id,
            pathway="even",
            kfac_structural_mask=kfac_structural_mask,
            kfac_scan_shared=kfac_scan_shared,
            kfac_repeat_ndim=2,
        )


class EdgeFWLUpdate(eqx.Module):
    ln_edge: _RMS
    ln_c: _RMS
    node_ctx_proj: BiasFreeLinear | None
    psi_L_in: Linear
    psi_L_out: Linear
    psi_R_in: Linear
    psi_R_out: Linear
    ln_path: _RMS
    ffn_in: Linear
    ffn_out: Linear
    alpha: Float[Array, "d_edge"]
    ngpt_alpha_max: float = eqx.field(static=True, default=0.5)

    def __init__(
        self,
        d_c: int,
        d_edge: int,
        *,
        key: PRNGKeyArray,
        alpha_init: float,
        alpha_max: float,
        channels: int = 64,
        edge_node_ctx_dim: int | None = None,
    ):
        node_ctx_dim = int(d_c) if edge_node_ctx_dim is None else int(edge_node_ctx_dim)
        if node_ctx_dim < 1:
            raise ValueError(
                f"tree edge_node_ctx_dim must be positive or None, got {edge_node_ctx_dim}"
            )
        d_pair = d_edge + 2 * node_ctx_dim
        d_psi_hidden = 2 * channels
        d_ffn_hidden = max(d_edge, 2 * channels)
        (
            k_psi_L_in,
            k_psi_L_out,
            k_psi_L_gate,
            k_psi_R_in,
            k_psi_R_out,
            k_psi_R_gate,
            k_ffn_in,
            k_ffn_out,
        ) = jax.random.split(key, 8)
        self.ln_edge = _RMS(d_edge)
        self.ln_c = _RMS(d_c)
        self.node_ctx_proj = (
            None
            if node_ctx_dim == int(d_c)
            else BiasFreeLinear(d_c, node_ctx_dim, key=jax.random.fold_in(key, 63262))
        )
        self.psi_L_in = Linear(d_pair, d_psi_hidden, key=k_psi_L_in)
        self.psi_L_out = Linear(d_psi_hidden, channels, key=k_psi_L_out)
        self.psi_R_in = Linear(d_pair, d_psi_hidden, key=k_psi_R_in)
        self.psi_R_out = Linear(d_psi_hidden, channels, key=k_psi_R_out)
        self.ln_path = _RMS(channels)
        self.ffn_in = Linear(d_pair + channels, d_ffn_hidden, key=k_ffn_in)
        self.ffn_out = Linear(d_ffn_hidden, d_edge, key=k_ffn_out)
        self.ngpt_alpha_max = float(alpha_max)
        self.alpha = float(alpha_init) * jnp.ones((int(d_edge),))

    def _psi_apply(
        self,
        pair_ij: Float[Array, "n n d_pair"],
        which: str,
        *,
        kfac_structural_mask=None,
        kfac_scan_shared: bool = False,
        kfac_repeat_ndim: int = 2,
    ) -> Float[Array, "n n C"]:
        if which == "L":
            l_in, l_out = (self.psi_L_in, self.psi_L_out)
        else:
            l_in, l_out = (self.psi_R_in, self.psi_R_out)
        hidden = fused_silu(
            _tagged_dense(
                l_in.weight,
                l_in.bias,
                pair_ij,
                tag_id=l_in._use_id,
                pathway="even",
                kfac_structural_mask=kfac_structural_mask,
                kfac_scan_shared=kfac_scan_shared,
                kfac_repeat_ndim=kfac_repeat_ndim,
            )
        )
        return _tagged_dense(
            l_out.weight,
            l_out.bias,
            hidden,
            tag_id=l_out._use_id,
            pathway="even",
            kfac_structural_mask=kfac_structural_mask,
            kfac_scan_shared=kfac_scan_shared,
            kfac_repeat_ndim=kfac_repeat_ndim,
        )

    def __call__(
        self,
        edge: Float[Array, "n n d_edge"],
        c_level: Float[Array, "n d_c"],
        mask: Float[Array, "n"] | None = None,
        *,
        kfac_scan_shared: bool = False,
    ) -> Float[Array, "n n d_edge"]:
        n = c_level.shape[0]
        d_c_dim = c_level.shape[-1]
        node_structural_mask = mask
        full_pair_structural_mask = (
            None if mask is None else mask[:, None] * mask[None, :]
        )
        edge_ln = _inline_norm_forward(
            self.ln_edge,
            edge,
            pathway="even",
            kfac_structural_mask=full_pair_structural_mask,
            kfac_scan_shared=kfac_scan_shared,
            kfac_repeat_ndim=2,
        )
        c_ln = _inline_norm_forward(
            self.ln_c,
            c_level,
            pathway="even",
            kfac_structural_mask=node_structural_mask,
            kfac_scan_shared=kfac_scan_shared,
            kfac_repeat_ndim=1,
        )
        c_ctx = (
            c_ln
            if self.node_ctx_proj is None
            else _tagged_dense_no_bias(
                self.node_ctx_proj.weight,
                c_ln,
                tag_id=self.node_ctx_proj._use_id,
                pathway="even",
                kfac_structural_mask=node_structural_mask,
                kfac_scan_shared=kfac_scan_shared,
                kfac_repeat_ndim=1,
            )
        )
        d_c_dim = c_ctx.shape[-1]
        c_i_b = jnp.broadcast_to(c_ctx[:, None, :], (n, n, d_c_dim))
        c_j_b = jnp.broadcast_to(c_ctx[None, :, :], (n, n, d_c_dim))
        pair_ij = jnp.concatenate([edge_ln, c_i_b, c_j_b], axis=-1)
        A = self._psi_apply(
            pair_ij,
            "L",
            kfac_structural_mask=full_pair_structural_mask,
            kfac_scan_shared=kfac_scan_shared,
        )
        B = self._psi_apply(
            pair_ij,
            "R",
            kfac_structural_mask=full_pair_structural_mask,
            kfac_scan_shared=kfac_scan_shared,
        )
        if mask is not None:
            m = mask.astype(A.dtype)
            A = A * (m[:, None, None] * m[None, :, None])
            B = B * m[None, :, None]
            n_eff = jnp.maximum(jnp.sum(m), 1.0).astype(A.dtype)
        else:
            n_eff = jnp.asarray(float(n), dtype=A.dtype)
        P = jnp.einsum("ikc,kjc->ijc", A, B)
        P = P / jnp.sqrt(n_eff)
        p_ij = _inline_norm_forward(
            self.ln_path,
            P,
            pathway="even",
            kfac_structural_mask=full_pair_structural_mask,
            kfac_scan_shared=kfac_scan_shared,
            kfac_repeat_ndim=2,
        )
        cat = jnp.concatenate([pair_ij, p_ij], axis=-1)
        hidden = fused_silu(
            _tagged_dense(
                self.ffn_in.weight,
                self.ffn_in.bias,
                cat,
                tag_id=self.ffn_in._use_id,
                pathway="even",
                kfac_structural_mask=full_pair_structural_mask,
                kfac_scan_shared=kfac_scan_shared,
                kfac_repeat_ndim=2,
            )
        )
        delta = _tagged_dense(
            self.ffn_out.weight,
            self.ffn_out.bias,
            hidden,
            tag_id=self.ffn_out._use_id,
            pathway="even",
            kfac_structural_mask=full_pair_structural_mask,
            kfac_scan_shared=kfac_scan_shared,
            kfac_repeat_ndim=2,
        )
        return delta

    def apply_residual(
        self,
        edge: Float[Array, "n n d_edge"],
        c_level: Float[Array, "n d_c"],
        mask: Float[Array, "n"] | None = None,
        *,
        kfac_scan_shared: bool = False,
    ) -> Float[Array, "n n d_edge"]:
        delta = self(edge, c_level, mask, kfac_scan_shared=kfac_scan_shared)
        update_mask = None
        if mask is not None:
            m = mask.astype(bool)
            update_mask = m[:, None] & m[None, :]
        return _tree_ngpt_residual(
            edge,
            delta,
            self.alpha,
            max_gain=self.ngpt_alpha_max,
            tag_id="",
            update_mask=update_mask,
            kfac_structural_mask=update_mask,
            kfac_scan_shared=kfac_scan_shared,
            kfac_repeat_ndim=2,
        )


class CausalRouterEdgeFWLUpdate(EdgeFWLUpdate):
    two_hop_channels: int = eqx.field(static=True)

    def __init__(
        self,
        d_c: int,
        d_edge: int,
        *,
        key: PRNGKeyArray,
        alpha_init: float,
        alpha_max: float,
        channels: int = 64,
        edge_node_ctx_dim: int | None = None,
    ):
        super().__init__(
            d_c,
            d_edge,
            key=key,
            alpha_init=alpha_init,
            alpha_max=alpha_max,
            channels=channels,
            edge_node_ctx_dim=edge_node_ctx_dim,
        )
        self.two_hop_channels = int(channels)

    def __call__(
        self,
        edge: Float[Array, "n n d_edge"],
        c_level: Float[Array, "n d_c"],
        mask: Float[Array, "n"] | None = None,
        *,
        kfac_scan_shared: bool = False,
    ) -> Float[Array, "n n d_edge"]:
        n = c_level.shape[0]
        node_structural_mask = mask
        full_pair_structural_mask = (
            None if mask is None else mask[:, None] * mask[None, :]
        )
        idx = jnp.arange(n, dtype=jnp.int32)
        causal_pair = idx[None, :] <= idx[:, None]
        if full_pair_structural_mask is None:
            full_pair_structural_mask = jnp.ones((n, n), dtype=bool)
        causal_pair_structural_mask = (
            full_pair_structural_mask.astype(bool) & causal_pair
        )
        edge_ln = _inline_norm_forward(
            self.ln_edge,
            edge,
            pathway="even",
            kfac_structural_mask=full_pair_structural_mask,
            kfac_scan_shared=kfac_scan_shared,
            kfac_repeat_ndim=2,
        )
        c_ln = _inline_norm_forward(
            self.ln_c,
            c_level,
            pathway="even",
            kfac_structural_mask=node_structural_mask,
            kfac_scan_shared=kfac_scan_shared,
            kfac_repeat_ndim=1,
        )
        c_ctx = (
            c_ln
            if self.node_ctx_proj is None
            else _tagged_dense_no_bias(
                self.node_ctx_proj.weight,
                c_ln,
                tag_id=self.node_ctx_proj._use_id,
                pathway="even",
                kfac_structural_mask=node_structural_mask,
                kfac_scan_shared=kfac_scan_shared,
                kfac_repeat_ndim=1,
            )
        )
        d_c_dim = c_ctx.shape[-1]
        c_i_b = jnp.broadcast_to(c_ctx[:, None, :], (n, n, d_c_dim))
        c_j_b = jnp.broadcast_to(c_ctx[None, :, :], (n, n, d_c_dim))
        pair_ij = jnp.concatenate([edge_ln, c_i_b, c_j_b], axis=-1)
        A = self._psi_apply(
            pair_ij,
            "L",
            kfac_structural_mask=causal_pair_structural_mask,
            kfac_scan_shared=kfac_scan_shared,
        )
        B = self._psi_apply(
            pair_ij,
            "R",
            kfac_structural_mask=full_pair_structural_mask,
            kfac_scan_shared=kfac_scan_shared,
        )
        if mask is not None:
            m = mask.astype(A.dtype)
            A = A * (m[:, None, None] * m[None, :, None])
            B = B * m[None, :, None]
            allowed = causal_pair.astype(A.dtype) * m[None, :]
        else:
            allowed = causal_pair.astype(A.dtype)
        A = A * causal_pair[..., None].astype(A.dtype)
        n_eff = jnp.maximum(jnp.sum(allowed, axis=1), 1.0).astype(A.dtype)
        P = jnp.einsum("ikc,kjc->ijc", A, B)
        P = P / jnp.sqrt(n_eff)[:, None, None]
        P = P * causal_pair[..., None].astype(P.dtype)
        p_ij = _inline_norm_forward(
            self.ln_path,
            P,
            pathway="even",
            kfac_structural_mask=causal_pair_structural_mask,
            kfac_scan_shared=kfac_scan_shared,
            kfac_repeat_ndim=2,
        )
        cat = jnp.concatenate([pair_ij, p_ij], axis=-1)
        hidden = fused_silu(
            _tagged_dense(
                self.ffn_in.weight,
                self.ffn_in.bias,
                cat,
                tag_id=self.ffn_in._use_id,
                pathway="even",
                kfac_structural_mask=causal_pair_structural_mask,
                kfac_scan_shared=kfac_scan_shared,
                kfac_repeat_ndim=2,
            )
        )
        delta = _tagged_dense(
            self.ffn_out.weight,
            self.ffn_out.bias,
            hidden,
            tag_id=self.ffn_out._use_id,
            pathway="even",
            kfac_structural_mask=causal_pair_structural_mask,
            kfac_scan_shared=kfac_scan_shared,
            kfac_repeat_ndim=2,
        )
        delta_mask = causal_pair.astype(delta.dtype)
        if mask is not None:
            delta_mask = delta_mask * (m[:, None] * m[None, :])
        return delta * delta_mask[..., None]

    def append_causal_row(
        self,
        edge: Float[Array, "n n d_edge"],
        c_level: Float[Array, "n d_c"],
        mask: Float[Array, "n"],
        row: Int[Array, ""],
        b_cache: Float[Array, "n n channels"],
        *,
        edge_row: Float[Array, "n d_edge"] | None = None,
        edge_col: Float[Array, "n d_edge"] | None = None,
        sequence_axis_name=None,
        sequence_mesh=None,
    ):
        n = c_level.shape[0]
        idx = jnp.arange(n, dtype=jnp.int32)
        row = jnp.asarray(row, dtype=jnp.int32)
        active = mask.astype(bool)
        allowed = active & (idx <= row)
        if edge_row is None:
            edge_row = edge[row]
        if edge_col is None:
            edge_col = edge[:, row]
        edge_row_ln = _inline_norm_forward(
            self.ln_edge,
            edge_row,
            pathway="even",
            kfac_structural_mask=allowed,
            kfac_repeat_ndim=1,
        )
        edge_col_ln = _inline_norm_forward(
            self.ln_edge,
            edge_col,
            pathway="even",
            kfac_structural_mask=active,
            kfac_repeat_ndim=1,
        )
        c_ln = _inline_norm_forward(
            self.ln_c,
            c_level,
            pathway="even",
            kfac_structural_mask=active,
            kfac_repeat_ndim=1,
        )
        c_ctx = (
            c_ln
            if self.node_ctx_proj is None
            else _tagged_dense_no_bias(
                self.node_ctx_proj.weight,
                c_ln,
                tag_id=self.node_ctx_proj._use_id,
                pathway="even",
                kfac_structural_mask=active,
                kfac_repeat_ndim=1,
            )
        )
        c_row = c_ctx[row]
        c_row_b = jnp.broadcast_to(c_row, c_ctx.shape)
        pair_row = jnp.concatenate([edge_row_ln, c_row_b, c_ctx], axis=-1)
        pair_col = jnp.concatenate([edge_col_ln, c_ctx, c_row_b], axis=-1)
        a_row = self._psi_apply(
            pair_row, "L", kfac_structural_mask=allowed, kfac_repeat_ndim=1
        )
        b_row = self._psi_apply(
            pair_row, "R", kfac_structural_mask=active, kfac_repeat_ndim=1
        )
        b_col = self._psi_apply(
            pair_col, "R", kfac_structural_mask=active, kfac_repeat_ndim=1
        )
        if sequence_axis_name is not None:
            from jax.sharding import NamedSharding, PartitionSpec as P

            lanes = (
                int(sequence_mesh.shape[sequence_axis_name])
                if sequence_mesh is not None
                else 1
            )
            row_spec = P(None, None)
            col_spec = P(
                sequence_axis_name if n >= lanes and n % lanes == 0 else None, None
            )
            if sequence_mesh is not None:
                row_spec = NamedSharding(sequence_mesh, row_spec)
                col_spec = NamedSharding(sequence_mesh, col_spec)
            b_row = jax.lax.with_sharding_constraint(b_row, row_spec)
            b_col = jax.lax.with_sharding_constraint(b_col, col_spec)
        b_cache = _replace_square_row_column(b_cache, row, b_row, b_col)
        a_row = a_row * allowed[:, None].astype(a_row.dtype)
        path = jnp.einsum("kc,kjc->jc", a_row, b_cache)
        n_eff = jnp.maximum(
            jnp.sum(allowed.astype(path.dtype)), jnp.asarray(1.0, dtype=path.dtype)
        )
        path = path / jnp.sqrt(n_eff)
        path = path * allowed[:, None].astype(path.dtype)
        path_ln = _inline_norm_forward(
            self.ln_path,
            path,
            pathway="even",
            kfac_structural_mask=allowed,
            kfac_repeat_ndim=1,
        )
        cat = jnp.concatenate([pair_row, path_ln], axis=-1)
        hidden = fused_silu(
            _tagged_dense(
                self.ffn_in.weight,
                self.ffn_in.bias,
                cat,
                tag_id=self.ffn_in._use_id,
                pathway="even",
                kfac_structural_mask=allowed,
                kfac_repeat_ndim=1,
            )
        )
        delta = _tagged_dense(
            self.ffn_out.weight,
            self.ffn_out.bias,
            hidden,
            tag_id=self.ffn_out._use_id,
            pathway="even",
            kfac_structural_mask=allowed,
            kfac_repeat_ndim=1,
        )
        delta = delta * allowed[:, None].astype(delta.dtype)
        updated = _tree_ngpt_residual(
            edge_row,
            delta,
            self.alpha,
            max_gain=self.ngpt_alpha_max,
            tag_id="",
            update_mask=allowed,
            kfac_structural_mask=allowed,
            kfac_repeat_ndim=1,
        )
        return updated, b_cache

    def apply_residual(
        self,
        edge: Float[Array, "n n d_edge"],
        c_level: Float[Array, "n d_c"],
        mask: Float[Array, "n"] | None = None,
        *,
        kfac_scan_shared: bool = False,
    ) -> Float[Array, "n n d_edge"]:
        delta = self(edge, c_level, mask, kfac_scan_shared=kfac_scan_shared)
        idx = jnp.arange(edge.shape[0], dtype=jnp.int32)
        update_mask = idx[None, :] <= idx[:, None]
        if mask is not None:
            m = mask.astype(bool)
            update_mask = update_mask & m[:, None] & m[None, :]
        return _tree_ngpt_residual(
            edge,
            delta,
            self.alpha,
            max_gain=self.ngpt_alpha_max,
            tag_id="",
            update_mask=update_mask,
            kfac_structural_mask=update_mask,
            kfac_scan_shared=kfac_scan_shared,
            kfac_repeat_ndim=2,
        )


class LevelEdgeAttn(eqx.Module):
    ln_scale: Float[Array, "d_c"]
    w_qkv: Float[Array, "d_c d_qkv"]
    w_o: Float[Array, "d_o_in d_c"]
    bias_mlp: MLP
    ffn_ln_scale: Float[Array, "d_c"]
    ffn_w1: Float[Array, "d_c d_ffn_hidden"]
    ffn_b1: Float[Array, "d_ffn_hidden"]
    ffn_w2: Float[Array, "d_ffn_hidden d_c"]
    ffn_b2: Float[Array, "d_c"]
    alpha_attn: Float[Array, "d_c"]
    alpha_ffn: Float[Array, "d_c"]
    n_heads: int = eqx.field(static=True)
    n_heads_kernel: int = eqx.field(static=True)
    d_attn: int = eqx.field(static=True)
    d_head: int = eqx.field(static=True)
    d_ffn_hidden: int = eqx.field(static=True)
    attn_impl: str = eqx.field(static=True)
    ln_eps: float = eqx.field(static=True)
    max_n: int = eqx.field(static=True)
    rope_base: float = eqx.field(static=True)
    rope_scaling: float = eqx.field(static=True)
    ngpt_alpha_max: float = eqx.field(static=True, default=0.5)
    _use_id_ln: str = eqx.field(static=True, default="")
    _use_id_qkv: str = eqx.field(static=True, default="")
    _use_id_o: str = eqx.field(static=True, default="")
    _use_id_ffn_ln: str = eqx.field(static=True, default="")
    _use_id_ffn1: str = eqx.field(static=True, default="")
    _use_id_ffn2: str = eqx.field(static=True, default="")

    def __init__(
        self,
        d_c: int,
        d_edge: int,
        *,
        key: PRNGKeyArray,
        alpha_init: float,
        alpha_max: float,
        n_heads: int = 4,
        attn_dim: int | None = None,
        attn_impl: str = "mhsea_tuned",
        bias_mlp_hidden: int | None = None,
        bias_mlp_n_blocks: int = 1,
        ffn_d_hidden: int | None = None,
        ln_eps: float = 1e-05,
        max_n: int = 128,
        rope_base: float = 10000.0,
        rope_scaling: float = 1.0,
    ):
        d_attn = int(d_c) if attn_dim is None else int(attn_dim)
        if d_attn < 1:
            raise ValueError(
                f"LevelEdgeAttn attn_dim must be positive or None, got {attn_dim}"
            )
        assert d_attn % n_heads == 0, (
            f"attn_dim ({d_attn}) must be divisible by n_heads ({n_heads})"
        )
        if attn_impl not in ("einsum", "mhsea_tuned"):
            raise ValueError("attn_impl must be 'einsum' or 'mhsea_tuned'")
        n_heads_kernel = 2 * n_heads
        d_head = d_attn // n_heads
        if d_head % 2 != 0:
            raise ValueError(
                f"LevelEdgeAttn requires even d_head for RoPE, got d_head={d_head} (= attn_dim={d_attn} / n_heads={n_heads})"
            )
        if max_n < 2:
            raise ValueError(f"LevelEdgeAttn max_n must be >= 2, got {max_n}")
        if rope_base <= 0.0:
            raise ValueError("LevelEdgeAttn rope_base must be positive")
        if rope_scaling <= 0.0:
            raise ValueError("LevelEdgeAttn rope_scaling must be positive")
        d_qkv_out = n_heads_kernel * d_head
        d_o_in = n_heads * d_head
        k_qkv, k_b, k_f1, k_o, k_f2 = jax.random.split(key, 5)
        self.w_qkv = jax.random.normal(k_qkv, (d_c, 3 * d_qkv_out)) * d_c ** (-0.5)
        self.w_o = jax.random.normal(k_o, (d_o_in, d_c)) * d_o_in ** (-0.5)
        if bias_mlp_hidden is None:
            bias_mlp_hidden = max(32, n_heads_kernel * 2)
        self.bias_mlp = MLP(
            d_edge, bias_mlp_hidden, n_heads_kernel, key=k_b, n_blocks=bias_mlp_n_blocks
        )
        self.ln_scale = jnp.ones((d_c,))
        d_ffn_eff = ffn_d_hidden if ffn_d_hidden is not None else 4 * d_c
        self.ffn_ln_scale = jnp.ones((d_c,))
        self.ffn_w1 = jax.random.normal(k_f1, (d_c, d_ffn_eff)) * d_c ** (-0.5)
        self.ffn_b1 = jnp.zeros((d_ffn_eff,))
        self.ffn_w2 = jax.random.normal(k_f2, (d_ffn_eff, d_c)) * d_ffn_eff ** (-0.5)
        self.ffn_b2 = jnp.zeros((d_c,))
        self.n_heads = n_heads
        self.n_heads_kernel = n_heads_kernel
        self.d_attn = d_attn
        self.d_head = d_head
        self.d_ffn_hidden = d_ffn_eff
        self.attn_impl = attn_impl
        self.ln_eps = ln_eps
        self.max_n = int(max_n)
        self.rope_base = float(rope_base)
        self.rope_scaling = float(rope_scaling)
        self.ngpt_alpha_max = float(alpha_max)
        self.alpha_attn = float(alpha_init) * jnp.ones((int(d_c),))
        self.alpha_ffn = float(alpha_init) * jnp.ones((int(d_c),))

    def __call__(
        self,
        c_level: Float[Array, "n d_c"],
        edge: Float[Array, "n n d_edge"],
        mask: Float[Array, "n"],
        level_idx=None,
        *,
        kfac_scan_shared: bool = False,
    ) -> Float[Array, "n d_c"]:
        n = c_level.shape[0]
        H_k = self.n_heads_kernel
        d_h = self.d_head
        pair_mask = mask[:, None] * mask[None, :]
        x = _tagged_rms_eqx_style(
            self.ln_scale,
            c_level,
            eps=self.ln_eps,
            tag_id=self._use_id_ln,
            pathway="even",
            kfac_structural_mask=mask,
            kfac_scan_shared=kfac_scan_shared,
            kfac_repeat_ndim=1,
        )
        qkv = _tagged_dense_no_bias(
            self.w_qkv,
            x,
            tag_id=self._use_id_qkv,
            pathway="even",
            kfac_structural_mask=mask,
            kfac_scan_shared=kfac_scan_shared,
            kfac_repeat_ndim=1,
        )
        qkv = qkv.reshape(n, 3, H_k, d_h)
        Q = qkv[:, 0]
        K = qkv[:, 1]
        V = qkv[:, 2]
        bmlp = self.bias_mlp
        b = _tagged_dense(
            bmlp.in_proj.weight,
            bmlp.in_proj.bias,
            edge,
            tag_id=bmlp.in_proj._use_id,
            pathway="even",
            kfac_structural_mask=pair_mask,
            kfac_scan_shared=kfac_scan_shared,
            kfac_repeat_ndim=2,
        )
        for nrm, l1, l2 in zip(bmlp.block_norms, bmlp.block_l1s, bmlp.block_l2s):
            normed = _inline_norm_forward(
                nrm,
                b,
                pathway="even",
                kfac_structural_mask=pair_mask,
                kfac_scan_shared=kfac_scan_shared,
                kfac_repeat_ndim=2,
            )
            inner = _tagged_dense(
                l1.weight,
                l1.bias,
                normed,
                tag_id=l1._use_id,
                pathway="even",
                kfac_structural_mask=pair_mask,
                kfac_scan_shared=kfac_scan_shared,
                kfac_repeat_ndim=2,
            )
            inner_act = bmlp._act(inner)
            inner_out = _tagged_dense(
                l2.weight,
                l2.bias,
                inner_act,
                tag_id=l2._use_id,
                pathway="even",
                kfac_structural_mask=pair_mask,
                kfac_scan_shared=kfac_scan_shared,
                kfac_repeat_ndim=2,
            )
            b = b + bmlp.inner_gain * inner_out
        b_normed = _inline_norm_forward(
            bmlp.out_norm,
            b,
            pathway="even",
            kfac_structural_mask=pair_mask,
            kfac_scan_shared=kfac_scan_shared,
            kfac_repeat_ndim=2,
        )
        coup_bias = _tagged_dense(
            bmlp.out_proj.weight,
            bmlp.out_proj.bias,
            b_normed,
            tag_id=bmlp.out_proj._use_id,
            pathway="even",
            kfac_structural_mask=pair_mask,
            kfac_scan_shared=kfac_scan_shared,
            kfac_repeat_ndim=2,
        )
        coup_bias = coup_bias / jnp.sqrt(d_h)
        if level_idx is not None:
            _idx = jnp.arange(n, dtype=jnp.int32)
            _lca = lca_alibi_bias(_idx, _idx, lca_fixed_slopes(H_k, dtype=x.dtype))
            coup_bias = coup_bias + jnp.transpose(_lca, (1, 2, 0))
        impl = self.attn_impl
        from .pallas_attention import (
            mhsea_tuned_edge_attention,
            reference_edge_attention,
        )

        if impl == "einsum":
            out = reference_edge_attention(Q, K, V, coup_bias, mask)
        elif impl == "mhsea_tuned":
            d_head_padded = max(16, d_h)
            pad_amount = d_head_padded - d_h
            scale = jnp.sqrt(jnp.float32(d_head_padded / d_h))
            pad_shape = (n, H_k, pad_amount)
            Q_p = jnp.concatenate([Q * scale, jnp.zeros(pad_shape, Q.dtype)], axis=-1)
            K_p = jnp.concatenate([K, jnp.zeros(pad_shape, K.dtype)], axis=-1)
            V_p = jnp.concatenate([V, jnp.zeros(pad_shape, V.dtype)], axis=-1)
            out = mhsea_tuned_edge_attention(Q_p, K_p, V_p, coup_bias, mask)[..., :d_h]
        else:
            raise ValueError("attn_impl must be 'einsum' or 'mhsea_tuned'")
        gate_heads = out[:, : self.n_heads, :]
        value_heads = out[:, self.n_heads :, :]
        out = jax.nn.sigmoid(gate_heads) * value_heads
        out_flat = out.reshape(n, self.n_heads * d_h)
        delta = _tagged_dense_no_bias(
            self.w_o,
            out_flat,
            tag_id=self._use_id_o,
            pathway="even",
            kfac_structural_mask=mask,
            kfac_scan_shared=kfac_scan_shared,
            kfac_repeat_ndim=1,
        )
        mask_q = mask.reshape(-1, 1).astype(delta.dtype)
        proposal_attn = mask_q * delta
        c_attn = _tree_ngpt_residual(
            c_level,
            proposal_attn,
            self.alpha_attn,
            max_gain=self.ngpt_alpha_max,
            tag_id="",
            update_mask=mask,
            kfac_structural_mask=mask,
            kfac_scan_shared=kfac_scan_shared,
            kfac_repeat_ndim=1,
        )
        x_ffn = _tagged_rms_eqx_style(
            self.ffn_ln_scale,
            c_attn,
            eps=self.ln_eps,
            tag_id=self._use_id_ffn_ln,
            pathway="even",
            kfac_structural_mask=mask,
            kfac_scan_shared=kfac_scan_shared,
            kfac_repeat_ndim=1,
        )
        h = _tagged_dense(
            self.ffn_w1,
            self.ffn_b1,
            x_ffn,
            tag_id=self._use_id_ffn1,
            pathway="even",
            kfac_structural_mask=mask,
            kfac_scan_shared=kfac_scan_shared,
            kfac_repeat_ndim=1,
        )
        h = fused_silu(h)
        delta_ffn = _tagged_dense(
            self.ffn_w2,
            self.ffn_b2,
            h,
            tag_id=self._use_id_ffn2,
            pathway="even",
            kfac_structural_mask=mask,
            kfac_scan_shared=kfac_scan_shared,
            kfac_repeat_ndim=1,
        )
        proposal_ffn = mask_q * delta_ffn
        return _tree_ngpt_residual(
            c_attn,
            proposal_ffn,
            self.alpha_ffn,
            max_gain=self.ngpt_alpha_max,
            tag_id="",
            update_mask=mask,
            kfac_structural_mask=mask,
            kfac_scan_shared=kfac_scan_shared,
            kfac_repeat_ndim=1,
        )


class MergeOp(eqx.Module):
    T: Float[Array, "G d_r d_r d_r"]
    _use_id_T: str = eqx.field(static=True, default="")
    mlp_c: MLP
    alpha_c: Float[Array, "d_c"]
    _use_id_alpha_c: str = eqx.field(static=True, default="")
    ngpt_alpha_max: float = eqx.field(static=True, default=0.5)
    eps: float = eqx.field(static=True)
    output_hypernet: HypernetMatrix
    edge_merge: EdgeMergeOp
    level_edge_attn: LevelEdgeAttn
    tree_edge_fwl: EdgeFWLUpdate

    def __init__(
        self,
        d_r: int,
        d_c: int,
        *,
        key: PRNGKeyArray,
        d_g: int,
        alpha_init: float,
        alpha_max: float,
        eps: float = 1e-06,
        merge_output_hypernet_rank: int = 128,
        d_m_merge: int | None = None,
        level_edge_attn_d_edge: int = 64,
        level_edge_attn_n_heads: int = 4,
        level_edge_attn_attn_dim: int | None = None,
        tree_edge_node_ctx_dim: int | None = None,
        level_edge_attn_attn_impl: str = "mhsea_tuned",
        level_edge_attn_edge_mlp_hidden: int | None = None,
        level_edge_attn_edge_mlp_n_blocks: int = 2,
        level_edge_attn_ffn_d_hidden: int | None = None,
        level_edge_attn_max_n: int = 128,
        level_edge_attn_rope_base: float = 10000.0,
        level_edge_attn_rope_scaling: float = 1.0,
        tree_edge_fwl_channels: int = 64,
        level_edge_attn_bias_mlp_hidden: int | None = None,
        level_edge_attn_bias_mlp_n_blocks: int = 1,
        merge_c_mlp_hidden: int | None = None,
    ):
        keys = jax.random.split(key, 11)
        k_t, k_c, k_h = keys[:3]
        k_em, k_lea, k_fwl = keys[4], keys[5], keys[10]
        d_m_eff = d_m_merge if d_m_merge is not None else d_r
        if d_m_eff % d_r != 0:
            raise ValueError(
                f"MergeOp: d_m_merge={d_m_merge} must be divisible by d_r={d_r} (HT carrier T is reshape-indexed as [G, d_r, d_r, d_r] with G = d_m_eff // d_r)."
            )
        G_merge = d_m_eff // d_r
        merge_edge_dim = 2 * int(level_edge_attn_d_edge)
        merge_clock_dim = int(d_c)
        merge_ngpt_dim = _TREE_NGPT_DEPTH_FEAT_DIM
        merge_extra_dim = int(d_g) + merge_edge_dim + merge_clock_dim + merge_ngpt_dim
        std = 1.0 / d_r
        self.T = jax.random.normal(k_t, (G_merge, d_r, d_r, d_r)) * std
        _c_hidden = (
            int(merge_c_mlp_hidden) if merge_c_mlp_hidden is not None else max(d_c, 16)
        )
        self.mlp_c = MLP(2 * d_c + merge_extra_dim, _c_hidden, d_c, key=k_c)
        self.ngpt_alpha_max = float(alpha_max)
        self.alpha_c = float(alpha_init) * jnp.ones((int(d_c),))
        self.output_hypernet = HypernetMatrix(
            d_in=d_m_eff,
            d_out=d_m_eff,
            d_e=d_c + merge_ngpt_dim,
            rank=merge_output_hypernet_rank,
            key=k_h,
        )
        self.eps = eps
        self.edge_merge = EdgeMergeOp(
            d_edge=level_edge_attn_d_edge,
            d_c=d_c,
            key=k_em,
            alpha_init=alpha_init,
            alpha_max=alpha_max,
            d_hidden=level_edge_attn_edge_mlp_hidden,
            n_blocks=level_edge_attn_edge_mlp_n_blocks,
            edge_node_ctx_dim=tree_edge_node_ctx_dim,
        )
        self.level_edge_attn = LevelEdgeAttn(
            d_c=d_c,
            d_edge=level_edge_attn_d_edge,
            key=k_lea,
            alpha_init=alpha_init,
            alpha_max=alpha_max,
            n_heads=level_edge_attn_n_heads,
            attn_dim=level_edge_attn_attn_dim,
            attn_impl=level_edge_attn_attn_impl,
            ffn_d_hidden=level_edge_attn_ffn_d_hidden,
            max_n=level_edge_attn_max_n,
            rope_base=level_edge_attn_rope_base,
            rope_scaling=level_edge_attn_rope_scaling,
            bias_mlp_hidden=level_edge_attn_bias_mlp_hidden,
            bias_mlp_n_blocks=level_edge_attn_bias_mlp_n_blocks,
        )
        self.tree_edge_fwl = EdgeFWLUpdate(
            d_c=d_c,
            d_edge=level_edge_attn_d_edge,
            key=k_fwl,
            alpha_init=alpha_init,
            alpha_max=alpha_max,
            channels=tree_edge_fwl_channels,
            edge_node_ctx_dim=tree_edge_node_ctx_dim,
        )

    def _merge_extra_inputs(
        self,
        c_a,
        c_b,
        g_emb,
        sibling_edge_lr,
        sibling_edge_rl,
        level_idx,
        pair_idx,
        pair_base,
        clock_depth,
        depth_feats,
    ):
        parts = [g_emb]
        parts.append(
            jnp.concatenate(
                [sibling_edge_lr.astype(c_a.dtype), sibling_edge_rl.astype(c_a.dtype)]
            )
        )
        parts.append(
            _tree_merge_clock(
                level_idx, pair_idx, pair_base, c_a.shape[-1], clock_depth, c_a.dtype
            )
        )
        parts.append(depth_feats.astype(c_a.dtype))
        return parts

    def _apply_c_skip(
        self,
        c_a,
        c_b,
        c_delta,
        *,
        kfac_structural_mask=None,
        kfac_scan_shared: bool = False,
    ):
        return _tree_ngpt_residual(
            0.5 * (c_a + c_b),
            c_delta,
            self.alpha_c,
            max_gain=self.ngpt_alpha_max,
            tag_id=self._use_id_alpha_c,
            pathway="even",
            kfac_structural_mask=kfac_structural_mask,
            kfac_scan_shared=kfac_scan_shared,
            kfac_repeat_ndim=1,
        )

    def context_candidate(
        self,
        c_a: Float[Array, "d_c"],
        c_b: Float[Array, "d_c"],
        g_emb: Float[Array, "d_g"],
        *,
        sibling_edge_lr: Float[Array, "d_edge"],
        sibling_edge_rl: Float[Array, "d_edge"],
        level_idx: Array,
        pair_idx: Array,
        pair_base: Array,
        clock_depth: Array,
        depth_feats: Array,
        kfac_structural_mask=None,
        kfac_g_structural_mask=None,
        kfac_scan_shared: bool = False,
    ):
        ffn_in = jnp.concatenate(
            [
                c_a,
                c_b,
                *self._merge_extra_inputs(
                    c_a,
                    c_b,
                    g_emb,
                    sibling_edge_lr,
                    sibling_edge_rl,
                    level_idx,
                    pair_idx,
                    pair_base,
                    clock_depth,
                    depth_feats,
                ),
            ]
        )
        mlp = self.mlp_c
        _we = _rownorm_cols
        x = _tagged_dense(
            mlp.in_proj.weight,
            mlp.in_proj.bias,
            ffn_in,
            tag_id=mlp.in_proj._use_id,
            pathway="even",
            weight_eff=_we(mlp.in_proj.weight),
            kfac_structural_mask=kfac_structural_mask,
            kfac_scan_shared=kfac_scan_shared,
            kfac_repeat_ndim=1,
        )
        for nrm, l1, l2 in zip(mlp.block_norms, mlp.block_l1s, mlp.block_l2s):
            normed = _inline_norm_forward(
                nrm,
                x,
                pathway="even",
                kfac_structural_mask=kfac_structural_mask,
                kfac_scan_shared=kfac_scan_shared,
                kfac_repeat_ndim=1,
            )
            inner = _tagged_dense(
                l1.weight,
                l1.bias,
                normed,
                tag_id=l1._use_id,
                pathway="even",
                weight_eff=_we(l1.weight),
                kfac_structural_mask=kfac_structural_mask,
                kfac_scan_shared=kfac_scan_shared,
                kfac_repeat_ndim=1,
            )
            inner_act = mlp._act(inner)
            inner_out = _tagged_dense(
                l2.weight,
                l2.bias,
                inner_act,
                tag_id=l2._use_id,
                pathway="even",
                weight_eff=_we(l2.weight),
                kfac_structural_mask=kfac_structural_mask,
                kfac_scan_shared=kfac_scan_shared,
                kfac_repeat_ndim=1,
            )
            x = x + mlp.inner_gain * inner_out
        out_normed = _inline_norm_forward(
            mlp.out_norm,
            x,
            pathway="even",
            kfac_structural_mask=kfac_structural_mask,
            kfac_scan_shared=kfac_scan_shared,
            kfac_repeat_ndim=1,
        )
        c_delta = _tagged_dense(
            mlp.out_proj.weight,
            mlp.out_proj.bias,
            out_normed,
            tag_id=mlp.out_proj._use_id,
            pathway="even",
            weight_eff=_we(mlp.out_proj.weight),
            kfac_structural_mask=kfac_structural_mask,
            kfac_scan_shared=kfac_scan_shared,
            kfac_repeat_ndim=1,
        )
        return self._apply_c_skip(
            c_a,
            c_b,
            c_delta,
            kfac_structural_mask=kfac_structural_mask,
            kfac_scan_shared=kfac_scan_shared,
        )

    def __call__(
        self,
        c_a: Float[Array, "d_c"],
        u_a: Float[Array, "d_m_eff"],
        s_a: Float[Array, ""],
        c_b: Float[Array, "d_c"],
        u_b: Float[Array, "d_m_eff"],
        s_b: Float[Array, ""],
        g_emb: Float[Array, "d_g"],
        *,
        sibling_edge_lr: Float[Array, "d_edge"],
        sibling_edge_rl: Float[Array, "d_edge"],
        level_idx: Array,
        pair_idx: Array,
        pair_base: Array,
        clock_depth: Array,
        depth_feats: Array,
        kfac_context_mask=None,
        kfac_g_context_mask=None,
        kfac_odd_mask=None,
        kfac_scan_shared: bool = False,
    ):
        c_p = self.context_candidate(
            c_a,
            c_b,
            g_emb,
            sibling_edge_lr=sibling_edge_lr,
            sibling_edge_rl=sibling_edge_rl,
            level_idx=level_idx,
            pair_idx=pair_idx,
            pair_base=pair_base,
            clock_depth=clock_depth,
            depth_feats=depth_feats,
            kfac_structural_mask=kfac_context_mask,
            kfac_g_structural_mask=kfac_g_context_mask,
            kfac_scan_shared=kfac_scan_shared,
        )
        raw = _quadrilinear_merge(
            self.T,
            u_a,
            u_b,
            tag_id=self._use_id_T,
            pathway="odd",
            kfac_structural_mask=kfac_odd_mask,
            kfac_scan_shared=kfac_scan_shared,
            kfac_repeat_ndim=1,
        )
        H = self.output_hypernet
        h_ctx = jnp.concatenate([c_p, depth_feats.astype(c_p.dtype)])
        h_p = _tagged_dense_no_bias(
            H.W_h,
            h_ctx,
            tag_id=H._use_id_W_h,
            pathway="hypernet_eside",
            kfac_structural_mask=kfac_odd_mask,
            kfac_scan_shared=kfac_scan_shared,
            kfac_repeat_ndim=1,
            kfac_context_primal_reused_over_walkers=True,
        )
        V_x = _tagged_dense_no_bias(
            H.V,
            raw,
            tag_id=H._use_id_V,
            pathway="odd",
            kfac_structural_mask=kfac_odd_mask,
            kfac_scan_shared=kfac_scan_shared,
            kfac_repeat_ndim=1,
        )
        m_p = h_p * V_x
        H_out = _tagged_dense_no_bias(
            H.U,
            m_p,
            tag_id=H._use_id_U,
            pathway="odd",
            kfac_structural_mask=kfac_odd_mask,
            kfac_scan_shared=kfac_scan_shared,
            kfac_repeat_ndim=1,
        )
        out = raw + H_out
        scale_sq = jnp.mean(out * out)
        scale = jnp.sqrt(scale_sq + self.eps)
        u_p = out / scale
        s_p = s_a + s_b + jnp.log(scale)
        return (c_p, u_p, s_p)


def project_tree_ngpt_rownorm(model):
    import equinox as _eqx

    def _collect(m):
        mlp = m.merge.mlp_c
        leaves = [mlp.in_proj.weight]
        for l1, l2 in zip(mlp.block_l1s, mlp.block_l2s):
            leaves.append(l1.weight)
            leaves.append(l2.weight)
        leaves.append(mlp.out_proj.weight)
        leaves.append(m.route_decoder.tree_merge.w1)
        leaves.append(m.route_decoder.tree_merge.w2)
        return tuple(leaves)

    targets = _collect(model)
    return _eqx.tree_at(_collect, model, _rownorm_project_jit(targets))


@partial(jax.jit, donate_argnums=0)
def _rownorm_project_jit(ws):
    return tuple((_rownorm_cols(w) for w in ws))


def merge_masked(
    c_a,
    u_a,
    s_a,
    m_a,
    c_b,
    u_b,
    s_b,
    m_b,
    merge,
    g_emb,
    *,
    k_a,
    k_b,
    sibling_edge_lr,
    sibling_edge_rl,
    level_idx,
    pair_idx,
    pair_base,
    clock_depth,
    depth_feats,
    kfac_g_context_mask=None,
    kfac_scan_shared=False,
):
    both = m_a * m_b
    only_a = m_a * (1.0 - m_b)
    only_b = (1.0 - m_a) * m_b
    both_k = k_a * k_b
    only_a_k = k_a * (1.0 - k_b)
    only_b_k = (1.0 - k_a) * k_b

    def gate(value, left, right):
        return jnp.where(
            both.astype(bool),
            value,
            jnp.where(
                only_a.astype(bool),
                left,
                jnp.where(only_b.astype(bool), right, jnp.zeros_like(value)),
            ),
        )

    def structural_gate(value, left, right):
        return jnp.where(
            both_k.astype(bool),
            value,
            jnp.where(
                only_a_k.astype(bool),
                left,
                jnp.where(only_b_k.astype(bool), right, jnp.zeros_like(value)),
            ),
        )

    c_new, u_new, s_new = merge(
        c_a,
        u_a,
        s_a,
        c_b,
        u_b,
        s_b,
        g_emb,
        sibling_edge_lr=sibling_edge_lr,
        sibling_edge_rl=sibling_edge_rl,
        level_idx=level_idx,
        pair_idx=pair_idx,
        pair_base=pair_base,
        clock_depth=clock_depth,
        depth_feats=depth_feats,
        kfac_context_mask=both_k,
        kfac_g_context_mask=kfac_g_context_mask,
        kfac_odd_mask=both,
        kfac_scan_shared=kfac_scan_shared,
    )
    c_out = structural_gate(c_new, c_a, c_b)
    out = (c_out, gate(u_new, u_a, u_b), gate(s_new, s_a, s_b), m_a + m_b - m_a * m_b)
    return out


def edge_merge_masked(
    e_2i_2j: Float[Array, "d_edge"],
    e_2i_2j1: Float[Array, "d_edge"],
    e_2i1_2j: Float[Array, "d_edge"],
    e_2i1_2j1: Float[Array, "d_edge"],
    m_2i: Float[Array, ""],
    m_2i1: Float[Array, ""],
    m_2j: Float[Array, ""],
    m_2j1: Float[Array, ""],
    c_2i: Float[Array, "d_c"],
    c_2i1: Float[Array, "d_c"],
    c_2j: Float[Array, "d_c"],
    c_2j1: Float[Array, "d_c"],
    edge_merge: EdgeMergeOp,
    *,
    k_2i: Float[Array, ""],
    k_2i1: Float[Array, ""],
    k_2j: Float[Array, ""],
    k_2j1: Float[Array, ""],
    kfac_scan_shared: bool = False,
) -> tuple[Float[Array, "d_edge"], Float[Array, ""]]:
    m_p = m_2i + m_2i1 - m_2i * m_2i1
    m_q = m_2j + m_2j1 - m_2j * m_2j1
    m_pq = m_p * m_q
    both_p = k_2i * k_2i1
    both_q = k_2j * k_2j1
    out_mask = both_p * both_q
    proposal = edge_merge(
        e_2i_2j,
        e_2i_2j1,
        e_2i1_2j,
        e_2i1_2j1,
        c_2i,
        c_2i1,
        c_2j,
        c_2j1,
        kfac_structural_mask=out_mask,
        kfac_scan_shared=kfac_scan_shared,
    )
    cell_weights = jnp.stack(
        [k_2i * k_2j, k_2i * k_2j1, k_2i1 * k_2j, k_2i1 * k_2j1]
    ).astype(proposal.dtype)
    child_edges = jnp.stack([e_2i_2j, e_2i_2j1, e_2i1_2j, e_2i1_2j1], axis=0)
    mean_denom = jnp.maximum(
        jnp.sum(cell_weights), jnp.asarray(1.0, dtype=proposal.dtype)
    )
    masked_mean = jnp.sum(child_edges * cell_weights[:, None], axis=0) / mean_denom
    e_pq = edge_merge.apply_skip(
        masked_mean,
        proposal,
        kfac_structural_mask=out_mask,
        kfac_scan_shared=kfac_scan_shared,
    )
    return (
        jnp.where(jnp.asarray(out_mask).astype(bool), e_pq, jnp.zeros_like(e_pq)),
        m_pq,
    )


def _next_pow2(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def _balanced_subtree_mask(m, n_pad: int):
    N = jnp.sum(m.astype(jnp.int32))
    powers = 2 ** jnp.arange(max(1, n_pad.bit_length()), dtype=jnp.int32)
    big = jnp.asarray(1 << 30, dtype=jnp.int32)
    next_p = jnp.min(jnp.where(powers >= jnp.maximum(N, 1), powers, big))
    return (jnp.arange(n_pad, dtype=jnp.int32) < next_p).astype(m.dtype)


def _tree_active_clock_depth(mask):
    n_active = jnp.maximum(jnp.sum(mask.astype(jnp.int32)), 1)
    depth = jnp.ceil(jnp.log2(n_active.astype(jnp.float32))).astype(jnp.int32)
    return jnp.maximum(depth, 1)


def balanced_tree_reduce_masked_scan(
    c: Float[Array, "n d_c"],
    u: Float[Array, "n d_m_eff"],
    s: Float[Array, "n"],
    m: Float[Array, "n"],
    merge: MergeOp,
    g_emb: Float[Array, "d_g"],
    *,
    edges_init: Float[Array, "n n d_edge"],
    gladder,
    g_stream0,
):
    from hamiltonzero.model.fp32 import compute_dtype as _compute_dtype

    _cdtype = _compute_dtype()
    if c.dtype != _cdtype:
        c = c.astype(_cdtype)
    if edges_init.dtype != _cdtype:
        edges_init = edges_init.astype(_cdtype)
    n = c.shape[0]
    if n == 1:
        return (c[0], u[0], s[0], m[0], edges_init[0, 0], g_stream0)
    n_pad = _next_pow2(n)
    n_levels = n_pad.bit_length() - 1
    pad_amount = n_pad - n
    if pad_amount > 0:
        c = jnp.pad(c, ((0, pad_amount),) + ((0, 0),) * (c.ndim - 1))
        u = jnp.pad(u, ((0, pad_amount),) + ((0, 0),) * (u.ndim - 1))
        s = jnp.pad(s, (0, pad_amount))
        m = jnp.pad(m, (0, pad_amount))
    k = _balanced_subtree_mask(m, n_pad)
    clock_depth = _tree_active_clock_depth(k)
    d_edge = edges_init.shape[-1]
    edges_padded = _tree_sphere(
        jnp.pad(edges_init, ((0, pad_amount), (0, pad_amount), (0, 0)))
    )
    initial_state = (c, u, s, m, k, edges_padded, g_stream0)
    n_pairs = n_pad // 2
    pidx = jnp.arange(n_pairs, dtype=jnp.int32)

    def _merge_one_pair(
        c_a_,
        u_a_,
        s_a_,
        c_b_,
        u_b_,
        s_b_,
        m_a_,
        m_b_,
        k_a_,
        k_b_,
        sibling_edge_lr_,
        sibling_edge_rl_,
        depth_feats_,
        pair_idx_,
        pair_base_,
        clock_depth_,
        level_idx_,
        g_emb_,
        g_structural_mask_,
    ):
        return merge_masked(
            c_a_,
            u_a_,
            s_a_,
            m_a_,
            c_b_,
            u_b_,
            s_b_,
            m_b_,
            merge,
            g_emb_,
            k_a=k_a_,
            k_b=k_b_,
            sibling_edge_lr=sibling_edge_lr_,
            sibling_edge_rl=sibling_edge_rl_,
            level_idx=level_idx_,
            pair_idx=pair_idx_,
            pair_base=pair_base_,
            clock_depth=clock_depth_,
            depth_feats=depth_feats_,
            kfac_g_context_mask=g_structural_mask_,
            kfac_scan_shared=True,
        )

    _vmap_in_axes = (0,) * 14 + (None,) * 5
    _vmapped_merge = jax.vmap(
        _merge_one_pair, in_axes=_vmap_in_axes, axis_name="tree_pair"
    )

    def _edge_one(
        e0,
        e1,
        e2,
        e3,
        m_p_a,
        m_p_b,
        m_q_a,
        m_q_b,
        k_p_a,
        k_p_b,
        k_q_a,
        k_q_b,
        c_p_a,
        c_p_b,
        c_q_a,
        c_q_b,
    ):
        return edge_merge_masked(
            e0,
            e1,
            e2,
            e3,
            m_p_a,
            m_p_b,
            m_q_a,
            m_q_b,
            c_p_a,
            c_p_b,
            c_q_a,
            c_q_b,
            merge.edge_merge,
            k_2i=k_p_a,
            k_2i1=k_p_b,
            k_2j=k_q_a,
            k_2j1=k_q_b,
            kfac_scan_shared=True,
        )

    _edge_inner = jax.vmap(
        _edge_one,
        in_axes=(0, 0, 0, 0, None, None, 0, 0, None, None, 0, 0, None, None, 0, 0),
        axis_name="tree_edge_q",
    )
    _edge_outer = jax.vmap(
        _edge_inner,
        in_axes=(0, 0, 0, 0, 0, 0, None, None, 0, 0, None, None, 0, 0, None, None),
        axis_name="tree_edge_p",
    )

    def body(state, xs_lv):
        level_idx, depth_feats_lv = xs_lv
        c, u, s, m, k, E, g_carry = state

        def _split(x):
            xr = x.reshape((n_pairs, 2) + x.shape[1:])
            return (xr[:, 0], xr[:, 1])

        m_a, m_b = _split(m)
        k_a, k_b = _split(k)
        level_active = jnp.any((k_a * k_b).astype(bool))
        c_a, c_b = _split(c)
        u_a, u_b = _split(u)
        s_a, s_b = _split(s)
        pair_args: list = [c_a, u_a, s_a, c_b, u_b, s_b]
        pair_args.extend([m_a, m_b])
        pair_args.extend([k_a, k_b])
        E_rs_for_merge = E.reshape(n_pairs, 2, n_pairs, 2, d_edge)
        pair_args.extend(
            [E_rs_for_merge[pidx, 0, pidx, 1, :], E_rs_for_merge[pidx, 1, pidx, 0, :]]
        )
        g_emb_lvl = _tagged_dense(
            gladder[2],
            gladder[3],
            g_carry,
            tag_id="gladder.tree.proj",
            pathway="even",
            kfac_structural_mask=level_active,
            kfac_scan_shared=True,
            kfac_repeat_ndim=0,
        )
        pair_args.append(depth_feats_lv)
        clock_pair_active = k_a + k_b - k_a * k_b
        pair_base = jnp.maximum(
            jnp.sum(clock_pair_active.astype(jnp.int32)),
            jnp.asarray(2, dtype=jnp.int32),
        )
        pair_args.extend(
            [pidx, pair_base, clock_depth, level_idx, g_emb_lvl, level_active]
        )
        merged = _vmapped_merge(*pair_args)
        c_p, u_p, s_p, m_p = merged
        k_p = k_a + k_b - k_a * k_b
        both_struct = k_a * k_b
        attn_mask = both_struct
        E_rs = E.reshape(n_pairs, 2, n_pairs, 2, d_edge)
        E_00 = E_rs[:, 0, :, 0, :]
        E_01 = E_rs[:, 0, :, 1, :]
        E_10 = E_rs[:, 1, :, 0, :]
        E_11 = E_rs[:, 1, :, 1, :]
        E_new, _m_edge_new = _edge_outer(
            E_00,
            E_01,
            E_10,
            E_11,
            m_a,
            m_b,
            m_a,
            m_b,
            k_a,
            k_b,
            k_a,
            k_b,
            c_a,
            c_b,
            c_a,
            c_b,
        )
        E_new = merge.tree_edge_fwl.apply_residual(
            E_new, c_p, attn_mask, kfac_scan_shared=True
        )
        edge_keep = (both_struct[:, None] * both_struct[None, :]).astype(bool)
        E_new = jnp.where(edge_keep[..., None], E_new, E_00)
        E_new = jnp.where(edge_keep[..., None], _tree_sphere(E_new), E_00)
        c_skip = c_p
        c_p = merge.level_edge_attn(
            c_p, E_new, attn_mask, level_idx=level_idx, kfac_scan_shared=True
        )
        c_p = jnp.where(attn_mask.astype(bool)[:, None], _tree_sphere(c_p), c_skip)
        _lvl_mask = k_p.astype(c_p.dtype)
        update_active = jnp.any(attn_mask.astype(bool))
        pool_structural_mask = _lvl_mask * update_active.astype(_lvl_mask.dtype)
        pooled = gladder[0](
            g_carry,
            c_p,
            _lvl_mask,
            kfac_structural_mask=pool_structural_mask,
            kfac_update_mask=update_active,
            kfac_scan_shared=True,
            kfac_repeat_ndim=1,
        )
        g_carry = gladder[1](
            g_carry,
            pooled,
            update_mask=update_active,
            kfac_structural_mask=update_active,
            kfac_scan_shared=True,
        )

        def _zpad(x_half):
            return jnp.concatenate([x_half, jnp.zeros_like(x_half)], axis=0)

        pad_amt = n_pad - n_pairs
        E_padded = jnp.pad(E_new, ((0, pad_amt), (0, pad_amt), (0, 0)))
        return (
            (
                _zpad(c_p),
                _zpad(u_p),
                _zpad(s_p),
                _zpad(m_p),
                _zpad(k_p),
                E_padded,
                g_carry,
            ),
            None,
        )

    depth_feat_levels = _tree_ngpt_level_counts(m, n_pairs, n_levels, c.dtype)
    final_state, _ = jax.lax.scan(
        body, initial_state, (jnp.arange(n_levels), depth_feat_levels)
    )
    c_f, u_f, s_f, m_f, _k_f, E_f, g_final = final_state
    return (c_f[0], u_f[0], s_f[0], m_f[0], E_f[0, 0], g_final)


class RootReadout(eqx.Module):
    output_hypernet: HypernetMatrix
    ln_e: _RMS

    def __init__(
        self,
        d_r: int,
        *,
        key: PRNGKeyArray,
        d_m_merge: int | None = None,
        d_edge: int,
        edge_rank: int = 64,
        d_g: int = 0,
        d_c: int,
    ):
        d_m_eff = d_m_merge if d_m_merge is not None else d_r
        if d_m_eff % d_r != 0:
            raise ValueError(
                f"RootReadout: d_m_merge={d_m_merge} must be divisible by d_r={d_r} (HT carrier dim must match MergeOp's d_m_eff)."
            )
        keys = jax.random.split(key, 5)
        d_e_ctx = int(d_edge) + int(d_c) + int(d_g)
        self.output_hypernet = HypernetMatrix(
            d_in=d_m_eff, d_out=2, d_e=d_e_ctx, rank=edge_rank, key=keys[0]
        )
        self.ln_e = _RMS(d_edge)

    def __call__(self, u_r, s_r, *, e_root, g_emb, c_root, kfac_structural_mask=None):
        e_norm = _inline_norm_forward(
            self.ln_e,
            e_root,
            pathway="even",
            kfac_structural_mask=kfac_structural_mask,
            kfac_scan_shared=False,
            kfac_repeat_ndim=0,
        )
        h_ctx = jnp.concatenate(
            [e_norm, c_root.astype(e_norm.dtype), g_emb.astype(e_norm.dtype)]
        )
        psi = self.output_hypernet.apply(
            h_ctx,
            u_r,
            e_pathway="even",
            kfac_structural_mask=kfac_structural_mask,
            kfac_scan_shared=False,
            kfac_repeat_ndim=0,
            kfac_context_primal_reused_over_walkers=True,
        )
        re = 0.5 * jnp.log(psi[0] * psi[0] + psi[1] * psi[1]) + s_r
        return (re, jnp.arctan2(psi[1], psi[0]))
