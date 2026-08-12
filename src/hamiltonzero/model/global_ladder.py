# Copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray

from .odd_ops import Linear, _RMS
from .tree import (
    _inline_norm_forward,
    _tagged_bounded_ngpt_gain,
    _tagged_dense,
    _tagged_dense_no_bias,
    _tree_sphere,
)


def _pool_masked_softmax(scores, mask, axis):
    neg = jnp.asarray(-1.0e30, dtype=scores.dtype)
    scores = jnp.where(mask > 0, scores, neg)
    return jax.nn.softmax(scores, axis=axis)


class GDescriptorPool(eqx.Module):
    W_q: Float[Array, "d_g hq"]
    K: Linear
    V: Linear
    ln_in: _RMS
    n_heads: int = eqx.field(static=True)
    d_k: int = eqx.field(static=True)
    d_v: int = eqx.field(static=True)
    tag: str = eqx.field(static=True, default="")

    def __init__(
        self,
        d_g: int,
        d_x: int,
        *,
        key: PRNGKeyArray,
        n_heads: int = 4,
        d_k: int = 64,
        d_v: int = 64,
        tag: str = "",
    ):
        kq, kk, kv = jax.random.split(key, 3)
        self.W_q = jax.random.normal(kq, (d_g, n_heads * d_k)) * (d_g**-0.5)
        self.K = Linear(d_x, n_heads * d_k, key=kk)
        self.V = Linear(d_x, n_heads * d_v, key=kv)
        self.ln_in = _RMS(d_x)
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.tag = tag

    @property
    def d_out(self) -> int:
        return self.n_heads * self.d_v

    def __call__(
        self,
        g,
        xs,
        mask,
        *,
        kfac_structural_mask=None,
        kfac_update_mask=None,
        kfac_scan_shared: bool = False,
        kfac_repeat_ndim: int = 1,
        kfac_context_primal_reused_over_walkers: bool = False,
    ):

        n = xs.shape[0]
        xn = _inline_norm_forward(
            self.ln_in,
            xs,
            pathway="even",
            kfac_structural_mask=kfac_structural_mask,
            kfac_scan_shared=kfac_scan_shared,
            kfac_repeat_ndim=kfac_repeat_ndim,
            kfac_context_primal_reused_over_walkers=(
                kfac_context_primal_reused_over_walkers
            ),
        )
        q_structural_mask = (
            jnp.any(jnp.asarray(kfac_structural_mask).astype(bool))
            if kfac_update_mask is None and kfac_structural_mask is not None
            else kfac_update_mask
        )
        q = _tagged_dense_no_bias(
            self.W_q,
            g,
            tag_id=f"{self.tag}.W_q",
            pathway="even",
            kfac_structural_mask=q_structural_mask,
            kfac_scan_shared=kfac_scan_shared,
            kfac_repeat_ndim=0,
            kfac_context_primal_reused_over_walkers=(
                kfac_context_primal_reused_over_walkers
            ),
        ).reshape(self.n_heads, self.d_k)
        k = _tagged_dense(
            self.K.weight,
            self.K.bias,
            xn,
            tag_id=self.K._use_id,
            pathway="even",
            kfac_structural_mask=kfac_structural_mask,
            kfac_scan_shared=kfac_scan_shared,
            kfac_repeat_ndim=kfac_repeat_ndim,
            kfac_context_primal_reused_over_walkers=(
                kfac_context_primal_reused_over_walkers
            ),
        ).reshape(n, self.n_heads, self.d_k)
        v = _tagged_dense(
            self.V.weight,
            self.V.bias,
            xn,
            tag_id=self.V._use_id,
            pathway="even",
            kfac_structural_mask=kfac_structural_mask,
            kfac_scan_shared=kfac_scan_shared,
            kfac_repeat_ndim=kfac_repeat_ndim,
            kfac_context_primal_reused_over_walkers=(
                kfac_context_primal_reused_over_walkers
            ),
        ).reshape(n, self.n_heads, self.d_v)
        scores = jnp.einsum("hd,nhd->hn", q, k) / jnp.sqrt(
            jnp.asarray(self.d_k, dtype=xs.dtype)
        )
        attn = _pool_masked_softmax(scores, mask[None, :], axis=-1)
        out = jnp.einsum("hn,nhv->hv", attn, v)
        return out.reshape(-1)


def _global_update_parameters(d_g: int, d_pool: int, tap_dim: int, key):
    tap = int(tap_dim)
    if tap < 1 or tap >= int(d_g):
        raise ValueError("global tap dimension must be positive and smaller than d_g")
    d_in = tap + int(d_pool)
    d_hidden = 2 * int(d_g)
    k1, k2, k3 = jax.random.split(key, 3)
    return (
        jax.random.normal(k1, (d_in, d_hidden)) * (d_in**-0.5),
        jnp.zeros((d_hidden,)),
        jax.random.normal(k2, (d_hidden, d_g)) * (d_hidden**-0.5),
        jnp.zeros((d_g,)),
        jnp.ones((d_in,)),
        jax.random.normal(k3, (d_g, tap)) * (d_g**-0.5),
    )


def _global_delta(
    update,
    g,
    pool,
    *,
    kfac_structural_mask,
    kfac_g_structural_mask,
    kfac_scan_shared,
    kfac_repeat_ndim,
    kfac_context_primal_reused_over_walkers,
):
    from hamiltonzero.model.tree import _tagged_rms_eqx_style
    from hamiltonzero.model.fused_silu import fused_silu

    g_mask = (
        kfac_structural_mask
        if kfac_g_structural_mask is None
        else kfac_g_structural_mask
    )
    g_in = _tagged_dense_no_bias(
        update.g_tap_w,
        g,
        tag_id=f"{update.tag}.gtap",
        pathway="even",
        kfac_structural_mask=g_mask,
        kfac_scan_shared=(
            kfac_scan_shared if kfac_g_structural_mask is None else False
        ),
        kfac_repeat_ndim=(kfac_repeat_ndim if kfac_g_structural_mask is None else 0),
        kfac_context_primal_reused_over_walkers=(
            kfac_context_primal_reused_over_walkers
        ),
    )
    x = jnp.concatenate([g_in, pool.astype(g.dtype)])
    x = _tagged_rms_eqx_style(
        update.ln_s,
        x,
        tag_id=f"{update.tag}.ln",
        pathway="even",
        kfac_structural_mask=kfac_structural_mask,
        kfac_scan_shared=kfac_scan_shared,
        kfac_repeat_ndim=kfac_repeat_ndim,
        kfac_context_primal_reused_over_walkers=(
            kfac_context_primal_reused_over_walkers
        ),
    )
    hidden = fused_silu(
        _tagged_dense(
            update.w1,
            update.b1,
            x,
            tag_id=f"{update.tag}.ffn1",
            pathway="even",
            kfac_structural_mask=kfac_structural_mask,
            kfac_scan_shared=kfac_scan_shared,
            kfac_repeat_ndim=kfac_repeat_ndim,
            kfac_context_primal_reused_over_walkers=(
                kfac_context_primal_reused_over_walkers
            ),
        )
    )
    return _tagged_dense(
        update.w2,
        update.b2,
        hidden,
        tag_id=f"{update.tag}.ffn2",
        pathway="even",
        kfac_structural_mask=kfac_structural_mask,
        kfac_scan_shared=kfac_scan_shared,
        kfac_repeat_ndim=kfac_repeat_ndim,
        kfac_context_primal_reused_over_walkers=(
            kfac_context_primal_reused_over_walkers
        ),
    )


class ResidualGlobalUpdate(eqx.Module):
    w1: Float[Array, "d_in d_hidden"]
    b1: Float[Array, "d_hidden"]
    w2: Float[Array, "d_hidden d_g"]
    b2: Float[Array, "d_g"]
    ln_s: Float[Array, "d_in"]
    g_tap_w: Float[Array, "d_g tap_dim"]
    residual_gain: float = eqx.field(static=True)
    tag: str = eqx.field(static=True)

    def __init__(
        self,
        d_g: int,
        d_pool: int,
        *,
        key: PRNGKeyArray,
        tap_dim: int,
        residual_gain: float,
        tag: str,
    ):
        (
            self.w1,
            self.b1,
            self.w2,
            self.b2,
            self.ln_s,
            self.g_tap_w,
        ) = _global_update_parameters(d_g, d_pool, tap_dim, key)
        self.residual_gain = float(residual_gain)
        self.tag = tag

    def __call__(
        self,
        g,
        pool,
        *,
        kfac_structural_mask=None,
        kfac_g_structural_mask=None,
        kfac_scan_shared: bool = False,
        kfac_repeat_ndim: int = 0,
        kfac_context_primal_reused_over_walkers: bool = False,
    ):
        delta = _global_delta(
            self,
            g,
            pool,
            kfac_structural_mask=kfac_structural_mask,
            kfac_g_structural_mask=kfac_g_structural_mask,
            kfac_scan_shared=kfac_scan_shared,
            kfac_repeat_ndim=kfac_repeat_ndim,
            kfac_context_primal_reused_over_walkers=(
                kfac_context_primal_reused_over_walkers
            ),
        )
        return g + self.residual_gain * delta


class BoundaryGlobalUpdate(eqx.Module):
    w1: Float[Array, "d_in d_hidden"]
    b1: Float[Array, "d_hidden"]
    w2: Float[Array, "d_hidden d_g"]
    b2: Float[Array, "d_g"]
    ln_s: Float[Array, "d_in"]
    g_tap_w: Float[Array, "d_g tap_dim"]
    tag: str = eqx.field(static=True)

    def __init__(
        self, d_g: int, d_pool: int, *, key: PRNGKeyArray, tap_dim: int, tag: str
    ):
        (
            self.w1,
            self.b1,
            self.w2,
            self.b2,
            self.ln_s,
            self.g_tap_w,
        ) = _global_update_parameters(d_g, d_pool, tap_dim, key)
        self.tag = tag

    def __call__(
        self,
        g,
        pool,
        *,
        kfac_structural_mask=None,
        kfac_g_structural_mask=None,
        kfac_scan_shared: bool = False,
        kfac_repeat_ndim: int = 0,
        kfac_context_primal_reused_over_walkers: bool = False,
    ):
        delta = _global_delta(
            self,
            g,
            pool,
            kfac_structural_mask=kfac_structural_mask,
            kfac_g_structural_mask=kfac_g_structural_mask,
            kfac_scan_shared=kfac_scan_shared,
            kfac_repeat_ndim=kfac_repeat_ndim,
            kfac_context_primal_reused_over_walkers=(
                kfac_context_primal_reused_over_walkers
            ),
        )
        return _tree_sphere(g + delta)


class TreeGlobalUpdate(eqx.Module):
    w1: Float[Array, "d_in d_hidden"]
    b1: Float[Array, "d_hidden"]
    w2: Float[Array, "d_hidden d_g"]
    b2: Float[Array, "d_g"]
    ln_s: Float[Array, "d_in"]
    alpha: Float[Array, "d_g"]
    g_tap_w: Float[Array, "d_g tap_dim"]
    alpha_max: float = eqx.field(static=True)
    tag: str = eqx.field(static=True)

    def __init__(
        self,
        d_g: int,
        d_pool: int,
        *,
        key: PRNGKeyArray,
        tap_dim: int,
        alpha_init: float,
        alpha_max: float,
        tag: str,
    ):
        (
            self.w1,
            self.b1,
            self.w2,
            self.b2,
            self.ln_s,
            self.g_tap_w,
        ) = _global_update_parameters(d_g, d_pool, tap_dim, key)
        self.alpha = float(alpha_init) * jnp.ones((d_g,))
        self.alpha_max = float(alpha_max)
        self.tag = tag

    def __call__(
        self,
        g,
        pool,
        update_mask=None,
        *,
        kfac_structural_mask=None,
        kfac_g_structural_mask=None,
        kfac_scan_shared: bool = False,
        kfac_repeat_ndim: int = 0,
        kfac_context_primal_reused_over_walkers: bool = False,
    ):
        delta = _global_delta(
            self,
            g,
            pool,
            kfac_structural_mask=kfac_structural_mask,
            kfac_g_structural_mask=kfac_g_structural_mask,
            kfac_scan_shared=kfac_scan_shared,
            kfac_repeat_ndim=kfac_repeat_ndim,
            kfac_context_primal_reused_over_walkers=(
                kfac_context_primal_reused_over_walkers
            ),
        )
        skip = _tree_sphere(g)
        proposal = _tree_sphere(delta)
        direction = proposal - skip
        gain = _tagged_bounded_ngpt_gain(
            self.alpha,
            direction,
            max_gain=self.alpha_max,
            tag_id=f"{self.tag}.alpha",
            pathway="even",
            kfac_structural_mask=kfac_structural_mask,
            kfac_scan_shared=kfac_scan_shared,
            kfac_repeat_ndim=kfac_repeat_ndim,
            kfac_context_primal_reused_over_walkers=(
                kfac_context_primal_reused_over_walkers
            ),
        )
        updated = _tree_sphere(skip + gain * direction)
        if update_mask is None:
            return updated
        active = jnp.asarray(update_mask).astype(bool)
        while active.ndim < updated.ndim:
            active = active[..., None]
        return jnp.where(active, updated, g)


class EdgeRowColGlobalUpdate(eqx.Module):
    row_pool: GDescriptorPool
    col_pool: GDescriptorPool
    set_pool: GDescriptorPool
    update: BoundaryGlobalUpdate

    def __init__(
        self,
        d_g: int,
        d_edge: int,
        *,
        key: PRNGKeyArray,
        n_heads: int = 4,
        d_k: int = 64,
        d_v: int = 64,
        tag: str = "",
        tap_dim: int,
    ):
        kr, kc, ks, ku = jax.random.split(key, 4)
        self.row_pool = GDescriptorPool(
            d_g, d_edge, key=kr, n_heads=n_heads, d_k=d_k, d_v=d_v, tag=f"{tag}.row"
        )
        self.col_pool = GDescriptorPool(
            d_g, d_edge, key=kc, n_heads=n_heads, d_k=d_k, d_v=d_v, tag=f"{tag}.col"
        )
        self.set_pool = GDescriptorPool(
            d_g,
            n_heads * d_v,
            key=ks,
            n_heads=n_heads,
            d_k=d_k,
            d_v=d_v,
            tag=f"{tag}.set",
        )
        self.update = BoundaryGlobalUpdate(
            d_g,
            n_heads * d_v,
            key=ku,
            tag=f"{tag}.upd",
            tap_dim=tap_dim,
        )

    def __call__(self, g, edge, mask):

        system_active = jnp.any(mask.astype(bool))
        rows = jax.vmap(
            lambda ei, mi: self.row_pool(
                g,
                ei,
                mask,
                kfac_structural_mask=mi * mask,
                kfac_update_mask=system_active,
                kfac_scan_shared=False,
                kfac_repeat_ndim=2,
            )
        )(edge, mask)
        cols = jax.vmap(
            lambda ej, mj: self.col_pool(
                g,
                ej,
                mask,
                kfac_structural_mask=mj * mask,
                kfac_update_mask=system_active,
                kfac_scan_shared=False,
                kfac_repeat_ndim=2,
            )
        )(jnp.swapaxes(edge, 0, 1), mask)
        descs = jnp.concatenate([rows, cols], axis=0)
        dmask = jnp.concatenate([mask, mask], axis=0)
        pooled = self.set_pool(
            g,
            descs,
            dmask,
            kfac_structural_mask=dmask,
            kfac_update_mask=system_active,
            kfac_scan_shared=False,
            kfac_repeat_ndim=1,
        )
        return self.update(
            g,
            pooled,
            kfac_structural_mask=system_active,
            kfac_scan_shared=False,
        )
