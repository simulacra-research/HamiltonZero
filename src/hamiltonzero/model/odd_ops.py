# Copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import math
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


def bounded_gain_logit(
    value: float,
    *,
    max_gain: float,
    init_fraction: float | None,
) -> float:
    maximum = float(max_gain)
    initial = float(value)
    if not (math.isfinite(maximum) and maximum > 0.0):
        raise ValueError(f"bounded update maximum must be positive, got {maximum}")
    if init_fraction is None:
        if not (math.isfinite(initial) and 0.0 < initial < maximum):
            raise ValueError(
                f"bounded update initial value must be in (0, {maximum}), got {initial}"
            )
        fraction = initial / maximum
    else:
        fraction = float(init_fraction)
        if not (math.isfinite(fraction) and 0.0 < fraction < 1.0):
            raise ValueError(
                f"bounded update initial fraction must be in (0, 1), got {fraction}"
            )
    return math.log(fraction) - math.log1p(-fraction)


class BiasFreeLinear(eqx.Module):
    weight: Float[Array, "in out"]
    _use_id: str = eqx.field(static=True, default="")

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        key: PRNGKeyArray,
        scale: float | None = None,
    ):
        std = scale if scale is not None else in_features ** (-0.5)
        self.weight = jax.random.normal(key, (in_features, out_features)) * std
        self._use_id = ""

    def __call__(
        self,
        x: Float[Array, "... in"],
        *,
        pathway: str | None = None,
        kfac_structural_mask=None,
        kfac_repeat_ndim: int = 0,
        kfac_context_primal_reused_over_walkers: bool = False,
    ) -> Float[Array, "... out"]:
        from hamiltonzero.model.tree import _tagged_dense_no_bias

        if pathway is None:
            pathway = "even"
        return _tagged_dense_no_bias(
            self.weight,
            x,
            tag_id=self._use_id,
            pathway=pathway,
            kfac_structural_mask=kfac_structural_mask,
            kfac_scan_shared=False,
            kfac_repeat_ndim=kfac_repeat_ndim,
            kfac_context_primal_reused_over_walkers=kfac_context_primal_reused_over_walkers,
        )


class Linear(eqx.Module):
    weight: Float[Array, "in out"]
    bias: Float[Array, "out"]
    _use_id: str = eqx.field(static=True, default="")

    def __init__(self, in_features: int, out_features: int, *, key: PRNGKeyArray):
        std = in_features ** (-0.5)
        self.weight = jax.random.normal(key, (in_features, out_features)) * std
        self.bias = jnp.zeros((out_features,))
        self._use_id = ""

    def __call__(
        self,
        x: Float[Array, "... in"],
        *,
        pathway: str | None = None,
        kfac_structural_mask=None,
        kfac_repeat_ndim: int = 0,
        kfac_context_primal_reused_over_walkers: bool = False,
    ) -> Float[Array, "... out"]:
        from hamiltonzero.model.tree import _tagged_dense

        if pathway is None:
            pathway = "even"
        return _tagged_dense(
            self.weight,
            self.bias,
            x,
            tag_id=self._use_id,
            pathway=pathway,
            kfac_structural_mask=kfac_structural_mask,
            kfac_scan_shared=False,
            kfac_repeat_ndim=kfac_repeat_ndim,
            kfac_context_primal_reused_over_walkers=kfac_context_primal_reused_over_walkers,
        )


class HypernetMatrix(eqx.Module):
    U: Float[Array, "R d_out"]
    V: Float[Array, "d_in R"]
    W_h: Float[Array, "d_e R"]
    _use_id_U: str = eqx.field(static=True, default="")
    _use_id_V: str = eqx.field(static=True, default="")
    _use_id_W_h: str = eqx.field(static=True, default="")

    def __init__(
        self, d_in: int, d_out: int, d_e: int, rank: int, *, key: PRNGKeyArray
    ):
        k_u, k_v, k_h = jax.random.split(key, 3)
        self.U = jax.random.normal(k_u, (rank, d_out)) * rank ** (-0.5)
        self.V = jax.random.normal(k_v, (d_in, rank)) * d_in ** (-0.5)
        self.W_h = jax.random.normal(k_h, (d_e, rank)) * d_e ** (-0.5)
        self._use_id_U = ""
        self._use_id_V = ""
        self._use_id_W_h = ""

    def apply(
        self,
        e: Float[Array, "d_e"],
        z: Float[Array, "d_in"],
        *,
        e_pathway: str | None = None,
        kfac_structural_mask=None,
        kfac_scan_shared: bool = False,
        kfac_repeat_ndim: int = 0,
        kfac_context_primal_reused_over_walkers: bool = False,
        kfac_all_primals_reused_over_walkers: bool = False,
    ) -> Float[Array, "d_out"]:
        from hamiltonzero.model.tree import _tagged_dense_no_bias

        eff_e_pathway = e_pathway if e_pathway is not None else "even"
        h = _tagged_dense_no_bias(
            self.W_h,
            e,
            tag_id=self._use_id_W_h,
            pathway=eff_e_pathway,
            kfac_structural_mask=kfac_structural_mask,
            kfac_scan_shared=kfac_scan_shared,
            kfac_repeat_ndim=kfac_repeat_ndim,
            kfac_context_primal_reused_over_walkers=kfac_context_primal_reused_over_walkers
            or kfac_all_primals_reused_over_walkers,
        )
        Vz = _tagged_dense_no_bias(
            self.V,
            z,
            tag_id=self._use_id_V,
            pathway="odd",
            kfac_structural_mask=kfac_structural_mask,
            kfac_scan_shared=kfac_scan_shared,
            kfac_repeat_ndim=kfac_repeat_ndim,
            kfac_context_primal_reused_over_walkers=kfac_all_primals_reused_over_walkers,
        )
        m = h * Vz
        return _tagged_dense_no_bias(
            self.U,
            m,
            tag_id=self._use_id_U,
            pathway="odd",
            kfac_structural_mask=kfac_structural_mask,
            kfac_scan_shared=kfac_scan_shared,
            kfac_repeat_ndim=kfac_repeat_ndim,
            kfac_context_primal_reused_over_walkers=kfac_all_primals_reused_over_walkers,
        )


class _RMS(eqx.Module):
    weight: Float[Array, "d"]
    eps: float = eqx.field(static=True, default=1e-05)
    _use_id: str = eqx.field(static=True, default="")

    def __init__(self, d: int):
        self.weight = jnp.ones((d,))
        self._use_id = ""

    def __call__(
        self,
        x: Float[Array, "... d"],
        *,
        pathway: str | None = None,
        kfac_structural_mask=None,
        kfac_repeat_ndim: int = 0,
        kfac_context_primal_reused_over_walkers: bool = False,
    ) -> Float[Array, "... d"]:
        from hamiltonzero.model.tree import _tagged_rms_eqx_style

        if pathway is None:
            pathway = "even"
        return _tagged_rms_eqx_style(
            self.weight,
            x,
            self.eps,
            tag_id=self._use_id,
            pathway=pathway,
            kfac_structural_mask=kfac_structural_mask,
            kfac_scan_shared=False,
            kfac_repeat_ndim=kfac_repeat_ndim,
            kfac_context_primal_reused_over_walkers=kfac_context_primal_reused_over_walkers,
        )


class MLP(eqx.Module):
    in_proj: Linear
    block_norms: list
    block_l1s: list
    block_l2s: list
    out_norm: _RMS
    out_proj: Linear
    inner_gain: float = eqx.field(static=True)

    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        d_out: int,
        *,
        key: PRNGKeyArray,
        n_blocks: int = 2,
        inner_gain: float = 1.0,
    ):
        keys = jax.random.split(key, 2 + 2 * n_blocks)
        self.in_proj = Linear(d_in, d_hidden, key=keys[0])
        self.block_norms = [_RMS(d_hidden) for _ in range(n_blocks)]
        self.block_l1s = [
            Linear(d_hidden, d_hidden, key=keys[1 + 2 * i]) for i in range(n_blocks)
        ]
        self.block_l2s = [
            Linear(d_hidden, d_hidden, key=keys[2 + 2 * i]) for i in range(n_blocks)
        ]
        self.out_norm = _RMS(d_hidden)
        self.out_proj = Linear(d_hidden, d_out, key=keys[-1])
        self.inner_gain = float(inner_gain)

    def _act(self, x):
        return x * jax.nn.sigmoid(x)

    def __call__(
        self,
        x: Float[Array, "... d_in"],
        *,
        pathway: str | None = None,
        kfac_structural_mask=None,
        kfac_repeat_ndim: int = 0,
        kfac_context_primal_reused_over_walkers: bool = False,
    ) -> Float[Array, "... d_out"]:
        kfac_kwargs = dict(
            kfac_structural_mask=kfac_structural_mask,
            kfac_repeat_ndim=kfac_repeat_ndim,
            kfac_context_primal_reused_over_walkers=kfac_context_primal_reused_over_walkers,
        )
        x = self.in_proj(x, pathway=pathway, **kfac_kwargs)
        for nrm, l1, l2 in zip(self.block_norms, self.block_l1s, self.block_l2s):
            normed = nrm(x, pathway=pathway, **kfac_kwargs)
            x = x + self.inner_gain * l2(
                self._act(l1(normed, pathway=pathway, **kfac_kwargs)),
                pathway=pathway,
                **kfac_kwargs,
            )
        out_normed = self.out_norm(x, pathway=pathway, **kfac_kwargs)
        return self.out_proj(out_normed, pathway=pathway, **kfac_kwargs)


class UnnormalizedMLP(eqx.Module):
    in_proj: Linear
    block_l1s: list
    block_l2s: list
    out_proj: Linear
    inner_gain: float = eqx.field(static=True)

    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        d_out: int,
        *,
        key: PRNGKeyArray,
        n_blocks: int = 1,
        inner_gain: float = 1.0,
    ):
        keys = jax.random.split(key, 2 + 2 * n_blocks)
        self.in_proj = Linear(d_in, d_hidden, key=keys[0])
        self.block_l1s = [
            Linear(d_hidden, d_hidden, key=keys[1 + 2 * i]) for i in range(n_blocks)
        ]
        self.block_l2s = [
            Linear(d_hidden, d_hidden, key=keys[2 + 2 * i]) for i in range(n_blocks)
        ]
        self.out_proj = Linear(d_hidden, d_out, key=keys[-1])
        self.inner_gain = float(inner_gain)

    def _act(self, x):
        return x * jax.nn.sigmoid(x)

    def __call__(
        self,
        x: Float[Array, "... d_in"],
        *,
        pathway: str | None = None,
        kfac_structural_mask=None,
        kfac_repeat_ndim: int = 0,
        kfac_context_primal_reused_over_walkers: bool = False,
    ) -> Float[Array, "... d_out"]:
        kfac_kwargs = dict(
            kfac_structural_mask=kfac_structural_mask,
            kfac_repeat_ndim=kfac_repeat_ndim,
            kfac_context_primal_reused_over_walkers=kfac_context_primal_reused_over_walkers,
        )
        x = self.in_proj(x, pathway=pathway, **kfac_kwargs)
        for l1, l2 in zip(self.block_l1s, self.block_l2s):
            x = x + self.inner_gain * l2(
                self._act(l1(x, pathway=pathway, **kfac_kwargs)),
                pathway=pathway,
                **kfac_kwargs,
            )
        return self.out_proj(x, pathway=pathway, **kfac_kwargs)
