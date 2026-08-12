# Copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


def _endpoint_one_sided(
    h0: Array,
    h1: Array,
    m0: Array,
    m1: Array,
) -> Array:
    d = ((2.0 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)
    d = jnp.where(jnp.sign(d) != jnp.sign(m0), jnp.zeros_like(d), d)
    clamp = (jnp.sign(m0) != jnp.sign(m1)) & (jnp.abs(d) > 3.0 * jnp.abs(m0))
    return jnp.where(clamp, 3.0 * m0, d)


def _hyman_knot_derivs(
    ts: Float[Array, "n"],
    ys: Float[Array, "n"],
) -> Float[Array, "n"]:
    h = jnp.diff(ts)
    m = jnp.diff(ys) / h
    h_prev = h[:-1]
    h_next = h[1:]
    m_prev = m[:-1]
    m_next = m[1:]
    d_natural = (h_next * m_prev + h_prev * m_next) / (h_prev + h_next)
    same_sign = m_prev * m_next > 0
    abs_min = jnp.minimum(jnp.abs(m_prev), jnp.abs(m_next))
    sign_m = jnp.sign(m_next)
    interior = jnp.where(
        same_sign,
        sign_m * jnp.minimum(jnp.abs(d_natural), 3.0 * abs_min),
        jnp.zeros_like(d_natural),
    )
    d_left = _endpoint_one_sided(h[0], h[1], m[0], m[1])
    d_right = _endpoint_one_sided(h[-1], h[-2], m[-1], m[-2])
    return jnp.concatenate([d_left[None], interior, d_right[None]])


def _hermite_eval(
    ts: Float[Array, "n"],
    ys: Float[Array, "n"],
    ds: Float[Array, "n"],
    t: Float[Array, ""],
) -> Float[Array, ""]:
    n = ts.shape[0]
    idx = jnp.clip(jnp.searchsorted(ts, t, side="right") - 1, 0, n - 2)
    x_lo = ts[idx]
    x_hi = ts[idx + 1]
    y_lo = ys[idx]
    y_hi = ys[idx + 1]
    d_lo = ds[idx]
    d_hi = ds[idx + 1]
    h = x_hi - x_lo
    tau = (t - x_lo) / h
    h00 = 2.0 * tau**3 - 3.0 * tau**2 + 1.0
    h10 = tau**3 - 2.0 * tau**2 + tau
    h01 = -2.0 * tau**3 + 3.0 * tau**2
    h11 = tau**3 - tau**2
    return y_lo * h00 + h * d_lo * h10 + y_hi * h01 + h * d_hi * h11


class MonotoneCubicInterpolation(eqx.Module):
    ts: Float[Array, "n"]
    ys: Float[Array, "n"]
    ds: Float[Array, "n"]

    def __init__(
        self,
        ts: Float[Array, "n"],
        ys: Float[Array, "n"],
    ) -> None:
        self.ts = ts
        self.ys = ys
        self.ds = _hyman_knot_derivs(ts, ys)

    def evaluate(self, t: Float[Array, ""]) -> Float[Array, ""]:
        return _hermite_eval(self.ts, self.ys, self.ds, t)
