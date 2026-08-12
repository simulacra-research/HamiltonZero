# Copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float


def quaternion_multiply(
    a: Float[Array, "... 4"],
    b: Float[Array, "... 4"],
) -> Float[Array, "... 4"]:
    w = (
        a[..., 0] * b[..., 0]
        - a[..., 1] * b[..., 1]
        - a[..., 2] * b[..., 2]
        - a[..., 3] * b[..., 3]
    )
    x = (
        a[..., 0] * b[..., 1]
        + a[..., 1] * b[..., 0]
        + a[..., 2] * b[..., 3]
        - a[..., 3] * b[..., 2]
    )
    y = (
        a[..., 0] * b[..., 2]
        - a[..., 1] * b[..., 3]
        + a[..., 2] * b[..., 0]
        + a[..., 3] * b[..., 1]
    )
    z = (
        a[..., 0] * b[..., 3]
        + a[..., 1] * b[..., 2]
        - a[..., 2] * b[..., 1]
        + a[..., 3] * b[..., 0]
    )
    return jnp.stack([w, x, y, z], axis=-1)


def quaternion_exp_tangent(
    v: Float[Array, "... 3"],
) -> Float[Array, "... 4"]:
    theta_sq = jnp.sum(v * v, axis=-1, keepdims=True)
    theta = jnp.sqrt(theta_sq)
    small = theta_sq < 1e-12
    sin_over_theta = jnp.where(
        small,
        1.0 - theta_sq / 6.0 + theta_sq * theta_sq / 120.0,
        jnp.sin(theta) / jnp.where(small, 1.0, theta),
    )
    real = jnp.where(
        small,
        1.0 - theta_sq / 2.0 + theta_sq * theta_sq / 24.0,
        jnp.cos(theta),
    )
    return jnp.concatenate([real, sin_over_theta * v], axis=-1)


def normalize_quaternion(
    q: Float[Array, "... 4"],
    eps: float = 1e-12,
) -> Float[Array, "... 4"]:
    return q / jnp.sqrt(jnp.sum(q * q, axis=-1, keepdims=True) + eps)


def quaternion_conjugate(
    q: Float[Array, "... 4"],
) -> Float[Array, "... 4"]:
    return jnp.concatenate([q[..., 0:1], -q[..., 1:]], axis=-1)


def quaternion_log(
    q: Float[Array, "... 4"],
    eps: float = 1e-8,
) -> Float[Array, "... 3"]:
    w = q[..., 0:1]
    v = q[..., 1:]
    sin_norm_sq = jnp.sum(v * v, axis=-1, keepdims=True)
    sin_norm = jnp.sqrt(sin_norm_sq)
    theta = jnp.arctan2(sin_norm, w)
    small = sin_norm < eps
    factor_small = 1.0 + sin_norm_sq / 6.0
    factor_normal = theta / jnp.where(
        small,
        jnp.ones_like(sin_norm),
        sin_norm,
    )
    factor = jnp.where(small, factor_small, factor_normal)
    return v * factor
