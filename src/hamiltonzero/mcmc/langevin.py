# Copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray

from .quaternion import (
    normalize_quaternion,
    quaternion_conjugate,
    quaternion_exp_tangent,
    quaternion_log,
    quaternion_multiply,
)


LogProbFn = Callable[[Float[Array, "N 4"]], Float[Array, ""]]
N_WRAP = 8


def _su2_left_frame(
    q: Float[Array, "N 4"],
) -> Float[Array, "N 3 4"]:
    q0, q1, q2, q3 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    e1 = jnp.stack([-q1, q0, q3, -q2], axis=-1)
    e2 = jnp.stack([-q2, -q3, q0, q1], axis=-1)
    e3 = jnp.stack([-q3, q2, -q1, q0], axis=-1)
    return jnp.stack([e1, e2, e3], axis=1)


def _drift_lie(
    q: Float[Array, "N 4"],
    grad_log_p: Float[Array, "N 4"],
) -> Float[Array, "N 3"]:
    e_unit = _su2_left_frame(q)
    return jnp.einsum("iaα,iα->ia", e_unit, grad_log_p)


def _log_jacobian_su2(
    norm: Float[Array, "..."],
) -> Float[Array, "..."]:
    eps = jnp.asarray(1e-9, dtype=norm.dtype)
    norm_safe = jnp.maximum(norm, eps)
    abs_sin = jnp.abs(jnp.sin(norm_safe))
    small = norm < jnp.asarray(1e-3, dtype=norm.dtype)
    val_taylor = -norm * norm / 6.0
    val_full = jnp.log(jnp.maximum(abs_sin, eps)) - jnp.log(norm_safe)
    log_sinc_abs = jnp.where(small, val_taylor, val_full)
    return 2.0 * log_sinc_abs


def _wrapped_log_q(
    xi_p: Float[Array, "N 3"],
    mu: Float[Array, "N 3"],
    sigma_sq: Float[Array, ""],
    mask: Int[Array, "N"],
) -> Float[Array, ""]:
    eps = jnp.asarray(1e-9, dtype=xi_p.dtype)
    norm = jnp.linalg.norm(xi_p, axis=-1, keepdims=True)
    small = norm < eps
    safe_norm = jnp.where(small, jnp.ones_like(norm), norm)
    default_u = jnp.broadcast_to(
        jnp.asarray([1.0, 0.0, 0.0], dtype=xi_p.dtype),
        xi_p.shape,
    )
    u_hat = jnp.where(small, default_u, xi_p / safe_norm)
    n_range = jnp.arange(-N_WRAP, N_WRAP + 1).astype(xi_p.dtype)
    two_pi = jnp.asarray(2.0 * jnp.pi, dtype=xi_p.dtype)
    branches = xi_p[None, :, :] + two_pi * n_range[:, None, None] * u_hat[None, :, :]
    diffs = branches - mu[None, :, :]
    log_gaussian = -jnp.sum(diffs * diffs, axis=-1) / (2.0 * sigma_sq)
    norm_branches = jnp.linalg.norm(branches, axis=-1)
    log_jacobian = _log_jacobian_su2(norm_branches)
    log_kernel = log_gaussian - log_jacobian
    log_q_per_site = jax.scipy.special.logsumexp(log_kernel, axis=0)
    site_mask_f = mask.astype(xi_p.dtype)
    return jnp.sum(log_q_per_site * site_mask_f)


def langevin_step_one_cached(
    key: PRNGKeyArray,
    q: Float[Array, "N 4"],
    log_p: Float[Array, ""],
    grad_log_p: Float[Array, "N 4"],
    beta: Float[Array, ""],
    sigma: Float[Array, ""],
    mask: Int[Array, "N"],
    log_p_fn: LogProbFn,
) -> tuple[
    Float[Array, "N 4"],
    Float[Array, ""],
    Float[Array, "N 4"],
    Float[Array, ""],
]:
    k_prop, k_acc = jax.random.split(key)
    n_spins = q.shape[0]
    site_mask_f = mask.astype(q.dtype)[:, None]
    site_mask_b = mask.astype(jnp.bool_)[:, None]
    drift_q = _drift_lie(q, grad_log_p)
    drift_q = drift_q * site_mask_f
    noise = jax.random.normal(k_prop, (n_spins, 3), dtype=q.dtype)
    noise = noise * site_mask_f
    sigma_sq = sigma * sigma
    mu_fwd = (sigma_sq / 2.0) * beta * drift_q
    xi_fwd = mu_fwd + sigma * noise
    delta = quaternion_exp_tangent(xi_fwd)
    q_prop = normalize_quaternion(quaternion_multiply(q, delta))
    delta_back = quaternion_multiply(quaternion_conjugate(q_prop), q)
    xi_p_back = quaternion_log(delta_back)
    xi_p_back = xi_p_back * site_mask_f
    xi_p_fwd = -xi_p_back
    log_p_at_qprop, grad_log_p_at_qprop = jax.value_and_grad(log_p_fn)(q_prop)
    drift_qprop = _drift_lie(q_prop, grad_log_p_at_qprop)
    drift_qprop = drift_qprop * site_mask_f
    mu_back = (sigma_sq / 2.0) * beta * drift_qprop
    log_q_fwd = _wrapped_log_q(xi_p_fwd, mu_fwd, sigma_sq, mask)
    log_q_back = _wrapped_log_q(xi_p_fwd, -mu_back, sigma_sq, mask)
    log_alpha = beta * (log_p_at_qprop - log_p) + log_q_back - log_q_fwd
    u = jnp.log(jax.random.uniform(k_acc, dtype=q.dtype))
    mh_accept = u < log_alpha
    nan_recovery = jnp.isnan(log_p) & jnp.logical_not(jnp.isnan(log_p_at_qprop))
    accept = mh_accept | nan_recovery
    q_new = jnp.where(accept, q_prop, q)
    q_new = jnp.where(site_mask_b, q_new, q)
    log_p_new = jnp.where(accept, log_p_at_qprop, log_p)
    grad_log_p_new = jnp.where(
        accept,
        grad_log_p_at_qprop,
        grad_log_p,
    )
    return q_new, log_p_new, grad_log_p_new, accept.astype(log_p.dtype)


__all__ = ["langevin_step_one_cached"]
