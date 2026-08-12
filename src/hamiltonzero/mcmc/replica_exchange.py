# Copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray

from ._monotone_cubic import MonotoneCubicInterpolation
from .langevin import langevin_step_one_cached
from .quaternion import normalize_quaternion


LogProbFn = Callable[[Float[Array, "N 4"]], Float[Array, ""]]


class REState(eqx.Module):
    q: Float[Array, "R N 4"]
    log_p: Float[Array, "R"]
    grad_log_p: Float[Array, "R N 4"]
    beta: Float[Array, "R"]
    sigma: Float[Array, "R"]
    step: Int[Array, ""]
    key: PRNGKeyArray
    n_local_accept: Float[Array, "R"]
    n_local: Int[Array, ""]
    n_swap_accept: Float[Array, "R_minus_1"]
    n_swap: Int[Array, "R_minus_1"]
    mask: Int[Array, "N"]
    m: Int[Array, "R"]
    n_haar_accept: Float[Array, "R"]
    n_haar: Int[Array, ""]


def annealing_geometric(
    n_replicas: int,
    base: float = 2.0**0.5,
) -> Float[Array, "R"]:
    if n_replicas < 2:
        raise ValueError("n_replicas must be at least two")
    if n_replicas == 2:
        return jnp.array([0.0, 1.0])
    powers = jnp.power(jnp.asarray(base), -jnp.arange(n_replicas - 1))
    tail = jnp.append(powers, 0.0)
    return tail[::-1]


def init_state(
    key: PRNGKeyArray,
    *,
    n_replicas: int,
    n_spins: int,
    sigma: float,
    mask: Int[Array, "N"],
    initial_m: int,
) -> REState:
    k_q, k_state = jax.random.split(key)
    q_raw = jax.random.normal(k_q, (n_replicas, n_spins, 4))
    q_init = normalize_quaternion(q_raw)
    beta = annealing_geometric(n_replicas)
    dtype = q_init.dtype
    sigma_arr = jnp.full((n_replicas,), sigma, dtype=dtype)
    log_p = jnp.full((n_replicas,), jnp.asarray(-1e30, dtype=dtype))
    grad_log_p = jnp.zeros_like(q_init)
    n_active = jnp.sum(mask).astype(jnp.int32)
    initial_m_clipped = jnp.minimum(
        jnp.maximum(jnp.asarray(initial_m, dtype=jnp.int32), jnp.int32(1)),
        jnp.maximum(n_active, jnp.int32(1)),
    )
    m_arr = jnp.broadcast_to(initial_m_clipped, (n_replicas,))
    is_hot = beta.astype(dtype) == jnp.asarray(0.0, dtype=dtype)
    m_arr = jnp.where(is_hot, n_active, m_arr)
    return REState(
        q=q_init,
        log_p=log_p,
        grad_log_p=grad_log_p,
        beta=beta.astype(dtype),
        sigma=sigma_arr,
        step=jnp.int32(0),
        key=k_state,
        n_local_accept=jnp.zeros(n_replicas, dtype=dtype),
        n_local=jnp.int32(0),
        n_swap_accept=jnp.zeros(n_replicas - 1, dtype=dtype),
        n_swap=jnp.zeros(n_replicas - 1, dtype=jnp.int32),
        mask=mask.astype(jnp.int32),
        m=m_arr,
        n_haar_accept=jnp.zeros(n_replicas, dtype=dtype),
        n_haar=jnp.int32(0),
    )


def _refresh_state_log_p_grad(
    state: REState,
    log_p_fn: LogProbFn,
) -> REState:
    log_p_and_grad = jax.vmap(jax.value_and_grad(log_p_fn))
    new_log_p, new_grad = log_p_and_grad(state.q)
    return eqx.tree_at(
        lambda value: (value.log_p, value.grad_log_p),
        state,
        (
            new_log_p.astype(state.log_p.dtype),
            new_grad.astype(state.grad_log_p.dtype),
        ),
    )


def _deo_permutation(
    edge_accept: Int[Array, "R_minus_1"],
) -> Int[Array, "R"]:
    replicas = edge_accept.shape[0] + 1
    zero = jnp.zeros((1,), dtype=edge_accept.dtype)
    left = jnp.concatenate([zero, edge_accept])
    right = jnp.concatenate([edge_accept, zero])
    index = jnp.arange(replicas, dtype=jnp.int32)
    index = jnp.where(left.astype(bool), index - 1, index)
    return jnp.where(right.astype(bool), index + 1, index)


def _swap_move(
    key: PRNGKeyArray,
    q: Float[Array, "R N 4"],
    log_p: Float[Array, "R"],
    grad_log_p: Float[Array, "R N 4"],
    beta: Float[Array, "R"],
    step: Int[Array, ""],
):
    replicas = q.shape[0]
    index = jnp.arange(replicas - 1)
    parity = step % 2
    edge_eligible = (index % 2) == parity
    lp_i, lp_j = log_p[:-1], log_p[1:]
    beta_i, beta_j = beta[:-1], beta[1:]
    log_alpha = (beta_j - beta_i) * (lp_i - lp_j)
    uniform = jnp.log(jax.random.uniform(key, (replicas - 1,), dtype=log_p.dtype))
    mh_edge = uniform < log_alpha
    nan_i = jnp.isnan(lp_i)
    nan_j = jnp.isnan(lp_j)
    nan_higher_only = nan_j & jnp.logical_not(nan_i)
    edge_accept = edge_eligible & (mh_edge | nan_higher_only)
    permutation = _deo_permutation(edge_accept.astype(jnp.int32))
    return (
        q[permutation],
        log_p[permutation],
        grad_log_p[permutation],
        edge_accept.astype(log_p.dtype),
        edge_eligible.astype(jnp.int32),
    )


def _global_haar_step(
    key: PRNGKeyArray,
    state: REState,
    log_p_fn: LogProbFn,
):
    replicas, n_sites, _ = state.q.shape
    dtype = state.q.dtype
    k_subset, k_haar, k_accept = jax.random.split(key, 3)
    scores = jax.random.uniform(k_subset, (replicas, n_sites), dtype=dtype)
    active = state.mask.astype(jnp.bool_)
    scores = jnp.where(
        active[None, :],
        scores,
        jnp.asarray(jnp.inf, dtype=dtype),
    )
    ranks = jnp.argsort(jnp.argsort(scores, axis=-1), axis=-1)
    move_mask = active[None, :] & (ranks < state.m[:, None])
    raw = jax.random.normal(k_haar, state.q.shape, dtype=dtype)
    q_haar = normalize_quaternion(raw)
    q_prop = jnp.where(move_mask[..., None], q_haar, state.q)
    log_p_prop, grad_log_p_prop = jax.vmap(jax.value_and_grad(log_p_fn))(q_prop)
    log_alpha = jnp.minimum(
        jnp.asarray(0.0, dtype=dtype),
        state.beta.astype(dtype) * (log_p_prop.astype(dtype) - state.log_p),
    )
    uniform = jax.random.uniform(k_accept, (replicas,), dtype=dtype)
    accept = jnp.log(uniform) <= log_alpha
    q_new = jnp.where(accept[:, None, None], q_prop, state.q)
    log_p_new = jnp.where(
        accept,
        log_p_prop.astype(dtype),
        state.log_p,
    )
    grad_log_p_new = jnp.where(
        accept[:, None, None],
        grad_log_p_prop,
        state.grad_log_p,
    )
    return (
        q_new,
        log_p_new,
        grad_log_p_new,
        accept.astype(state.n_haar_accept.dtype),
    )


def global_haar_step(
    state: REState,
    log_p_fn: LogProbFn,
) -> REState:
    k_haar, k_next = jax.random.split(state.key, 2)
    q_new, log_p_new, grad_log_p_new, accept = _global_haar_step(
        k_haar,
        state,
        log_p_fn,
    )
    return eqx.tree_at(
        lambda value: (
            value.q,
            value.log_p,
            value.grad_log_p,
            value.key,
            value.n_haar_accept,
            value.n_haar,
        ),
        state,
        (
            q_new,
            log_p_new,
            grad_log_p_new,
            k_next,
            state.n_haar_accept + accept,
            state.n_haar + jnp.int32(1),
        ),
    )


def step_re_langevin_cached(
    state: REState,
    log_p_fn: LogProbFn,
) -> REState:
    k_local, k_swap, k_next = jax.random.split(state.key, 3)
    local_keys = jax.random.split(k_local, state.q.shape[0])

    def local_move(key, q, log_p, grad_log_p, beta, sigma):
        return langevin_step_one_cached(
            key,
            q,
            log_p,
            grad_log_p,
            beta,
            sigma,
            state.mask,
            log_p_fn,
        )

    q_local, log_p_local, grad_local, accepts = jax.vmap(local_move)(
        local_keys,
        state.q,
        state.log_p,
        state.grad_log_p,
        state.beta,
        state.sigma,
    )
    q_swap, log_p_swap, grad_swap, edge_accept, edge_eligible = _swap_move(
        k_swap,
        q_local,
        log_p_local,
        grad_local,
        state.beta,
        state.step,
    )
    return REState(
        q=q_swap,
        log_p=log_p_swap,
        grad_log_p=grad_swap,
        beta=state.beta,
        sigma=state.sigma,
        step=state.step + 1,
        key=k_next,
        n_local_accept=state.n_local_accept + accepts,
        n_local=state.n_local + 1,
        n_swap_accept=state.n_swap_accept + edge_accept,
        n_swap=state.n_swap + edge_eligible,
        mask=state.mask,
        m=state.m,
        n_haar_accept=state.n_haar_accept,
        n_haar=state.n_haar,
    )


def run_re_langevin_cached(
    state: REState,
    log_p_fn: LogProbFn,
    n_steps: int,
) -> REState:
    state = _refresh_state_log_p_grad(state, log_p_fn)
    state = global_haar_step(state, log_p_fn)

    def body(carry, _unused):
        return step_re_langevin_cached(carry, log_p_fn), None

    state, _unused = jax.lax.scan(
        body,
        state,
        xs=None,
        length=n_steps,
    )
    return state


def _local_accept_rate(state: REState) -> Float[Array, "R"]:
    attempts = state.n_local.astype(state.sigma.dtype)[..., None]
    return state.n_local_accept / jnp.maximum(attempts, 1.0)


def _swap_accept_rate(state: REState) -> Float[Array, "R_minus_1"]:
    attempts = state.n_swap.astype(state.sigma.dtype)
    return state.n_swap_accept / jnp.maximum(attempts, 1.0)


def adapt_sigma(
    state: REState,
    *,
    target: float = 0.234,
    factor: float = 1.1,
    lo: float = 1e-4,
    hi: float = 3.14,
) -> REState:
    rate = _local_accept_rate(state)
    sigma_new = jnp.where(
        rate > target,
        state.sigma * factor,
        state.sigma / factor,
    )
    sigma_new = jnp.clip(sigma_new, lo, hi)
    return eqx.tree_at(
        lambda value: (
            value.sigma,
            value.n_local_accept,
            value.n_local,
        ),
        state,
        (
            sigma_new,
            jnp.zeros_like(state.n_local_accept),
            jnp.zeros_like(state.n_local),
        ),
    )


def _haar_accept_rate(state: REState) -> Float[Array, "R"]:
    denominator = jnp.maximum(
        state.n_haar.astype(state.n_haar_accept.dtype),
        jnp.asarray(1.0, dtype=state.n_haar_accept.dtype),
    )
    return state.n_haar_accept / denominator[..., None]


def adapt_m(
    state: REState,
    *,
    target: float = 0.234,
) -> REState:
    rate = _haar_accept_rate(state)
    n_active = jnp.sum(state.mask).astype(jnp.int32)
    delta = jnp.where(rate > target, jnp.int32(1), jnp.int32(-1))
    m_new = jnp.clip(state.m + delta, jnp.int32(1), n_active)
    is_hot = state.beta == jnp.asarray(0.0, dtype=state.beta.dtype)
    m_new = jnp.where(is_hot, n_active, m_new)
    return eqx.tree_at(
        lambda value: (
            value.m,
            value.n_haar_accept,
            value.n_haar,
        ),
        state,
        (
            m_new,
            jnp.zeros_like(state.n_haar_accept),
            jnp.zeros_like(state.n_haar),
        ),
    )


def _estimate_lambda_values(
    rejection_rates: Float[Array, "R_minus_1"],
    offset: float = 1e-4,
) -> Float[Array, "R"]:
    rejection_rates = jnp.maximum(rejection_rates, offset)
    extended = jnp.concatenate(
        [jnp.zeros_like(rejection_rates[..., :1]), rejection_rates],
        axis=-1,
    )
    return jnp.cumsum(extended, axis=-1)


def _annealing_optimal_hyman(
    n_replicas: int,
    previous_schedule: Float[Array, "R"],
    rejection_rates: Float[Array, "R_minus_1"],
    *,
    offset: float = 1e-4,
    ema: float = 0.99,
) -> Float[Array, "R"]:
    lambda_values = _estimate_lambda_values(
        rejection_rates,
        offset=offset,
    )
    lambda_fn = MonotoneCubicInterpolation(
        ts=previous_schedule,
        ys=lambda_values,
    )
    lambda_max = lambda_values[-1]
    dtype = previous_schedule.dtype
    indices = jnp.arange(1, n_replicas - 1, dtype=dtype)
    targets = indices * (lambda_max / (n_replicas - 1))
    lower = jnp.asarray(offset, dtype=dtype)
    upper = jnp.asarray(1.0 - offset, dtype=dtype)
    tolerance = jnp.asarray(offset * 1e-3, dtype=dtype)
    max_iterations = jnp.int32(50)

    def bisect(target):
        def condition(state):
            lo, hi, iteration = state
            return (iteration < max_iterations) & ((hi - lo) > tolerance)

        def body(state):
            lo, hi, iteration = state
            midpoint = (lo + hi) * jnp.asarray(0.5, dtype=dtype)
            value = lambda_fn.evaluate(midpoint) - target
            move_lower = value < 0.0
            new_lower = jnp.where(move_lower, midpoint, lo)
            new_upper = jnp.where(move_lower, hi, midpoint)
            return new_lower, new_upper, iteration + 1

        lo_final, hi_final, _iteration = jax.lax.while_loop(
            condition,
            body,
            (lower, upper, jnp.int32(0)),
        )
        return (lo_final + hi_final) * jnp.asarray(0.5, dtype=dtype)

    interior = jax.vmap(bisect)(targets)
    new_schedule = jnp.concatenate(
        [
            jnp.zeros((1,), dtype=dtype),
            interior,
            jnp.ones((1,), dtype=dtype),
        ]
    )
    return jnp.clip(
        (1.0 - ema) * new_schedule + ema * previous_schedule,
        0.0,
        1.0,
    ).astype(dtype)


def adapt_beta_equi_rej(
    state: REState,
    *,
    ema: float = 0.99,
) -> REState:
    rejection = 1.0 - _swap_accept_rate(state)
    beta_new = _annealing_optimal_hyman(
        n_replicas=int(state.beta.shape[0]),
        previous_schedule=state.beta,
        rejection_rates=rejection,
        ema=ema,
    )
    return eqx.tree_at(
        lambda value: (
            value.beta,
            value.n_swap_accept,
            value.n_swap,
        ),
        state,
        (
            beta_new,
            jnp.zeros_like(state.n_swap_accept),
            jnp.zeros_like(state.n_swap),
        ),
    )


__all__ = [
    "REState",
    "adapt_beta_equi_rej",
    "adapt_m",
    "adapt_sigma",
    "init_state",
    "run_re_langevin_cached",
]
