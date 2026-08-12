# Copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import fields
from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from .replica_exchange import (
    REState,
    adapt_beta_equi_rej,
    adapt_m,
    adapt_sigma,
    init_state,
    run_re_langevin_cached,
)


def _state_axes(mask_axis: int | None) -> REState:
    return REState(
        q=0,
        log_p=0,
        grad_log_p=0,
        beta=None,
        sigma=None,
        step=None,
        key=0,
        n_local_accept=0,
        n_local=0,
        n_swap_accept=0,
        n_swap=0,
        mask=mask_axis,
        m=None,
        n_haar_accept=0,
        n_haar=0,
    )


def init_batched_state(
    key: jax.Array,
    context: Any,
    batch_size: int,
    n_replicas: int,
    initial_m: int = 1,
    initial_sigma: float = 0.3,
) -> REState:
    mask = jnp.asarray(context.mask, dtype=jnp.int32)
    keys = jax.random.split(key, batch_size)
    return jax.vmap(
        lambda walker_key: init_state(
            walker_key,
            n_replicas=n_replicas,
            n_spins=int(mask.shape[-1]),
            sigma=initial_sigma,
            mask=mask,
            initial_m=initial_m,
        ),
        out_axes=_state_axes(None),
    )(keys)


def _log_probability(
    model: Callable,
    context: Any,
    q: jax.Array,
) -> jax.Array:
    real, _phase = model(q, context, 0.0)
    return 2.0 * real


def _run_one(
    state: REState,
    model: Callable,
    context: Any,
    n_steps: int,
) -> REState:
    return run_re_langevin_cached(
        state,
        lambda q: _log_probability(model, context, q),
        n_steps,
    )


def run_batched(
    model: Callable,
    context: Any,
    state: REState,
    n_steps: int,
    walker_chunk_size: int | None = None,
) -> REState:
    if walker_chunk_size is None or walker_chunk_size >= state.q.shape[0]:
        return jax.vmap(
            lambda walker: _run_one(walker, model, context, n_steps),
            in_axes=(_state_axes(None),),
            out_axes=_state_axes(None),
        )(state)

    shared_names = {"mask", "beta", "sigma", "step", "m"}
    shared = {name: getattr(state, name) for name in shared_names}
    walker_fields = {
        item.name: getattr(state, item.name)
        for item in fields(state)
        if item.name not in shared_names
    }

    def run_walker(walker: dict[str, jax.Array]) -> dict[str, jax.Array]:
        new_state = _run_one(
            REState(**shared, **walker),
            model,
            context,
            n_steps,
        )
        return {
            item.name: getattr(new_state, item.name)
            for item in fields(new_state)
            if item.name not in {"mask", "beta", "sigma", "m"}
        }

    mapped = jax.lax.map(
        run_walker,
        walker_fields,
        batch_size=int(walker_chunk_size),
    )
    step = mapped.pop("step")[0]
    return REState(**{**shared, "step": step}, **mapped)


def adapt_batched(
    state: REState,
    *,
    beta_history_weight: float = 0.9,
    sigma_target: float = 0.574,
    sigma_scale: float = 1.1,
    haar_target: float = 0.234,
) -> REState:
    pooled = REState(
        q=state.q[0],
        log_p=state.log_p[0],
        grad_log_p=state.grad_log_p[0],
        beta=state.beta,
        sigma=state.sigma,
        step=state.step,
        key=state.key[0],
        n_local_accept=state.n_local_accept.sum(axis=0),
        n_local=state.n_local.sum(axis=0).astype(state.n_local.dtype),
        n_swap_accept=state.n_swap_accept.sum(axis=0),
        n_swap=state.n_swap.sum(axis=0).astype(state.n_swap.dtype),
        mask=state.mask,
        m=state.m,
        n_haar_accept=state.n_haar_accept.sum(axis=0),
        n_haar=state.n_haar.sum(axis=0).astype(state.n_haar.dtype),
    )
    adapted = adapt_sigma(
        pooled,
        target=sigma_target,
        factor=sigma_scale,
    )
    adapted = adapt_m(adapted, target=haar_target)
    adapted = adapt_beta_equi_rej(
        adapted,
        ema=beta_history_weight,
    )

    def target(value: REState):
        return (
            value.sigma,
            value.beta,
            value.m,
            value.n_local_accept,
            value.n_local,
            value.n_swap_accept,
            value.n_swap,
            value.n_haar_accept,
            value.n_haar,
        )

    return eqx.tree_at(
        target,
        state,
        (
            adapted.sigma,
            adapted.beta,
            adapted.m,
            jnp.zeros_like(state.n_local_accept),
            jnp.zeros_like(state.n_local),
            jnp.zeros_like(state.n_swap_accept),
            jnp.zeros_like(state.n_swap),
            jnp.zeros_like(state.n_haar_accept),
            jnp.zeros_like(state.n_haar),
        ),
    )


def cold_samples(state: REState) -> jax.Array:
    return state.q[:, -1]


__all__ = [
    "REState",
    "adapt_batched",
    "cold_samples",
    "init_batched_state",
    "run_batched",
]
