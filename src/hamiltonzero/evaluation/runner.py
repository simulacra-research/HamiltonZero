# Copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from hamiltonzero.config import EvalConfig

from .backend import EvalBackend, MCMCPopulation
from .statistics import EnergyWindow, P01_TWO_SIDED, select_winner_per_physical
from .types import ContestCandidate, ContestResult, EvalMetric, EvalResult


def _permute_q_prefix(q, permutations):
    index = jnp.broadcast_to(permutations[:, None, None, :, None], q.shape)
    return jnp.take_along_axis(q, index, axis=3)


def _collapse_to_winner(q_virtual, permutations, winner_idx, *, P, K):
    winner_idx = jnp.asarray(winner_idx, dtype=jnp.int32)
    inverse = jnp.argsort(permutations, axis=-1)
    q_canonical = _permute_q_prefix(q_virtual, inverse)
    batch_per_candidate = q_canonical.shape[1]
    tail = q_canonical.shape[2:]
    q_physical = q_canonical.reshape((P, K, batch_per_candidate) + tail).reshape(
        (P, K * batch_per_candidate) + tail
    )
    permutations_pk = permutations.reshape((P, K, -1))
    route = jnp.take_along_axis(
        permutations_pk,
        winner_idx[:, None, None],
        axis=1,
    )[:, 0]
    return _permute_q_prefix(q_physical, route), route


def _gather_winner_ladder(values, winner_idx, *, P, K):
    winner_idx = jnp.asarray(winner_idx, dtype=jnp.int32)
    values_pk = values.reshape((P, K) + values.shape[1:])
    index = winner_idx[(slice(None), None) + (None,) * (values_pk.ndim - 2)]
    return jnp.take_along_axis(values_pk, index, axis=1)[:, 0]


def _as_batched_route(permutation):
    value = jnp.asarray(permutation, dtype=jnp.int32)
    if value.ndim == 1:
        value = value[None, :]
    if value.ndim != 2 or value.shape[0] != 1:
        raise ValueError("single-system evaluation requires a route with shape [1, N]")
    return value


def _compose_walker_route(old_inverse, route):
    return jnp.take_along_axis(
        jnp.asarray(old_inverse, dtype=jnp.int32),
        route,
        axis=-1,
    )


def _adapt(backend: EvalBackend, state: Any, config: EvalConfig):
    return backend.adapt_mcmc(state, config.mcmc)


def _burn_in(
    backend: EvalBackend,
    state: Any,
    model: Any,
    context: Any,
    config: EvalConfig,
    *,
    iterations: int,
    replica_steps: int,
):
    for _ in range(int(iterations)):
        state = backend.step_mcmc(
            state,
            model,
            context,
            replica_steps=int(replica_steps),
            walker_chunk_size=int(config.mcmc.walker_chunk_size),
        )
        backend.block_until_ready(backend.cold_walkers(state))
        state = _adapt(backend, state, config)
    return state


def _measure(
    backend: EvalBackend,
    state: Any,
    model: Any,
    context: Any,
    config: EvalConfig,
    *,
    started: float,
    metric_sink: Callable[[EvalMetric], None] | None,
):
    window = EnergyWindow(
        config.measurements,
        systems=1,
        batch_size=config.mcmc.batch_size,
    )
    for step in range(config.measurements):
        step_started = time.perf_counter()
        state = backend.step_mcmc(
            state,
            model,
            context,
            replica_steps=int(config.mcmc.steps),
            walker_chunk_size=int(config.mcmc.walker_chunk_size),
        )
        q_cold = backend.cold_walkers(state)
        total, exchange, _casimir, field = backend.custom_lap_energy(
            model,
            context,
            q_cold,
            config.energy,
        )
        backend.block_until_ready(total)
        window.push(total, exchange, field)
        state = _adapt(backend, state, config)
        if metric_sink is not None:
            energy = np.asarray(total).real
            metric_sink(
                EvalMetric(
                    step=step,
                    energy=float(np.mean(energy)),
                    energy_std=float(np.std(energy)),
                    step_walltime=float(time.perf_counter() - step_started),
                    walltime=float(time.perf_counter() - started),
                )
            )
    return state, window


def _ordinary(
    backend: EvalBackend,
    model: Any,
    context: Any,
    canonical,
    mcmc_key,
    config: EvalConfig,
):
    state = backend.initialize_mcmc(mcmc_key, model, context, config.mcmc)
    candidates = backend.beam_candidates(
        model,
        canonical.context,
        beam_width=int(config.contest_beam_width),
        top_k=1,
        temperature=float(config.route_temperature),
    )
    permutations = jnp.asarray(candidates.permutations, dtype=jnp.int32)
    if permutations.ndim != 3 or permutations.shape[:2] != (1, 1):
        raise ValueError("ordinary eval router must return shape [1, 1, N]")
    route = permutations[:, 0]
    walker_route = _compose_walker_route(canonical.old_inverse, route)
    routed_context = backend.route_context(
        canonical.context,
        route,
        compact_custom_lap=False,
    )
    state = backend.route_mcmc(state, walker_route)
    wavefunction = backend.compile_single(model, routed_context)
    backend.block_until_ready(wavefunction)
    logp = float(np.asarray(candidates.log_probabilities)[0, 0])
    return wavefunction, routed_context, state, route, logp, None


def _compiled_finetune_ordinary(
    backend: EvalBackend,
    model: Any,
    context: Any,
    canonical,
    embedded_route,
    mcmc_key,
    config: EvalConfig,
):
    state = backend.initialize_mcmc(mcmc_key, model, context, config.mcmc)
    route = _as_batched_route(embedded_route)
    if route.shape[-1] != canonical.context.mask.shape[-1]:
        raise ValueError(
            "compiled fine-tune route width does not match the evaluation system"
        )
    walker_route = _compose_walker_route(canonical.old_inverse, route)
    routed_context = backend.route_context(
        canonical.context,
        route,
        compact_custom_lap=False,
    )
    state = backend.route_mcmc(state, walker_route)
    wavefunction = backend.compile_embedded(model)
    backend.block_until_ready(wavefunction)
    return wavefunction, routed_context, state, route, None, None


def _contest(
    backend: EvalBackend,
    model: Any,
    canonical,
    root_key,
    config: EvalConfig,
):
    K = int(config.contest_candidates)
    batch_per_candidate = int(config.mcmc.batch_size) // K
    candidates = backend.beam_candidates(
        model,
        canonical.context,
        beam_width=int(config.contest_beam_width),
        top_k=K,
        temperature=float(config.route_temperature),
    )
    beam_permutations = jnp.asarray(candidates.permutations, dtype=jnp.int32)
    if beam_permutations.ndim != 3 or beam_permutations.shape[:2] != (1, K):
        raise ValueError(f"contest router must return shape [1, {K}, N]")
    n_sites = int(beam_permutations.shape[-1])
    permutations = beam_permutations.reshape((K, n_sites))
    virtual_context = backend.virtual_context(canonical.context, permutations)
    race_mcmc = replace(config.mcmc, batch_size=batch_per_candidate)
    state = backend.initialize_mcmc(
        jax.random.fold_in(root_key, 7411),
        model,
        virtual_context,
        race_mcmc,
    )
    wavefunctions = backend.compile_candidates(
        model,
        canonical.context,
        permutations,
    )
    backend.block_until_ready(wavefunctions)
    for _ in range(int(config.contest_preburn)):
        state = backend.step_mcmc(
            state,
            wavefunctions,
            virtual_context,
            replica_steps=int(config.mcmc.burn_in_replica_steps),
            walker_chunk_size=int(config.mcmc.walker_chunk_size),
        )
        backend.block_until_ready(backend.cold_walkers(state))
        state = backend.adapt_mcmc(state, race_mcmc)
    race_window = EnergyWindow(
        config.contest_measurements,
        systems=K,
        batch_size=batch_per_candidate,
    )
    for _ in range(int(config.contest_measurements)):
        state = backend.step_mcmc(
            state,
            wavefunctions,
            virtual_context,
            replica_steps=int(config.mcmc.steps),
            walker_chunk_size=int(config.mcmc.walker_chunk_size),
        )
        state = backend.adapt_mcmc(state, race_mcmc)
        q_cold = backend.cold_walkers(state)
        total, exchange, _casimir, field = backend.custom_lap_energy(
            wavefunctions,
            virtual_context,
            q_cold,
            config.energy,
        )
        backend.block_until_ready(total)
        race_window.push(total, exchange, field)
    energies = np.asarray(
        [[race_window.tail_mean("total", candidate) for candidate in range(K)]]
    )
    tailstd = np.asarray(
        [[race_window.tail_std("total", candidate) for candidate in range(K)]]
    )
    beam_logp = np.asarray(candidates.log_probabilities, dtype=float)
    winners, ties, reasons, _bands, standard_errors = select_winner_per_physical(
        energies,
        tailstd,
        beam_logp,
        batch_per_candidate,
        z=P01_TWO_SIDED,
        ucb_z=float(config.contest_se_multiplier),
    )
    winner = int(winners[0])
    wavefunction = backend.select_candidate(wavefunctions, winner)
    population = backend.mcmc_population(state)
    q_final, route = _collapse_to_winner(
        population.q,
        permutations,
        winners,
        P=1,
        K=K,
    )
    sigma = _gather_winner_ladder(population.sigma, winners, P=1, K=K)
    beta = _gather_winner_ladder(population.beta, winners, P=1, K=K)
    routed_context = backend.route_context(
        canonical.context,
        route,
        compact_custom_lap=False,
    )
    final_state = backend.initialize_mcmc(
        jax.random.fold_in(root_key, 7919),
        wavefunction,
        routed_context,
        config.mcmc,
    )
    final_state = backend.replace_mcmc_population(
        final_state,
        MCMCPopulation(q=q_final, sigma=sigma, beta=beta),
    )
    backend.block_until_ready(backend.cold_walkers(final_state))
    contest_candidates = tuple(
        ContestCandidate(
            index=index,
            route_log_probability=float(beam_logp[0, index]),
            energy=float(energies[0, index]),
            standard_error=float(standard_errors[0, index]),
            walker_tail_std=float(tailstd[0, index]),
            in_tie_set=bool(ties[0, index]),
        )
        for index in range(K)
    )
    contest = ContestResult(
        winner=winner,
        reason=reasons[0],
        candidates=contest_candidates,
    )
    backend.release_context(virtual_context)
    return (
        wavefunction,
        routed_context,
        final_state,
        route,
        float(beam_logp[0, winner]),
        contest,
    )


def _large_n(
    backend: EvalBackend,
    model: Any,
    context: Any,
    canonical,
    mcmc_key,
    config: EvalConfig,
):
    state = backend.initialize_mcmc(mcmc_key, model, context, config.mcmc)
    compiled = backend.compile_large_n(
        model,
        canonical.context,
        sequence_shards=int(config.large_n_sequence_shards),
        pair_tile_size=int(config.large_n_pair_tile_size),
        temperature=float(config.route_temperature),
    )
    route = _as_batched_route(compiled.permutation)
    walker_route = _compose_walker_route(canonical.old_inverse, route)
    routed_context = backend.route_context(
        canonical.context,
        route,
        compact_custom_lap=True,
    )
    state = backend.route_mcmc(state, walker_route)
    backend.block_until_ready(compiled.wavefunction)
    logp = float(np.asarray(compiled.log_probability))
    return compiled.wavefunction, routed_context, state, route, logp, None


def _validate(config: EvalConfig) -> None:
    if config.contest and config.large_n:
        raise ValueError("contest and large_n are mutually exclusive")
    if int(config.measurements) < 1:
        raise ValueError("measurements must be positive")
    if int(config.mcmc.batch_size) < 1:
        raise ValueError("MCMC batch size must be positive")
    if int(config.mcmc.replicas) < 2:
        raise ValueError("MCMC requires at least two replicas")
    if int(config.mcmc.steps) < 1:
        raise ValueError("MCMC replica steps must be positive")
    if int(config.mcmc.burn_in_replica_steps) < 1:
        raise ValueError("burn-in replica steps must be positive")
    if int(config.mcmc.walker_chunk_size) < 1:
        raise ValueError("walker chunk size must be positive")
    if config.contest:
        K = int(config.contest_candidates)
        W = int(config.contest_beam_width)
        if K < 2 or W < K:
            raise ValueError("contest requires beam_width >= candidates >= 2")
        if int(config.mcmc.batch_size) % K:
            raise ValueError("MCMC batch size must be divisible by candidates")
        if int(config.mcmc.batch_size) // K < 32:
            raise ValueError("contest requires at least 32 walkers per candidate")
        if int(config.contest_preburn) < 0:
            raise ValueError("contest preburn must be non-negative")
        if int(config.contest_measurements) < 1:
            raise ValueError("contest measurements must be positive")
    if int(config.large_n_sequence_shards) < 0:
        raise ValueError("large-N sequence shards must be non-negative")
    if int(config.large_n_pair_tile_size) < 1:
        raise ValueError("large-N pair tile size must be positive")


def evaluate(
    config: EvalConfig,
    backend: EvalBackend,
    *,
    metric_sink: Callable[[EvalMetric], None] | None = None,
) -> EvalResult:
    _validate(config)
    started = time.perf_counter()
    root_key = jax.random.PRNGKey(int(config.seed))
    model_key, mcmc_key = jax.random.split(root_key)
    context = backend.load_system(config.system, config.energy)
    model = backend.load_model(
        config.checkpoint,
        config.model,
        model_key,
        context,
        contextualizer_attention=config.contextualizer_attention,
    )
    canonical = backend.canonicalize_context(context)
    embedded_route = backend.embedded_route(model)
    if embedded_route is not None and (config.contest or config.large_n):
        raise ValueError(
            "compiled fine-tune checkpoints support ordinary eval only; "
            "contest and large_n require a router checkpoint"
        )
    if embedded_route is not None:
        prepared = _compiled_finetune_ordinary(
            backend,
            model,
            context,
            canonical,
            embedded_route,
            mcmc_key,
            config,
        )
        path = "ordinary"
    elif config.contest:
        prepared = _contest(
            backend,
            model,
            canonical,
            root_key,
            config,
        )
        path = "contest"
    elif config.large_n:
        prepared = _large_n(
            backend,
            model,
            context,
            canonical,
            mcmc_key,
            config,
        )
        path = "large_n"
    else:
        prepared = _ordinary(
            backend,
            model,
            context,
            canonical,
            mcmc_key,
            config,
        )
        path = "ordinary"
    wavefunction, routed_context, state, route, route_logp, contest = prepared
    del model, context, canonical, embedded_route, prepared
    wavefunction, routed_context, state = backend.prepare_singular(
        wavefunction,
        routed_context,
        state,
    )
    state = _burn_in(
        backend,
        state,
        wavefunction,
        routed_context,
        config,
        iterations=int(config.mcmc.burn_in),
        replica_steps=int(config.mcmc.burn_in_replica_steps),
    )
    _state, window = _measure(
        backend,
        state,
        wavefunction,
        routed_context,
        config,
        started=started,
        metric_sink=metric_sink,
    )
    route_host = np.asarray(route, dtype=np.int32)
    if route_host.shape[0] != 1:
        raise ValueError("single-system eval produced more than one route")
    return EvalResult(
        path=path,
        route=tuple(int(value) for value in route_host[0]),
        route_log_probability=(None if route_logp is None else float(route_logp)),
        measurements=int(window.count),
        walltime_seconds=float(time.perf_counter() - started),
        energy=window.metrics("total"),
        channels={
            "exchange": window.metrics("exchange"),
            "field": window.metrics("field"),
        },
        contest=contest,
    )
