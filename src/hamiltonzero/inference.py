# Copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from hamiltonzero.config import EnergyConfig, EvalMCMCConfig, ModelConfig
from hamiltonzero.evaluation.runtime import DefaultEvalBackend
from hamiltonzero.evaluation.runner import _compose_walker_route
from hamiltonzero.hamiltonian import SpinHamiltonian
from hamiltonzero.observables import local_spin


class EnergySamples(NamedTuple):
    total: Any
    exchange: Any
    casimir: Any
    field: Any


class CompiledOrder(NamedTuple):
    leaf_to_input: Any
    input_to_leaf: Any


@dataclass(slots=True)
class _InferenceRuntime:
    backend: DefaultEvalBackend
    wavefunction: Any = None
    context: Any = None


@dataclass(frozen=True, slots=True)
class PreparedInference:
    wavefunction: Any = field(repr=False)
    context: Any = field(repr=False)
    energy_frame: Any = field(repr=False)
    route: Any
    route_log_probability: float
    _initial_context: Any = field(repr=False)
    _walker_route: Any = field(repr=False)
    _runtime: _InferenceRuntime = field(repr=False)


def prepare(
    system: SpinHamiltonian,
    checkpoint: str | Path,
    model_key,
    *,
    model: ModelConfig = ModelConfig(),
    route_temperature: float = 4.0,
    eps: float = 0.1,
) -> tuple[PreparedInference, CompiledOrder]:
    if not isinstance(system, SpinHamiltonian):
        raise TypeError("system must be a SpinHamiltonian")
    temperature = float(route_temperature)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("route_temperature must be finite and positive")
    backend = DefaultEvalBackend()
    initial_context = backend.build_system(system, EnergyConfig(eps=float(eps)))
    foundation = backend.load_model(
        Path(checkpoint),
        model,
        model_key,
        initial_context,
        contextualizer_attention=None,
    )
    if backend.embedded_route(foundation) is not None:
        raise ValueError("prepare requires a foundation router checkpoint")
    canonical = backend.canonicalize_context(initial_context)
    candidates = backend.beam_candidates(
        foundation,
        canonical.context,
        beam_width=8,
        top_k=1,
        temperature=temperature,
    )
    permutations = jnp.asarray(candidates.permutations, dtype=jnp.int32)
    if permutations.ndim != 3 or permutations.shape[:2] != (1, 1):
        raise ValueError("router must return one route with shape [1, 1, N]")
    route = permutations[:, 0]
    walker_route = _compose_walker_route(canonical.old_inverse, route)
    routed_context = backend.route_context(
        canonical.context,
        route,
        compact_custom_lap=False,
    )
    wavefunction = backend.compile_single(foundation, routed_context)
    compact_context = backend.route_context(
        canonical.context,
        route,
        compact_custom_lap=True,
    )
    backend.block_until_ready(wavefunction)
    route = route[0]
    order = CompiledOrder(
        leaf_to_input=route,
        input_to_leaf=jnp.argsort(route).astype(jnp.int32),
    )
    prepared = PreparedInference(
        wavefunction=wavefunction,
        context=compact_context,
        energy_frame=compact_context.energy_frame,
        route=order.leaf_to_input,
        route_log_probability=float(np.asarray(candidates.log_probabilities)[0, 0]),
        _initial_context=initial_context,
        _walker_route=walker_route,
        _runtime=_InferenceRuntime(backend),
    )
    return prepared, order


def _mcmc_config(
    *,
    batch_size: int,
    replicas: int,
    steps: int,
    burn_in: int,
    walker_chunk_size: int,
    burn_in_replica_steps: int = 2,
    initial_sigma: float = 0.3,
    initial_haar_sites: int = 1,
) -> EvalMCMCConfig:
    if int(batch_size) < 1:
        raise ValueError("batch_size must be positive")
    if int(replicas) < 2:
        raise ValueError("replicas must be at least two")
    if int(steps) < 1:
        raise ValueError("steps must be positive")
    if int(burn_in) < 0:
        raise ValueError("burn_in must be non-negative")
    if int(walker_chunk_size) < 1:
        raise ValueError("walker_chunk_size must be positive")
    if int(burn_in_replica_steps) < 1:
        raise ValueError("replica_steps must be positive")
    if not np.isfinite(float(initial_sigma)) or float(initial_sigma) <= 0.0:
        raise ValueError("initial_sigma must be finite and positive")
    if int(initial_haar_sites) < 1:
        raise ValueError("initial_haar_sites must be positive")
    return EvalMCMCConfig(
        batch_size=int(batch_size),
        replicas=int(replicas),
        steps=int(steps),
        burn_in=int(burn_in),
        burn_in_replica_steps=int(burn_in_replica_steps),
        walker_chunk_size=int(walker_chunk_size),
        initial_sigma=float(initial_sigma),
        initial_haar_sites=int(initial_haar_sites),
    )


def burn_in(
    prepared: PreparedInference,
    key,
    *,
    batch_size: int = 256,
    replicas: int = 8,
    burn_in: int = 1024,
    replica_steps: int = 2,
    walker_chunk_size: int = 16,
    initial_sigma: float = 0.3,
    initial_haar_sites: int = 1,
):
    config = _mcmc_config(
        batch_size=batch_size,
        replicas=replicas,
        steps=1,
        burn_in=burn_in,
        walker_chunk_size=walker_chunk_size,
        burn_in_replica_steps=replica_steps,
        initial_sigma=initial_sigma,
        initial_haar_sites=initial_haar_sites,
    )
    runtime = prepared._runtime
    backend = runtime.backend
    state = backend.initialize_mcmc(
        key,
        prepared.wavefunction,
        prepared._initial_context,
        config,
    )
    state = backend.route_mcmc(state, prepared._walker_route)
    runtime.wavefunction, runtime.context, state = backend.prepare_singular(
        prepared.wavefunction,
        prepared.context,
        state,
    )
    for _ in range(config.burn_in):
        state = backend.step_mcmc(
            state,
            runtime.wavefunction,
            runtime.context,
            replica_steps=config.burn_in_replica_steps,
            walker_chunk_size=config.walker_chunk_size,
        )
        backend.block_until_ready(backend.cold_walkers(state))
        state = backend.adapt_mcmc(state, config)
    q_cold = backend.cold_walkers(state)
    backend.block_until_ready(q_cold)
    return state, q_cold


def step(
    prepared: PreparedInference,
    state,
    *,
    steps: int = 24,
    walker_chunk_size: int = 16,
):
    config = _mcmc_config(
        batch_size=int(state.q.shape[0]),
        replicas=int(state.q.shape[1]),
        steps=steps,
        burn_in=0,
        walker_chunk_size=walker_chunk_size,
    )
    runtime = prepared._runtime
    backend = runtime.backend
    if runtime.wavefunction is None or runtime.context is None:
        raise RuntimeError("burn_in must be called before step")
    state = backend.step_mcmc(
        state,
        runtime.wavefunction,
        runtime.context,
        replica_steps=config.steps,
        walker_chunk_size=config.walker_chunk_size,
    )
    q_cold = backend.cold_walkers(state)
    backend.block_until_ready(q_cold)
    state = backend.adapt_mcmc(state, config)
    return state, q_cold


def energy(
    prepared: PreparedInference,
    q,
    *,
    chunk_size: int = 512,
):
    if int(chunk_size) < 1:
        raise ValueError("chunk_size must be positive")
    runtime = prepared._runtime
    if runtime.wavefunction is None or runtime.context is None:
        raise RuntimeError("burn_in must be called before energy")
    values = runtime.backend.custom_lap_energy(
        runtime.wavefunction,
        runtime.context,
        q,
        EnergyConfig(chunk_size=int(chunk_size)),
    )
    runtime.backend.block_until_ready(values)
    return EnergySamples(*(value[0] for value in values))


def spin(
    prepared: PreparedInference,
    q,
    *,
    chunk_size: int | None = 512,
):
    runtime = prepared._runtime
    if runtime.wavefunction is None or runtime.context is None:
        raise RuntimeError("burn_in must be called before spin")
    values = local_spin(
        runtime.wavefunction,
        runtime.context,
        q,
        chunk_size=chunk_size,
    )
    runtime.backend.block_until_ready(values)
    return values


__all__ = [
    "EnergySamples",
    "CompiledOrder",
    "PreparedInference",
    "burn_in",
    "energy",
    "prepare",
    "spin",
    "step",
]
