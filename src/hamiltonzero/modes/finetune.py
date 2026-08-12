# Copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import time
from dataclasses import dataclass
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from hamiltonzero.checkpoint import load_mcmc, load_model, save_model
from hamiltonzero.compiled.model import (
    CompiledFinetuneWaveFunction,
    compile_finetune_model,
)
from hamiltonzero.config import FineTuneConfig
from hamiltonzero.data import build_context_and_energy, load_system
from hamiltonzero.energy import vmc_energy_custom_lap_finetune
from hamiltonzero.energy.frame import compile_energy_frame
from hamiltonzero.mcmc import (
    REState,
    adapt_batched,
    cold_samples,
    init_batched_state,
    run_batched,
)
from hamiltonzero.model import build_model
from hamiltonzero.optim import (
    KFACBundle,
    apply_finetune_kfac_step,
    init_finetune_kfac_state,
    learning_rate,
    process_finetune_targets,
)
from hamiltonzero.router import (
    batch_context,
    route_context,
    route_state,
    select_frozen_route,
    strip_router,
)


@dataclass(frozen=True, slots=True)
class FineTuneMetric:
    step: int
    energy: float
    energy_std: float
    step_walltime: float
    walltime: float


@dataclass(frozen=True, slots=True)
class FineTuneResult:
    model: CompiledFinetuneWaveFunction
    route_perm: jax.Array
    mcmc_state: REState
    kfac: KFACBundle
    last_metric: FineTuneMetric | None


def _adapt(state: REState, config: FineTuneConfig) -> REState:
    return adapt_batched(
        state,
        beta_history_weight=config.mcmc.beta_history_weight,
        sigma_target=config.mcmc.langevin_target_acceptance,
        sigma_scale=config.mcmc.sigma_scale,
        haar_target=config.mcmc.haar_target_acceptance,
    )


def _burn_in(
    model: CompiledFinetuneWaveFunction,
    context,
    state: REState,
    config: FineTuneConfig,
) -> REState:
    replica_steps = config.mcmc.burn_in_replica_steps
    step_mcmc = eqx.filter_jit(
        functools.partial(
            run_batched,
            n_steps=replica_steps,
            walker_chunk_size=config.mcmc.walker_chunk_size,
        )
    )
    adapt = eqx.filter_jit(functools.partial(_adapt, config=config))
    for iteration in range(config.mcmc.burn_in):
        state = step_mcmc(
            model,
            context,
            state,
        )
        if iteration > 0 and iteration % config.mcmc.adapt_every == 0:
            state = adapt(state)
    return state


def _replicate(value, sharding: NamedSharding):
    return jax.device_put(
        value,
        jax.tree_util.tree_map(lambda _leaf: sharding, value),
    )


def _place_state(state: REState, mesh: Mesh) -> REState:
    replicated = NamedSharding(mesh, P())
    batched = NamedSharding(mesh, P("batch"))
    shardings = REState(
        q=batched,
        log_p=batched,
        grad_log_p=batched,
        beta=replicated,
        sigma=replicated,
        step=replicated,
        key=batched,
        n_local_accept=batched,
        n_local=batched,
        n_swap_accept=batched,
        n_swap=batched,
        mask=replicated,
        m=replicated,
        n_haar_accept=batched,
        n_haar=batched,
    )
    return jax.device_put(state, shardings)


def _metric(
    step: int,
    total,
    step_started: float,
    run_started: float,
) -> FineTuneMetric:
    jax.block_until_ready(total)
    return FineTuneMetric(
        step=step,
        energy=float(jax.device_get(jnp.mean(total.real))),
        energy_std=float(jax.device_get(jnp.std(total.real))),
        step_walltime=time.perf_counter() - step_started,
        walltime=time.perf_counter() - run_started,
    )


def run_finetune(
    config: FineTuneConfig,
    *,
    metric_sink: Callable[[FineTuneMetric], None] | None = None,
) -> FineTuneResult:
    key = jax.random.PRNGKey(config.seed)
    key_model, key_mcmc = jax.random.split(key)
    system = load_system(config.system)
    context, energy_inputs = build_context_and_energy(
        system,
        n_max=None,
        mu=config.energy.mu,
        eps=config.energy.eps,
    )
    template = build_model(
        config.model,
        key_model,
        n_max=int(context.mask.shape[-1]),
    )
    eager_model = load_model(config.checkpoint, template)
    state = init_batched_state(
        key_mcmc,
        context,
        batch_size=config.mcmc.batch_size,
        n_replicas=config.mcmc.replicas,
        initial_m=config.mcmc.initial_haar_sites,
        initial_sigma=config.mcmc.initial_sigma,
    )
    reused = False
    if config.mcmc.reuse_mcmc is not None:
        state = load_mcmc(config.mcmc.reuse_mcmc, state)
        reused = True
    key, _route_key = jax.random.split(key)
    freeze_route = eqx.filter_jit(
        functools.partial(
            select_frozen_route,
            tau=config.route_temperature,
        )
    )
    route_perm = freeze_route(
        eager_model,
        context,
    )
    energy_frame = compile_energy_frame(
        energy_inputs,
        context.mask,
        context.bmask,
        route_perm,
    )
    context = route_context(context, route_perm)
    state = route_state(state, route_perm)
    eager_model = strip_router(eager_model)
    key, key_expand = jax.random.split(key)
    compile_model = eqx.filter_jit(
        functools.partial(
            compile_finetune_model,
            leaf_rank=config.leaf_rank,
            merge_rank=config.merge_rank,
        )
    )
    model = compile_model(
        eager_model,
        context,
        physical_perm=route_perm,
        key=key_expand,
    )
    del eager_model, template, _route_key
    devices = tuple(jax.devices())
    if config.mcmc.batch_size % len(devices):
        raise ValueError(
            f"batch_size={config.mcmc.batch_size} must be divisible by "
            f"the {len(devices)} visible devices"
        )
    mesh = Mesh(np.asarray(devices, dtype=object), ("batch",))
    replicated = NamedSharding(mesh, P())
    model = _replicate(model, replicated)
    context = _replicate(context, replicated)
    energy_frame = _replicate(energy_frame, replicated)
    state = _place_state(state, mesh)
    context_batch = _replicate(batch_context(context), replicated)
    q_cold = cold_samples(state)
    kfac_data = NamedSharding(mesh, P(None, "batch"))
    q_kfac = jax.device_put(q_cold[None], kfac_data)
    energy_seed = jax.device_put(
        jnp.zeros((1, config.mcmc.batch_size), dtype=jnp.complex64),
        kfac_data,
    )
    kfac = init_finetune_kfac_state(
        config.kfac,
        model,
        q_kfac,
        energy_seed,
        context_batch,
        t=0.0,
        key=jax.random.fold_in(key, 0xCAFE),
        multi_device=mesh.size > 1,
    )
    if not reused:
        state = _burn_in(model, context, state, config)
    step_mcmc = eqx.filter_jit(
        functools.partial(
            run_batched,
            n_steps=config.mcmc.steps,
            walker_chunk_size=config.mcmc.walker_chunk_size,
        )
    )
    adapt = eqx.filter_jit(functools.partial(_adapt, config=config))
    local_energy = eqx.filter_jit(
        functools.partial(
            vmc_energy_custom_lap_finetune,
            chunk_size=config.energy.chunk_size,
        )
    )
    run_started = time.perf_counter()
    last_metric = None
    for step in range(config.steps):
        step_started = time.perf_counter()
        state = step_mcmc(
            model,
            context,
            state,
        )
        if step > 0 and step % config.mcmc.adapt_every == 0:
            state = adapt(state)
        q_cold = cold_samples(state)
        total, _exchange, _casimir, _field = local_energy(
            model,
            energy_frame,
            q_cold,
        )
        target = process_finetune_targets(
            total[None],
            context_batch.s_norm,
            mad_width=config.kfac.mad_clip_width,
        )
        key, key_kfac = jax.random.split(key)
        model, kfac = apply_finetune_kfac_step(
            kfac,
            model,
            jax.device_put(q_cold[None], kfac_data),
            jax.device_put(target, kfac_data),
            context_batch,
            t=0.0,
            key=key_kfac,
            momentum=config.kfac.momentum,
            learning_rate=learning_rate(config.kfac, step),
            damping=config.kfac.damping,
        )
        jax.block_until_ready(model)
        last_metric = _metric(step, total, step_started, run_started)
        if metric_sink is not None:
            metric_sink(last_metric)
    save_model(
        config.output,
        model,
        kind="compiled_finetune",
        metadata={
            "leaf_rank": int(config.leaf_rank),
            "merge_rank": int(config.merge_rank),
            "n_max": int(context.mask.shape[-1]),
        },
    )
    return FineTuneResult(
        model=model,
        route_perm=route_perm,
        mcmc_state=state,
        kfac=kfac,
        last_metric=last_metric,
    )


__all__ = [
    "FineTuneMetric",
    "FineTuneResult",
    "run_finetune",
]
