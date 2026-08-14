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
    step_mcmc,
    adapt,
) -> REState:
    for iteration in range(config.mcmc.burn_in):
        state = step_mcmc(
            state,
            model,
            context,
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
    return jax.device_put(state, _state_sharding(mesh))


def _state_specs() -> REState:
    walkers = P("batch")
    replicated = P()
    return REState(
        q=walkers,
        log_p=walkers,
        grad_log_p=walkers,
        beta=replicated,
        sigma=replicated,
        step=replicated,
        key=walkers,
        n_local_accept=walkers,
        n_local=walkers,
        n_swap_accept=walkers,
        n_swap=walkers,
        mask=replicated,
        m=replicated,
        n_haar_accept=walkers,
        n_haar=walkers,
    )


def _state_sharding(mesh: Mesh) -> REState:
    replicated = NamedSharding(mesh, P())
    batched = NamedSharding(mesh, P("batch"))
    return REState(
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


def _build_mcmc_entry(
    mesh: Mesh,
    state_sharding,
    model_sharding,
    context_sharding,
    *,
    replica_steps: int,
    walker_chunk_size: int | None,
):
    specs = _state_specs()

    def local_step(state, model, context):
        out = run_batched(
            model,
            context,
            state,
            n_steps=int(replica_steps),
            walker_chunk_size=walker_chunk_size,
        )
        local_count = jnp.asarray(out.q.shape[0], dtype=jnp.int32)
        global_count = jax.lax.psum(local_count, "batch")
        guard = global_count.astype(out.q.dtype) * jnp.asarray(0.0, out.q.dtype)
        return eqx.tree_at(lambda value: value.q, out, out.q + guard)

    mapped = jax.shard_map(
        local_step,
        mesh=mesh,
        in_specs=(specs, P(), P()),
        out_specs=specs,
        check_vma=False,
    )
    return jax.jit(
        mapped,
        in_shardings=(state_sharding, model_sharding, context_sharding),
        out_shardings=state_sharding,
        donate_argnums=(0,),
    )


def _build_energy_entry(
    mesh: Mesh,
    model_sharding,
    frame_sharding,
    *,
    chunk_size: int,
):
    q_spec = P("batch", None, None)
    output_spec = P("batch")

    def local_energy(model, frame, q):
        outputs = vmc_energy_custom_lap_finetune(
            model,
            frame,
            q,
            chunk_size=int(chunk_size),
        )
        local_count = jnp.asarray(q.shape[0], dtype=jnp.int32)
        global_count = jax.lax.psum(local_count, "batch")
        guard = global_count.astype(jnp.float32) * jnp.asarray(0.0, jnp.float32)
        return tuple(value + guard.astype(value.dtype) for value in outputs)

    mapped = jax.shard_map(
        local_energy,
        mesh=mesh,
        in_specs=(P(), P(), q_spec),
        out_specs=(output_spec,) * 4,
        check_vma=False,
    )
    q_sharding = NamedSharding(mesh, q_spec)
    output_sharding = NamedSharding(mesh, output_spec)
    return jax.jit(
        mapped,
        in_shardings=(model_sharding, frame_sharding, q_sharding),
        out_shardings=(output_sharding,) * 4,
    )


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
    if config.mcmc.reuse_mcmc is not None:
        state = load_mcmc(config.mcmc.reuse_mcmc, state)
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
    state_sharding = _state_sharding(mesh)
    model_sharding = jax.tree_util.tree_map(lambda _value: replicated, model)
    context_sharding = jax.tree_util.tree_map(lambda _value: replicated, context)
    frame_sharding = jax.tree_util.tree_map(lambda _value: replicated, energy_frame)
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
    mcmc_entries = {}

    def mcmc_entry(replica_steps: int):
        entry_key = (int(replica_steps), config.mcmc.walker_chunk_size)
        entry = mcmc_entries.get(entry_key)
        if entry is None:
            entry = _build_mcmc_entry(
                mesh,
                state_sharding,
                model_sharding,
                context_sharding,
                replica_steps=entry_key[0],
                walker_chunk_size=entry_key[1],
            )
            mcmc_entries[entry_key] = entry
        return entry

    adapt = jax.jit(
        functools.partial(_adapt, config=config),
        in_shardings=(state_sharding,),
        out_shardings=state_sharding,
    )
    local_energy = _build_energy_entry(
        mesh,
        model_sharding,
        frame_sharding,
        chunk_size=config.energy.chunk_size,
    )
    target_entry = jax.jit(
        functools.partial(
            process_finetune_targets,
            mad_width=config.kfac.mad_clip_width,
        ),
        in_shardings=(kfac_data, replicated),
        out_shardings=kfac_data,
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
    state = _burn_in(
        model,
        context,
        state,
        config,
        mcmc_entry(config.mcmc.burn_in_replica_steps),
        adapt,
    )
    step_mcmc = mcmc_entry(config.mcmc.steps)
    run_started = time.perf_counter()
    last_metric = None
    for step in range(config.steps):
        step_started = time.perf_counter()
        state = step_mcmc(
            state,
            model,
            context,
        )
        if step > 0 and step % config.mcmc.adapt_every == 0:
            state = adapt(state)
        q_cold = cold_samples(state)
        total, _exchange, _casimir, _field = local_energy(
            model,
            energy_frame,
            q_cold,
        )
        target = target_entry(
            total[None],
            context_batch.s_norm,
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
