# Copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from hamiltonzero.checkpoint import load_mcmc, save_model
from hamiltonzero.compiled.tree import compile_physical_tree_reference
from hamiltonzero.compiled.trunk import bind_shared_kernel, compile_shared_trunk
from hamiltonzero.compiled.types import (
    CompiledWaveFunction,
)
from hamiltonzero.config import TrainConfig
from hamiltonzero.data import build_context_and_energy, load_systems
from hamiltonzero.energy import vmc_energy_custom_lap_compiled
from hamiltonzero.energy.frame import compile_energy_frame
from hamiltonzero.mcmc import (
    REState,
    adapt_batched,
    cold_samples,
    init_batched_state,
    run_batched,
)
from hamiltonzero.model import MultiSystemContext, build_model
from hamiltonzero.model.tree import project_tree_ngpt_rownorm
from hamiltonzero.optim import (
    KFACBundle,
    apply_router_kfac_step,
    init_router_kfac_state,
    learning_rate,
    process_route_targets,
)
from hamiltonzero.router import (
    ROUTE_SAMPLES,
    bind_router_kernel,
    build_beam16,
    build_route_sampler,
    compile_router_static,
    rebase_cold_samples,
    reframe_state_context,
    snis_mode_baseline,
)


@dataclass(frozen=True, slots=True)
class TrainMetric:
    step: int
    system: int
    energy: float
    energy_std: float
    step_walltime: float
    walltime: float


@dataclass(frozen=True, slots=True)
class TrainResult:
    model: object
    kfac: KFACBundle
    mcmc_states: tuple[REState | None, ...]
    last_metric: TrainMetric | None


@dataclass(slots=True)
class _SystemState:
    sampler: REState
    context: MultiSystemContext
    perms: jax.Array


def _identity_perms(n_max: int):
    return jnp.broadcast_to(
        jnp.arange(n_max, dtype=jnp.int32),
        (ROUTE_SAMPLES, n_max),
    )


def _route_sharding(mesh: Mesh, value):
    return jax.tree_util.tree_map(
        lambda x: NamedSharding(
            mesh,
            P("systems", *([None] * (x.ndim - 1)))
            if x.ndim and x.shape[0] == ROUTE_SAMPLES
            else P(),
        ),
        value,
    )


def _replicate(mesh: Mesh, value):
    replicated = NamedSharding(mesh, P())
    return jax.device_put(value, jax.tree_util.tree_map(lambda _: replicated, value))


def _place_routes(mesh: Mesh, value):
    return jax.device_put(value, _route_sharding(mesh, value))


def _host_pool(value):
    def materialize(x):
        if not eqx.is_array(x):
            return x
        result = np.asarray(jax.device_get(x))
        if not result.flags.writeable:
            result = result.copy()
        return result

    return jax.tree_util.tree_map(materialize, value)


def _host_system_state(state: _SystemState) -> _SystemState:
    return _SystemState(
        sampler=_host_pool(state.sampler),
        context=_host_pool(state.context),
        perms=_host_pool(state.perms),
    )


def _activate_system(mesh: Mesh, state: _SystemState) -> _SystemState:
    return _SystemState(
        sampler=_place_routes(mesh, state.sampler),
        context=_place_routes(mesh, state.context),
        perms=_place_routes(mesh, state.perms),
    )


def _compile_trees(model, trunk, perms):
    return jax.vmap(lambda perm: compile_physical_tree_reference(model, trunk, perm))(
        perms
    )


def _compile_frames(inputs, mask, bmask, perms):
    return jax.vmap(
        lambda perm: compile_energy_frame(
            inputs,
            mask,
            bmask,
            perm,
        )
    )(perms)


def _run_routes(kernel, trees, contexts, state, n_steps, chunk_size):
    def one(tree, context, sampler):
        return run_batched(
            CompiledWaveFunction(kernel=kernel, tree=tree),
            context,
            sampler,
            n_steps,
            walker_chunk_size=chunk_size,
        )

    return jax.vmap(one)(trees, contexts, state)


def _adapt_routes(state, config):
    return jax.vmap(
        lambda sampler: adapt_batched(
            sampler,
            beta_history_weight=config.mcmc.beta_history_weight,
            sigma_target=config.mcmc.langevin_target_acceptance,
            sigma_scale=config.mcmc.sigma_scale,
            haar_target=config.mcmc.haar_target_acceptance,
        )
    )(state)


def _sampled_energy(kernel, trees, frames, q, chunk_size):
    def one(tree, frame, q_row):
        return vmc_energy_custom_lap_compiled(
            kernel,
            tree,
            frame,
            q_row,
            chunk_size=chunk_size,
        )

    return jax.vmap(one)(trees, frames, q)


def _mode_energy(kernel, tree, frame, q_canonical, mode_perm, chunk_size):
    q_mode = jnp.take(q_canonical, mode_perm, axis=-2)

    def one(q_row):
        with jax.default_matmul_precision("default"):
            log_p = 2.0 * jax.vmap(
                lambda walker: CompiledWaveFunction(kernel, tree)(walker)[0]
            )(q_row)
        total, exchange, casimir, field = vmc_energy_custom_lap_compiled(
            kernel,
            tree,
            frame,
            q_row,
            chunk_size=chunk_size,
        )
        return total, exchange, casimir, field, log_p

    return jax.vmap(one)(q_mode)


def _initial_system_state(
    model,
    context,
    config,
    mcmc_key,
    system_index,
    *,
    n_systems,
    mesh,
    compile_plan,
    compile_trees,
    run_routes,
):
    walkers = config.mcmc.batch_size // ROUTE_SAMPLES
    cpu = jax.devices("cpu")[0]
    mcmc_key = jax.device_put(mcmc_key, cpu)
    with jax.default_device(cpu):
        lane_indices = system_index * ROUTE_SAMPLES + jnp.arange(
            ROUTE_SAMPLES, dtype=jnp.int32
        )
        keys = jax.vmap(lambda index: jax.random.fold_in(mcmc_key, index))(lane_indices)
        sampler = jax.vmap(
            lambda lane_key: init_batched_state(
                lane_key,
                context,
                batch_size=walkers,
                n_replicas=config.mcmc.replicas,
                initial_m=config.mcmc.initial_haar_sites,
                initial_sigma=config.mcmc.initial_sigma,
            )
        )(keys)
        contexts = MultiSystemContext.stack([context] * ROUTE_SAMPLES)
        perms = _identity_perms(config.n_max)
    if config.mcmc.reuse_mcmc is not None:
        source = config.mcmc.reuse_mcmc
        if source.is_dir():
            source = source / f"{system_index}.eqx"
        elif n_systems != 1:
            raise ValueError(
                "multisystem --reuse-mcmc must point to a directory of "
                "<system-index>.eqx files"
            )
        sampler = load_mcmc(source, sampler)
    sampler = _place_routes(mesh, sampler)
    contexts = _place_routes(mesh, contexts)
    perms = _place_routes(mesh, perms)
    trunk = compile_plan(model, context)[0]
    kernel = bind_shared_kernel(model)
    trees = compile_trees(model, trunk, perms)
    if config.mcmc.reuse_mcmc is None:
        for iteration in range(config.mcmc.burn_in):
            sampler = run_routes(
                kernel,
                trees,
                contexts,
                sampler,
                config.mcmc.burn_in_replica_steps,
                config.mcmc.walker_chunk_size,
            )
            if iteration and iteration % config.mcmc.adapt_every == 0:
                sampler = _adapt_routes(sampler, config)
    return _SystemState(sampler=sampler, context=contexts, perms=perms)


def _metric(step, system_index, total, step_started, run_started):
    jax.block_until_ready(total)
    return TrainMetric(
        step=step,
        system=system_index,
        energy=float(jax.device_get(jnp.mean(total.real))),
        energy_std=float(jax.device_get(jnp.std(total.real))),
        step_walltime=time.perf_counter() - step_started,
        walltime=time.perf_counter() - run_started,
    )


def run_train(
    config: TrainConfig,
    *,
    metric_sink: Callable[[TrainMetric], None] | None = None,
) -> TrainResult:
    if config.mcmc.batch_size % ROUTE_SAMPLES:
        raise ValueError("mcmc.batch_size must be divisible by K=8")
    systems = load_systems(config.systems)
    if not systems:
        raise ValueError("training requires at least one system")
    systems_data = [
        _host_pool(
            build_context_and_energy(
                system,
                n_max=config.n_max,
                mu=config.energy.mu,
                eps=config.energy.eps,
            )
        )
        for system in systems
    ]
    contexts = [context for context, _energy in systems_data]
    energy_inputs = [energy for _context, energy in systems_data]
    key = jax.random.PRNGKey(config.seed)
    key_model, key_mcmc = jax.random.split(key)
    model = build_model(config.model, key_model, n_max=config.n_max)
    devices = tuple(jax.devices())
    if len(devices) < ROUTE_SAMPLES:
        raise ValueError(
            "learned-router train requires eight devices for the K=8 systems mesh"
        )
    mesh = Mesh(np.asarray(devices[:ROUTE_SAMPLES], dtype=object), ("systems",))
    model = _replicate(mesh, model)
    compile_plan = jax.jit(
        lambda model_value, context_value: (
            compile_shared_trunk(model_value, context_value),
        )
    )
    compile_trees = jax.jit(_compile_trees)
    compile_frames = jax.jit(_compile_frames)
    run_routes = jax.jit(
        _run_routes,
        static_argnums=(4, 5),
        donate_argnums=(3,),
    )
    sampled_energy = jax.jit(_sampled_energy, static_argnums=(4,))
    mode_energy = jax.jit(_mode_energy, static_argnums=(5,))
    reframe = jax.jit(reframe_state_context, donate_argnums=(0, 1))
    system_states: list[_SystemState | None] = [None] * len(systems)

    def get_system(index: int):
        cached = system_states[index]
        if cached is None:
            return _initial_system_state(
                model,
                contexts[index],
                config,
                key_mcmc,
                index,
                n_systems=len(systems),
                mesh=mesh,
                compile_plan=compile_plan,
                compile_trees=compile_trees,
                run_routes=run_routes,
            )
        return _activate_system(mesh, cached)

    first = get_system(0)
    q_seed = jax.vmap(cold_samples)(first.sampler)
    energy_seed = _place_routes(mesh, np.zeros(q_seed.shape[:2], dtype=np.complex64))
    kfac = init_router_kfac_state(
        config.kfac,
        model,
        q_seed,
        energy_seed,
        first.context,
        t=0.0,
        key=jax.random.fold_in(key, 0xCAFE),
        multi_device=True,
        route_tau=config.router.temperature,
        route_loss_weight=config.router.loss_weight,
    )
    system_states[0] = _host_system_state(first)
    del first, q_seed, energy_seed
    order_rng = np.random.default_rng(config.seed)
    order = np.arange(len(systems), dtype=np.int32)
    order_rng.shuffle(order)
    run_started = time.perf_counter()
    last_metric = None
    for step in range(config.steps):
        step_started = time.perf_counter()
        if step and step % len(order) == 0:
            order_rng.shuffle(order)
        system_index = int(order[step % len(order)])
        state = get_system(system_index)
        trunk = compile_plan(model, contexts[system_index])[0]
        router_kernel = bind_router_kernel(model)
        router_static = compile_router_static(
            router_kernel,
            trunk,
            contexts[system_index].route_quotient_node_key,
            contexts[system_index].route_quotient_edge_key,
            contexts[system_index].needs_fwl2,
        )
        tau = jnp.asarray(config.router.temperature, dtype=jnp.float32)
        router_kernel = _replicate(mesh, router_kernel)
        router_static = _replicate(mesh, router_static)
        key, key_route = jax.random.split(key)
        new_perms = build_route_sampler(mesh, router_kernel.decoder, router_static)(
            router_kernel.decoder, router_static, key_route, tau
        )
        mode_perm = build_beam16(mesh, router_kernel.decoder, router_static)(
            router_kernel.decoder, router_static, tau
        )
        state.sampler, state.context = reframe(
            state.sampler, state.context, state.perms, new_perms
        )
        state.perms = new_perms
        kernel = bind_shared_kernel(model)
        trees = compile_trees(model, trunk, new_perms)
        frames = compile_frames(
            energy_inputs[system_index],
            contexts[system_index].mask,
            contexts[system_index].bmask,
            new_perms,
        )
        state.sampler = run_routes(
            kernel,
            trees,
            state.context,
            state.sampler,
            config.mcmc.steps,
            config.mcmc.walker_chunk_size,
        )
        if step and step % config.mcmc.adapt_every == 0:
            state.sampler = _adapt_routes(state.sampler, config)
        q_cold = jax.vmap(cold_samples)(state.sampler)
        total, _exchange, _casimir, _field = sampled_energy(
            kernel,
            trees,
            frames,
            q_cold,
            config.energy.chunk_size,
        )
        baseline_is_sampled = bool(
            np.asarray(
                jax.device_get(
                    jnp.all(
                        new_perms.astype(jnp.int32)
                        == mode_perm[None, :].astype(jnp.int32)
                    )
                )
            )
        )
        if baseline_is_sampled:
            baseline_total = total
            baseline_weights = jnp.full(
                total.shape,
                1.0 / total.shape[-1],
                dtype=total.real.dtype,
            )
        else:
            mode_tree = compile_physical_tree_reference(model, trunk, mode_perm)
            mode_frame = compile_energy_frame(
                energy_inputs[system_index],
                contexts[system_index].mask,
                contexts[system_index].bmask,
                mode_perm,
            )
            q_canonical = rebase_cold_samples(q_cold, new_perms)
            baseline_total, _bx, _bc, _bf, candidate_log_p = mode_energy(
                kernel,
                mode_tree,
                mode_frame,
                q_canonical,
                mode_perm,
                config.energy.chunk_size,
            )
            sampled_log_p = state.sampler.log_p[..., -1]
            baseline_weights = snis_mode_baseline(
                baseline_total, candidate_log_p, sampled_log_p
            )
        target, advantage = process_route_targets(
            total,
            baseline_total,
            state.context.s_norm,
            baseline_weights,
            mad_width=config.kfac.mad_clip_width,
        )
        key, key_kfac = jax.random.split(key)
        model, kfac = apply_router_kfac_step(
            kfac,
            model,
            q_cold,
            target,
            state.context,
            t=0.0,
            key=key_kfac,
            momentum=config.kfac.momentum,
            learning_rate=learning_rate(config.kfac, step),
            damping=config.kfac.damping,
            route_advantage=advantage,
            route_tau=config.router.temperature,
        )
        model = project_tree_ngpt_rownorm(model)
        jax.block_until_ready(model)
        system_states[system_index] = _host_system_state(state)
        last_metric = _metric(step, system_index, total, step_started, run_started)
        if metric_sink is not None:
            metric_sink(last_metric)
    save_model(
        config.output,
        model,
        kind="router",
        metadata={"n_max": config.n_max},
    )
    return TrainResult(
        model=model,
        kfac=kfac,
        mcmc_states=tuple(
            None if value is None else value.sampler for value in system_states
        ),
        last_metric=last_metric,
    )


__all__ = [
    "TrainMetric",
    "TrainResult",
    "run_train",
]
