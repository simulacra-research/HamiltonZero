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

from hamiltonzero.checkpoint import load_mcmc, load_model, save_model
from hamiltonzero.compiled.tree import (
    bind_physical_compiler_kernel,
    compile_physical_tree_from_shared_trunk,
)
from hamiltonzero.compiled.trunk import (
    bind_shared_kernel,
    bind_trunk_compiler_kernel,
    compile_shared_trunk_from_kernel,
)
from hamiltonzero.compiled.types import (
    CompiledWaveFunction,
)
from hamiltonzero.config import TrainConfig
from hamiltonzero.data import build_context_and_energy, load_systems
from hamiltonzero.energy import vmc_energy_custom_lap_compiled
from hamiltonzero.energy.custom_lap import build_W_levels
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


def _systems_sharding(mesh: Mesh, value):
    sharding = NamedSharding(mesh, P("systems"))
    return jax.tree_util.tree_map(lambda _value: sharding, value)


def _replicated_sharding(mesh: Mesh, value):
    sharding = NamedSharding(mesh, P())
    return jax.tree_util.tree_map(lambda _value: sharding, value)


def _replicate(mesh: Mesh, value):
    return jax.device_put(value, _replicated_sharding(mesh, value))


def _place_routes(mesh: Mesh, value):
    return jax.device_put(value, _systems_sharding(mesh, value))


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


def _abstract(value):
    return jax.tree_util.tree_map(
        lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype),
        value,
    )


def _owner_reduce(value):
    owner = jax.lax.axis_index("systems") == 0
    if jnp.issubdtype(value.dtype, jnp.bool_):
        return jax.lax.pmax(jnp.where(owner, value, jnp.zeros_like(value)), "systems")
    return jax.lax.psum(jnp.where(owner, value, jnp.zeros_like(value)), "systems")


def _build_owner_entry(mesh: Mesh, function, templates):
    output_template = jax.eval_shape(function, *_abstract(templates))
    input_specs = jax.tree_util.tree_map(lambda _value: P(), templates)
    output_specs = jax.tree_util.tree_map(lambda _value: P(), output_template)

    def local(*values):
        owner = jax.lax.axis_index("systems") == 0
        output = jax.lax.cond(
            owner,
            lambda args: function(*args),
            lambda _args: jax.tree_util.tree_map(
                lambda value: jnp.zeros(value.shape, value.dtype),
                output_template,
            ),
            values,
        )
        return jax.tree_util.tree_map(_owner_reduce, output)

    mapped = jax.shard_map(
        local,
        mesh=mesh,
        in_specs=input_specs,
        out_specs=output_specs,
        check_vma=False,
    )
    return jax.jit(
        mapped,
        in_shardings=_replicated_sharding(mesh, templates),
        out_shardings=_replicated_sharding(mesh, output_template),
    )


def _compile_tree_local(physical_kernel, trunk, perms):
    tree = compile_physical_tree_from_shared_trunk(
        physical_kernel,
        trunk,
        perms[0],
    )
    return jax.tree_util.tree_map(lambda value: value[None], tree)


def _build_compile_trees(mesh: Mesh, physical_kernel, trunk, perms):
    local_perms = jax.ShapeDtypeStruct((1, perms.shape[1]), perms.dtype)
    local_output = jax.eval_shape(
        _compile_tree_local,
        _abstract(physical_kernel),
        _abstract(trunk),
        local_perms,
    )
    mapped = jax.shard_map(
        _compile_tree_local,
        mesh=mesh,
        in_specs=(P(), P(), P("systems")),
        out_specs=jax.tree_util.tree_map(lambda _value: P("systems"), local_output),
        check_vma=False,
    )
    output_template = jax.eval_shape(
        mapped,
        _abstract(physical_kernel),
        _abstract(trunk),
        _abstract(perms),
    )
    return jax.jit(
        mapped,
        in_shardings=(
            _replicated_sharding(mesh, physical_kernel),
            _replicated_sharding(mesh, trunk),
            NamedSharding(mesh, P("systems")),
        ),
        out_shardings=_systems_sharding(mesh, output_template),
    )


def _compile_frame_local(inputs, mask, bmask, perms):
    frame = compile_energy_frame(inputs, mask, bmask, perms[0])
    return jax.tree_util.tree_map(lambda value: value[None], frame)


def _build_compile_frames(mesh: Mesh, inputs, mask, bmask, perms):
    local_perms = jax.ShapeDtypeStruct((1, perms.shape[1]), perms.dtype)
    local_output = jax.eval_shape(
        _compile_frame_local,
        _abstract(inputs),
        _abstract(mask),
        _abstract(bmask),
        local_perms,
    )
    mapped = jax.shard_map(
        _compile_frame_local,
        mesh=mesh,
        in_specs=(P(), P(), P(), P("systems")),
        out_specs=jax.tree_util.tree_map(lambda _value: P("systems"), local_output),
        check_vma=False,
    )
    output_template = jax.eval_shape(
        mapped,
        _abstract(inputs),
        _abstract(mask),
        _abstract(bmask),
        _abstract(perms),
    )
    return jax.jit(
        mapped,
        in_shardings=(
            _replicated_sharding(mesh, inputs),
            NamedSharding(mesh, P()),
            NamedSharding(mesh, P()),
            NamedSharding(mesh, P("systems")),
        ),
        out_shardings=_systems_sharding(mesh, output_template),
    )


def _run_routes_local(state, kernel, trees, n_steps, chunk_size):
    sampler = jax.tree_util.tree_map(lambda value: value[0], state)
    tree = jax.tree_util.tree_map(lambda value: value[0], trees)
    sampler = run_batched(
        CompiledWaveFunction(kernel=kernel, tree=tree),
        None,
        sampler,
        int(n_steps),
        walker_chunk_size=chunk_size,
    )
    return jax.tree_util.tree_map(lambda value: value[None], sampler)


def _build_run_routes(
    mesh: Mesh,
    state,
    kernel,
    trees,
    *,
    n_steps: int,
    chunk_size: int | None,
):
    state_specs = jax.tree_util.tree_map(lambda _value: P("systems"), state)
    tree_specs = jax.tree_util.tree_map(lambda _value: P("systems"), trees)

    def local(state_value, kernel_value, trees_value):
        output = _run_routes_local(
            state_value,
            kernel_value,
            trees_value,
            int(n_steps),
            chunk_size,
        )
        local_count = jnp.asarray(
            output.q.shape[0] * output.q.shape[1],
            dtype=jnp.int32,
        )
        global_count = jax.lax.psum(local_count, "systems")
        guard = global_count.astype(output.q.dtype) * jnp.asarray(0.0, output.q.dtype)
        return eqx.tree_at(
            lambda value: value.q,
            output,
            output.q + guard,
        )

    mapped = jax.shard_map(
        local,
        mesh=mesh,
        in_specs=(state_specs, P(), tree_specs),
        out_specs=state_specs,
        check_vma=False,
    )
    return jax.jit(
        mapped,
        in_shardings=(
            _systems_sharding(mesh, state),
            _replicated_sharding(mesh, kernel),
            _systems_sharding(mesh, trees),
        ),
        out_shardings=_systems_sharding(mesh, state),
        donate_argnums=(0,),
    )


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
        n_sites = int(q_row.shape[-2])
        frame = eqx.tree_at(
            lambda value: value.w_levels,
            frame,
            tuple(build_W_levels(frame.custom_lap_J_eff, n_sites)),
        )
        return vmc_energy_custom_lap_compiled(
            kernel,
            tree,
            frame,
            q_row,
            chunk_size=chunk_size,
        )

    return jax.vmap(one)(trees, frames, q)


def _build_sampled_energy(mesh: Mesh, chunk_size: int):
    systems = P("systems")
    system_batch = P("systems", None)

    def local_energy(kernel, trees, frames, q):
        outputs = _sampled_energy(
            kernel,
            trees,
            frames,
            q,
            int(chunk_size),
        )
        local_count = jnp.asarray(q.shape[0] * q.shape[1], dtype=jnp.int32)
        global_count = jax.lax.psum(local_count, "systems")
        guard = global_count.astype(jnp.float32) * jnp.asarray(0.0, jnp.float32)
        return tuple(value + guard.astype(value.dtype) for value in outputs)

    mapped = jax.shard_map(
        local_energy,
        mesh=mesh,
        in_specs=(P(), systems, systems, system_batch),
        out_specs=(system_batch,) * 4,
        check_vma=False,
    )
    replicated_sharding = NamedSharding(mesh, P())
    systems_sharding = NamedSharding(mesh, systems)
    system_batch_sharding = NamedSharding(mesh, system_batch)
    return jax.jit(
        mapped,
        in_shardings=(
            replicated_sharding,
            systems_sharding,
            systems_sharding,
            system_batch_sharding,
        ),
        out_shardings=(system_batch_sharding,) * 4,
    )


def _build_reframe(mesh: Mesh, state, context, perms):
    state_specs = jax.tree_util.tree_map(lambda _value: P("systems"), state)
    context_specs = jax.tree_util.tree_map(lambda _value: P("systems"), context)

    def local(state_value, context_value, old_perms, new_perms):
        return reframe_state_context(
            state_value,
            context_value,
            old_perms,
            new_perms,
        )

    mapped = jax.shard_map(
        local,
        mesh=mesh,
        in_specs=(
            state_specs,
            context_specs,
            P("systems"),
            P("systems"),
        ),
        out_specs=(state_specs, context_specs),
        check_vma=False,
    )
    return jax.jit(
        mapped,
        in_shardings=(
            _systems_sharding(mesh, state),
            _systems_sharding(mesh, context),
            NamedSharding(mesh, P("systems")),
            NamedSharding(mesh, P("systems")),
        ),
        out_shardings=(
            _systems_sharding(mesh, state),
            _systems_sharding(mesh, context),
        ),
    )


def _build_rebase(mesh: Mesh, q, perms):
    mapped = jax.shard_map(
        rebase_cold_samples,
        mesh=mesh,
        in_specs=(P("systems"), P("systems")),
        out_specs=P("systems"),
        check_vma=False,
    )
    return jax.jit(
        mapped,
        in_shardings=(
            NamedSharding(mesh, P("systems")),
            NamedSharding(mesh, P("systems")),
        ),
        out_shardings=NamedSharding(mesh, P("systems")),
    )


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


def _build_mode_energy(mesh: Mesh, kernel, tree, frame, q, mode_perm, chunk_size):
    system_batch = P("systems", None)

    def local(kernel_value, tree_value, frame_value, q_value, perm_value):
        return _mode_energy(
            kernel_value,
            tree_value,
            frame_value,
            q_value,
            perm_value,
            int(chunk_size),
        )

    mapped = jax.shard_map(
        local,
        mesh=mesh,
        in_specs=(P(), P(), P(), system_batch, P()),
        out_specs=(system_batch,) * 5,
        check_vma=False,
    )
    output_sharding = NamedSharding(mesh, system_batch)
    return jax.jit(
        mapped,
        in_shardings=(
            _replicated_sharding(mesh, kernel),
            _replicated_sharding(mesh, tree),
            _replicated_sharding(mesh, frame),
            NamedSharding(mesh, system_batch),
            NamedSharding(mesh, P()),
        ),
        out_shardings=(output_sharding,) * 5,
    )


def _compile_mode(physical_kernel, trunk, inputs, mask, bmask, perm):
    tree = compile_physical_tree_from_shared_trunk(physical_kernel, trunk, perm)
    frame = compile_energy_frame(inputs, mask, bmask, perm)
    return tree, frame


def _build_exact_skip(mesh: Mesh):
    def local(sampled, mode):
        equal = jnp.all(sampled.astype(jnp.int32) == mode[None].astype(jnp.int32))
        return jax.lax.pmin(equal.astype(jnp.int32), "systems").astype(jnp.bool_)

    mapped = jax.shard_map(
        local,
        mesh=mesh,
        in_specs=(P("systems"), P()),
        out_specs=P(),
        check_vma=False,
    )
    replicated = NamedSharding(mesh, P())
    return jax.jit(
        mapped,
        in_shardings=(NamedSharding(mesh, P("systems")), replicated),
        out_shardings=replicated,
    )


def _initial_model(config: TrainConfig, key):
    model = build_model(config.model, key, n_max=config.n_max)
    if config.checkpoint is not None:
        model = load_model(config.checkpoint, model)
    return model


def _run_route_burn_in(
    sampler,
    kernel,
    trees,
    config,
    run_routes,
    adapt,
):
    for iteration in range(config.mcmc.burn_in):
        sampler = run_routes(
            sampler,
            kernel,
            trees,
        )
        if iteration and iteration % config.mcmc.adapt_every == 0:
            sampler = adapt(sampler)
    return sampler


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
    get_run_routes,
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
    trunk = compile_plan(bind_trunk_compiler_kernel(model), context)
    physical_kernel = bind_physical_compiler_kernel(model)
    trees = compile_trees(physical_kernel, trunk, perms)
    run_routes = get_run_routes(
        sampler,
        bind_shared_kernel(model),
        trees,
        config.mcmc.burn_in_replica_steps,
    )
    adapt = jax.jit(
        lambda state: _adapt_routes(state, config),
        in_shardings=(_systems_sharding(mesh, sampler),),
        out_shardings=_systems_sharding(mesh, sampler),
    )
    sampler = _run_route_burn_in(
        sampler,
        bind_shared_kernel(model),
        trees,
        config,
        run_routes,
        adapt,
    )
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
    if config.mcmc.burn_in < 0:
        raise ValueError("mcmc.burn_in must be non-negative")
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
    model = _initial_model(config, key_model)
    devices = tuple(jax.devices())
    if len(devices) != ROUTE_SAMPLES:
        raise ValueError("learned-router train requires exactly eight visible devices")
    mesh = Mesh(np.asarray(devices, dtype=object), ("systems",))
    model = _replicate(mesh, model)
    trunk_kernel = bind_trunk_compiler_kernel(model)
    compile_plan = _build_owner_entry(
        mesh,
        compile_shared_trunk_from_kernel,
        (trunk_kernel, contexts[0]),
    )
    trunk_template = compile_plan(trunk_kernel, contexts[0])
    physical_kernel = bind_physical_compiler_kernel(model)
    perms_template = _place_routes(mesh, _identity_perms(config.n_max))
    compile_trees = _build_compile_trees(
        mesh,
        physical_kernel,
        trunk_template,
        perms_template,
    )
    compile_frames = _build_compile_frames(
        mesh,
        energy_inputs[0],
        contexts[0].mask,
        contexts[0].bmask,
        perms_template,
    )
    sampled_energy = _build_sampled_energy(mesh, config.energy.chunk_size)
    exact_skip = _build_exact_skip(mesh)
    mcmc_entries = {}

    def get_run_routes(state, kernel, trees, n_steps):
        entry_key = (int(n_steps), config.mcmc.walker_chunk_size)
        entry = mcmc_entries.get(entry_key)
        if entry is None:
            entry = _build_run_routes(
                mesh,
                state,
                kernel,
                trees,
                n_steps=entry_key[0],
                chunk_size=entry_key[1],
            )
            mcmc_entries[entry_key] = entry
        return entry

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
                get_run_routes=get_run_routes,
            )
        return _activate_system(mesh, cached)

    first = get_system(0)
    q_seed = jax.vmap(cold_samples)(first.sampler)
    system_batch_sharding = NamedSharding(mesh, P("systems", None))
    systems_sharding = NamedSharding(mesh, P("systems"))
    q_seed = jax.device_put(q_seed, system_batch_sharding)
    energy_seed = jax.device_put(
        np.zeros(q_seed.shape[:2], dtype=np.complex64),
        system_batch_sharding,
    )
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
    reframe = _build_reframe(mesh, first.sampler, first.context, first.perms)
    rebase = _build_rebase(mesh, q_seed, first.perms)
    adapt_routes = jax.jit(
        lambda state: _adapt_routes(state, config),
        in_shardings=(_systems_sharding(mesh, first.sampler),),
        out_shardings=_systems_sharding(mesh, first.sampler),
    )
    target_entry = jax.jit(
        lambda sampled, baseline, sigma, weights: process_route_targets(
            sampled,
            baseline,
            sigma,
            weights,
            mad_width=config.kfac.mad_clip_width,
        ),
        in_shardings=(
            system_batch_sharding,
            system_batch_sharding,
            systems_sharding,
            system_batch_sharding,
        ),
        out_shardings=(system_batch_sharding, systems_sharding),
    )
    snis_entry = jax.jit(
        snis_mode_baseline,
        in_shardings=(
            system_batch_sharding,
            system_batch_sharding,
            system_batch_sharding,
        ),
        out_shardings=system_batch_sharding,
    )
    router_kernel_template = bind_router_kernel(model)
    compile_router = _build_owner_entry(
        mesh,
        compile_router_static,
        (
            router_kernel_template,
            trunk_template,
            contexts[0].route_quotient_node_key,
            contexts[0].route_quotient_edge_key,
            contexts[0].needs_fwl2,
        ),
    )
    router_static_template = compile_router(
        router_kernel_template,
        trunk_template,
        contexts[0].route_quotient_node_key,
        contexts[0].route_quotient_edge_key,
        contexts[0].needs_fwl2,
    )
    route_sampler = build_route_sampler(
        mesh,
        router_kernel_template.decoder,
        router_static_template,
    )
    mode_sampler = build_beam16(
        mesh,
        router_kernel_template.decoder,
        router_static_template,
    )
    mode_perm_template = jnp.arange(config.n_max, dtype=jnp.int32)
    compile_mode = _build_owner_entry(
        mesh,
        _compile_mode,
        (
            physical_kernel,
            trunk_template,
            energy_inputs[0],
            contexts[0].mask,
            contexts[0].bmask,
            mode_perm_template,
        ),
    )
    mode_energy = None
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
        trunk = compile_plan(
            bind_trunk_compiler_kernel(model),
            contexts[system_index],
        )
        router_kernel = bind_router_kernel(model)
        router_static = compile_router(
            router_kernel,
            trunk,
            contexts[system_index].route_quotient_node_key,
            contexts[system_index].route_quotient_edge_key,
            contexts[system_index].needs_fwl2,
        )
        tau = jnp.asarray(config.router.temperature, dtype=jnp.float32)
        key, key_route = jax.random.split(key)
        new_perms = route_sampler(
            router_kernel.decoder, router_static, key_route, tau
        )
        mode_perm = mode_sampler(
            router_kernel.decoder, router_static, tau
        )
        state.sampler, state.context = reframe(
            state.sampler, state.context, state.perms, new_perms
        )
        state.perms = new_perms
        kernel = bind_shared_kernel(model)
        physical_kernel = bind_physical_compiler_kernel(model)
        trees = compile_trees(
            physical_kernel,
            trunk,
            new_perms,
        )
        frames = compile_frames(
            energy_inputs[system_index],
            contexts[system_index].mask,
            contexts[system_index].bmask,
            new_perms,
        )
        state.sampler = get_run_routes(
            state.sampler,
            kernel,
            trees,
            config.mcmc.steps,
        )(state.sampler, kernel, trees)
        if step and step % config.mcmc.adapt_every == 0:
            state.sampler = adapt_routes(state.sampler)
        q_cold = jax.vmap(cold_samples)(state.sampler)
        q_cold = jax.device_put(q_cold, system_batch_sharding)
        total, _exchange, _casimir, _field = sampled_energy(
            kernel,
            trees,
            frames,
            q_cold,
        )
        baseline_is_sampled = bool(np.asarray(jax.device_get(exact_skip(new_perms, mode_perm))))
        if baseline_is_sampled:
            baseline_total = total
            baseline_weights = jnp.ones_like(total.real) / total.shape[-1]
        else:
            mode_tree, mode_frame = compile_mode(
                physical_kernel,
                trunk,
                energy_inputs[system_index],
                contexts[system_index].mask,
                contexts[system_index].bmask,
                mode_perm,
            )
            q_canonical = rebase(q_cold, new_perms)
            if mode_energy is None:
                mode_energy = _build_mode_energy(
                    mesh,
                    kernel,
                    mode_tree,
                    mode_frame,
                    q_canonical,
                    mode_perm,
                    config.energy.chunk_size,
                )
            baseline_total, _bx, _bc, _bf, candidate_log_p = mode_energy(
                kernel,
                mode_tree,
                mode_frame,
                q_canonical,
                mode_perm,
            )
            sampled_log_p = state.sampler.log_p[..., -1]
            sampled_log_p = jax.device_put(sampled_log_p, system_batch_sharding)
            baseline_weights = snis_entry(
                baseline_total, candidate_log_p, sampled_log_p
            )
        target, advantage = target_entry(
            total,
            baseline_total,
            state.context.s_norm,
            baseline_weights,
        )
        target = jax.device_put(target, system_batch_sharding)
        advantage = jax.device_put(advantage, systems_sharding)
        state.context = jax.device_put(
            state.context,
            _systems_sharding(mesh, state.context),
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
