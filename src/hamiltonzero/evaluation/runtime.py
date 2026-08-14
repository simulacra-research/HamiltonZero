# Copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from hamiltonzero.checkpoint import (
    load_model as load_checkpoint_model,
    load_model_metadata,
)
from hamiltonzero.compiled.api import (
    compile_wavefunction,
)
from hamiltonzero.compiled.model import (
    CompiledFinetuneWaveFunction,
    build_finetune_template_model,
)
from hamiltonzero.compiled.types import (
    CompiledWaveFunction,
    CompiledWaveFunctions,
    EnergyInputs,
)
from hamiltonzero.compiled.tree import (
    bind_physical_compiler_kernel,
    compile_physical_tree_from_shared_trunk,
)
from hamiltonzero.compiled.trunk import bind_shared_kernel, compile_shared_trunk
from hamiltonzero.config import EnergyConfig, EvalMCMCConfig, ModelConfig
from hamiltonzero.data.systems import (
    build_context_and_energy,
    load_system as load_spin_hamiltonian,
)
from hamiltonzero.energy import vmc_energy_custom_lap_compiled
from hamiltonzero.energy.custom_lap import build_W_levels
from hamiltonzero.energy.frame import compile_energy_frame, route_energy_inputs
from hamiltonzero.mcmc.runtime import (
    adapt_batched,
    cold_samples,
    init_batched_state,
    run_batched,
)
from hamiltonzero.model.api import build_model
from hamiltonzero.model.context import MultiSystemContext
from hamiltonzero.model.model import _shallow_replace
from hamiltonzero.router.api import (
    route_context as apply_route_context,
    route_state,
)
from hamiltonzero.router.compiled import bind_router_kernel, compile_router_static
from hamiltonzero.router.permutation import (
    permute_ctx_prefix,
    permute_multi_ctx_prefix,
)

from .backend import (
    BeamCandidates,
    CanonicalContext,
    LargeNCompilation,
    MCMCPopulation,
)
from .large_n import compile_large_n_eval_wavefunction


class CompiledEvalContext(eqx.Module):
    mask: Any
    bmask: Any
    route_perm: Any
    energy_frame: Any


class _BeamArrays(NamedTuple):
    permutations: jax.Array
    log_probabilities: jax.Array


def _attention_name(value: str) -> str:
    if value == "tuned":
        return "mhsea_tuned"
    if value == "einsum":
        return "einsum"
    raise ValueError("contextualizer attention must be 'tuned' or 'einsum'")


def _with_contextualizer_attention(model, value: str | None):
    if value is None:
        return model
    contextualizer = model.route_contextualizer
    implementation = _attention_name(value)
    layers = _shallow_replace(
        contextualizer.layers,
        attn_impl=implementation,
    )
    return _shallow_replace(
        model,
        route_contextualizer=_shallow_replace(contextualizer, layers=layers),
    )


def _required_positive_int(metadata: dict[str, Any], name: str) -> int:
    if name not in metadata or isinstance(metadata[name], bool):
        raise ValueError(f"compiled fine-tune checkpoint metadata requires {name!r}")
    value = int(metadata[name])
    if value < 1:
        raise ValueError(
            f"compiled fine-tune checkpoint metadata {name!r} must be positive"
        )
    return value


def _compile_energy_frame(energy_inputs, mask, bmask, permutation):
    return compile_energy_frame(
        energy_inputs,
        mask,
        bmask,
        jnp.asarray(permutation, dtype=jnp.int32),
    )


_compile_frame = jax.jit(_compile_energy_frame)


@jax.jit
def _compile_frame_rows(energy_inputs, mask, bmask, permutations):
    return jax.vmap(
        lambda permutation: _compile_energy_frame(
            energy_inputs,
            mask,
            bmask,
            permutation,
        )
    )(permutations)


def _compiled_wavefunctions_vmap_axes(model: CompiledWaveFunctions):
    return CompiledWaveFunctions(
        kernel=jax.tree_util.tree_map(lambda _value: None, model.kernel),
        trees=jax.tree_util.tree_map(lambda _value: 0, model.trees),
    )


@partial(
    jax.jit,
    static_argnames=("batch_size", "replicas", "initial_m"),
)
def _initialize_single(
    key,
    context,
    initial_sigma,
    *,
    batch_size: int,
    replicas: int,
    initial_m: int,
):
    return init_batched_state(
        jax.random.fold_in(key, jnp.int32(0)),
        context,
        batch_size=batch_size,
        n_replicas=replicas,
        initial_m=initial_m,
        initial_sigma=initial_sigma,
    )


@partial(
    jax.jit,
    static_argnames=("batch_size", "replicas", "initial_m"),
)
def _initialize_rows(
    key,
    context,
    initial_sigma,
    *,
    batch_size: int,
    replicas: int,
    initial_m: int,
):
    indices = jnp.arange(context.mask.shape[0], dtype=jnp.int32)
    return jax.vmap(
        lambda index, context_row: init_batched_state(
            jax.random.fold_in(key, index),
            context_row,
            batch_size=batch_size,
            n_replicas=replicas,
            initial_m=initial_m,
            initial_sigma=initial_sigma,
        )
    )(indices, context)


def _step_single(
    state,
    model,
    context,
    *,
    replica_steps: int,
    walker_chunk_size: int,
):
    return run_batched(
        model,
        context,
        state,
        replica_steps,
        walker_chunk_size=walker_chunk_size,
    )


def _step_compiled_rows(
    state,
    model,
    context,
    *,
    replica_steps: int,
    walker_chunk_size: int,
):
    return jax.vmap(
        lambda state_row, model_row, context_row: run_batched(
            model_row,
            context_row,
            state_row,
            replica_steps,
            walker_chunk_size=walker_chunk_size,
        ),
        in_axes=(0, _compiled_wavefunctions_vmap_axes(model), 0),
    )(state, model, context)


def _adapt_single(
    state,
    beta_history_weight,
    sigma_target,
    sigma_scale,
    haar_target,
):
    return adapt_batched(
        state,
        beta_history_weight=beta_history_weight,
        sigma_target=sigma_target,
        sigma_scale=sigma_scale,
        haar_target=haar_target,
    )


def _adapt_rows(
    state,
    beta_history_weight,
    sigma_target,
    sigma_scale,
    haar_target,
):
    return jax.vmap(
        lambda row: adapt_batched(
            row,
            beta_history_weight=beta_history_weight,
            sigma_target=sigma_target,
            sigma_scale=sigma_scale,
            haar_target=haar_target,
        )
    )(state)


def _compiled_energy_single(kernel, tree, frame, q, *, chunk_size: int):
    return vmc_energy_custom_lap_compiled(
        kernel,
        tree,
        frame,
        q,
        chunk_size=chunk_size,
    )


def _build_compiled_energy_single(mesh: Mesh, chunk_size: int):
    batch = P("batch", None, None)
    batch_output = P("batch")

    def local_energy(kernel, tree, frame, q):
        outputs = _compiled_energy_single(
            kernel,
            tree,
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
        in_specs=(P(), P(), P(), batch),
        out_specs=(batch_output,) * 4,
        check_vma=False,
    )
    replicated_sharding = NamedSharding(mesh, P())
    batch_sharding = NamedSharding(mesh, batch)
    batch_output_sharding = NamedSharding(mesh, batch_output)
    return jax.jit(
        mapped,
        in_shardings=(
            replicated_sharding,
            replicated_sharding,
            replicated_sharding,
            batch_sharding,
        ),
        out_shardings=(batch_output_sharding,) * 4,
    )


def _compiled_energy_rows(kernel, trees, frames, q, *, chunk_size: int):
    def energy_row(tree, frame, q_row):
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

    return jax.vmap(energy_row)(trees, frames, q)


def _build_compiled_energy_rows(mesh: Mesh, chunk_size: int):
    systems = P("systems")
    system_batch = P("systems", None)

    def local_energy(kernel, trees, frames, q):
        outputs = _compiled_energy_rows(
            kernel,
            trees,
            frames,
            q,
            chunk_size=int(chunk_size),
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


_compile_single = jax.jit(compile_wavefunction)


def _systems_mesh(count: int) -> Mesh:
    devices = tuple(jax.devices())
    lanes = min(len(devices), int(count))
    if int(count) % lanes:
        raise ValueError(
            f"contest K={count} must be divisible by visible devices={lanes}"
        )
    return Mesh(np.asarray(devices[:lanes], dtype=object), ("systems",))


def _batch_mesh(batch_size: int) -> Mesh:
    devices = tuple(jax.devices())
    lanes = min(len(devices), int(batch_size))
    while int(batch_size) % lanes:
        lanes -= 1
    return Mesh(np.asarray(devices[:lanes], dtype=object), ("batch",))


def _single_state_sharding(mesh: Mesh, state):
    walkers = NamedSharding(mesh, P("batch"))
    replicated = NamedSharding(mesh, P())
    return type(state)(
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


def _single_state_specs(state):
    walkers = P("batch")
    replicated = P()
    return type(state)(
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


def _row_state_specs(state):
    systems = P("systems")
    return jax.tree_util.tree_map(lambda _value: systems, state)


def _row_model_specs(model):
    return CompiledWaveFunctions(
        kernel=jax.tree_util.tree_map(lambda _value: P(), model.kernel),
        trees=jax.tree_util.tree_map(lambda _value: P("systems"), model.trees),
    )


def _row_model_sharding(mesh: Mesh, model):
    return CompiledWaveFunctions(
        kernel=_replicated_sharding(mesh, model.kernel),
        trees=_system_sharding(mesh, model.trees),
    )


def _build_rows_mcmc_step(
    mesh: Mesh,
    state,
    model,
    context,
    *,
    replica_steps: int,
    walker_chunk_size: int,
):
    state_specs = _row_state_specs(state)
    model_specs = _row_model_specs(model)
    context_specs = jax.tree_util.tree_map(lambda _value: P("systems"), context)
    state_sharding = _system_sharding(mesh, state)
    model_sharding = _row_model_sharding(mesh, model)
    context_sharding = _system_sharding(mesh, context)

    def local_step(state_local, model_local, context_local):
        out = _step_compiled_rows(
            state_local,
            model_local,
            context_local,
            replica_steps=int(replica_steps),
            walker_chunk_size=int(walker_chunk_size),
        )
        local_count = jnp.asarray(out.q.shape[0] * out.q.shape[1], dtype=jnp.int32)
        global_count = jax.lax.psum(local_count, "systems")
        guard = global_count.astype(out.q.dtype) * jnp.asarray(0.0, out.q.dtype)
        return eqx.tree_at(lambda value: value.q, out, out.q + guard)

    mapped = jax.shard_map(
        local_step,
        mesh=mesh,
        in_specs=(state_specs, model_specs, context_specs),
        out_specs=state_specs,
        check_vma=False,
    )
    return jax.jit(
        mapped,
        in_shardings=(state_sharding, model_sharding, context_sharding),
        out_shardings=state_sharding,
        donate_argnums=(0,),
    )


def _build_singular_mcmc_step(
    mesh: Mesh,
    state,
    state_sharding,
    model_sharding,
    context_sharding,
    *,
    replica_steps: int,
    walker_chunk_size: int,
):
    state_specs = _single_state_specs(state)

    def local_step(state_local, model_local, context_local):
        out = _step_single(
            state_local,
            model_local,
            context_local,
            replica_steps=int(replica_steps),
            walker_chunk_size=int(walker_chunk_size),
        )
        local_count = jnp.asarray(out.q.shape[0], dtype=jnp.int32)
        global_count = jax.lax.psum(local_count, "batch")
        guard = global_count.astype(out.q.dtype) * jnp.asarray(0.0, out.q.dtype)
        return eqx.tree_at(lambda value: value.q, out, out.q + guard)

    mapped = jax.shard_map(
        local_step,
        mesh=mesh,
        in_specs=(state_specs, P(), P()),
        out_specs=state_specs,
        check_vma=False,
    )
    return jax.jit(
        mapped,
        in_shardings=(state_sharding, model_sharding, context_sharding),
        out_shardings=state_sharding,
        donate_argnums=(0,),
    )


def _system_sharding(mesh: Mesh, value):
    return jax.tree_util.tree_map(
        lambda array: NamedSharding(
            mesh,
            P("systems", *([None] * (array.ndim - 1))),
        ),
        value,
    )


def _replicated_sharding(mesh: Mesh, value):
    replicated = NamedSharding(mesh, P())
    return jax.tree_util.tree_map(lambda _array: replicated, value)


def _place_system_rows(mesh: Mesh, value):
    return jax.device_put(value, _system_sharding(mesh, value))


def _abstract(tree):
    return jax.tree_util.tree_map(
        lambda value: jax.ShapeDtypeStruct(value.shape, value.dtype),
        tree,
    )


def _beam_mesh(width: int) -> Mesh:
    devices = tuple(jax.devices())
    lanes = len(devices) if int(width) % len(devices) == 0 else 1
    return Mesh(np.asarray(devices[:lanes], dtype=object), ("systems",))


def _distributed_beam_local(decoder, static, tau, *, width: int, lanes: int):
    permutations, log_probabilities = decoder.beam_search(
        static.node_input,
        static.raw_edge,
        static.routable_mask,
        global_feat=static.global_input,
        tau=tau,
        beam_width=width,
        real_mask=static.real_mask,
        first_orbit_ids=(
            static.quotient_node_key,
            static.quotient_edge_key,
            static.needs_fwl2,
        ),
        router_static=static,
        distributed_axis_name="systems",
        distributed_lanes=lanes,
    )
    return _BeamArrays(permutations, log_probabilities)


def _build_distributed_beam(mesh: Mesh, decoder, static, width: int):
    lanes = int(mesh.shape["systems"])
    if tuple(mesh.axis_names) != ("systems",) or int(width) % lanes:
        raise ValueError("distributed eval beam requires a divisible systems mesh")
    mapped = jax.shard_map(
        partial(
            _distributed_beam_local,
            width=int(width),
            lanes=lanes,
        ),
        mesh=mesh,
        in_specs=(
            jax.tree_util.tree_map(lambda _leaf: P(), decoder),
            jax.tree_util.tree_map(lambda _leaf: P(), static),
            P(),
        ),
        out_specs=_BeamArrays(P(), P()),
        check_vma=False,
    )
    replicated = NamedSharding(mesh, P())
    return jax.jit(
        mapped,
        in_shardings=(
            jax.tree_util.tree_map(lambda _leaf: replicated, decoder),
            jax.tree_util.tree_map(lambda _leaf: replicated, static),
            replicated,
        ),
        out_shardings=_BeamArrays(
            replicated,
            replicated,
        ),
    )


def _compile_eval_router_static(model, context):
    trunk = compile_shared_trunk(model, context)
    kernel = bind_router_kernel(model)
    return compile_router_static(
        kernel,
        trunk,
        context.route_quotient_node_key,
        context.route_quotient_edge_key,
        context.needs_fwl2,
    )


class DefaultEvalBackend:
    def __init__(self) -> None:
        self._energy_inputs_by_context: dict[int, EnergyInputs] = {}
        self._energy_frames: dict[int, Any] = {}
        self._context_meshes: dict[int, Mesh] = {}
        self._contest_mesh: Mesh | None = None
        self._contest_step_entries: dict[tuple[int, int], Any] = {}
        self._contest_adapt_entry: Any | None = None
        self._singular_mesh: Mesh | None = None
        self._singular_state_sharding: Any | None = None
        self._singular_model_sharding: Any | None = None
        self._singular_context_sharding: Any | None = None
        self._singular_step_entries: dict[tuple[int, int], Any] = {}
        self._singular_adapt_entry: Any | None = None
        self._singular_energy_entries: dict[int, Any] = {}
        self._contest_energy_entries: dict[int, Any] = {}

    def build_system(self, system, energy: EnergyConfig):
        context, energy_inputs = build_context_and_energy(
            system,
            n_max=None,
            mu=energy.mu,
            eps=energy.eps,
        )
        self._energy_inputs_by_context[id(context)] = energy_inputs
        return context

    def load_system(self, path: Path, energy: EnergyConfig):
        return self.build_system(load_spin_hamiltonian(path), energy)

    def load_model(
        self,
        checkpoint: Path,
        config: ModelConfig,
        key,
        context,
        *,
        contextualizer_attention: str | None,
    ):
        metadata = load_model_metadata(checkpoint) or {}
        kind = metadata.get("kind", "router")
        n_sites = int(context.mask.shape[-1])
        eager_template = build_model(config, key, n_max=n_sites)
        if kind == "router":
            eager_template = _with_contextualizer_attention(
                eager_template,
                contextualizer_attention,
            )
            return load_checkpoint_model(checkpoint, eager_template)
        if kind != "compiled_finetune":
            raise ValueError(f"unsupported checkpoint kind {kind!r}")
        leaf_rank = _required_positive_int(metadata, "leaf_rank")
        merge_rank = _required_positive_int(metadata, "merge_rank")
        checkpoint_n = int(metadata.get("n_max", n_sites))
        if checkpoint_n != n_sites:
            raise ValueError(
                "compiled fine-tune checkpoint width does not match the "
                f"evaluation system: checkpoint={checkpoint_n}, system={n_sites}"
            )
        template = build_finetune_template_model(
            eager_template,
            n_sites,
            leaf_rank=leaf_rank,
            merge_rank=merge_rank,
        )
        return load_checkpoint_model(checkpoint, template)

    def canonicalize_context(self, context) -> CanonicalContext:
        route = jnp.asarray(context.route_perm, dtype=jnp.int32)
        if route.ndim != 1:
            raise ValueError("single-system context route must have shape [N]")
        inverse = jnp.argsort(route).astype(jnp.int32)
        canonical = permute_ctx_prefix(context, inverse)
        identity = jnp.arange(route.shape[0], dtype=jnp.int32)
        canonical = eqx.tree_at(
            lambda value: value.route_perm,
            canonical,
            identity,
        )
        energy_inputs = self._energy_inputs_by_context.get(id(context))
        if energy_inputs is None:
            raise RuntimeError("energy inputs are unavailable for this context")
        self._energy_inputs_by_context[id(canonical)] = route_energy_inputs(
            energy_inputs,
            inverse,
        )
        return CanonicalContext(
            context=canonical,
            old_inverse=inverse[None, :],
        )

    def embedded_route(self, model):
        if isinstance(model, CompiledFinetuneWaveFunction):
            return jnp.asarray(model.perm, dtype=jnp.int32)
        return None

    def route_context(
        self,
        context,
        permutation,
        *,
        compact_custom_lap: bool,
    ):
        permutation = jnp.asarray(permutation, dtype=jnp.int32)
        if permutation.ndim == 2:
            if permutation.shape[0] != 1:
                raise ValueError("single-system route must have shape [1, N]")
            permutation = permutation[0]
        if permutation.ndim != 1:
            raise ValueError("single-system route must have shape [N]")
        energy_inputs = self._energy_inputs_by_context.get(id(context))
        if energy_inputs is None:
            raise RuntimeError("energy inputs are unavailable for this context")
        frame = _compile_frame(
            energy_inputs,
            context.mask,
            context.bmask,
            permutation,
        )
        if compact_custom_lap:
            routed = CompiledEvalContext(
                mask=frame.masks.real,
                bmask=frame.masks.balanced,
                route_perm=permutation,
                energy_frame=frame,
            )
            self._energy_frames[id(routed)] = frame
            return routed
        routed = apply_route_context(context, permutation)
        self._energy_frames[id(routed)] = frame
        return routed

    def virtual_context(self, context, permutations):
        permutations = jnp.asarray(permutations, dtype=jnp.int32)
        if permutations.ndim != 2:
            raise ValueError("candidate permutations must have shape [K, N]")
        count = permutations.shape[0]
        batched = MultiSystemContext.from_single(context)
        tiled = jax.tree_util.tree_map(
            lambda value: (
                jnp.repeat(value, count, axis=0)
                if eqx.is_array(value) and value.ndim >= 1
                else value
            ),
            batched,
        )
        tiled = eqx.tree_at(
            lambda value: value.route_perm,
            tiled,
            permutations,
        )
        routed = permute_multi_ctx_prefix(tiled, permutations)
        energy_inputs = self._energy_inputs_by_context.get(id(context))
        if energy_inputs is None:
            raise RuntimeError("energy inputs are unavailable for this context")
        frames = _compile_frame_rows(
            energy_inputs,
            context.mask,
            context.bmask,
            permutations,
        )
        mesh = _systems_mesh(count)
        routed = _place_system_rows(mesh, routed)
        frames = _place_system_rows(mesh, frames)
        self._energy_frames[id(routed)] = frames
        self._context_meshes[id(routed)] = mesh
        self._contest_mesh = mesh
        self._contest_step_entries.clear()
        self._contest_adapt_entry = None
        self._contest_energy_entries.clear()
        return routed

    def release_context(self, context) -> None:
        self._energy_inputs_by_context.pop(id(context), None)
        self._energy_frames.pop(id(context), None)
        self._context_meshes.pop(id(context), None)
        self._contest_mesh = None
        self._contest_step_entries.clear()
        self._contest_adapt_entry = None
        self._contest_energy_entries.clear()

    def beam_candidates(
        self,
        model,
        context,
        *,
        beam_width: int,
        top_k: int,
        temperature: float,
    ) -> BeamCandidates:
        decoder = getattr(model, "route_decoder", None)
        from hamiltonzero.model.route_pointer import TreePrefixPointerMHSEA

        if not isinstance(decoder, TreePrefixPointerMHSEA):
            raise ValueError("eval requires the learned-quotient TreePrefix decoder")
        if int(top_k) > int(beam_width):
            raise ValueError("top_k cannot exceed beam_width")
        mesh = _beam_mesh(int(beam_width))
        static = eqx.filter_jit(_compile_eval_router_static)(model, context)
        replicated = NamedSharding(mesh, P())
        decoder, static, tau = jax.device_put(
            (
                decoder,
                static,
                jnp.asarray(temperature, dtype=jnp.float32),
            ),
            replicated,
        )
        result = _build_distributed_beam(
            mesh,
            decoder,
            static,
            int(beam_width),
        )(decoder, static, tau)
        jax.block_until_ready(result.permutations)
        permutations = result.permutations[: int(top_k)].astype(jnp.int32)
        log_probabilities = result.log_probabilities[: int(top_k)].astype(jnp.float32)
        return BeamCandidates(
            permutations=permutations[None],
            log_probabilities=log_probabilities[None],
        )

    def compile_single(self, model, routed_context):
        n_sites = int(routed_context.mask.shape[-1])
        return _compile_single(
            model,
            routed_context,
            jnp.arange(n_sites, dtype=jnp.int32),
        )

    def compile_embedded(self, model):
        if not isinstance(model, CompiledFinetuneWaveFunction):
            raise TypeError("embedded eval compilation requires a fine-tune checkpoint")
        return CompiledWaveFunction(
            kernel=model.kernel,
            tree=model.as_compiled_tree(),
        )

    def compile_candidates(self, model, canonical_context, permutations):
        if self._contest_mesh is None:
            raise RuntimeError("contest context must be built before compilation")
        mesh = self._contest_mesh
        physical_kernel = bind_physical_compiler_kernel(model)
        shared_trunk = jax.jit(compile_shared_trunk)(model, canonical_context)
        jax.block_until_ready(shared_trunk)
        physical_sharding = _replicated_sharding(mesh, physical_kernel)
        trunk_sharding = _replicated_sharding(mesh, shared_trunk)
        permutation_sharding = NamedSharding(mesh, P("systems", None))
        physical_kernel = jax.device_put(physical_kernel, physical_sharding)
        shared_trunk = jax.device_put(shared_trunk, trunk_sharding)
        permutations = jax.device_put(
            jnp.asarray(permutations, dtype=jnp.int32),
            permutation_sharding,
        )

        def compile_all(kernel, trunk, candidate_permutations):
            return jax.vmap(
                lambda permutation: compile_physical_tree_from_shared_trunk(
                    kernel,
                    trunk,
                    permutation,
                )
            )(candidate_permutations)

        tree_template = jax.eval_shape(
            compile_all,
            _abstract(physical_kernel),
            _abstract(shared_trunk),
            _abstract(permutations),
        )
        tree_sharding = _system_sharding(mesh, tree_template)
        trees = jax.jit(
            compile_all,
            in_shardings=(
                physical_sharding,
                trunk_sharding,
                permutation_sharding,
            ),
            out_shardings=tree_sharding,
        )(physical_kernel, shared_trunk, permutations)
        shared_kernel = bind_shared_kernel(model)
        shared_kernel = jax.device_put(
            shared_kernel,
            _replicated_sharding(mesh, shared_kernel),
        )
        compiled = CompiledWaveFunctions(shared_kernel, trees)
        jax.block_until_ready(compiled)
        return compiled

    def select_candidate(self, wavefunctions, winner: int):
        if self._contest_mesh is None:
            raise RuntimeError("contest mesh is unavailable for winner selection")
        mesh = self._contest_mesh
        count = int(wavefunctions.trees.perm.shape[0])
        index = int(winner)
        if index < 0 or index >= count:
            raise IndexError(
                f"winner index {index} outside candidate range [0, {count})"
            )
        tree_sharding = _system_sharding(mesh, wavefunctions.trees)
        winner_sharding = NamedSharding(mesh, P())

        def gather(trees, selected_index):
            return jax.tree_util.tree_map(
                lambda value: jax.lax.dynamic_index_in_dim(
                    value,
                    selected_index,
                    axis=0,
                    keepdims=False,
                ),
                trees,
            )

        output_template = jax.eval_shape(
            gather,
            _abstract(wavefunctions.trees),
            jax.ShapeDtypeStruct((), jnp.int32),
        )
        tree = jax.jit(
            gather,
            in_shardings=(tree_sharding, winner_sharding),
            out_shardings=_replicated_sharding(mesh, output_template),
        )(
            wavefunctions.trees,
            jax.device_put(jnp.asarray(index, jnp.int32), winner_sharding),
        )
        selected = CompiledWaveFunction(wavefunctions.kernel, tree)
        jax.block_until_ready(selected)
        return selected

    def compile_large_n(
        self,
        model,
        canonical_context,
        *,
        sequence_shards: int,
        pair_tile_size: int,
        temperature: float,
    ) -> LargeNCompilation:
        result = compile_large_n_eval_wavefunction(
            model,
            canonical_context,
            seq_shards=int(sequence_shards),
            pair_tile_size=int(pair_tile_size),
            tau=float(temperature),
        )
        device = jax.devices()[0]
        return LargeNCompilation(
            wavefunction=jax.device_put(result.wavefunction, device),
            permutation=jax.device_put(result.perm, device),
            log_probability=jax.device_put(result.logp, device),
        )

    def prepare_singular(self, model, context, state):
        if state.q.ndim != 4:
            raise ValueError(
                "post-selection eval MCMC state must have shape [B, R, N, 4]"
            )
        frame = self._frames(context)
        mesh = _batch_mesh(int(state.q.shape[0]))
        state_sharding = _single_state_sharding(mesh, state)
        model_sharding = _replicated_sharding(mesh, model)
        context_sharding = _replicated_sharding(mesh, context)
        model = jax.device_put(model, model_sharding)
        context = jax.device_put(context, context_sharding)
        state = jax.device_put(state, state_sharding)
        self._energy_frames[id(context)] = frame
        jax.block_until_ready((model, context, state.q))
        self._singular_mesh = mesh
        self._singular_state_sharding = state_sharding
        self._singular_model_sharding = model_sharding
        self._singular_context_sharding = context_sharding
        self._singular_step_entries.clear()
        self._singular_adapt_entry = None
        self._singular_energy_entries.clear()
        return model, context, state

    def initialize_mcmc(
        self,
        key,
        model,
        context,
        config: EvalMCMCConfig,
    ):
        del model
        if isinstance(context, MultiSystemContext):
            state = _initialize_rows(
                key,
                context,
                jnp.asarray(config.initial_sigma, dtype=jnp.float32),
                batch_size=int(config.batch_size),
                replicas=int(config.replicas),
                initial_m=int(config.initial_haar_sites),
            )
            mesh = self._context_meshes.get(id(context))
            return _place_system_rows(mesh, state) if mesh is not None else state
        return _initialize_single(
            key,
            context,
            batch_size=int(config.batch_size),
            replicas=int(config.replicas),
            initial_m=int(config.initial_haar_sites),
            initial_sigma=jnp.asarray(config.initial_sigma, dtype=jnp.float32),
        )

    def step_mcmc(
        self,
        state,
        model,
        context,
        *,
        replica_steps: int,
        walker_chunk_size: int,
    ):
        if state.q.ndim != 5:
            if (
                self._singular_mesh is None
                or self._singular_state_sharding is None
                or self._singular_model_sharding is None
                or self._singular_context_sharding is None
            ):
                raise RuntimeError("singular eval placement has not been prepared")
            key = (int(replica_steps), int(walker_chunk_size))
            step = self._singular_step_entries.get(key)
            if step is None:
                step = _build_singular_mcmc_step(
                    self._singular_mesh,
                    state,
                    self._singular_state_sharding,
                    self._singular_model_sharding,
                    self._singular_context_sharding,
                    replica_steps=key[0],
                    walker_chunk_size=key[1],
                )
                self._singular_step_entries[key] = step
            return step(state, model, context)
        if not isinstance(model, CompiledWaveFunctions):
            raise TypeError("multirow eval MCMC requires compiled wavefunctions")
        mesh = self._context_meshes.get(id(context))
        if mesh is None:
            raise RuntimeError("contest MCMC mesh is unavailable")
        key = (int(replica_steps), int(walker_chunk_size))
        step = self._contest_step_entries.get(key)
        if step is None:
            step = _build_rows_mcmc_step(
                mesh,
                state,
                model,
                context,
                replica_steps=key[0],
                walker_chunk_size=key[1],
            )
            self._contest_step_entries[key] = step
        return step(state, model, context)

    def adapt_mcmc(self, state, config: EvalMCMCConfig):
        arguments = (
            state,
            jnp.asarray(config.beta_history_weight, dtype=jnp.float32),
            jnp.asarray(config.langevin_target_acceptance, dtype=jnp.float32),
            jnp.asarray(config.sigma_scale, dtype=jnp.float32),
            jnp.asarray(config.haar_target_acceptance, dtype=jnp.float32),
        )
        if state.q.ndim == 5:
            if self._contest_mesh is None:
                raise RuntimeError("contest adaptation mesh is unavailable")
            if self._contest_adapt_entry is None:
                replicated = NamedSharding(self._contest_mesh, P())
                state_sharding = _system_sharding(self._contest_mesh, state)
                self._contest_adapt_entry = jax.jit(
                    _adapt_rows,
                    in_shardings=(
                        state_sharding,
                        replicated,
                        replicated,
                        replicated,
                        replicated,
                    ),
                    out_shardings=state_sharding,
                )
            return self._contest_adapt_entry(*arguments)
        if self._singular_mesh is None or self._singular_state_sharding is None:
            raise RuntimeError("singular eval placement has not been prepared")
        if self._singular_adapt_entry is None:
            replicated = NamedSharding(self._singular_mesh, P())
            self._singular_adapt_entry = jax.jit(
                _adapt_single,
                in_shardings=(
                    self._singular_state_sharding,
                    replicated,
                    replicated,
                    replicated,
                    replicated,
                ),
                out_shardings=self._singular_state_sharding,
            )
        return self._singular_adapt_entry(*arguments)

    def route_mcmc(self, state, permutation):
        return route_state(state, permutation)

    def mcmc_population(self, state) -> MCMCPopulation:
        return MCMCPopulation(q=state.q, sigma=state.sigma, beta=state.beta)

    def replace_mcmc_population(
        self,
        state,
        population: MCMCPopulation,
    ):
        q = population.q
        sigma = population.sigma
        beta = population.beta
        if q.ndim == state.q.ndim + 1 and q.shape[0] == 1:
            q = jax.device_put(q[0], jax.devices()[0])
        if sigma.ndim == state.sigma.ndim + 1 and sigma.shape[0] == 1:
            sigma = jax.device_put(sigma[0], jax.devices()[0])
        if beta.ndim == state.beta.ndim + 1 and beta.shape[0] == 1:
            beta = jax.device_put(beta[0], jax.devices()[0])
        return eqx.tree_at(
            lambda value: (value.q, value.sigma, value.beta),
            state,
            (
                q.astype(state.q.dtype),
                sigma.astype(state.sigma.dtype),
                beta.astype(state.beta.dtype),
            ),
        )

    def cold_walkers(self, state):
        return (
            jax.vmap(cold_samples)(state) if state.q.ndim == 5 else cold_samples(state)
        )

    def _frames(self, context):
        if isinstance(context, CompiledEvalContext):
            return context.energy_frame
        frames = self._energy_frames.get(id(context))
        if frames is not None:
            return frames
        energy_inputs = self._energy_inputs_by_context.get(id(context))
        if energy_inputs is None:
            raise RuntimeError("energy frame is unavailable for this context")
        if isinstance(context, MultiSystemContext):
            raise RuntimeError("multi-system energy frames must be compiled explicitly")
        n_sites = int(context.mask.shape[-1])
        frames = _compile_frame(
            energy_inputs,
            context.mask,
            context.bmask,
            jnp.arange(n_sites, dtype=jnp.int32),
        )
        self._energy_frames[id(context)] = frames
        return frames

    def custom_lap_energy(
        self,
        model,
        context,
        q,
        config: EnergyConfig,
    ):
        chunk_size = int(config.chunk_size)
        if isinstance(model, CompiledWaveFunctions):
            frames = self._frames(context)
            mesh = self._context_meshes.get(id(context))
            if mesh is None:
                raise RuntimeError("contest energy mesh is unavailable")
            energy = self._contest_energy_entries.get(chunk_size)
            if energy is None:
                energy = _build_compiled_energy_rows(mesh, chunk_size)
                self._contest_energy_entries[chunk_size] = energy
            return energy(
                model.kernel,
                model.trees,
                frames,
                q,
            )
        if not isinstance(model, CompiledWaveFunction):
            raise TypeError("singular eval energy requires a compiled wavefunction")
        if (
            self._singular_mesh is None
            or self._singular_model_sharding is None
            or self._singular_context_sharding is None
        ):
            raise RuntimeError("singular eval placement has not been prepared")
        q_sharding = NamedSharding(
            self._singular_mesh,
            P("batch", None, None),
        )
        q = jax.device_put(q, q_sharding)
        frames = self._frames(context)
        frame_sharding = _replicated_sharding(
            self._singular_mesh,
            frames,
        )
        frames = jax.device_put(frames, frame_sharding)
        energy = self._singular_energy_entries.get(chunk_size)
        if energy is None:
            energy = _build_compiled_energy_single(
                self._singular_mesh,
                chunk_size,
            )
            self._singular_energy_entries[chunk_size] = energy
        outputs = energy(
            model.kernel,
            model.tree,
            frames,
            q,
        )
        return tuple(jnp.expand_dims(value, axis=0) for value in outputs)

    def block_until_ready(self, value) -> None:
        jax.block_until_ready(value)


def build_eval_backend() -> DefaultEvalBackend:
    return DefaultEvalBackend()


__all__ = [
    "DefaultEvalBackend",
    "build_eval_backend",
]
