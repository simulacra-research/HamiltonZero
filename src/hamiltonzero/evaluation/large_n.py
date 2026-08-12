# Copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import gc
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from hamiltonzero.compiled.tree import bind_physical_compiler_kernel
from hamiltonzero.compiled.trunk import (
    bind_shared_kernel,
    bind_trunk_compiler_kernel,
)
from hamiltonzero.compiled.types import CompiledWaveFunction
from hamiltonzero.model.route_pointer import TreePrefixPointerMHSEA
from .sequence_trunk import (
    build_sequence_pair_permute,
    build_sequence_parallel_contextualizer,
    build_sequence_parallel_edge_global_update,
    build_sequence_parallel_physical_leaf,
    build_sequence_parallel_physical_reducer,
    build_sequence_parallel_shared_trunk,
)
from .greedy_router import build_compact_greedy_router


class LargeNCompiledEvalResult(NamedTuple):
    wavefunction: CompiledWaveFunction
    perm: jax.Array
    logp: jax.Array


def _sequence_mesh(n: int, requested_shards: int) -> Mesh:
    devices = tuple(jax.devices())
    shards = len(devices) if requested_shards == 0 else requested_shards
    if shards < 1 or shards > len(devices):
        raise ValueError(
            f"compiled eval requested {shards} seq shards, but JAX exposes "
            f"{len(devices)} devices"
        )
    if n % shards:
        raise ValueError(f"N={n} must be divisible by seq_shards={shards}")
    local_rows = n // shards
    if n & (n - 1) or local_rows & (local_rows - 1):
        raise ValueError(
            "large-N physical compilation requires a power-of-two padded "
            f"width and power-of-two rows per lane; got N={n}, "
            f"local_rows={local_rows}"
        )
    return Mesh(np.asarray(devices[:shards], dtype=object), ("seq",))


def _replicate(tree, sharding: NamedSharding):
    return jax.device_put(tree, jax.tree_util.tree_map(lambda _leaf: sharding, tree))


class _LargeNXlaTreePrefixPointer(TreePrefixPointerMHSEA):
    def _resolve_heavy_attn_impl(self, n: int):
        del n
        return None

    def _resolve_tree_attn_impl(self):
        return None

    def _route_attention(
        self,
        q,
        k,
        v,
        edge_bias,
        key_mask,
        *,
        impl,
        key_mask_only=False,
        attention_mask=None,
        sequence_axis_name=None,
        sequence_mesh=None,
    ):
        del impl, key_mask_only
        dtype = q.dtype
        valid = key_mask.astype(bool)[:, None, :]
        if attention_mask is not None:
            valid = valid & attention_mask.astype(bool)
        has_key = jnp.any(valid, axis=-1)
        q_c = q.astype(jnp.float32)
        k_c = k.astype(jnp.float32)
        v_c = v.astype(jnp.float32)
        bias_c = edge_bias.astype(jnp.float32)
        if sequence_axis_name is not None:

            def sharding(*axes):
                spec = P(*axes)
                return (
                    NamedSharding(sequence_mesh, spec)
                    if sequence_mesh is not None
                    else spec
                )

            q_c = jax.lax.with_sharding_constraint(
                q_c, sharding(None, sequence_axis_name, None, None)
            )
            k_c = jax.lax.with_sharding_constraint(
                k_c, sharding(None, None, None, None)
            )
            v_c = jax.lax.with_sharding_constraint(
                v_c, sharding(None, None, None, None)
            )
            bias_c = jax.lax.with_sharding_constraint(
                bias_c, sharding(None, sequence_axis_name, None, None)
            )
        logits = jnp.einsum("bihd,bjhd->bhij", q_c, k_c)
        logits = logits / jnp.sqrt(jnp.asarray(self.d_head, dtype=jnp.float32))
        logits = logits + jnp.transpose(bias_c, (0, 3, 1, 2))
        if sequence_axis_name is not None:
            logits = jax.lax.with_sharding_constraint(
                logits, sharding(None, None, sequence_axis_name, None)
            )
        logits = jnp.where(
            valid[:, None, :, :],
            logits,
            jnp.asarray(-1.0e30, dtype=jnp.float32),
        )
        if sequence_axis_name is not None:
            logits = jax.lax.with_sharding_constraint(
                logits, sharding(None, None, sequence_axis_name, None)
            )
        alpha = jax.nn.softmax(logits, axis=-1)
        if sequence_axis_name is not None:
            alpha = jax.lax.with_sharding_constraint(
                alpha, sharding(None, None, sequence_axis_name, None)
            )
        out = jnp.einsum("bhij,bjhd->bihd", alpha, v_c)
        if sequence_axis_name is not None:
            out = jax.lax.with_sharding_constraint(
                out, sharding(None, sequence_axis_name, None, None)
            )
        out = out.astype(dtype)
        return jnp.where(has_key[..., None, None], out, jnp.zeros_like(out))


def _large_n_xla_decoder_view(decoder: TreePrefixPointerMHSEA):
    compiled = object.__new__(_LargeNXlaTreePrefixPointer)
    compiled.__dict__.update(decoder.__dict__)
    return compiled


def _validate_large_n_model(model) -> None:
    decoder = getattr(model, "route_decoder", None)
    failures: list[str] = []
    if not isinstance(decoder, TreePrefixPointerMHSEA):
        failures.append("route decoder must be TreePrefixPointerMHSEA")
    if getattr(model, "route_contextualizer", None) is None:
        failures.append("route contextualizer must be enabled")
    if getattr(model, "gladder_fork_route", None) is None:
        failures.append("route global fork must be enabled")
    if getattr(model, "readout_leaf_context", None) is None:
        failures.append("physical contextualizer must be enabled")
    if failures:
        raise ValueError(
            "unsupported large-N eval compiler model: " + "; ".join(failures)
        )


def _validate_context(ctx) -> int:
    jdp = jnp.asarray(ctx.J_double_prime)
    mask = jnp.asarray(ctx.mask)
    bmask = jnp.asarray(ctx.bmask)
    h_prime = jnp.asarray(ctx.h_prime)
    if jdp.ndim != 3 or jdp.shape[-1] != 10:
        raise ValueError("ctx.J_double_prime must have shape [N,N,10]")
    n = int(jdp.shape[0])
    if jdp.shape[1] != n:
        raise ValueError("ctx.J_double_prime pair axes must be square")
    if mask.shape != (n,) or bmask.shape != (n,):
        raise ValueError("ctx.mask and ctx.bmask must both have shape [N]")
    if h_prime.shape != (n, 3):
        raise ValueError("ctx.h_prime must have shape [N,3]")
    if jdp.dtype != jnp.float32 or h_prime.dtype != jnp.float32:
        raise TypeError(
            "large-N eval keeps streamed pair/frontier arithmetic in fp32; "
            f"got J={jdp.dtype}, h={h_prime.dtype}"
        )
    return n


def compile_large_n_eval_wavefunction(
    model,
    ctx,
    *,
    seq_shards: int = 0,
    pair_tile_size: int = 128,
    tau: float = 1.0,
) -> LargeNCompiledEvalResult:

    _validate_large_n_model(model)
    n = _validate_context(ctx)
    if pair_tile_size < 1:
        raise ValueError("pair_tile_size must be positive")
    mesh = _sequence_mesh(n, int(seq_shards))
    rep = NamedSharding(mesh, P())
    seq_edge = NamedSharding(mesh, P("seq", None, None))

    trunk_kernel = _replicate(bind_trunk_compiler_kernel(model), rep)
    route_contextualizer = _replicate(model.route_contextualizer, rep)
    route_global_fork = _replicate(model.gladder_fork_route, rep)
    decoder = _replicate(_large_n_xla_decoder_view(model.route_decoder), rep)
    physical_kernel = _replicate(bind_physical_compiler_kernel(model), rep)

    jdp = jax.device_put(jnp.asarray(ctx.J_double_prime), seq_edge)
    h_prime, real_mask, structural_mask = jax.device_put(
        (
            jnp.asarray(ctx.h_prime),
            jnp.asarray(ctx.mask),
            jnp.asarray(ctx.bmask),
        ),
        rep,
    )

    trunk_entry = build_sequence_parallel_shared_trunk(
        mesh=mesh,
        kernel_template=trunk_kernel,
        featurizer_tile_size=int(pair_tile_size),
    )
    trunk = trunk_entry(trunk_kernel, jdp, h_prime, real_mask, structural_mask)

    route_context_entry = build_sequence_parallel_contextualizer(
        mesh=mesh,
        contextualizer_template=route_contextualizer,
        g_template=trunk.global_stream,
        tile_size=int(pair_tile_size),
    )
    route_node, route_edge, route_g = route_context_entry(
        route_contextualizer,
        trunk.node_raw,
        trunk.edge_raw,
        real_mask,
        structural_mask,
        trunk.global_stream,
    )
    route_global_entry = build_sequence_parallel_edge_global_update(
        mesh=mesh,
        module_template=route_global_fork,
        tile_size=int(pair_tile_size),
    )
    route_global = route_global_entry(
        route_global_fork, route_g, route_edge, structural_mask
    )

    route_entry = build_compact_greedy_router(
        mesh=mesh,
        decoder_template=decoder,
        pair_tile_size=int(pair_tile_size),
    )
    perm, logp = route_entry(
        decoder,
        route_node,
        route_edge,
        structural_mask,
        route_global,
        jnp.asarray(tau, dtype=jnp.float32),
        real_mask,
    )

    jax.block_until_ready((perm, logp))
    del route_node, route_edge, route_g, route_global

    pair_permute = build_sequence_pair_permute(mesh=mesh)
    routed_node, routed_edge = pair_permute(trunk.node_raw, trunk.edge_raw, perm)
    leaf_real = real_mask[perm]
    global_stream = trunk.global_stream

    jax.block_until_ready((routed_node, routed_edge, leaf_real, global_stream))
    del (
        trunk,
        trunk_kernel,
        route_contextualizer,
        route_global_fork,
        decoder,
        jdp,
        h_prime,
        real_mask,
        trunk_entry,
        route_context_entry,
        route_global_entry,
        route_entry,
        pair_permute,
    )
    gc.collect()

    physical_context_entry = build_sequence_parallel_contextualizer(
        mesh=mesh,
        contextualizer_template=physical_kernel.contextualizer,
        g_template=global_stream,
        tile_size=int(pair_tile_size),
    )
    physical_node, physical_edge, physical_context_g = physical_context_entry(
        physical_kernel.contextualizer,
        routed_node,
        routed_edge,
        leaf_real,
        structural_mask,
        global_stream,
    )
    jax.block_until_ready((physical_node, physical_edge, physical_context_g))
    del routed_node, routed_edge, global_stream, physical_context_entry
    gc.collect()

    physical_global_entry = build_sequence_parallel_edge_global_update(
        mesh=mesh,
        module_template=physical_kernel.global_fork,
        tile_size=int(pair_tile_size),
    )
    physical_global = physical_global_entry(
        physical_kernel.global_fork,
        physical_context_g,
        physical_edge,
        structural_mask,
    )
    jax.block_until_ready(physical_global)
    del physical_context_g, physical_global_entry
    gc.collect()

    physical_leaf_entry = build_sequence_parallel_physical_leaf(
        mesh=mesh,
        kernel_template=physical_kernel,
    )
    leaf_h, c_rows = physical_leaf_entry(
        physical_kernel,
        physical_node,
        physical_global,
    )
    jax.block_until_ready((leaf_h, c_rows))
    del physical_node, physical_leaf_entry
    gc.collect()

    physical_reducer_entry = build_sequence_parallel_physical_reducer(
        mesh=mesh,
        kernel_template=physical_kernel,
        edge_template=physical_edge,
        replicate_threshold=min(512, n),
        contextualizer_tile_size=int(pair_tile_size),
    )
    tree = physical_reducer_entry(
        physical_kernel,
        physical_edge,
        leaf_h,
        c_rows,
        leaf_real,
        structural_mask,
        physical_global,
        perm,
    )
    jax.block_until_ready(tree)

    return LargeNCompiledEvalResult(
        wavefunction=CompiledWaveFunction(bind_shared_kernel(model), tree),
        perm=perm,
        logp=logp,
    )


__all__ = [
    "LargeNCompiledEvalResult",
    "compile_large_n_eval_wavefunction",
]
