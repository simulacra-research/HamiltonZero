# Copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools

import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

ROUTE_SAMPLES = 8
GLOBAL_BEAM_WIDTH = 16


def _decode_one(decoder, static, key, tau):
    return decoder._decode(
        static.node_input,
        static.raw_edge,
        static.routable_mask,
        tau=tau,
        key=key,
        real_mask=static.real_mask,
        first_orbit_ids=(
            static.quotient_node_key,
            static.quotient_edge_key,
            static.needs_fwl2,
        ),
        router_static=static,
    )


def _beam_local(decoder, static, tau, *, lanes):
    permutations, _log_probabilities = decoder.beam_search(
        static.node_input,
        static.raw_edge,
        static.routable_mask,
        global_feat=static.global_input,
        tau=tau,
        beam_width=GLOBAL_BEAM_WIDTH,
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
    return permutations[0]


def build_beam16(mesh: Mesh, decoder, static):
    lanes = int(mesh.shape["systems"])
    if tuple(mesh.axis_names) != ("systems",) or GLOBAL_BEAM_WIDTH % lanes:
        raise ValueError("beam16 requires a one-dimensional divisible systems mesh")
    mapped = jax.shard_map(
        functools.partial(_beam_local, lanes=lanes),
        mesh=mesh,
        in_specs=(
            jax.tree_util.tree_map(lambda _: P(), decoder),
            jax.tree_util.tree_map(lambda _: P(), static),
            P(),
        ),
        out_specs=P(),
        check_vma=False,
    )
    replicated = NamedSharding(mesh, P())
    return jax.jit(
        mapped,
        in_shardings=(
            jax.tree_util.tree_map(lambda _: replicated, decoder),
            jax.tree_util.tree_map(lambda _: replicated, static),
            replicated,
        ),
        out_shardings=replicated,
    )


def build_route_sampler(mesh: Mesh, decoder, static):
    if tuple(mesh.axis_names) != ("systems",) or mesh.shape["systems"] != ROUTE_SAMPLES:
        raise ValueError("learned-router train requires an eight-lane systems mesh")
    replicated = NamedSharding(mesh, P())
    route_vector = NamedSharding(mesh, P("systems", None))
    local_specs = (
        jax.tree_util.tree_map(lambda _: P(), decoder),
        jax.tree_util.tree_map(lambda _: P(), static),
        P(),
        P(),
    )

    def local(decoder_value, static_value, key, tau):
        lane_key = jax.random.fold_in(key, jax.lax.axis_index("systems"))
        sample_key = jax.random.split(lane_key, 1)[0]
        permutation = _decode_one(
            decoder_value,
            static_value,
            sample_key,
            tau,
        )
        return permutation[None]

    mapped = jax.shard_map(
        local,
        mesh=mesh,
        in_specs=local_specs,
        out_specs=P("systems", None),
        check_vma=False,
    )
    return jax.jit(
        mapped,
        in_shardings=(
            jax.tree_util.tree_map(lambda _: replicated, decoder),
            jax.tree_util.tree_map(lambda _: replicated, static),
            replicated,
            replicated,
        ),
        out_shardings=route_vector,
    )


__all__ = [
    "GLOBAL_BEAM_WIDTH",
    "ROUTE_SAMPLES",
    "build_beam16",
    "build_route_sampler",
]
