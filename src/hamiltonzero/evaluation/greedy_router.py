# Copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from collections.abc import Callable
from functools import partial

import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from hamiltonzero.model.route_pointer import TreePrefixPointerMHSEA


def _ring_permute_rows_local(values, route_ids, *, axis_size: int):

    local_rows = values.shape[0]
    lane = jax.lax.axis_index("seq").astype(jax.numpy.int32)
    local_ids = jax.lax.dynamic_slice_in_dim(
        route_ids,
        lane * local_rows,
        local_rows,
        axis=0,
    )
    owner = local_ids // local_rows
    within_owner = local_ids % local_rows
    output0 = jax.numpy.zeros_like(values)

    def take_from(panel, origin, output):
        selected = panel[within_owner]
        mask = owner == origin
        while mask.ndim < selected.ndim:
            mask = mask[..., None]
        return jax.numpy.where(mask, selected, output)

    origin0 = lane
    output0 = take_from(values, origin0, output0)
    permutation = [(i, (i + 1) % axis_size) for i in range(axis_size)]

    def step(carry, _):
        panel, origin, output = carry
        panel = jax.lax.ppermute(panel, "seq", permutation)
        origin = (origin - jax.numpy.asarray(1, jax.numpy.int32)) % axis_size
        output = take_from(panel, origin, output)
        return (panel, origin, output), None

    (_, _, output), _ = jax.lax.scan(
        step,
        (values, origin0, output0),
        xs=None,
        length=axis_size - 1,
    )
    return output


def _make_row_permute(mesh: Mesh) -> Callable:
    axis_size = int(mesh.shape["seq"])
    spec = P("seq", None, None)
    if axis_size == 1:
        return lambda values, route_ids: values[route_ids]
    return jax.shard_map(
        partial(_ring_permute_rows_local, axis_size=axis_size),
        mesh=mesh,
        in_specs=(spec, P()),
        out_specs=spec,
        check_vma=False,
    )


def build_compact_greedy_router(
    *,
    mesh: Mesh,
    decoder_template: TreePrefixPointerMHSEA,
    pair_tile_size: int = 128,
) -> Callable:

    if tuple(mesh.axis_names) != ("seq",):
        raise ValueError("compact greedy router requires a one-dimensional 'seq' mesh")
    if int(mesh.shape["seq"]) < 1:
        raise ValueError("compact greedy router requires at least one seq lane")
    if not isinstance(decoder_template, TreePrefixPointerMHSEA):
        raise TypeError("decoder_template must be TreePrefixPointerMHSEA")
    if isinstance(pair_tile_size, bool) or int(pair_tile_size) < 1:
        raise ValueError("pair_tile_size must be a positive integer")

    rep = NamedSharding(mesh, P())
    seq_vec = NamedSharding(mesh, P("seq", None))
    seq_edge = NamedSharding(mesh, P("seq", None, None))
    decoder_rep = jax.tree_util.tree_map(lambda _leaf: rep, decoder_template)
    row_permute = _make_row_permute(mesh)

    def decode(decoder, h, edge, mask, global_feat, tau, real_mask):

        h = jax.lax.with_sharding_constraint(h, seq_vec)
        edge = jax.lax.with_sharding_constraint(edge, seq_edge)
        perm, logp = decoder._decode_greedy_compact(
            h,
            edge,
            mask,
            global_feat=global_feat,
            tau=tau,
            real_mask=real_mask,
            sequence_mesh=mesh,
            pair_tile_size=int(pair_tile_size),
            row_permute_fn=row_permute,
        )
        return perm, logp

    return jax.jit(
        decode,
        in_shardings=(
            decoder_rep,
            seq_vec,
            seq_edge,
            rep,
            rep,
            rep,
            rep,
        ),
        out_shardings=(rep, rep),
    )


__all__ = ["build_compact_greedy_router"]
