# Copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from hamiltonzero.model import tree_sphere

from .types import (
    CanonicalHamiltonian,
    FactorizedQSide,
    QSideLeafInput,
    SharedKernel,
    SharedTrunk,
    TrunkCompilerKernel,
)


def _qside(hypernet) -> FactorizedQSide:
    return FactorizedQSide(V=hypernet.V, U=hypernet.U)


def bind_shared_kernel(model) -> SharedKernel:
    return SharedKernel(
        q_to_odd=QSideLeafInput(weight=model.q_to_odd.weight),
        leaf_factors=(_qside(model.leaf.P_u),),
        leaf_combiner_factors=(),
        merge_T=model.merge.T,
        merge_factors=(_qside(model.merge.output_hypernet),),
        readout_factors=(_qside(model.readout.output_hypernet),),
        merge_eps=float(model.merge.eps),
    )


def bind_trunk_compiler_kernel(model) -> TrunkCompilerKernel:
    return TrunkCompilerKernel(
        featurizer=model.featurizer,
        trunk=model.trunk,
        shared_global=model.gladder_post,
    )


class _TrunkMaskContext(eqx.Module):
    mask: jax.Array


def compile_canonical_shared_trunk(
    kernel: TrunkCompilerKernel,
    canonical: CanonicalHamiltonian,
) -> SharedTrunk:
    if canonical.node_mask.ndim != 2 or canonical.node_mask.shape[0] != 1:
        raise ValueError("compiled shared trunk requires exact physical P=1")
    if canonical.balanced_mask.shape != canonical.node_mask.shape:
        raise ValueError("balanced_mask must match node_mask shape")
    graph = canonical.graph_inputs
    if graph.node.shape[:2] != canonical.node_mask.shape:
        raise ValueError("canonical graph node width must match node_mask")
    if graph.edge.shape[:3] != (
        1,
        canonical.node_mask.shape[1],
        canonical.node_mask.shape[1],
    ):
        raise ValueError("canonical graph edge width must match node_mask")

    def one(edge_input, node_input, real_mask, balanced_mask):
        edge_feat, local_feat, global_feat = kernel.featurizer(
            edge_input,
            real_mask,
            node_input,
        )
        g_seed = tree_sphere(global_feat.astype(local_feat.dtype))
        node_raw, edge_raw, g_seed = kernel.trunk(
            _TrunkMaskContext(real_mask),
            edge_feat,
            local_feat,
            g_seed,
        )
        global_stream = kernel.shared_global(
            g_seed.astype(edge_raw.dtype), edge_raw, real_mask
        )
        return SharedTrunk(
            node_raw=node_raw,
            edge_raw=edge_raw,
            global_raw=global_feat,
            global_stream=global_stream,
            real_mask=real_mask,
            balanced_mask=balanced_mask,
        )

    return jax.vmap(one)(
        graph.edge,
        graph.node,
        canonical.node_mask,
        canonical.balanced_mask,
    )


def select_single_physical_trunk(trunk: SharedTrunk) -> SharedTrunk:
    leaves = jax.tree_util.tree_leaves(trunk)
    if not leaves or any(x.ndim < 1 or x.shape[0] != 1 for x in leaves):
        raise ValueError("production SharedTrunk must have exact leading P=1")
    return jax.tree_util.tree_map(lambda x: x[0], trunk)


def compile_shared_trunk(model, ctx) -> SharedTrunk:
    edge_feat, local_feat, global_feat = model.featurizer(
        ctx.J_double_prime,
        ctx.mask,
        ctx.h_prime,
    )
    g_seed = tree_sphere(global_feat.astype(local_feat.dtype))
    node_raw, edge_raw, g_seed = model.trunk(
        ctx,
        edge_feat,
        local_feat,
        g_seed,
    )
    global_stream = model._gladder_g_stream(edge_raw, ctx.mask, g_seed)
    return SharedTrunk(
        node_raw=node_raw,
        edge_raw=edge_raw,
        global_raw=global_feat,
        global_stream=global_stream,
        real_mask=ctx.mask,
        balanced_mask=ctx.bmask,
    )
