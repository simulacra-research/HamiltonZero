# Copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import jax
import jax.numpy as jnp

from hamiltonzero.compiled.types import SharedTrunk
from hamiltonzero.model.route_pointer import TreePrefixPointerMHSEA, lca_gaussian_decay

from .types import RouterKernel, RouterStatic


def bind_router_kernel(model) -> RouterKernel:
    decoder = model.route_decoder
    if not isinstance(decoder, TreePrefixPointerMHSEA):
        raise ValueError("learned-router train requires TreePrefixPointerMHSEA")
    return RouterKernel(
        contextualizer=model.route_contextualizer,
        global_fork=model.gladder_fork_route,
        decoder=decoder,
    )


def compile_router_static(
    kernel: RouterKernel,
    trunk: SharedTrunk,
    quotient_node_key,
    quotient_edge_key,
    needs_fwl2,
) -> RouterStatic:
    decoder = kernel.decoder
    node_input, raw_edge, global_input = kernel.contextualizer.with_edge(
        trunk.node_raw,
        trunk.edge_raw,
        trunk.real_mask,
        trunk.balanced_mask,
        g=trunk.global_stream,
    )
    global_input = kernel.global_fork(global_input, raw_edge, trunk.balanced_mask)
    (node_raw, node_projected), _ = decoder._prepare_nodes(
        node_input, trunk.balanced_mask
    )
    global_raw, global_projected = decoder._project_global(
        global_input, node_input.dtype
    )
    ((_, _, initial_suffix, _, _), edge_messages) = (
        decoder._initial_summaries_and_edge_messages(
            raw_edge, trunk.balanced_mask, node_input.dtype
        )
    )
    prefix_edge_messages, suffix_edge_messages = edge_messages
    n = node_input.shape[0]
    idx = jnp.arange(n, dtype=jnp.int32)
    return RouterStatic(
        node_input=node_raw,
        node_projected=node_projected,
        global_input=global_raw,
        global_projected=global_projected,
        raw_edge=raw_edge,
        initial_suffix=initial_suffix,
        prefix_edge_messages=prefix_edge_messages,
        suffix_edge_messages=suffix_edge_messages,
        order_decay=lca_gaussian_decay(
            idx, idx, decoder.order_decay_w[0], decoder.order_decay_b[0]
        ),
        virtual_decay=lca_gaussian_decay(
            idx, idx, decoder.virt_decay_w[0], decoder.virt_decay_b[0]
        ),
        tree_pair_messages=decoder._tree_pair_messages(raw_edge, trunk.balanced_mask),
        static_bias_tables=decoder._pack_heavy_static_bias_tables(raw_edge),
        quadratic_base_static=(),
        forked_global_static=(),
        quotient_node_key=quotient_node_key,
        quotient_edge_key=quotient_edge_key,
        real_mask=trunk.real_mask,
        routable_mask=trunk.balanced_mask,
        needs_fwl2=jnp.asarray(needs_fwl2, dtype=jnp.bool_),
    )


__all__ = ["bind_router_kernel", "compile_router_static"]
