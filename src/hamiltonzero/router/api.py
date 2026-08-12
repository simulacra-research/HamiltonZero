# Copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from hamiltonzero.model.context import MultiSystemContext, SpinContext
from .permutation import permute_ctx_prefix, permute_multi_ctx_prefix, permute_q_prefix


def batch_context(context: SpinContext) -> MultiSystemContext:
    return MultiSystemContext.from_single(context)


def select_frozen_route(
    model,
    context: SpinContext,
    *,
    tau: float,
):
    node, edge, global_features = model.route_features(context)
    permutations, _log_probabilities = model.route_decoder.beam_search(
        node,
        edge,
        context.bmask,
        global_feat=global_features,
        tau=tau,
        beam_width=8,
        real_mask=context.mask,
        first_orbit_ids=(
            context.route_quotient_node_key,
            context.route_quotient_edge_key,
        ),
    )
    return permutations[0].astype(jnp.int32)


def route_context(context, perm):
    if isinstance(context, MultiSystemContext):
        perms = perm if perm.ndim == 2 else perm[None, :]
        routed = permute_multi_ctx_prefix(context, perms)
        return eqx.tree_at(lambda c: c.route_perm, routed, perms)
    routed = permute_ctx_prefix(context, perm)
    return eqx.tree_at(lambda c: c.route_perm, routed, perm)


def route_state(state, perm):
    if state.q.ndim == 5:
        perms = perm if perm.ndim == 2 else perm[None, :]
        q = permute_q_prefix(state.q, perms)
        grad = permute_q_prefix(state.grad_log_p, perms)
        mask = jnp.take_along_axis(state.mask, perms, axis=-1)
    else:
        p = perm if perm.ndim == 1 else perm[0]
        q = permute_q_prefix(state.q[None, ...], p[None, :])[0]
        grad = permute_q_prefix(state.grad_log_p[None, ...], p[None, :])[0]
        mask = jnp.take_along_axis(state.mask, p, axis=-1)
    return eqx.tree_at(
        lambda s: (s.q, s.grad_log_p, s.mask),
        state,
        (q, grad, mask),
    )


def strip_router(model):
    return eqx.tree_at(
        lambda m: (m.route_decoder, m.route_contextualizer),
        model,
        (None, None),
    )


__all__ = [
    "batch_context",
    "route_context",
    "route_state",
    "select_frozen_route",
    "strip_router",
]
