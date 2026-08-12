# Copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from .permutation import permute_multi_ctx_prefix, permute_q_prefix


def reframe_state_context(state, context, old_perms, new_perms):
    if old_perms.shape != new_perms.shape:
        raise ValueError("old and new route permutations must have equal shapes")
    old_inverse = jnp.argsort(old_perms, axis=-1)
    composed = jnp.take_along_axis(old_inverse, new_perms, axis=-1)
    q = permute_q_prefix(state.q, composed)
    grad = permute_q_prefix(state.grad_log_p, composed)
    mask = jnp.take_along_axis(state.mask, composed, axis=-1)
    state = eqx.tree_at(
        lambda value: (value.q, value.grad_log_p, value.mask),
        state,
        (q, grad, mask),
    )
    context = permute_multi_ctx_prefix(context, composed)
    context = eqx.tree_at(
        lambda value: value.route_perm,
        context,
        new_perms,
    )
    return state, context


def rebase_cold_samples(q_routed, perms):
    inverse = jnp.argsort(perms, axis=-1)
    index = jnp.broadcast_to(inverse[:, None, :, None], q_routed.shape)
    return jnp.take_along_axis(q_routed, index, axis=-2)


__all__ = ["rebase_cold_samples", "reframe_state_context"]
