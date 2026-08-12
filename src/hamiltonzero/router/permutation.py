# Copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from hamiltonzero.model.context import MultiSystemContext, SpinContext


def permute_ctx_prefix(ctx: SpinContext, perm: Int[Array, "n"]) -> SpinContext:
    return eqx.tree_at(
        lambda c: (
            c.h_prime,
            c.J_double_prime,
            c.mask,
            c.route_quotient_node_key,
            c.route_quotient_edge_key,
        ),
        ctx,
        (
            ctx.h_prime[perm],
            ctx.J_double_prime[perm][:, perm, :],
            ctx.mask[perm],
            ctx.route_quotient_node_key[perm],
            (
                ctx.route_quotient_edge_key
                if ctx.route_quotient_edge_key.shape[-1] == 0
                else ctx.route_quotient_edge_key[perm][:, perm]
            ),
        ),
    )


def permute_multi_ctx_prefix(
    ctx: MultiSystemContext,
    perms: Int[Array, "s n"],
) -> MultiSystemContext:
    return eqx.tree_at(
        lambda c: (
            c.h_prime,
            c.J_double_prime,
            c.mask,
            c.route_quotient_node_key,
            c.route_quotient_edge_key,
        ),
        ctx,
        (
            jax.vmap(lambda x, p: x[p])(ctx.h_prime, perms),
            jax.vmap(lambda x, p: x[p][:, p, :])(ctx.J_double_prime, perms),
            jax.vmap(lambda x, p: x[p])(ctx.mask, perms),
            jax.vmap(lambda x, p: x[p])(ctx.route_quotient_node_key, perms),
            (
                ctx.route_quotient_edge_key
                if ctx.route_quotient_edge_key.shape[-1] == 0
                else jax.vmap(lambda x, p: x[p][:, p])(
                    ctx.route_quotient_edge_key, perms
                )
            ),
        ),
    )


def permute_q_prefix(q: Float[Array, "s b r n d"], perms: Int[Array, "s n"]):
    idx = jnp.broadcast_to(perms[:, None, None, :, None], q.shape)
    return jnp.take_along_axis(q, idx, axis=3)


__all__ = [
    "permute_ctx_prefix",
    "permute_multi_ctx_prefix",
    "permute_q_prefix",
]
