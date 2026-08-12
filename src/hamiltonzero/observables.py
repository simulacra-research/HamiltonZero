# Copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp

from hamiltonzero.energy.kernel import _right_su2_chart_jet


def _local_spin_single(
    wavefunction: Any,
    q_routed: jax.Array,
) -> jax.Array:
    n_sites = q_routed.shape[0]

    def f_entry(z):
        q_perturbed = _right_su2_chart_jet(q_routed, z)
        real, imaginary = wavefunction(q_perturbed, None, 0.0)
        return jnp.stack([real, imaginary])

    jac_pair = jax.jacrev(f_entry)(jnp.zeros((n_sites, 3), dtype=q_routed.dtype))
    g_lie = jac_pair[0] + 1j * jac_pair[1]
    return -0.5j * g_lie


def local_spin(
    wavefunction: Any,
    context: Any,
    q_routed: jax.Array,
    *,
    chunk_size: int | None = 512,
) -> jax.Array:
    q_routed = jnp.asarray(q_routed)
    if q_routed.ndim < 2 or q_routed.shape[-1] != 4:
        raise ValueError("q_routed must have shape [..., N, 4]")
    if context.mask.shape[-1] != q_routed.shape[-2]:
        raise ValueError("context mask and q_routed must have the same site width")
    lead = q_routed.shape[:-2]
    n_items = math.prod(lead) if lead else 1
    flat = q_routed.reshape((n_items,) + q_routed.shape[-2:])

    with jax.default_matmul_precision("highest"):
        if chunk_size is None or chunk_size >= n_items:
            values = jax.vmap(lambda q: _local_spin_single(wavefunction, q))(flat)
        else:
            if chunk_size < 1:
                raise ValueError("chunk_size must be positive or None")
            values = jax.lax.map(
                lambda q: _local_spin_single(wavefunction, q),
                flat,
                batch_size=int(chunk_size),
            )
    values = values.reshape(lead + q_routed.shape[-2:-1] + (3,))
    return values * jnp.asarray(context.mask, dtype=values.real.dtype)[..., None]


__all__ = [
    "local_spin",
]
