# Copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from hamiltonzero.compiled.types import EnergyFrame, EnergyInputs, EnergyMasks
from hamiltonzero.energy.custom_lap import build_W_levels


def _real_dtype(dtype):
    return jnp.real(jnp.zeros((), dtype)).dtype


def _host_eigvalsh(x):
    out_dtype = _real_dtype(x.dtype)
    out_shape = jax.ShapeDtypeStruct(x.shape[:-1], out_dtype)

    def callback(a):
        values = np.linalg.eigvalsh(np.asarray(a))
        return values.astype(np.dtype(out_dtype))

    return jax.pure_callback(
        callback,
        out_shape,
        x,
        vmap_method="sequential",
    )


def _compute_custom_lap_views_batched(J_full, mask, mu, eps):
    n = J_full.shape[1]
    n_systems = J_full.shape[0]
    dtype = J_full.dtype
    J_matrix = jnp.transpose(J_full, (0, 1, 3, 2, 4)).reshape(n_systems, 3 * n, 3 * n)
    J_matrix_symmetric = 0.5 * (J_matrix + jnp.swapaxes(J_matrix, -1, -2))
    mask_3 = jnp.repeat(mask.astype(dtype), 3, axis=-1)
    eye_3n = jnp.eye(3 * n, dtype=dtype)
    masked_identity = eye_3n[None] * mask_3[:, None, :]
    J_eff = (J_matrix_symmetric - (mu + eps)[:, None, None] * masked_identity) / 4.0
    J_eff = 0.5 * (J_eff + jnp.swapaxes(J_eff, -1, -2))

    eigh_epsilon = jnp.asarray(1e-6, dtype=_real_dtype(dtype))
    eigenvalues = _host_eigvalsh(J_eff + eigh_epsilon * eye_3n[None]) - eigh_epsilon
    lam_max = jnp.max(eigenvalues, axis=-1)
    delta_mu = jax.nn.relu(4.0 * (lam_max + eps))
    shift = delta_mu / 4.0
    J_eff = J_eff - shift[:, None, None] * eye_3n[None]
    mu_eff = mu + delta_mu
    casimir_per_site = (mu_eff + eps) * jnp.asarray(0.75, dtype=dtype)
    radial_const = -casimir_per_site * mask.astype(dtype).sum(axis=-1)
    return J_eff, radial_const


def _compute_custom_lap_views(J_full, mask, mu, eps):
    values = _compute_custom_lap_views_batched(
        J_full[None],
        mask[None],
        jnp.asarray(mu)[None],
        jnp.asarray(eps)[None],
    )
    return tuple(value[0] for value in values)


def build_energy_inputs(J_full, h, mask, mu, eps) -> EnergyInputs:
    dtype = J_full.dtype
    mu_array = jnp.asarray(mu, dtype=dtype)
    eps_array = jnp.asarray(eps, dtype=dtype)
    J_eff, radial_const = _compute_custom_lap_views(
        J_full,
        mask,
        mu_array,
        eps_array,
    )
    return EnergyInputs(
        custom_lap_J_eff=J_eff,
        custom_lap_radial_const=radial_const,
        one_body_fields=(h,),
    )


def block_permutation3(perm: Array) -> Array:

    components = jnp.arange(3, dtype=perm.dtype)
    return (perm[:, None] * 3 + components[None, :]).reshape(-1)


def route_one_body_field(field: Array, perm: Array) -> Array:

    return jnp.take(field, perm, axis=0)


def route_one_body_fields(
    fields: Sequence[Array],
    perm: Array,
) -> tuple[Array, ...]:

    return tuple(route_one_body_field(field, perm) for field in fields)


def route_J_eff(J_eff: Array, perm: Array) -> Array:

    block_perm = block_permutation3(perm)
    return jnp.take(jnp.take(J_eff, block_perm, axis=0), block_perm, axis=1)


def route_energy_inputs(energy_inputs: EnergyInputs, perm: Array) -> EnergyInputs:
    return EnergyInputs(
        custom_lap_J_eff=route_J_eff(energy_inputs.custom_lap_J_eff, perm),
        custom_lap_radial_const=energy_inputs.custom_lap_radial_const,
        one_body_fields=route_one_body_fields(energy_inputs.one_body_fields, perm),
    )


def compile_energy_frame(
    energy_inputs: EnergyInputs,
    real_mask: Array,
    balanced_mask: Array,
    perm: Array,
) -> EnergyFrame:

    J_eff = route_J_eff(energy_inputs.custom_lap_J_eff, perm)
    n_sites = int(perm.shape[0])
    frame = EnergyFrame(
        custom_lap_J_eff=J_eff,
        w_levels=tuple(build_W_levels(J_eff, n_sites)),
        custom_lap_radial_const=energy_inputs.custom_lap_radial_const,
        one_body_fields=route_one_body_fields(energy_inputs.one_body_fields, perm),
        masks=EnergyMasks(
            real=route_one_body_field(real_mask, perm),
            balanced=balanced_mask,
        ),
    )
    return frame


__all__ = [
    "block_permutation3",
    "build_energy_inputs",
    "compile_energy_frame",
    "route_energy_inputs",
    "route_J_eff",
    "route_one_body_field",
    "route_one_body_fields",
]
