# Copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from hamiltonzero.compiled.execute import execute_wavefunction
from hamiltonzero.energy.custom_lap import (
    custom_forward_laplacian_with_jac,
    use_custom_lap,
)


def _right_su2_chart_jet(
    q: Float[Array, "N 4"],
    z: Float[Array, "N 3"],
) -> Float[Array, "N 4"]:

    q0, q1, q2, q3 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    zx, zy, zz = z[:, 0], z[:, 1], z[:, 2]
    u0 = 1.0 - 0.5 * (zx * zx + zy * zy + zz * zz)
    u1, u2, u3 = zz, zy, zx
    return jnp.stack(
        [
            q0 * u0 - q1 * u1 - q2 * u2 - q3 * u3,
            q0 * u1 + q1 * u0 + q2 * u3 - q3 * u2,
            q0 * u2 - q1 * u3 + q2 * u0 + q3 * u1,
            q0 * u3 + q1 * u2 - q2 * u1 + q3 * u0,
        ],
        axis=-1,
    )


def _custom_lap_single_finetune(
    model: Callable,
    q: Float[Array, "N 4"],
    t,
    energy_frame: Any,
):
    if len(energy_frame.one_body_fields) != 1:
        raise ValueError("custom-Laplacian energy requires exactly one one-body field")
    return _custom_lap_single_from_frame(
        model,
        None,
        q,
        t,
        J_eff=energy_frame.custom_lap_J_eff,
        W_levels=energy_frame.w_levels,
        field_xyz=energy_frame.one_body_fields[0],
        radial_const=energy_frame.custom_lap_radial_const,
    )


def _custom_lap_single_prebuilt(
    model: Callable,
    q: Float[Array, "N 4"],
    t,
    frame: Any,
):
    if len(frame.one_body_fields) != 1:
        raise ValueError("custom-Laplacian energy requires exactly one one-body field")
    return _custom_lap_single_from_frame(
        model,
        None,
        q,
        t,
        J_eff=frame.custom_lap_J_eff,
        W_levels=frame.w_levels,
        field_xyz=frame.one_body_fields[0],
        radial_const=frame.custom_lap_radial_const,
    )


def _custom_lap_single_from_frame(
    model: Callable,
    ctx: Any,
    q: Float[Array, "N 4"],
    t,
    *,
    J_eff,
    W_levels,
    field_xyz,
    radial_const,
):
    N = q.shape[0]

    def f_entry(z):
        q_pert = _right_su2_chart_jet(q, z)
        re, im = model(q_pert, ctx, t)
        return jnp.stack([re, im])

    with use_custom_lap():
        _value, jac_pair, lap_pair = custom_forward_laplacian_with_jac(
            f_entry,
            W_levels,
            N,
        )(jnp.zeros((N, 3), dtype=q.dtype))

    tr_total = lap_pair[0] + 1j * lap_pair[1]

    g_lie = jac_pair[:, 0] + 1j * jac_pair[:, 1]
    quad_total = jnp.einsum("a,ab,b->", g_lie, J_eff.astype(g_lie.dtype), g_lie)

    la_xyz = 0.5 * g_lie.reshape(N, 3)
    field = (1j * jnp.einsum("ic,ic->", field_xyz.astype(g_lie.dtype), la_xyz)).astype(
        g_lie.dtype
    )

    total = tr_total + quad_total + radial_const.astype(g_lie.dtype) + field

    zero = jnp.zeros_like(total)
    exchange = total - field
    return total, exchange, zero, field


def _vmc_energy_custom_lap_finetune(
    model: Callable,
    energy_frame: Any,
    q: Float[Array, "... N 4"],
    t=0.0,
    *,
    chunk_size: int | None = None,
) -> tuple[
    Complex[Array, "..."],
    Complex[Array, "..."],
    Complex[Array, "..."],
    Complex[Array, "..."],
]:

    def single(qq, tt):
        return _custom_lap_single_finetune(model, qq, tt, energy_frame)

    return _run_custom_lap_batch(single, q, t, chunk_size)


def _vmc_energy_custom_lap_prebuilt(
    kernel: Any,
    tree: Any,
    energy_frame: Any,
    q: Float[Array, "... N 4"],
    *,
    chunk_size: int | None = None,
) -> tuple[
    Complex[Array, "..."],
    Complex[Array, "..."],
    Complex[Array, "..."],
    Complex[Array, "..."],
]:
    def model(q_pert, _ctx, _t):
        return execute_wavefunction(kernel, tree, q_pert)

    def single(qq, tt):
        return _custom_lap_single_prebuilt(model, qq, tt, energy_frame)

    return _run_custom_lap_batch(single, q, 0.0, chunk_size)


def _run_custom_lap_batch(single, q, t, chunk_size):
    n_sites, n_dims = q.shape[-2], q.shape[-1]
    assert n_dims == 4, f"expected quaternion last dim 4, got {n_dims}"
    lead = q.shape[:-2]
    n_items = 1
    for d in lead:
        n_items *= d

    q_flat = q.reshape(n_items, n_sites, 4)
    t_arr = jnp.asarray(t, dtype=q.dtype)
    t_bcast = jnp.broadcast_to(t_arr, lead if lead else ())
    t_flat = t_bcast.reshape(n_items) if lead else jnp.broadcast_to(t_arr, (n_items,))

    with jax.default_matmul_precision("highest"):
        if chunk_size is None or chunk_size >= n_items:
            total, exchange, casimir, field = jax.vmap(single)(q_flat, t_flat)
        else:
            total, exchange, casimir, field = jax.lax.map(
                lambda x: single(x[0], x[1]),
                (q_flat, t_flat),
                batch_size=chunk_size,
            )

    out_shape = lead if lead else ()
    return (
        total.reshape(out_shape),
        exchange.reshape(out_shape),
        casimir.reshape(out_shape),
        field.reshape(out_shape),
    )


__all__ = []
