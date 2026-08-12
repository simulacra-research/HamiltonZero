# Copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextvars

import jax
import jax.numpy as jnp
from jax.extend import core
from jax.interpreters import batching


_custom_lap_active_var: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "spin_custom_lap_active",
    default=False,
)


def custom_lap_active() -> bool:
    return _custom_lap_active_var.get()


def enter_custom_lap():
    return _custom_lap_active_var.set(True)


def restore_custom_lap(token):
    _custom_lap_active_var.reset(token)


quadrilinear_merge_p = core.Primitive("quadrilinear_merge_lap")
quadrilinear_merge_p.multiple_results = False


def _quadrilinear_merge_impl(T, u_a, u_b):
    groups, rank, _, _ = T.shape
    if u_a.ndim == 1:
        u_a_2d = u_a.reshape(groups, rank)
        u_b_2d = u_b.reshape(groups, rank)
        contracted = jnp.einsum("ijkl,ik->ijl", T, u_a_2d)
        out_2d = jnp.einsum("ijl,il->ij", contracted, u_b_2d)
        return out_2d.reshape(-1)
    batch = u_a.shape[0]
    u_a_3d = u_a.reshape(batch, groups, rank)
    u_b_3d = u_b.reshape(batch, groups, rank)
    out_3d = jnp.einsum("ijkl,Bik,Bil->Bij", T, u_a_3d, u_b_3d)
    return out_3d.reshape(batch, -1)


def _quadrilinear_merge_abstract_eval(T_aval, u_a_aval, u_b_aval):
    del u_b_aval
    return jax.core.ShapedArray(u_a_aval.shape, T_aval.dtype)


quadrilinear_merge_p.def_impl(_quadrilinear_merge_impl)
quadrilinear_merge_p.def_abstract_eval(_quadrilinear_merge_abstract_eval)


def _quadrilinear_merge_batched(args, dims):
    T, u_a, u_b = args
    T_axis, u_a_axis, u_b_axis = dims
    if T_axis is not None:
        raise ValueError("quadrilinear merge parameters cannot be batched")
    batch = None
    if u_a_axis is not None:
        u_a = jnp.moveaxis(u_a, u_a_axis, 0)
        batch = u_a.shape[0]
    if u_b_axis is not None:
        u_b = jnp.moveaxis(u_b, u_b_axis, 0)
        batch = u_b.shape[0] if batch is None else batch
    if batch is None:
        return quadrilinear_merge_p.bind(T, u_a, u_b), None
    if u_a_axis is None:
        u_a = jnp.broadcast_to(u_a[None], (batch,) + u_a.shape)
    if u_b_axis is None:
        u_b = jnp.broadcast_to(u_b[None], (batch,) + u_b.shape)
    u_a_flat = u_a.reshape((-1, u_a.shape[-1]))
    u_b_flat = u_b.reshape((-1, u_b.shape[-1]))
    out_flat = quadrilinear_merge_p.bind(T, u_a_flat, u_b_flat)
    out = out_flat.reshape(u_a.shape[:-1] + (out_flat.shape[-1],))
    return out, 0


batching.primitive_batchers[quadrilinear_merge_p] = _quadrilinear_merge_batched


__all__ = [
    "custom_lap_active",
    "quadrilinear_merge_p",
]
