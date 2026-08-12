# Copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from hamiltonzero.model import normalize_leaf_carriers

from .types import CARRY_LEFT, CARRY_RIGHT, EMPTY, MERGE


def _single(values: tuple[Any, ...], name: str) -> Any:
    if len(values) != 1:
        raise ValueError(
            f"compiled HT executor requires exactly one {name}; got {len(values)}"
        )
    return values[0]


def _factorized_apply(factor: Any, h: jax.Array, x: jax.Array) -> jax.Array:
    odd_dtype = jnp.float32
    x_compute = x if x.dtype == odd_dtype else x.astype(odd_dtype)
    V = factor.V if factor.V.dtype == odd_dtype else factor.V.astype(odd_dtype)
    U = factor.U if factor.U.dtype == odd_dtype else factor.U.astype(odd_dtype)
    mixed = (x_compute @ V) * h
    mixed = mixed if mixed.dtype == odd_dtype else mixed.astype(odd_dtype)
    return mixed @ U


def _compiled_quadrilinear_merge(
    T: jax.Array,
    u_a: jax.Array,
    u_b: jax.Array,
) -> jax.Array:
    from hamiltonzero.energy import custom_lap_active, quadrilinear_merge_p

    odd_dtype = jnp.float32
    T = T if T.dtype == odd_dtype else T.astype(odd_dtype)
    u_a = u_a if u_a.dtype == odd_dtype else u_a.astype(odd_dtype)
    u_b = u_b if u_b.dtype == odd_dtype else u_b.astype(odd_dtype)
    if custom_lap_active():
        return quadrilinear_merge_p.bind(T, u_a, u_b)
    G, d_r, _, _ = T.shape
    leading = u_a.shape[:-1]
    u_a_flat = u_a.reshape((-1, G, d_r))
    u_b_flat = u_b.reshape((-1, G, d_r))
    out_flat = jnp.einsum("ijkl,Bik,Bil->Bij", T, u_a_flat, u_b_flat)
    return out_flat.reshape((*leading, G * d_r))


def _opcode_gates(opcodes: jax.Array, dtype: jnp.dtype) -> tuple[jax.Array, ...]:
    both = (opcodes == MERGE).astype(dtype)
    left = (opcodes == CARRY_LEFT).astype(dtype)
    right = (opcodes == CARRY_RIGHT).astype(dtype)
    return both, left, right


def _gate_reference(
    candidate: jax.Array,
    left_value: jax.Array,
    right_value: jax.Array,
    opcodes: jax.Array,
    *,
    feature_axis: bool,
) -> jax.Array:
    both, left, right = _opcode_gates(opcodes, candidate.dtype)
    if feature_axis:
        both, left, right = both[..., None], left[..., None], right[..., None]
    pad = candidate.ndim - both.ndim
    shape = (1,) * pad + both.shape
    both, left, right = both.reshape(shape), left.reshape(shape), right.reshape(shape)
    return both * candidate + left * left_value + right * right_value


def execute_wavefunction(kernel: Any, tree: Any, q_routed: jax.Array):
    if len(tree.leaf_combiner_h) != 0 or len(tree.readout_combiner_h) != 0:
        raise ValueError("single-head compiled HT executor does not accept combiners")
    if len(tree.merge_h) != len(tree.opcodes):
        raise ValueError("merge_h and opcodes must have one entry per tree level")
    q_weight = kernel.q_to_odd.weight
    leaf_factor = _single(kernel.leaf_factors, "leaf factor")
    merge_factor = _single(kernel.merge_factors, "merge factor")
    readout_factor = _single(kernel.readout_factors, "readout factor")
    leaf_h = _single(tree.leaf_h, "leaf conditioner")
    readout_h = _single(tree.readout_h, "readout conditioner")
    odd_dtype = jnp.float32
    q_compute = q_routed if q_routed.dtype == odd_dtype else q_routed.astype(odd_dtype)
    q_weight = q_weight if q_weight.dtype == odd_dtype else q_weight.astype(odd_dtype)
    z = q_compute @ q_weight
    u_raw = _factorized_apply(leaf_factor, leaf_h, z)
    u, log_rms = normalize_leaf_carriers(u_raw)
    s = jnp.zeros(u.shape[:-1], dtype=odd_dtype)
    s = s + log_rms.astype(s.dtype)
    for h_level, opcodes in zip(tree.merge_h, tree.opcodes, strict=True):
        if u.shape[-2] != 2 * h_level.shape[-2]:
            raise ValueError("compiled merge level has incompatible shrinking shape")
        u_left, u_right = u[..., 0::2, :], u[..., 1::2, :]
        s_left, s_right = s[..., 0::2], s[..., 1::2]
        raw = _compiled_quadrilinear_merge(kernel.merge_T, u_left, u_right)
        out = raw + _factorized_apply(merge_factor, h_level, raw)
        scale = jnp.sqrt(jnp.mean(out * out, axis=-1) + kernel.merge_eps)
        candidate_u = out / scale[..., None]
        candidate_s = s_left + s_right + jnp.log(scale)
        u = _gate_reference(candidate_u, u_left, u_right, opcodes, feature_axis=True)
        s = _gate_reference(candidate_s, s_left, s_right, opcodes, feature_axis=False)
    if u.shape[-2] != 1:
        raise ValueError("compiled tree did not reduce to one root")
    u_root = u[..., 0, :]
    s_root = s[..., 0]
    psi = _factorized_apply(readout_factor, readout_h, u_root)
    psi_re, psi_im = psi[..., 0], psi[..., 1]
    log_abs = 0.5 * jnp.log(psi_re * psi_re + psi_im * psi_im) + s_root
    phase = jnp.arctan2(psi_im, psi_re)
    return log_abs, phase
