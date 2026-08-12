# Copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from functools import partial
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from .utils import compiler_params

_MIN_SEQUENCE_LENGTH = 64
_SUPPORTED_HEAD_DIMS = frozenset((8, 16, 32, 64))
_INTERNAL_DTYPE = jnp.dtype(jnp.float32)


def _mask_value() -> jax.Array:
    return jnp.float32(-10000.0)


def _mhsea_tangent_kernel(
    q_ref,
    k_ref,
    e_ref,
    v_ref,
    mask_ref,
    primal_out_ref,
    lse_ref,
    qt_ref,
    kt_ref,
    et_ref,
    vt_ref,
    tangent_ref,
    *,
    q_block_len: int,
    kv_block_len: int,
    precision,
):
    q_tile = pl.program_id(1)
    head_dim = q_ref.shape[-1]
    q_slice = pl.dslice(q_tile * q_block_len, q_block_len)
    q_mask = mask_ref[q_slice]
    q_zero = jnp.asarray(0, dtype=q_ref.dtype)
    qt_zero = jnp.asarray(0, dtype=qt_ref.dtype)
    q = jnp.where(q_mask[:, None], q_ref[:, :], q_zero)
    qt = jnp.where(q_mask[:, None], qt_ref[:, :], qt_zero)
    mean = jnp.zeros((q_block_len,), dtype=jnp.float32)
    score_d_v = jnp.zeros((q_block_len, head_dim), dtype=jnp.float32)
    p_vt = jnp.zeros((q_block_len, head_dim), dtype=jnp.float32)
    lse = lse_ref[:].astype(jnp.float32)

    def _visit_kv_tile(kv_tile, carry):
        mean, score_d_v, p_vt = carry
        kv_slice = pl.dslice(kv_tile * kv_block_len, kv_block_len)
        kv_mask = mask_ref[kv_slice]
        square_mask = q_mask[:, None] & kv_mask[None, :]
        k_zero = jnp.asarray(0, dtype=k_ref.dtype)
        v_zero = jnp.asarray(0, dtype=v_ref.dtype)
        kt_zero = jnp.asarray(0, dtype=kt_ref.dtype)
        vt_zero = jnp.asarray(0, dtype=vt_ref.dtype)
        k = jnp.where(kv_mask[:, None], k_ref[kv_slice, :], k_zero)
        v = jnp.where(kv_mask[:, None], v_ref[kv_slice, :], v_zero)
        kt = jnp.where(kv_mask[:, None], kt_ref[kv_slice, :], kt_zero)
        vt = jnp.where(kv_mask[:, None], vt_ref[kv_slice, :], vt_zero)
        e = e_ref[:, kv_slice]
        et = et_ref[:, kv_slice]
        score_raw = pl.dot(q, k, trans_b=True, precision=precision) + e
        score = jnp.where(square_mask, score_raw, _mask_value())
        p = jnp.exp(score.astype(jnp.float32) - lse[:, None])
        score_d_raw = (
            pl.dot(qt, k, trans_b=True, precision=precision)
            + pl.dot(q, kt, trans_b=True, precision=precision)
            + et
        )
        score_d = jnp.where(square_mask, score_d_raw, jnp.float32(0))
        p_score_d = p * score_d.astype(jnp.float32)
        mean += jnp.sum(p_score_d, axis=1, dtype=jnp.float32)
        score_d_v += pl.dot(
            p_score_d.astype(v_ref.dtype), v, precision=precision
        ).astype(jnp.float32)
        p_vt += pl.dot(p.astype(vt_ref.dtype), vt, precision=precision).astype(
            jnp.float32
        )
        return (mean, score_d_v, p_vt)

    mean, score_d_v, p_vt = jax.lax.fori_loop(
        0, k_ref.shape[0] // kv_block_len, _visit_kv_tile, (mean, score_d_v, p_vt)
    )
    primal_out = primal_out_ref[:, :].astype(jnp.float32)
    tangent = score_d_v + p_vt - mean[:, None] * primal_out
    tangent = jnp.where(q_mask[:, None], tangent, jnp.float32(0))
    tangent_ref[:, :] = tangent.astype(tangent_ref.dtype)


def _require_shape(name: str, value: jax.Array, expected: tuple[int, ...]) -> None:
    if value.shape != expected:
        raise ValueError(f"{name}.shape must be {expected}, got {value.shape}")


def _require_external_f32(name: str, value: jax.Array) -> None:
    if jnp.dtype(value.dtype) != jnp.dtype(jnp.float32):
        raise TypeError(f"{name}.dtype must be float32, got {value.dtype}")


def mhsea_tangent_pallas(
    q,
    k,
    e,
    v,
    mask,
    primal_out,
    lse,
    qt,
    kt,
    et,
    vt,
    *,
    q_block_len,
    kv_block_len,
    num_warps,
    num_stages,
    precision,
):
    if q.ndim != 4:
        raise ValueError(f"q must have rank 4 [B, N, H, D], got shape {q.shape}")
    batch_size, seq_len, num_heads, head_dim = q.shape
    qkv_shape = (batch_size, seq_len, num_heads, head_dim)
    edge_shape = (batch_size, seq_len, num_heads, seq_len)
    if seq_len < _MIN_SEQUENCE_LENGTH:
        raise ValueError(
            f"sequence length must be >= {_MIN_SEQUENCE_LENGTH}, got {seq_len}; q_block_len and kv_block_len must divide it exactly"
        )
    if head_dim not in _SUPPORTED_HEAD_DIMS:
        raise ValueError(
            f"head dimension must be one of {sorted(_SUPPORTED_HEAD_DIMS)}, got {head_dim}"
        )
    if q_block_len <= 0 or seq_len % q_block_len:
        raise ValueError(
            f"q_block_len must be a positive divisor of N={seq_len}, got {q_block_len}"
        )
    if kv_block_len <= 0 or seq_len % kv_block_len:
        raise ValueError(
            f"kv_block_len must be a positive divisor of N={seq_len}, got {kv_block_len}"
        )
    if num_warps <= 0:
        raise ValueError(f"num_warps must be positive, got {num_warps}")
    if num_stages <= 0:
        raise ValueError(f"num_stages must be positive, got {num_stages}")
    internal_dtype = _INTERNAL_DTYPE
    for name, value in (
        ("q", q),
        ("k", k),
        ("v", v),
        ("primal_out", primal_out),
        ("qt", qt),
        ("kt", kt),
        ("vt", vt),
    ):
        _require_shape(name, value, qkv_shape)
        _require_external_f32(name, value)
    for name, value in (("e", e), ("et", et)):
        _require_shape(name, value, edge_shape)
        _require_external_f32(name, value)
    _require_shape("mask", mask, (batch_size, seq_len))
    _require_shape("lse", lse, (batch_size, seq_len, num_heads))
    _require_external_f32("lse", lse)
    q_like_spec = pl.BlockSpec(
        (None, q_block_len, None, head_dim),
        lambda batch, q_tile, head: (batch, q_tile, head, 0),
    )
    kv_like_spec = pl.BlockSpec(
        (None, seq_len, None, head_dim),
        lambda batch, _q_tile, head: (batch, 0, head, 0),
    )
    edge_spec = pl.BlockSpec(
        (None, q_block_len, None, seq_len),
        lambda batch, q_tile, head: (batch, q_tile, head, 0),
    )
    mask_spec = pl.BlockSpec((None, seq_len), lambda batch, _q_tile, _head: (batch, 0))
    lse_spec = pl.BlockSpec(
        (None, q_block_len, None), lambda batch, q_tile, head: (batch, q_tile, head)
    )
    kernel = pl.pallas_call(
        partial(
            _mhsea_tangent_kernel,
            q_block_len=q_block_len,
            kv_block_len=kv_block_len,
            precision=precision,
        ),
        grid=(batch_size, seq_len // q_block_len, num_heads),
        in_specs=[
            q_like_spec,
            kv_like_spec,
            edge_spec,
            kv_like_spec,
            mask_spec,
            q_like_spec,
            lse_spec,
            q_like_spec,
            kv_like_spec,
            edge_spec,
            kv_like_spec,
        ],
        out_specs=q_like_spec,
        out_shape=jax.ShapeDtypeStruct(qkv_shape, jnp.float32),
        compiler_params=compiler_params(num_warps=num_warps, num_stages=num_stages),
        debug=False,
        interpret=False,
        name="mhsea_tuned_tangent",
    )
    return kernel(
        q.astype(internal_dtype),
        k.astype(internal_dtype),
        e.astype(internal_dtype),
        v.astype(internal_dtype),
        mask.astype(jnp.bool_),
        primal_out,
        lse,
        qt.astype(internal_dtype),
        kt.astype(internal_dtype),
        et.astype(internal_dtype),
        vt.astype(internal_dtype),
    )
