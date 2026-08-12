# Copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import jax
import jax.core as _shaped_core
import jax.numpy as jnp
from jax.extend import core as _ext_core
from jax.interpreters import ad, batching, mlir

from .custom_gradients import mhsea_backward, mhsea_forward
from .mhsea_full_ad_jvp import mhsea_jvp_pallas


_TF32_PRECISION = jax.lax.Precision.DEFAULT


def _validate_primals(q, k, e, v, mask) -> None:
    assert q.ndim == k.ndim == v.ndim == 4
    assert e.ndim == 4
    assert mask.ndim == 2
    assert q.dtype == k.dtype == e.dtype == v.dtype
    assert jnp.dtype(q.dtype) == jnp.dtype(jnp.float32)
    batch_size, seq_len, num_heads, _head_dim = q.shape
    assert k.shape == q.shape
    assert v.shape == q.shape
    assert e.shape == (batch_size, seq_len, num_heads, seq_len)
    assert mask.shape == (batch_size, seq_len)


mhsea_full_ad_p = _ext_core.Primitive("mhsea_full_ad")
mhsea_full_ad_p.multiple_results = True


def _outer_impl(q, k, e, v, mask):
    _validate_primals(q, k, e, v, mask)
    seq_len = q.shape[1]
    out, residuals = mhsea_forward(
        q,
        k,
        e,
        v,
        mask,
        q_block_len=seq_len,
        num_warps=8,
        num_stages=3,
        precision=_TF32_PRECISION,
    )
    return out, residuals[5]


def _outer_abstract_eval(q, k, e, v, mask):
    _validate_primals(q, k, e, v, mask)
    batch_size, seq_len, num_heads, _head_dim = q.shape
    return (
        _shaped_core.ShapedArray(v.shape, v.dtype),
        _shaped_core.ShapedArray((batch_size, seq_len, num_heads), q.dtype),
    )


def _outer_jvp(primals, tangents):
    q, k, e, v, mask = primals
    qt, kt, et, vt, _mask_t = tangents
    qt = ad.instantiate_zeros(qt)
    kt = ad.instantiate_zeros(kt)
    et = ad.instantiate_zeros(et)
    vt = ad.instantiate_zeros(vt)
    primal_out, lse = mhsea_full_ad_p.bind(q, k, e, v, mask)
    tangent_out = mhsea_full_ad_lin_p.bind(
        q, k, e, v, mask, primal_out, lse, qt, kt, et, vt
    )
    return (primal_out, lse), (
        tangent_out,
        ad.Zero(_shaped_core.ShapedArray(lse.shape, lse.dtype)),
    )


def _merge_first_two(value):
    return value.reshape(value.shape[0] * value.shape[1], *value.shape[2:])


def _split_first_two(value, mapped_size: int):
    return value.reshape(mapped_size, value.shape[0] // mapped_size, *value.shape[1:])


def _prepare_batched_args(args, axes):
    mapped_size = None
    moved = []
    for value, axis in zip(args, axes):
        if axis is None:
            moved.append(value)
            continue
        value = jnp.moveaxis(value, axis, 0)
        if mapped_size is None:
            mapped_size = value.shape[0]
        else:
            assert value.shape[0] == mapped_size
        moved.append(value)
    if mapped_size is None:
        return None, None
    broadcasted = [
        jnp.broadcast_to(value[None], (mapped_size, *value.shape))
        if axis is None
        else value
        for value, axis in zip(moved, axes)
    ]
    return mapped_size, [_merge_first_two(value) for value in broadcasted]


def _outer_batch(args, axes):
    mapped_size, flat = _prepare_batched_args(args, axes)
    if mapped_size is None:
        out, lse = mhsea_full_ad_p.bind(*args)
        return (out, lse), (None, None)
    out_flat, lse_flat = mhsea_full_ad_p.bind(*flat)
    return (
        _split_first_two(out_flat, mapped_size),
        _split_first_two(lse_flat, mapped_size),
    ), (0, 0)


def _outer_mlir(ctx, *args, **kwargs):
    return mlir.lower_fun(_outer_impl, multiple_results=True)(ctx, *args, **kwargs)


mhsea_full_ad_p.def_impl(_outer_impl)
mhsea_full_ad_p.def_abstract_eval(_outer_abstract_eval)
ad.primitive_jvps[mhsea_full_ad_p] = _outer_jvp
batching.primitive_batchers[mhsea_full_ad_p] = _outer_batch
mlir.register_lowering(mhsea_full_ad_p, _outer_mlir)


mhsea_full_ad_lin_p = _ext_core.Primitive("mhsea_full_ad_lin")
mhsea_full_ad_lin_p.multiple_results = False


def _inner_impl(q, k, e, v, mask, primal_out, lse, qt, kt, et, vt):
    del primal_out, lse
    _out, tangent = mhsea_jvp_pallas(
        q, k, e, v, mask, qt, kt, et, vt, precision=_TF32_PRECISION
    )
    return tangent


def _inner_abstract_eval(q, k, e, v, mask, primal_out, lse, qt, kt, et, vt):
    del q, k, e, mask, primal_out, lse, qt, kt, et, vt
    return _shaped_core.ShapedArray(v.shape, v.dtype)


def _inner_transpose(cot, q, k, e, v, mask, primal_out, lse, qt, kt, et, vt):
    if isinstance(cot, ad.Zero):
        zeros = tuple(
            ad.Zero(_shaped_core.ShapedArray(value.shape, value.dtype))
            for value in (q, k, e, v)
        )
        return (None,) * 7 + zeros
    dq, dk, de, dv = mhsea_backward(
        q.shape[1],
        2,
        1,
        _TF32_PRECISION,
        (q, k, e, v, mask, lse, primal_out),
        cot,
    )
    return (None,) * 7 + (dq, dk, de, dv)


def _inner_batch(args, axes):
    mapped_size, flat = _prepare_batched_args(args, axes)
    if mapped_size is None:
        return mhsea_full_ad_lin_p.bind(*args), None
    out_flat = mhsea_full_ad_lin_p.bind(*flat)
    return _split_first_two(out_flat, mapped_size), 0


def _inner_mlir(ctx, *args, **kwargs):
    return mlir.lower_fun(_inner_impl, multiple_results=False)(ctx, *args, **kwargs)


mhsea_full_ad_lin_p.def_impl(_inner_impl)
mhsea_full_ad_lin_p.def_abstract_eval(_inner_abstract_eval)
ad.primitive_transposes[mhsea_full_ad_lin_p] = _inner_transpose
batching.primitive_batchers[mhsea_full_ad_lin_p] = _inner_batch
mlir.register_lowering(mhsea_full_ad_lin_p, _inner_mlir)


def mhsea_with_full_ad(q, k, e, v, mask) -> jax.Array:
    _validate_primals(q, k, e, v, mask)
    out, _lse = mhsea_full_ad_p.bind(q, k, e, v, mask)
    return out


__all__ = ["mhsea_with_full_ad"]
