# Copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import jax
import jax.core as _shaped_core
import jax.numpy as jnp
from jax.extend import core as _ext_core
from jax.interpreters import ad, batching, mlir
from .custom_gradients import mhsea_backward, mhsea_forward
from .mhsea_tuned_jvp import mhsea_tangent_pallas

_SUPPORTED_TUNED_HEAD_DIMS = frozenset((8, 16, 32, 64))
_TF32_PRECISION = jax.lax.Precision.DEFAULT


def _forward_config(seq_len: int) -> tuple[int, int, int]:
    if seq_len <= 128:
        default_q_block, default_warps, default_stages = (32, 4, 2)
    else:
        default_q_block, default_warps, default_stages = (16, 8, 3)
    return (default_q_block, default_warps, default_stages)


_BACKWARD_CONFIG = (32, 4, 1)
_JVP_CONFIG = (32, 32, 4, 2)


def _validate_primals(q, k, e, v, mask) -> None:
    assert q.ndim == k.ndim == v.ndim == 4
    assert e.ndim == 4
    assert mask.ndim == 2
    assert q.dtype == k.dtype == e.dtype == v.dtype, (
        f"mhsea_tuned_ad requires q/k/e/v to share one storage dtype; got {q.dtype}, {k.dtype}, {e.dtype}, {v.dtype}"
    )
    assert jnp.dtype(q.dtype) == jnp.dtype(jnp.float32), (
        f"mhsea_tuned_ad currently requires float32 public tensors (TF32 dot execution); got {q.dtype}"
    )
    batch_len, seq_len, num_heads, head_len = q.shape
    assert k.shape == q.shape, f"k shape {k.shape} != q shape {q.shape}"
    assert v.shape == q.shape, (
        f"mhsea_tuned_ad currently requires Vdim == q/k head dim and matching leading dimensions; got v={v.shape}, q={q.shape}"
    )
    assert e.shape == (batch_len, seq_len, num_heads, seq_len), (
        f"e shape {e.shape} != ({batch_len}, {seq_len}, {num_heads}, {seq_len})"
    )
    assert mask.shape == (batch_len, seq_len), (
        f"mask shape {mask.shape} != ({batch_len}, {seq_len})"
    )
    assert head_len == v.shape[-1]


def _supports_tuned_shape(q) -> bool:
    return (
        jnp.dtype(q.dtype) == jnp.dtype(jnp.float32)
        and q.shape[1] >= 64
        and (q.shape[1] % 32 == 0)
        and (q.shape[-1] in _SUPPORTED_TUNED_HEAD_DIMS)
    )


def _supports_full_ad_shape(q) -> bool:
    seq_len = q.shape[1]
    return (
        jnp.dtype(q.dtype) == jnp.dtype(jnp.float32)
        and 0 < seq_len < 64
        and not (seq_len & (seq_len - 1))
        and (q.shape[-1] in _SUPPORTED_TUNED_HEAD_DIMS)
    )


mhsea_tuned_ad_p = _ext_core.Primitive("mhsea_tuned_ad")
mhsea_tuned_ad_p.multiple_results = True


def _outer_impl(q, k, e, v, mask):
    _validate_primals(q, k, e, v, mask)
    q_block_len, num_warps, num_stages = _forward_config(q.shape[1])
    out, residuals = mhsea_forward(
        q,
        k,
        e,
        v,
        mask,
        q_block_len=q_block_len,
        num_warps=num_warps,
        num_stages=num_stages,
        precision=_TF32_PRECISION,
    )
    lse = residuals[5]
    out = jnp.where(mask[:, :, None, None].astype(jnp.bool_), out, 0.0)
    return (out, lse)


def _outer_abstract_eval(q, k, e, v, mask):
    _validate_primals(q, k, e, v, mask)
    batch_len, seq_len, num_heads, _head_len = q.shape
    return (
        _shaped_core.ShapedArray(v.shape, v.dtype),
        _shaped_core.ShapedArray((batch_len, seq_len, num_heads), q.dtype),
    )


def _outer_jvp(primals, tangents):
    q, k, e, v, mask = primals
    qt, kt, et, vt, _mask_t = tangents
    qt = ad.instantiate_zeros(qt)
    kt = ad.instantiate_zeros(kt)
    et = ad.instantiate_zeros(et)
    vt = ad.instantiate_zeros(vt)
    primal_out, lse = mhsea_tuned_ad_p.bind(q, k, e, v, mask)
    tangent_out = mhsea_tuned_ad_lin_p.bind(
        q, k, e, v, mask, primal_out, lse, qt, kt, et, vt
    )
    return (
        (primal_out, lse),
        (tangent_out, ad.Zero(_shaped_core.ShapedArray(lse.shape, lse.dtype))),
    )


def _merge_first_two(x):
    return x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])


def _split_first_two(x, mapped_size: int):
    return x.reshape(mapped_size, x.shape[0] // mapped_size, *x.shape[1:])


def _prepare_batched_args(args, axes):
    mapped_size = None
    moved = []
    for x, axis in zip(args, axes):
        if axis is None:
            moved.append(x)
            continue
        x = jnp.moveaxis(x, axis, 0)
        if mapped_size is None:
            mapped_size = x.shape[0]
        else:
            assert x.shape[0] == mapped_size
        moved.append(x)
    if mapped_size is None:
        return (None, None)
    broadcasted = [
        jnp.broadcast_to(x[None], (mapped_size, *x.shape)) if axis is None else x
        for x, axis in zip(moved, axes)
    ]
    return (mapped_size, [_merge_first_two(x) for x in broadcasted])


def _outer_batch(args, axes):
    mapped_size, flat = _prepare_batched_args(args, axes)
    if mapped_size is None:
        out, lse = mhsea_tuned_ad_p.bind(*args)
        return ((out, lse), (None, None))
    out_flat, lse_flat = mhsea_tuned_ad_p.bind(*flat)
    return (
        (
            _split_first_two(out_flat, mapped_size),
            _split_first_two(lse_flat, mapped_size),
        ),
        (0, 0),
    )


def _outer_mlir(ctx, *args, **kwargs):
    return mlir.lower_fun(_outer_impl, multiple_results=True)(ctx, *args, **kwargs)


mhsea_tuned_ad_p.def_impl(_outer_impl)
mhsea_tuned_ad_p.def_abstract_eval(_outer_abstract_eval)
ad.primitive_jvps[mhsea_tuned_ad_p] = _outer_jvp
batching.primitive_batchers[mhsea_tuned_ad_p] = _outer_batch
mlir.register_lowering(mhsea_tuned_ad_p, _outer_mlir)
mhsea_tuned_ad_lin_p = _ext_core.Primitive("mhsea_tuned_ad_lin")
mhsea_tuned_ad_lin_p.multiple_results = False


def _inner_impl(q, k, e, v, mask, primal_out, lse, qt, kt, et, vt):
    q_block_len, kv_block_len, num_warps, num_stages = _JVP_CONFIG
    return mhsea_tangent_pallas(
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
        q_block_len=q_block_len,
        kv_block_len=kv_block_len,
        num_warps=num_warps,
        num_stages=num_stages,
        precision=_TF32_PRECISION,
    )


def _inner_abstract_eval(q, k, e, v, mask, primal_out, lse, qt, kt, et, vt):
    del q, k, e, mask, primal_out, lse, qt, kt, et, vt
    return _shaped_core.ShapedArray(v.shape, v.dtype)


def _inner_transpose(cot, q, k, e, v, mask, primal_out, lse, qt, kt, et, vt):
    if isinstance(cot, ad.Zero):
        zeros = tuple(
            (ad.Zero(_shaped_core.ShapedArray(x.shape, x.dtype)) for x in (q, k, e, v))
        )
        return (None,) * 7 + zeros
    q_block_len, num_warps, num_stages = _BACKWARD_CONFIG
    dq, dk, de, dv = mhsea_backward(
        q_block_len,
        num_warps,
        num_stages,
        _TF32_PRECISION,
        (q, k, e, v, mask, lse, primal_out),
        cot,
    )
    return (None,) * 7 + (dq, dk, de, dv)


def _inner_batch(args, axes):
    mapped_size, flat = _prepare_batched_args(args, axes)
    if mapped_size is None:
        return (mhsea_tuned_ad_lin_p.bind(*args), None)
    out_flat = mhsea_tuned_ad_lin_p.bind(*flat)
    return (_split_first_two(out_flat, mapped_size), 0)


def _inner_mlir(ctx, *args, **kwargs):
    return mlir.lower_fun(_inner_impl, multiple_results=False)(ctx, *args, **kwargs)


mhsea_tuned_ad_lin_p.def_impl(_inner_impl)
mhsea_tuned_ad_lin_p.def_abstract_eval(_inner_abstract_eval)
ad.primitive_transposes[mhsea_tuned_ad_lin_p] = _inner_transpose
batching.primitive_batchers[mhsea_tuned_ad_lin_p] = _inner_batch
mlir.register_lowering(mhsea_tuned_ad_lin_p, _inner_mlir)


def mhsea_with_tuned_ad(q, k, e, v, mask) -> jax.Array:
    if not _supports_tuned_shape(q):
        if not _supports_full_ad_shape(q):
            raise ValueError(
                f"mhsea requires a power-of-two N and head dimension in {sorted(_SUPPORTED_TUNED_HEAD_DIMS)}"
            )
        from .mhsea_full_ad import mhsea_with_full_ad

        out = mhsea_with_full_ad(q, k, e, v, mask)
        return jnp.where(mask[:, :, None, None].astype(jnp.bool_), out, 0.0)
    _validate_primals(q, k, e, v, mask)
    out, _lse = mhsea_tuned_ad_p.bind(q, k, e, v, mask)
    return out


__all__ = ["mhsea_with_tuned_ad"]
