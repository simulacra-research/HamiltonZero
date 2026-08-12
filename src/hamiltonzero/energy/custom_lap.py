# Copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.extend import core
from jax.extend.core import Literal


class JLP(NamedTuple):
    value: Any
    jac: Any
    lap: Any
    level: int
    has_chunk_axis: bool
    chunk_idx: int = -1


def _is_jlp(x) -> bool:
    return isinstance(x, JLP)


from hamiltonzero.model._custom_lap_primitives import (
    custom_lap_active,
    enter_custom_lap,
    restore_custom_lap,
    quadrilinear_merge_p,
)


class use_custom_lap:
    def __enter__(self):
        self._cla_token = enter_custom_lap()
        return self

    def __exit__(self, *exc):
        restore_custom_lap(self._cla_token)


def build_W_levels(W_full, N: int) -> list:

    assert W_full.shape == (3 * N, 3 * N), (
        f"W_full must be [3N, 3N]; got {W_full.shape}"
    )
    assert (N & (N - 1)) == 0, f"N must be a power of 2; got {N}"
    levels = []
    k = 1
    while k <= N:
        n_chunks = N // k
        W_reshaped = W_full.reshape(n_chunks, 3 * k, n_chunks, 3 * k)
        idx = jnp.arange(n_chunks)
        W_k = W_reshaped[idx, :, idx, :]
        levels.append(W_k)
        k *= 2
    return levels


def _level_idx(k: int) -> int:

    assert k > 0 and (k & (k - 1)) == 0, f"k must be a power of 2; got {k}"
    return k.bit_length() - 1


def _W_at_level(W_levels, k: int):
    return W_levels[_level_idx(k)]


_RULE_REGISTRY: dict[core.Primitive, Callable] = {}


def _params_except(params, *drop, **defaults):

    out = {k: params[k] for k in params if k not in drop}
    for k, v in defaults.items():
        out.setdefault(k, v)
    return out


def _shape_bind_params(params, *drop):

    out = {k: params[k] for k in params if k not in drop and k != "out_sharding"}
    out.setdefault("sharding", params.get("out_sharding", None))
    return out


def _select_W_for_jlp(
    level: int, chunk_idx: int, has_chunk_axis: bool, W_levels, M_jac: int
):

    W_k = _W_at_level(W_levels, level)
    if has_chunk_axis:
        assert W_k.shape[0] == M_jac, (
            f"has_chunk_axis: W_k chunks {W_k.shape[0]} must equal jac M {M_jac}"
        )
        return W_k
    if chunk_idx >= 0:
        assert M_jac == 1
        return W_k[chunk_idx : chunk_idx + 1]

    assert W_k.shape[0] == M_jac, (
        f"multi-chunk: W_k chunks {W_k.shape[0]} must equal jac M {M_jac}"
    )
    return W_k


def _jac_self_quad_form(
    jac, level: int, chunk_idx: int, has_chunk_axis: bool, W_levels
):

    M = jac.shape[1]
    W_used = _select_W_for_jlp(level, chunk_idx, has_chunk_axis, W_levels, M)
    n_trailing = jac.ndim - 2
    if n_trailing == 0:
        out = jnp.einsum("mc,cmn,nc->c", jac, W_used, jac)
    elif n_trailing == 1:
        out = jnp.einsum("mca,cmn,nca->ca", jac, W_used, jac)
    elif n_trailing == 2:
        out = jnp.einsum("mcab,cmn,ncab->cab", jac, W_used, jac)
    elif n_trailing == 3:
        out = jnp.einsum("mcabd,cmn,ncabd->cabd", jac, W_used, jac)
    else:
        raise NotImplementedError(
            f"_jac_self_quad_form: trailing rank {n_trailing} not supported"
        )
    if not has_chunk_axis:
        out = out.sum(axis=0)
    return out


def _make_unary_rule(prim, f_prime_fn, f_dprime_fn):

    def rule(invals, params, W_levels):
        [x] = invals
        assert _is_jlp(x)
        v_out = prim.bind(x.value, **params)
        fp = f_prime_fn(x.value)
        jac_out = fp * x.jac
        fpp = f_dprime_fn(x.value)
        cross = _jac_self_quad_form(
            x.jac, x.level, x.chunk_idx, x.has_chunk_axis, W_levels
        )
        lap_out = fp * x.lap + fpp * cross
        return JLP(
            value=v_out,
            jac=jac_out,
            lap=lap_out,
            level=x.level,
            has_chunk_axis=x.has_chunk_axis,
            chunk_idx=x.chunk_idx,
        )

    _RULE_REGISTRY[prim] = rule


_make_unary_rule(lax.sin_p, jnp.cos, lambda x: -jnp.sin(x))
_make_unary_rule(lax.cos_p, lambda x: -jnp.sin(x), lambda x: -jnp.cos(x))
_make_unary_rule(
    lax.tanh_p,
    lambda x: 1.0 - jnp.tanh(x) ** 2,
    lambda x: -2.0 * jnp.tanh(x) * (1.0 - jnp.tanh(x) ** 2),
)
_make_unary_rule(lax.exp_p, jnp.exp, jnp.exp)
_make_unary_rule(lax.log_p, lambda x: 1.0 / x, lambda x: -1.0 / (x * x))
_make_unary_rule(lax.neg_p, lambda x: -jnp.ones_like(x), lambda x: jnp.zeros_like(x))
_make_unary_rule(lax.abs_p, lambda x: jnp.sign(x), lambda x: jnp.zeros_like(x))
_make_unary_rule(
    lax.sqrt_p, lambda x: 0.5 / jnp.sqrt(x), lambda x: -0.25 / (x * jnp.sqrt(x))
)
_make_unary_rule(
    lax.rsqrt_p,
    lambda x: -0.5 / (x * jnp.sqrt(x)),
    lambda x: 0.75 / (x * x * jnp.sqrt(x)),
)


def _logistic_prime(x):
    s = jax.nn.sigmoid(x)
    return s * (1.0 - s)


def _logistic_dprime(x):
    s = jax.nn.sigmoid(x)
    return s * (1.0 - s) * (1.0 - 2.0 * s)


_logistic_p = lax.logistic_p
_make_unary_rule(_logistic_p, _logistic_prime, _logistic_dprime)


def _integer_pow_rule(invals, params, W_levels):
    [x] = invals
    y = params["y"]
    assert _is_jlp(x)
    v_out = lax.integer_pow_p.bind(x.value, **params)
    if y == 0:
        return JLP(
            value=jnp.ones_like(x.value),
            jac=jnp.zeros_like(x.jac),
            lap=jnp.zeros_like(x.lap),
            level=x.level,
            has_chunk_axis=x.has_chunk_axis,
            chunk_idx=x.chunk_idx,
        )
    if y == 1:
        return x
    fp = float(y) * lax.integer_pow_p.bind(x.value, y=y - 1)
    jac_out = fp * x.jac
    fpp = float(y * (y - 1)) * lax.integer_pow_p.bind(x.value, y=max(y - 2, 0))
    cross = _jac_self_quad_form(x.jac, x.level, x.chunk_idx, x.has_chunk_axis, W_levels)
    lap_out = fp * x.lap + fpp * cross
    return JLP(
        value=v_out,
        jac=jac_out,
        lap=lap_out,
        level=x.level,
        has_chunk_axis=x.has_chunk_axis,
        chunk_idx=x.chunk_idx,
    )


_RULE_REGISTRY[lax.integer_pow_p] = _integer_pow_rule


def _convert_rule(invals, params, W_levels):
    [x] = invals
    assert _is_jlp(x)
    new_dtype = params["new_dtype"]
    return JLP(
        value=x.value.astype(new_dtype),
        jac=x.jac.astype(new_dtype),
        lap=x.lap.astype(new_dtype),
        level=x.level,
        has_chunk_axis=x.has_chunk_axis,
        chunk_idx=x.chunk_idx,
    )


_RULE_REGISTRY[lax.convert_element_type_p] = _convert_rule


def _broadcast_jac_lap_for_op(x, target_shape):

    lap_out = jnp.broadcast_to(x.lap, target_shape)
    jac_out = _broadcast_jac_to_value_shape(x.jac, target_shape, x.has_chunk_axis)
    return jac_out, lap_out


def _broadcast_jac_to_value_shape(jac, target_value_shape, has_chunk_axis: bool):

    if has_chunk_axis:
        new_shape = (jac.shape[0], target_value_shape[0]) + tuple(
            target_value_shape[1:]
        )
    else:
        new_shape = (jac.shape[0], jac.shape[1]) + tuple(target_value_shape)
    while jac.ndim < len(new_shape):
        jac = jnp.expand_dims(jac, axis=jac.ndim)
    return jnp.broadcast_to(jac, new_shape)


def _promote_jlp_one_level(x: JLP) -> JLP:

    k = x.level
    new_k = 2 * k
    if x.has_chunk_axis:
        assert x.chunk_idx in (0, 1), (
            f"_promote_jlp_one_level (chunked): chunk_idx must be 0 or 1; got {x}"
        )
        is_left = x.chunk_idx == 0
        zero_shape = (3 * k,) + x.jac.shape[1:]
        zeros = jnp.zeros(zero_shape, dtype=x.jac.dtype)
        if is_left:
            new_jac = jnp.concatenate([x.jac, zeros], axis=0)
        else:
            new_jac = jnp.concatenate([zeros, x.jac], axis=0)
        return JLP(
            value=x.value,
            jac=new_jac,
            lap=x.lap,
            level=new_k,
            has_chunk_axis=True,
            chunk_idx=-1,
        )

    assert x.chunk_idx >= 0, (
        f"_promote_jlp_one_level: per-node requires chunk_idx>=0; got {x}"
    )
    is_left = x.chunk_idx % 2 == 0
    new_chunk_idx = x.chunk_idx // 2
    zero_shape = (3 * k,) + x.jac.shape[1:]
    zeros = jnp.zeros(zero_shape, dtype=x.jac.dtype)
    if is_left:
        new_jac = jnp.concatenate([x.jac, zeros], axis=0)
    else:
        new_jac = jnp.concatenate([zeros, x.jac], axis=0)
    return JLP(
        value=x.value,
        jac=new_jac,
        lap=x.lap,
        level=new_k,
        has_chunk_axis=False,
        chunk_idx=new_chunk_idx,
    )


def _align_jlp_levels(a: JLP, b: JLP):

    while a.level < b.level:
        a = _promote_jlp_one_level(a)
    while b.level < a.level:
        b = _promote_jlp_one_level(b)
    assert a.has_chunk_axis == b.has_chunk_axis, (
        "_align_jlp_levels: chunk-axis mismatch"
    )
    if a.has_chunk_axis:
        if (
            a.chunk_idx in (0, 1)
            and b.chunk_idx in (0, 1)
            and a.chunk_idx != b.chunk_idx
        ):
            a = _promote_jlp_one_level(a)
            b = _promote_jlp_one_level(b)
        return a, b

    while a.chunk_idx != b.chunk_idx:
        a = _promote_jlp_one_level(a)
        b = _promote_jlp_one_level(b)
    return a, b


def _add_or_sub_rule(sign: float):
    def rule(invals, params, W_levels):
        a, b = invals
        if _is_jlp(a) and _is_jlp(b):
            need_promote = (
                (a.level != b.level)
                or (
                    not a.has_chunk_axis
                    and not b.has_chunk_axis
                    and a.chunk_idx != b.chunk_idx
                )
                or (
                    a.has_chunk_axis
                    and b.has_chunk_axis
                    and a.chunk_idx in (0, 1)
                    and b.chunk_idx in (0, 1)
                    and a.chunk_idx != b.chunk_idx
                )
            )
            if need_promote:
                a, b = _align_jlp_levels(a, b)
            assert a.has_chunk_axis == b.has_chunk_axis, "add/sub: chunk-axis mismatch"
            v_out = a.value + sign * b.value
            a_jac_b = _broadcast_jac_to_value_shape(
                a.jac, v_out.shape, a.has_chunk_axis
            )
            b_jac_b = _broadcast_jac_to_value_shape(
                b.jac, v_out.shape, b.has_chunk_axis
            )
            jac_out = a_jac_b + sign * b_jac_b
            a_lap_b = jnp.broadcast_to(a.lap, v_out.shape)
            b_lap_b = jnp.broadcast_to(b.lap, v_out.shape)
            lap_out = a_lap_b + sign * b_lap_b
            return JLP(
                value=v_out,
                jac=jac_out,
                lap=lap_out,
                level=a.level,
                has_chunk_axis=a.has_chunk_axis,
                chunk_idx=a.chunk_idx,
            )
        if _is_jlp(a):
            v_out = a.value + sign * b
            jac_out, lap_out = _broadcast_jac_lap_for_op(a, v_out.shape)
            return JLP(
                value=v_out,
                jac=jac_out,
                lap=lap_out,
                level=a.level,
                has_chunk_axis=a.has_chunk_axis,
                chunk_idx=a.chunk_idx,
            )

        v_out = a + sign * b.value
        jac_out, lap_out = _broadcast_jac_lap_for_op(b, v_out.shape)
        return JLP(
            value=v_out,
            jac=sign * jac_out,
            lap=sign * lap_out,
            level=b.level,
            has_chunk_axis=b.has_chunk_axis,
            chunk_idx=b.chunk_idx,
        )

    return rule


_RULE_REGISTRY[lax.add_p] = _add_or_sub_rule(+1.0)
_RULE_REGISTRY[lax.sub_p] = _add_or_sub_rule(-1.0)


def _mul_rule(invals, params, W_levels):
    a, b = invals
    if _is_jlp(a) and _is_jlp(b):
        need_promote = (a.level != b.level) or (
            not a.has_chunk_axis and not b.has_chunk_axis and a.chunk_idx != b.chunk_idx
        )
        if need_promote:
            a, b = _align_jlp_levels(a, b)
        assert a.has_chunk_axis == b.has_chunk_axis

        v_out = a.value * b.value
        a_jac_b = _broadcast_jac_to_value_shape(a.jac, v_out.shape, a.has_chunk_axis)
        b_jac_b = _broadcast_jac_to_value_shape(b.jac, v_out.shape, b.has_chunk_axis)
        a_val_b = jnp.broadcast_to(a.value, v_out.shape)
        b_val_b = jnp.broadcast_to(b.value, v_out.shape)
        jac_out = a_val_b * b_jac_b + b_val_b * a_jac_b
        a_lap_b = jnp.broadcast_to(a.lap, v_out.shape)
        b_lap_b = jnp.broadcast_to(b.lap, v_out.shape)
        cross = _jac_cross_quad_form(
            a_jac_b, b_jac_b, a.level, a.chunk_idx, a.has_chunk_axis, W_levels
        )
        lap_out = a_val_b * b_lap_b + b_val_b * a_lap_b + 2.0 * cross
        return JLP(
            value=v_out,
            jac=jac_out,
            lap=lap_out,
            level=a.level,
            has_chunk_axis=a.has_chunk_axis,
            chunk_idx=a.chunk_idx,
        )
    if _is_jlp(a):
        v_out = a.value * b
        jac_b, lap_b = _broadcast_jac_lap_for_op(a, v_out.shape)
        return JLP(
            value=v_out,
            jac=b * jac_b,
            lap=b * lap_b,
            level=a.level,
            has_chunk_axis=a.has_chunk_axis,
            chunk_idx=a.chunk_idx,
        )

    v_out = a * b.value
    jac_b, lap_b = _broadcast_jac_lap_for_op(b, v_out.shape)
    return JLP(
        value=v_out,
        jac=a * jac_b,
        lap=a * lap_b,
        level=b.level,
        has_chunk_axis=b.has_chunk_axis,
        chunk_idx=b.chunk_idx,
    )


def _jac_cross_quad_form(
    jac_a, jac_b, level: int, chunk_idx: int, has_chunk_axis: bool, W_levels
):

    M = jac_a.shape[1]
    W_used = _select_W_for_jlp(level, chunk_idx, has_chunk_axis, W_levels, M)
    n_trailing = jac_a.ndim - 2
    if n_trailing == 0:
        out = jnp.einsum("mc,cmn,nc->c", jac_a, W_used, jac_b)
    elif n_trailing == 1:
        out = jnp.einsum("mca,cmn,nca->ca", jac_a, W_used, jac_b)
    elif n_trailing == 2:
        out = jnp.einsum("mcab,cmn,ncab->cab", jac_a, W_used, jac_b)
    elif n_trailing == 3:
        out = jnp.einsum("mcabd,cmn,ncabd->cabd", jac_a, W_used, jac_b)
    else:
        raise NotImplementedError(
            f"_jac_cross_quad_form: trailing rank {n_trailing} not supported"
        )
    if not has_chunk_axis:
        out = out.sum(axis=0)
    return out


_RULE_REGISTRY[lax.mul_p] = _mul_rule


def _div_rule(invals, params, W_levels):
    a, b = invals
    if _is_jlp(a) and _is_jlp(b):
        need_promote = (a.level != b.level) or (
            not a.has_chunk_axis and not b.has_chunk_axis and a.chunk_idx != b.chunk_idx
        )
        if need_promote:
            a, b = _align_jlp_levels(a, b)
        assert a.has_chunk_axis == b.has_chunk_axis
        v_out = a.value / b.value
        a_val_b = jnp.broadcast_to(a.value, v_out.shape)
        b_val_b = jnp.broadcast_to(b.value, v_out.shape)
        a_jac_b = _broadcast_jac_to_value_shape(a.jac, v_out.shape, a.has_chunk_axis)
        b_jac_b = _broadcast_jac_to_value_shape(b.jac, v_out.shape, b.has_chunk_axis)
        inv_b = 1.0 / b_val_b
        jac_out = inv_b * a_jac_b - (a_val_b * inv_b * inv_b) * b_jac_b
        a_lap_b = jnp.broadcast_to(a.lap, v_out.shape)
        b_lap_b = jnp.broadcast_to(b.lap, v_out.shape)
        lap_first = inv_b * a_lap_b - (a_val_b * inv_b * inv_b) * b_lap_b
        cross_bb = _jac_self_quad_form(
            b_jac_b, b.level, b.chunk_idx, b.has_chunk_axis, W_levels
        )
        cross_ab = _jac_cross_quad_form(
            a_jac_b, b_jac_b, a.level, a.chunk_idx, a.has_chunk_axis, W_levels
        )
        lap_second = (2.0 * a_val_b * inv_b**3) * cross_bb + (
            -2.0 * inv_b * inv_b
        ) * cross_ab
        lap_out = lap_first + lap_second
        return JLP(
            value=v_out,
            jac=jac_out,
            lap=lap_out,
            level=a.level,
            has_chunk_axis=a.has_chunk_axis,
            chunk_idx=a.chunk_idx,
        )
    if _is_jlp(a):
        inv_b = 1.0 / b
        v_out = a.value * inv_b
        jac_b, lap_b = _broadcast_jac_lap_for_op(a, v_out.shape)
        return JLP(
            value=v_out,
            jac=inv_b * jac_b,
            lap=inv_b * lap_b,
            level=a.level,
            has_chunk_axis=a.has_chunk_axis,
            chunk_idx=a.chunk_idx,
        )

    v_out = a / b.value
    inv_b = 1.0 / b.value
    fp = -a * inv_b * inv_b
    jac_out = fp * b.jac
    fpp = 2.0 * a * inv_b**3
    cross = _jac_self_quad_form(b.jac, b.level, b.chunk_idx, b.has_chunk_axis, W_levels)
    lap_out = fp * b.lap + fpp * cross
    return JLP(
        value=v_out,
        jac=jac_out,
        lap=lap_out,
        level=b.level,
        has_chunk_axis=b.has_chunk_axis,
        chunk_idx=b.chunk_idx,
    )


_RULE_REGISTRY[lax.div_p] = _div_rule


def _atan2_rule(invals, params, W_levels):
    a_arg, b_arg = invals
    if _is_jlp(a_arg) and _is_jlp(b_arg):
        need_promote = (a_arg.level != b_arg.level) or (
            not a_arg.has_chunk_axis
            and not b_arg.has_chunk_axis
            and a_arg.chunk_idx != b_arg.chunk_idx
        )
        if need_promote:
            a_arg, b_arg = _align_jlp_levels(a_arg, b_arg)
        assert a_arg.has_chunk_axis == b_arg.has_chunk_axis
        a, b = a_arg, b_arg
    elif _is_jlp(a_arg):
        a = a_arg
        b = _trivial_jlp_like(b_arg, a_arg)
    else:
        b = b_arg
        a = _trivial_jlp_like(a_arg, b_arg)
    v_out = jnp.arctan2(a.value, b.value)
    r_sq = a.value**2 + b.value**2
    inv_r2 = 1.0 / r_sq
    fa = b.value * inv_r2
    fb = -a.value * inv_r2
    aj = _broadcast_jac_to_value_shape(a.jac, v_out.shape, a.has_chunk_axis)
    bj = _broadcast_jac_to_value_shape(b.jac, v_out.shape, b.has_chunk_axis)
    jac_out = fa * aj + fb * bj
    faa = -2.0 * a.value * b.value * inv_r2 * inv_r2
    fbb = 2.0 * a.value * b.value * inv_r2 * inv_r2
    fab = (a.value**2 - b.value**2) * inv_r2 * inv_r2
    al = jnp.broadcast_to(a.lap, v_out.shape)
    bl = jnp.broadcast_to(b.lap, v_out.shape)
    cross_aa = _jac_self_quad_form(aj, a.level, a.chunk_idx, a.has_chunk_axis, W_levels)
    cross_bb = _jac_self_quad_form(bj, b.level, b.chunk_idx, b.has_chunk_axis, W_levels)
    cross_ab = _jac_cross_quad_form(
        aj, bj, a.level, a.chunk_idx, a.has_chunk_axis, W_levels
    )
    lap_out = fa * al + fb * bl + faa * cross_aa + fbb * cross_bb + 2.0 * fab * cross_ab
    return JLP(
        value=v_out,
        jac=jac_out,
        lap=lap_out,
        level=a.level,
        has_chunk_axis=a.has_chunk_axis,
        chunk_idx=a.chunk_idx,
    )


def _trivial_jlp_like(plain_val, template_jlp: JLP) -> JLP:

    val = jnp.asarray(plain_val)
    if template_jlp.has_chunk_axis:
        jac_shape = (template_jlp.jac.shape[0], template_jlp.jac.shape[1]) + tuple(
            val.shape[1:]
        )
    else:
        jac_shape = (template_jlp.jac.shape[0], template_jlp.jac.shape[1]) + tuple(
            val.shape
        )
    return JLP(
        value=val,
        jac=jnp.zeros(jac_shape, dtype=val.dtype),
        lap=jnp.zeros_like(val),
        level=template_jlp.level,
        has_chunk_axis=template_jlp.has_chunk_axis,
        chunk_idx=template_jlp.chunk_idx,
    )


_RULE_REGISTRY[lax.atan2_p] = _atan2_rule


def _dot_general_rule(invals, params, W_levels):
    lhs, rhs = invals
    dimension_numbers = params["dimension_numbers"]
    (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers

    if _is_jlp(lhs) and _is_jlp(rhs):
        raise NotImplementedError(
            "dot_general: JLP × JLP outside quadrilinear_merge_p is not supported. "
            "If a model op needs same-level bilinear, route it through the merge "
            "primitive or rewrite as elementwise mul + reduce_sum."
        )
    if _is_jlp(lhs):
        return _dot_general_jlp_plain(
            lhs, rhs, dimension_numbers, params, jlp_is_lhs=True
        )
    return _dot_general_jlp_plain(rhs, lhs, dimension_numbers, params, jlp_is_lhs=False)


def _dot_general_jlp_plain(
    jlp_arg, plain_arg, dimension_numbers, params, jlp_is_lhs: bool
):

    (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers
    dot_kw = {
        "precision": params.get("precision", None),
        "preferred_element_type": params.get("preferred_element_type", None),
        "out_sharding": params.get("out_sharding", None),
    }

    if jlp_is_lhs:
        v_out = lax.dot_general(jlp_arg.value, plain_arg, dimension_numbers, **dot_kw)
        lap_out = lax.dot_general(jlp_arg.lap, plain_arg, dimension_numbers, **dot_kw)
    else:
        v_out = lax.dot_general(plain_arg, jlp_arg.value, dimension_numbers, **dot_kw)
        lap_out = lax.dot_general(plain_arg, jlp_arg.lap, dimension_numbers, **dot_kw)

    jac = jlp_arg.jac
    leading_3k = jac.shape[0]

    if jlp_arg.has_chunk_axis:
        jac_for_dot = jac
        shift = 1
    else:
        jac_for_dot = jac.reshape((leading_3k,) + tuple(jlp_arg.value.shape))
        shift = 1

    if jlp_is_lhs:
        new_lhs_contract = tuple(a + shift for a in lhs_contract)
        new_rhs_contract = tuple(rhs_contract)
        new_lhs_batch = tuple(a + shift for a in lhs_batch)
        new_rhs_batch = tuple(rhs_batch)
        new_dim_nums = (
            (new_lhs_contract, new_rhs_contract),
            (new_lhs_batch, new_rhs_batch),
        )
        jac_out_raw = lax.dot_general(jac_for_dot, plain_arg, new_dim_nums, **dot_kw)

        n_batch = len(new_lhs_batch)
        pos_3k = n_batch
    else:
        new_lhs_contract = tuple(lhs_contract)
        new_rhs_contract = tuple(a + shift for a in rhs_contract)
        new_lhs_batch = tuple(lhs_batch)
        new_rhs_batch = tuple(a + shift for a in rhs_batch)
        new_dim_nums = (
            (new_lhs_contract, new_rhs_contract),
            (new_lhs_batch, new_rhs_batch),
        )
        jac_out_raw = lax.dot_general(plain_arg, jac_for_dot, new_dim_nums, **dot_kw)

        n_batch = len(new_lhs_batch)
        lhs_ndim = jnp.asarray(plain_arg).ndim
        n_lhs_nonbatch = lhs_ndim - n_batch - len(new_lhs_contract)
        pos_3k = n_batch + n_lhs_nonbatch

    if pos_3k != 0:
        jac_out = jnp.moveaxis(jac_out_raw, pos_3k, 0)
    else:
        jac_out = jac_out_raw

    if jlp_arg.has_chunk_axis:
        new_has_chunk_axis = True
        new_chunk_idx = -1
    else:
        jac_out = jac_out[:, None]
        new_has_chunk_axis = False
        new_chunk_idx = jlp_arg.chunk_idx

    return JLP(
        value=v_out,
        jac=jac_out,
        lap=lap_out,
        level=jlp_arg.level,
        has_chunk_axis=new_has_chunk_axis,
        chunk_idx=new_chunk_idx,
    )


_RULE_REGISTRY[lax.dot_general_p] = _dot_general_rule


def _broadcast_in_dim_rule(invals, params, W_levels):
    [x] = invals
    assert _is_jlp(x)
    shape = params["shape"]
    broadcast_dimensions = params["broadcast_dimensions"]
    extra = _shape_bind_params(params, "shape", "broadcast_dimensions")

    v_out = lax.broadcast_in_dim_p.bind(
        x.value, shape=shape, broadcast_dimensions=broadcast_dimensions, **extra
    )
    lap_out = lax.broadcast_in_dim_p.bind(
        x.lap, shape=shape, broadcast_dimensions=broadcast_dimensions, **extra
    )

    if x.has_chunk_axis:
        new_shape = (x.jac.shape[0],) + tuple(shape)
        new_bd = (0,) + tuple(d + 1 for d in broadcast_dimensions)
        jac_out = lax.broadcast_in_dim_p.bind(
            x.jac.reshape((x.jac.shape[0],) + x.value.shape),
            shape=new_shape,
            broadcast_dimensions=new_bd,
            **extra,
        )

        new_has_chunk_axis = True
        new_chunk_idx = -1
    else:
        new_shape = (x.jac.shape[0], 1) + tuple(shape)
        new_bd = (0, 1) + tuple(d + 2 for d in broadcast_dimensions)
        jac_out = lax.broadcast_in_dim_p.bind(
            x.jac,
            shape=new_shape,
            broadcast_dimensions=new_bd,
            **extra,
        )
        new_has_chunk_axis = False
        new_chunk_idx = x.chunk_idx

    return JLP(
        value=v_out,
        jac=jac_out,
        lap=lap_out,
        level=x.level,
        has_chunk_axis=new_has_chunk_axis,
        chunk_idx=new_chunk_idx,
    )


_RULE_REGISTRY[lax.broadcast_in_dim_p] = _broadcast_in_dim_rule


def _reduce_sum_rule(invals, params, W_levels):
    [x] = invals
    assert _is_jlp(x)
    axes = tuple(params["axes"])
    extra = _params_except(params, "axes", out_sharding=None)
    v_out = lax.reduce_sum_p.bind(x.value, axes=axes, **extra)
    lap_out = lax.reduce_sum_p.bind(x.lap, axes=axes, **extra)
    if x.has_chunk_axis:
        if 0 in axes:
            non_chunk_axes = tuple(a for a in axes if a != 0)
            jac_reduce_axes = tuple(a + 1 for a in non_chunk_axes)
            if jac_reduce_axes:
                jac_out = lax.reduce_sum_p.bind(x.jac, axes=jac_reduce_axes, **extra)
            else:
                jac_out = x.jac
            new_has_chunk_axis = False
            new_chunk_idx = -1
        else:
            jac_reduce_axes = tuple(a + 1 for a in axes)
            jac_out = lax.reduce_sum_p.bind(x.jac, axes=jac_reduce_axes, **extra)
            new_has_chunk_axis = True
            new_chunk_idx = -1
    else:
        jac_reduce_axes = tuple(a + 2 for a in axes)
        jac_out = lax.reduce_sum_p.bind(x.jac, axes=jac_reduce_axes, **extra)
        new_has_chunk_axis = False
        new_chunk_idx = x.chunk_idx
    return JLP(
        value=v_out,
        jac=jac_out,
        lap=lap_out,
        level=x.level,
        has_chunk_axis=new_has_chunk_axis,
        chunk_idx=new_chunk_idx,
    )


_RULE_REGISTRY[lax.reduce_sum_p] = _reduce_sum_rule


def _reduce_max_rule(invals, params, W_levels):
    [x] = invals
    assert _is_jlp(x)
    axes = tuple(params["axes"])
    extra = _params_except(params, "axes", out_sharding=None)
    v_out = lax.reduce_max_p.bind(x.value, axes=axes, **extra)

    keep_shape = list(x.value.shape)
    for a in axes:
        keep_shape[a] = 1
    v_max_kept = lax.reduce_max_p.bind(x.value, axes=axes, **extra).reshape(
        tuple(keep_shape)
    )
    mask = (x.value == v_max_kept).astype(x.value.dtype)

    mask_sum_axes = lax.reduce_sum_p.bind(
        mask,
        axes=axes,
        **extra,
    ).reshape(tuple(keep_shape))
    mask = mask / (mask_sum_axes + 1e-30)

    def _gated_reduce(jac_arr, jac_axes):

        n_lead = jac_arr.ndim - x.value.ndim
        mask_b = mask.reshape((1,) * n_lead + mask.shape)

        return lax.reduce_sum_p.bind(
            jac_arr * mask_b,
            axes=jac_axes,
            **extra,
        )

    if x.has_chunk_axis:
        if 0 in axes:
            non_chunk_axes = tuple(a for a in axes if a != 0)
            jac_reduce_axes = tuple(a + 1 for a in non_chunk_axes) + (1,)
            jac_out = _gated_reduce(x.jac, jac_reduce_axes)
            new_has_chunk_axis = False
            new_chunk_idx = -1
        else:
            jac_reduce_axes = tuple(a + 1 for a in axes)
            jac_out = _gated_reduce(x.jac, jac_reduce_axes)
            new_has_chunk_axis = True
            new_chunk_idx = -1
    else:
        jac_reduce_axes = tuple(a + 2 for a in axes)
        jac_out = _gated_reduce(x.jac, jac_reduce_axes)
        new_has_chunk_axis = False
        new_chunk_idx = x.chunk_idx
    lap_out = jnp.zeros_like(v_out)
    return JLP(
        value=v_out,
        jac=jac_out,
        lap=lap_out,
        level=x.level,
        has_chunk_axis=new_has_chunk_axis,
        chunk_idx=new_chunk_idx,
    )


_RULE_REGISTRY[lax.reduce_max_p] = _reduce_max_rule


def _make_minmax_rule(sign_for_first):

    def rule(invals, params, W_levels):
        a, b = invals
        if _is_jlp(a) and _is_jlp(b):
            need_promote = (
                (a.level != b.level)
                or (
                    not a.has_chunk_axis
                    and not b.has_chunk_axis
                    and a.chunk_idx != b.chunk_idx
                )
                or (
                    a.has_chunk_axis
                    and b.has_chunk_axis
                    and a.chunk_idx in (0, 1)
                    and b.chunk_idx in (0, 1)
                    and a.chunk_idx != b.chunk_idx
                )
            )
            if need_promote:
                a, b = _align_jlp_levels(a, b)
            cmp = (a.value - b.value) * sign_for_first
            mask_a = (cmp > 0).astype(a.value.dtype)
            mask_b = 1.0 - mask_a
            v_out = mask_a * a.value + mask_b * b.value

            jac_a_b = _broadcast_jac_to_value_shape(
                a.jac, v_out.shape, a.has_chunk_axis
            )
            jac_b_b = _broadcast_jac_to_value_shape(
                b.jac, v_out.shape, b.has_chunk_axis
            )
            n_lead_a = jac_a_b.ndim - mask_a.ndim
            n_lead_b = jac_b_b.ndim - mask_b.ndim
            mask_a_lead = mask_a.reshape((1,) * n_lead_a + mask_a.shape)
            mask_b_lead = mask_b.reshape((1,) * n_lead_b + mask_b.shape)
            jac_out = mask_a_lead * jac_a_b + mask_b_lead * jac_b_b
            lap_out = mask_a * jnp.broadcast_to(
                a.lap, v_out.shape
            ) + mask_b * jnp.broadcast_to(b.lap, v_out.shape)
            return JLP(
                value=v_out,
                jac=jac_out,
                lap=lap_out,
                level=a.level,
                has_chunk_axis=a.has_chunk_axis,
                chunk_idx=a.chunk_idx,
            )
        if _is_jlp(a):
            cmp = (a.value - b) * sign_for_first
            mask_a = (cmp > 0).astype(a.value.dtype)
            v_out = mask_a * a.value + (1 - mask_a) * b
            n_lead = a.jac.ndim - mask_a.ndim
            mask_a_lead = mask_a.reshape((1,) * n_lead + mask_a.shape)
            jac_out = mask_a_lead * a.jac
            lap_out = mask_a * a.lap
            return JLP(
                value=v_out,
                jac=jac_out,
                lap=lap_out,
                level=a.level,
                has_chunk_axis=a.has_chunk_axis,
                chunk_idx=a.chunk_idx,
            )

        cmp = (a - b.value) * sign_for_first
        mask_a = (cmp > 0).astype(b.value.dtype)
        v_out = mask_a * a + (1 - mask_a) * b.value
        mask_b = 1 - mask_a
        n_lead = b.jac.ndim - mask_b.ndim
        mask_b_lead = mask_b.reshape((1,) * n_lead + mask_b.shape)
        jac_out = mask_b_lead * b.jac
        lap_out = mask_b * b.lap
        return JLP(
            value=v_out,
            jac=jac_out,
            lap=lap_out,
            level=b.level,
            has_chunk_axis=b.has_chunk_axis,
            chunk_idx=b.chunk_idx,
        )

    return rule


_RULE_REGISTRY[lax.max_p] = _make_minmax_rule(+1.0)
_RULE_REGISTRY[lax.min_p] = _make_minmax_rule(-1.0)


def _reshape_jac_for_value(x: JLP, new_value_shape: tuple, new_dimensions):

    leading_3k = x.jac.shape[0]
    if x.has_chunk_axis:
        M = x.value.shape[0]
        old_trailing = x.value.shape[1:]

        if len(new_value_shape) >= 1 and new_value_shape[0] == M:
            jac_axes_perm = None
            if new_dimensions is not None:
                jac_axes_perm = (0,) + tuple(d + 1 for d in new_dimensions)
                jac_pre = jnp.transpose(x.jac, jac_axes_perm)
            else:
                jac_pre = x.jac
            new_jac_shape = (leading_3k,) + tuple(new_value_shape)
            jac_new = jnp.reshape(jac_pre, new_jac_shape)

            return jac_new, True, -1

        if M == 1:
            assert new_dimensions is None, (
                "reshape with M=1-squeeze + transpose not yet supported"
            )

            new_jac_shape = (leading_3k, 1) + tuple(new_value_shape)

            jac_new = jnp.reshape(x.jac, new_jac_shape)
            return jac_new, False, 0
        raise NotImplementedError(
            f"reshape: cannot reshape JLP value {x.value.shape} (has_chunk_axis, M={M}) "
            f"to {new_value_shape} — chunk axis would be fused/lost."
        )

    assert new_dimensions is None or all(d >= 0 for d in new_dimensions)
    jac_pre = x.jac
    if new_dimensions is not None:
        jac_axes_perm = (0, 1) + tuple(d + 2 for d in new_dimensions)
        jac_pre = jnp.transpose(jac_pre, jac_axes_perm)
    new_jac_shape = (leading_3k, 1) + tuple(new_value_shape)
    jac_new = jnp.reshape(jac_pre, new_jac_shape)
    return jac_new, False, x.chunk_idx


def _reshape_rule(invals, params, W_levels):
    [x] = invals
    assert _is_jlp(x)
    new_sizes = tuple(params["new_sizes"])
    dimensions = params.get("dimensions")

    extra = _shape_bind_params(params, "new_sizes", "dimensions")
    v_out = lax.reshape_p.bind(
        x.value, new_sizes=new_sizes, dimensions=dimensions, **extra
    )
    lap_out = lax.reshape_p.bind(
        x.lap, new_sizes=new_sizes, dimensions=dimensions, **extra
    )
    jac_new, new_has_chunk, new_idx = _reshape_jac_for_value(x, new_sizes, dimensions)
    return JLP(
        value=v_out,
        jac=jac_new,
        lap=lap_out,
        level=x.level,
        has_chunk_axis=new_has_chunk,
        chunk_idx=new_idx,
    )


_RULE_REGISTRY[lax.reshape_p] = _reshape_rule


def _transpose_rule(invals, params, W_levels):
    [x] = invals
    assert _is_jlp(x)
    perm = tuple(params["permutation"])
    extra = {k: params[k] for k in params if k != "permutation"}
    v_out = lax.transpose_p.bind(x.value, permutation=perm, **extra)
    lap_out = lax.transpose_p.bind(x.lap, permutation=perm, **extra)

    if x.has_chunk_axis:
        if perm[0] != 0:
            raise NotImplementedError(
                "transpose: chunk axis (value axis 0) must remain at position 0; "
                f"got permutation {perm}."
            )

        jac_perm = (0, 1) + tuple(p + 1 for p in perm[1:])
    else:
        jac_perm = (0, 1) + tuple(p + 2 for p in perm)
    jac_new = lax.transpose_p.bind(x.jac, permutation=jac_perm, **extra)
    return JLP(
        value=v_out,
        jac=jac_new,
        lap=lap_out,
        level=x.level,
        has_chunk_axis=x.has_chunk_axis,
        chunk_idx=x.chunk_idx,
    )


_RULE_REGISTRY[lax.transpose_p] = _transpose_rule


def _slice_rule(invals, params, W_levels):
    [x] = invals
    assert _is_jlp(x)
    start = tuple(params["start_indices"])
    limit = tuple(params["limit_indices"])
    strides = params.get("strides")
    extra = {
        k: params[k]
        for k in params
        if k not in ("start_indices", "limit_indices", "strides")
    }
    v_out = lax.slice_p.bind(
        x.value, start_indices=start, limit_indices=limit, strides=strides, **extra
    )
    lap_out = lax.slice_p.bind(
        x.lap, start_indices=start, limit_indices=limit, strides=strides, **extra
    )

    if x.has_chunk_axis:
        new_chunk_idx = x.chunk_idx
        old_M = x.value.shape[0]
        stride0 = strides[0] if strides is not None else 1
        v_out_M = v_out.shape[0]
        chunk_axis_touched = start[0] != 0 or limit[0] != old_M or stride0 != 1
        if chunk_axis_touched:
            if v_out_M == 1:
                new_chunk_idx = start[0]
            elif stride0 > 1 and v_out_M * stride0 == old_M and start[0] in (0, 1):
                new_chunk_idx = start[0]
            elif v_out_M != old_M:
                raise NotImplementedError(
                    f"slice on chunk axis: unsupported sub-range "
                    f"start={start[0]}, limit={limit[0]}, stride={stride0}, "
                    f"old_M={old_M}, v_out_M={v_out_M}"
                )
        jac_start = (0, start[0]) + tuple(start[1:])
        jac_limit = (x.jac.shape[0], limit[0]) + tuple(limit[1:])
        jac_strides = None if strides is None else (1, strides[0]) + tuple(strides[1:])
        jac_out = lax.slice_p.bind(
            x.jac,
            start_indices=jac_start,
            limit_indices=jac_limit,
            strides=jac_strides,
            **extra,
        )
        return JLP(
            value=v_out,
            jac=jac_out,
            lap=lap_out,
            level=x.level,
            has_chunk_axis=True,
            chunk_idx=new_chunk_idx,
        )

    jac_start = (0, 0) + tuple(start)
    jac_limit = (x.jac.shape[0], x.jac.shape[1]) + tuple(limit)
    jac_strides = None if strides is None else (1, 1) + tuple(strides)
    jac_out = lax.slice_p.bind(
        x.jac,
        start_indices=jac_start,
        limit_indices=jac_limit,
        strides=jac_strides,
        **extra,
    )
    return JLP(
        value=v_out,
        jac=jac_out,
        lap=lap_out,
        level=x.level,
        has_chunk_axis=False,
        chunk_idx=x.chunk_idx,
    )


_RULE_REGISTRY[lax.slice_p] = _slice_rule


def _squeeze_rule(invals, params, W_levels):
    [x] = invals
    assert _is_jlp(x)
    dims = tuple(params["dimensions"])
    extra = {k: params[k] for k in params if k != "dimensions"}
    v_out = lax.squeeze_p.bind(x.value, dimensions=dims, **extra)
    lap_out = lax.squeeze_p.bind(x.lap, dimensions=dims, **extra)
    if x.has_chunk_axis and 0 in dims:
        non_chunk_dims = tuple(d for d in dims if d != 0)
        jac_dims = tuple(d + 1 for d in non_chunk_dims)

        if jac_dims:
            jac_out = lax.squeeze_p.bind(x.jac, dimensions=jac_dims, **extra)
        else:
            jac_out = x.jac
        return JLP(
            value=v_out,
            jac=jac_out,
            lap=lap_out,
            level=x.level,
            has_chunk_axis=False,
            chunk_idx=x.chunk_idx,
        )
    if x.has_chunk_axis:
        jac_dims = tuple(d + 1 for d in dims)
    else:
        jac_dims = tuple(d + 2 for d in dims)
    if jac_dims:
        jac_out = lax.squeeze_p.bind(x.jac, dimensions=jac_dims, **extra)
    else:
        jac_out = x.jac
    return JLP(
        value=v_out,
        jac=jac_out,
        lap=lap_out,
        level=x.level,
        has_chunk_axis=x.has_chunk_axis,
        chunk_idx=x.chunk_idx,
    )


_RULE_REGISTRY[lax.squeeze_p] = _squeeze_rule


_stack_p = lax.stack_p


def _stack_rule(invals, params, W_levels):
    del W_levels
    axis = int(params["axis"])
    extra = {k: params[k] for k in params if k != "axis"}

    jlp_inputs = [v for v in invals if _is_jlp(v)]
    levels = {v.level for v in jlp_inputs}
    assert len(levels) == 1, f"stack: mixed JLP levels {levels}"
    level = next(iter(levels))
    has_chunk = jlp_inputs[0].has_chunk_axis
    chunk_idx = jlp_inputs[0].chunk_idx
    for v in jlp_inputs[1:]:
        assert v.has_chunk_axis == has_chunk and v.chunk_idx == chunk_idx, (
            "stack: inconsistent chunk metadata"
        )

    if has_chunk and axis == 0:
        raise NotImplementedError(
            "stack: inserting an axis before the JLP chunk axis is not supported"
        )

    values = [v.value if _is_jlp(v) else v for v in invals]
    value_out = _stack_p.bind(*values, axis=axis, **extra)

    laps = [v.lap if _is_jlp(v) else jnp.zeros_like(v) for v in invals]
    lap_out = _stack_p.bind(*laps, axis=axis, **extra)

    jac_axis = axis + (1 if has_chunk else 2)
    ref_jac = jlp_inputs[0].jac
    jacs = [v.jac if _is_jlp(v) else jnp.zeros_like(ref_jac) for v in invals]
    jac_out = _stack_p.bind(*jacs, axis=jac_axis, **extra)
    return JLP(
        value=value_out,
        jac=jac_out,
        lap=lap_out,
        level=level,
        has_chunk_axis=has_chunk,
        chunk_idx=chunk_idx,
    )


_RULE_REGISTRY[_stack_p] = _stack_rule


def _concatenate_rule(invals, params, W_levels):
    dim = params["dimension"]
    extra = {k: params[k] for k in params if k != "dimension"}

    jlp_inputs = [v for v in invals if _is_jlp(v)]
    levels = set(v.level for v in jlp_inputs)
    assert len(levels) == 1, f"concatenate: mixed JLP levels {levels}"
    level = next(iter(levels))
    has_chunk = jlp_inputs[0].has_chunk_axis
    chunk_idx = jlp_inputs[0].chunk_idx
    for v in jlp_inputs[1:]:
        assert v.has_chunk_axis == has_chunk and v.chunk_idx == chunk_idx, (
            "concatenate: inconsistent chunk metadata"
        )
    if has_chunk and dim == 0:
        raise NotImplementedError("concatenate on chunk axis not supported")

    values = [v.value if _is_jlp(v) else v for v in invals]
    v_out = lax.concatenate_p.bind(*values, dimension=dim, **extra)
    laps = [v.lap if _is_jlp(v) else jnp.zeros_like(v) for v in invals]
    lap_out = lax.concatenate_p.bind(*laps, dimension=dim, **extra)

    jac_dim = dim + 2 if not has_chunk else dim + 1
    jacs = []
    for v in invals:
        if _is_jlp(v):
            jacs.append(v.jac)
        else:
            shape_v = v.shape if hasattr(v, "shape") else jnp.asarray(v).shape
            if has_chunk:
                M = jlp_inputs[0].jac.shape[1]
                jac_shape = (jlp_inputs[0].jac.shape[0], M) + tuple(shape_v[1:])
            else:
                M = jlp_inputs[0].jac.shape[1]
                jac_shape = (jlp_inputs[0].jac.shape[0], M) + tuple(shape_v)
            jacs.append(jnp.zeros(jac_shape, dtype=jlp_inputs[0].jac.dtype))
    jac_out = lax.concatenate_p.bind(*jacs, dimension=jac_dim, **extra)
    return JLP(
        value=v_out,
        jac=jac_out,
        lap=lap_out,
        level=level,
        has_chunk_axis=has_chunk,
        chunk_idx=chunk_idx,
    )


_RULE_REGISTRY[lax.concatenate_p] = _concatenate_rule


def _jit_p_rule(invals, params, W_levels):

    inner_jaxpr = params["jaxpr"]
    j = inner_jaxpr.jaxpr
    consts = inner_jaxpr.consts
    env: dict = {}
    for cv, c in zip(j.constvars, consts):
        env[cv] = c
    for iv, x in zip(j.invars, invals):
        env[iv] = x
    for eqn in j.eqns:
        outvals = _eval_eqn(eqn, env, W_levels)
        for ov, ov_val in zip(eqn.outvars, outvals):
            env[ov] = ov_val
    return [env[ov] for ov in j.outvars]


from jax._src import pjit as _pjit_module

_RULE_REGISTRY[_pjit_module.jit_p] = _jit_p_rule


def _quadrilinear_merge_rule(invals, params, W_levels):

    T, u_a, u_b = invals
    assert not _is_jlp(T), "quadrilinear_merge: T must be plain"
    assert _is_jlp(u_a) and _is_jlp(u_b), "quadrilinear_merge: u_a, u_b must be JLPs"
    assert u_a.level == u_b.level, (
        f"quadrilinear_merge: leg levels differ {u_a.level} vs {u_b.level}"
    )
    assert u_a.has_chunk_axis and u_b.has_chunk_axis, (
        "quadrilinear_merge requires the compiled chunk axis"
    )

    k = u_a.level
    new_k = 2 * k
    G, d_r, _, _ = T.shape
    d_m_eff = G * d_r
    M = u_a.value.shape[0]
    assert u_b.value.shape[0] == M, "quadrilinear_merge: chunked legs must agree on M"
    u_a_2d = u_a.value.reshape(M, G, d_r)
    u_b_2d = u_b.value.reshape(M, G, d_r)
    raw_value_2d = jnp.einsum("ijkl,mik,mil->mij", T, u_a_2d, u_b_2d)
    raw_value = raw_value_2d.reshape(M, d_m_eff)
    ua_jac_2d = u_a.jac.reshape(u_a.jac.shape[0], M, G, d_r)
    ub_jac_2d = u_b.jac.reshape(u_b.jac.shape[0], M, G, d_r)
    jac_upper_2d = jnp.einsum(
        "ijkl,Amik,mil->Amij",
        T,
        ua_jac_2d,
        u_b_2d,
    )
    jac_lower_2d = jnp.einsum(
        "ijkl,mik,Bmil->Bmij",
        T,
        u_a_2d,
        ub_jac_2d,
    )
    jac_upper = jac_upper_2d.reshape(jac_upper_2d.shape[0], M, d_m_eff)
    jac_lower = jac_lower_2d.reshape(jac_lower_2d.shape[0], M, d_m_eff)
    jac_out = jnp.concatenate([jac_upper, jac_lower], axis=0)
    ua_lap_2d = u_a.lap.reshape(M, G, d_r)
    ub_lap_2d = u_b.lap.reshape(M, G, d_r)
    term1_2d = jnp.einsum(
        "ijkl,mik,mil->mij",
        T,
        ua_lap_2d,
        u_b_2d,
    )
    term2_2d = jnp.einsum(
        "ijkl,mik,mil->mij",
        T,
        u_a_2d,
        ub_lap_2d,
    )
    W_2k = _W_at_level(W_levels, new_k)
    W_off = W_2k[:, : 3 * k, 3 * k :]
    cross_2d = jnp.einsum(
        "ijkl,Amik,Bmil,mAB->mij",
        T,
        ua_jac_2d,
        ub_jac_2d,
        W_off,
    )
    lap_out_2d = term1_2d + term2_2d + 2.0 * cross_2d
    lap_out = lap_out_2d.reshape(M, d_m_eff)
    return JLP(
        value=raw_value,
        jac=jac_out,
        lap=lap_out,
        level=new_k,
        has_chunk_axis=True,
        chunk_idx=-1,
    )


_RULE_REGISTRY[quadrilinear_merge_p] = _quadrilinear_merge_rule


def _eval_eqn(eqn, env, W_levels):
    invals = []
    for v in eqn.invars:
        if isinstance(v, Literal):
            invals.append(v.val)
        else:
            invals.append(env[v])
    has_jlp = any(_is_jlp(x) for x in invals)
    if not has_jlp:
        bind_params = eqn.primitive.get_bind_params(eqn.params)
        outvals = eqn.primitive.bind(*invals, **bind_params)
        if not eqn.primitive.multiple_results:
            outvals = [outvals]
        return outvals
    rule = _RULE_REGISTRY.get(eqn.primitive)
    if rule is None:
        raise NotImplementedError(
            f"custom_lap: no rule for primitive {eqn.primitive.name!r}. "
            f"eqn = {eqn}. Register a rule in energy/custom_lap.py."
        )
    outvals = rule(invals, eqn.params, W_levels)
    if not eqn.primitive.multiple_results:
        outvals = [outvals]
    return outvals


def _trace_z(fn, z, N, W_levels):

    z = jnp.asarray(z)
    assert z.shape == (N, 3), f"z must be [N={N}, 3]; got {z.shape}"
    eye3 = jnp.eye(3, dtype=z.dtype)
    z_jac = jnp.broadcast_to(eye3[:, None, :], (3, N, 3))
    z_lap = jnp.zeros((N, 3), dtype=z.dtype)
    z_jlp = JLP(
        value=z,
        jac=z_jac,
        lap=z_lap,
        level=1,
        has_chunk_axis=True,
        chunk_idx=-1,
    )
    closed = jax.make_jaxpr(fn)(z)
    jaxpr = closed.jaxpr
    consts = closed.consts
    env: dict = {}
    for cv, c in zip(jaxpr.constvars, consts):
        env[cv] = c
    env[jaxpr.invars[0]] = z_jlp
    for eqn in jaxpr.eqns:
        outvals = _eval_eqn(eqn, env, W_levels)
        for ov, ov_val in zip(eqn.outvars, outvals):
            env[ov] = ov_val
    return [env[ov] for ov in jaxpr.outvars]


def _canonical_jac(jlp: JLP):

    if jlp.jac.shape[1] != 1:
        jac = jnp.swapaxes(jlp.jac, 0, 1)
        return jac.reshape((jlp.jac.shape[0] * jlp.jac.shape[1],) + jlp.jac.shape[2:])
    return jlp.jac[:, 0]


def custom_forward_laplacian_with_jac(fn: Callable, W_levels: list, N: int) -> Callable:

    def lap_jac_fn(z):
        outs = _trace_z(fn, z, N, W_levels)
        if len(outs) == 1:
            out = outs[0]
            if _is_jlp(out):
                return out.value, _canonical_jac(out), out.lap
            value = out
            return (
                value,
                jnp.zeros((3 * N,) + value.shape, dtype=value.dtype),
                jnp.zeros_like(value),
            )
        values = tuple(o.value if _is_jlp(o) else o for o in outs)
        jacs = tuple(
            _canonical_jac(o)
            if _is_jlp(o)
            else jnp.zeros((3 * N,) + o.shape, dtype=o.dtype)
            for o in outs
        )
        laps = tuple(o.lap if _is_jlp(o) else jnp.zeros_like(o) for o in outs)
        return values, jacs, laps

    return lap_jac_fn


__all__ = [
    "JLP",
    "build_W_levels",
    "custom_forward_laplacian_with_jac",
    "custom_lap_active",
    "use_custom_lap",
]
