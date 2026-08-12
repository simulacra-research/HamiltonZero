# Copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Callable, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import kfac_jax

from hamiltonzero.optim import compat as _kfac_compat
from hamiltonzero.optim import spin_blocks as _spin_blocks
from hamiltonzero.optim.blocks import (
    make_graph_patterns,
)


_GRAPH_PATTERNS = make_graph_patterns()
_FISHER_SIGN_STREAM = 1262895427
_FISHER_CHANNELS = 3
_ROUTE_SAMPLES = 8


class KFACBundle(NamedTuple):
    optimizer: Any
    loss_fn: Callable
    state: Any


def _make_fisher_signs(key, q_cold):
    sign_key = jax.random.fold_in(key, _FISHER_SIGN_STREAM)
    signs = jax.random.rademacher(
        sign_key,
        q_cold.shape[:2] + (_FISHER_CHANNELS,),
        dtype=q_cold.dtype,
    )
    q_sharding = getattr(q_cold, "sharding", None)
    if isinstance(q_sharding, jax.sharding.NamedSharding):
        q_spec = tuple(q_sharding.spec)
        sharding = jax.sharding.NamedSharding(
            q_sharding.mesh,
            jax.sharding.PartitionSpec(*q_spec[:2], None),
        )
        signs = jax.device_put(signs, sharding)
    elif isinstance(q_sharding, jax.sharding.SingleDeviceSharding):
        signs = jax.device_put(signs, q_sharding)
    return signs


def _signed_identity(value, signs):
    stopped = jax.lax.stop_gradient(value)
    return stopped + signs.astype(value.dtype) * (value - stopped)


def _register_fisher_output(value, signs):
    kfac_jax.register_normal_predictive_distribution(
        _signed_identity(value, signs).reshape(-1, 1)
    )


def _center_preprocessed_energy(energy, pmap_axis_name):
    if pmap_axis_name is None:
        energy_sum = jnp.sum(energy, axis=1)
        count = jnp.asarray(energy.shape[1], dtype=energy.real.dtype)
    else:
        from kfac_jax._src.utils import parallel as kfac_parallel

        energy_sum = kfac_parallel.psum_if_pmap(
            jnp.sum(energy, axis=1),
            pmap_axis_name,
        )
        count = kfac_parallel.psum_if_pmap(
            jnp.asarray(energy.shape[1], dtype=energy.real.dtype),
            pmap_axis_name,
        )
    mean = energy_sum / jnp.maximum(count, 1.0)
    delta = energy - mean[:, None]
    return jax.lax.stop_gradient(delta), mean


def _router_loss(apply_fn, route_loss_weight: float):
    def apply_walkers(params, q_cold, context, t, tau):
        systems, walkers = q_cold.shape[:2]
        if systems == 1:
            context_single = jax.tree.map(
                lambda x: x[0] if isinstance(x, jnp.ndarray) and x.ndim > 0 else x,
                context,
            )
            re, im, route_logp = jax.vmap(
                lambda p, q, time: apply_fn(
                    p,
                    q,
                    context_single,
                    time,
                    tau,
                ),
                in_axes=(None, 0, None),
            )(params, q_cold[0], t)
            return (
                re.reshape(1, walkers),
                im.reshape(1, walkers),
                route_logp.reshape(1, walkers),
            )

        def apply_system(params_value, context_value, q_value, t_value):
            return jax.vmap(
                lambda p, q, time: apply_fn(
                    p,
                    q,
                    context_value,
                    time,
                    tau,
                ),
                in_axes=(None, 0, None),
            )(params_value, q_value, t_value)

        return jax.vmap(
            apply_system,
            in_axes=(None, 0, 0, None),
        )(params, context, q_cold, t)

    @jax.custom_jvp
    def total_energy(params, batch):
        _q, energy, _context, _t, _tau, _advantage, _signs = batch
        _delta, mean = _center_preprocessed_energy(energy, None)
        return jnp.mean(mean.real)

    @total_energy.defjvp
    def total_energy_jvp(primals, tangents):
        params, batch = primals
        params_t, _batch_t = tangents
        q_cold, energy, context, t, tau, advantage, fisher_signs = batch
        (re, im, route_logp), (tan_re, tan_im, tan_route_logp) = jax.jvp(
            lambda p: apply_walkers(p, q_cold, context, t, tau),
            (params,),
            (params_t,),
        )
        _register_fisher_output(re, fisher_signs[..., 0])
        _register_fisher_output(im, fisher_signs[..., 1])
        _register_fisher_output(route_logp, fisher_signs[..., 2])
        delta, mean = _center_preprocessed_energy(energy, None)
        real_num = jnp.sum(tan_re * delta.real)
        imag_num = jnp.sum(tan_im * delta.imag)
        n_eff = jnp.maximum(
            jnp.sum(jnp.abs(delta) > 0).astype(tan_re.dtype),
            1.0,
        )
        loss_tangent = 2.0 * (real_num + imag_num) / n_eff
        advantage = jax.lax.stop_gradient(advantage.reshape((-1,)))
        route_tangent = jnp.mean(
            advantage * jnp.mean(tan_route_logp, axis=1).reshape((-1,))
        )
        loss_tangent = (
            loss_tangent
            + jnp.asarray(
                route_loss_weight,
                dtype=loss_tangent.dtype,
            )
            * route_tangent
        )
        return jnp.mean(mean.real), loss_tangent

    return total_energy


def _finetune_loss(apply_fn, pmap_axis_name):
    def apply_walkers(params, q_cold, context, t):
        context = jax.tree.map(
            lambda x: x[0] if isinstance(x, jnp.ndarray) and x.ndim > 0 else x,
            context,
        )
        re, im = jax.vmap(
            lambda p, q, time: apply_fn(p, q, context, time),
            in_axes=(None, 0, None),
        )(params, q_cold[0], t)
        batch_size = q_cold.shape[1]
        return re.reshape(1, batch_size), im.reshape(1, batch_size)

    @jax.custom_jvp
    def total_energy(params, batch):
        _q, energy, _context, _t, _signs = batch
        _delta, mean = _center_preprocessed_energy(
            energy,
            pmap_axis_name,
        )
        return jnp.mean(mean.real)

    @total_energy.defjvp
    def total_energy_jvp(primals, tangents):
        params, batch = primals
        params_t, _batch_t = tangents
        q_cold, energy, context, t, fisher_signs = batch
        (re, im), (tan_re, tan_im) = jax.jvp(
            lambda p: apply_walkers(p, q_cold, context, t),
            (params,),
            (params_t,),
        )
        _register_fisher_output(re, fisher_signs[..., 0])
        _register_fisher_output(im, fisher_signs[..., 1])
        delta, mean = _center_preprocessed_energy(energy, pmap_axis_name)
        local_real_num = jnp.sum(tan_re * delta.real)
        local_imag_num = jnp.sum(tan_im * delta.imag)
        local_n_eff = jnp.sum(jnp.abs(delta) > 0).astype(tan_re.dtype)
        if pmap_axis_name is None:
            real_num = local_real_num
            imag_num = local_imag_num
            n_eff = local_n_eff
        else:
            from kfac_jax._src.utils import parallel as kfac_parallel

            real_num = kfac_parallel.psum_if_pmap(
                local_real_num,
                pmap_axis_name,
            )
            imag_num = kfac_parallel.psum_if_pmap(
                local_imag_num,
                pmap_axis_name,
            )
            n_eff = kfac_parallel.psum_if_pmap(
                local_n_eff,
                pmap_axis_name,
            )
        loss_tangent = 2.0 * (real_num + imag_num) / jnp.maximum(n_eff, 1.0)
        return jnp.mean(mean.real), loss_tangent

    return total_energy


def _configure_kfac():
    kfac_jax.utils.set_use_cholesky_inversion(True)


def _new_optimizer(config, loss_fn, *, multi_device: bool, axis_name):
    _configure_kfac()
    return kfac_jax.Optimizer(
        jax.value_and_grad(loss_fn),
        learning_rate_schedule=None,
        damping_schedule=None,
        norm_constraint=float(config.norm_constraint),
        multi_device=multi_device,
        pmap_axis_name=axis_name if multi_device else None,
        value_func_has_aux=False,
        value_func_has_rng=False,
        register_only_generic=False,
        auto_register_kwargs={
            "graph_patterns": _GRAPH_PATTERNS,
            "allow_multiple_registrations": True,
        },
        include_norms_in_stats=False,
        estimation_mode="fisher_exact",
        share_curvature_and_grad_forward=False,
        num_burnin_steps=0,
        batch_size_extractor=lambda batch, *_: batch[0].shape[0] * batch[0].shape[1],
        min_damping=float(config.minimum_damping),
        inverse_update_period=int(config.inverse_update_period),
        curvature_update_period=int(config.curvature_update_period),
        curvature_ema=float(config.curvature_ema),
        l2_reg=float(config.l2_regularization),
    )


def _partition(model):
    return eqx.partition(model, jax.tree.map(eqx.is_inexact_array, model))


def _assert_no_naive_full(optimizer, state):
    blocks = list(enumerate(getattr(state, "blocks_states", []) or []))
    if not blocks:
        try:
            blocks = list(enumerate(optimizer._estimator.blocks))
        except AttributeError:
            blocks = []
    bad = []
    for index, block in blocks:
        name = type(block).__name__
        if "NaiveFull" in name:
            bad.append((index, name, getattr(block, "parameters_shapes", None)))
    if bad:
        details = "; ".join(
            f"block[{index}] {name} shapes={shapes}" for index, name, shapes in bad
        )
        raise RuntimeError(f"KFAC produced unsupported NaiveFull blocks: {details}")


def _router_initial_advantage(energy):
    rewards = jnp.mean(energy.real, axis=1)
    grouped = rewards.reshape((-1, _ROUTE_SAMPLES))
    centered = grouped - jnp.mean(grouped, axis=1, keepdims=True)
    return jax.lax.stop_gradient(
        (float(_ROUTE_SAMPLES) / float(_ROUTE_SAMPLES - 1) * centered).reshape((-1,))
    )


def init_router_kfac_state(
    config,
    model,
    q_cold,
    energy,
    context,
    *,
    t: float,
    key,
    multi_device: bool,
    route_tau,
    route_loss_weight: float,
):
    params, static = _partition(model)

    def apply_fn(params_value, q, context_value, t_value, tau_value):
        combined = eqx.combine(params_value, static)
        return combined.call_with_route_logprob(
            q,
            context_value,
            t_value,
            tau=tau_value,
        )

    loss_fn = _router_loss(apply_fn, route_loss_weight)
    optimizer = _new_optimizer(
        config,
        loss_fn,
        multi_device=multi_device,
        axis_name="systems",
    )
    fisher_signs = _make_fisher_signs(key, q_cold)
    batch = (
        q_cold,
        energy,
        context,
        jnp.asarray(t, dtype=q_cold.dtype),
        jnp.asarray(route_tau, dtype=q_cold.dtype),
        _router_initial_advantage(energy),
        fisher_signs,
    )
    _configure_kfac()
    state = optimizer.init(params, key, batch)
    _assert_no_naive_full(optimizer, state)
    return KFACBundle(optimizer=optimizer, loss_fn=loss_fn, state=state)


def init_finetune_kfac_state(
    config,
    model,
    q_cold,
    energy,
    context,
    *,
    t: float,
    key,
    multi_device: bool,
):
    params, static = _partition(model)

    def apply_fn(params_value, q, context_value, t_value):
        combined = eqx.combine(params_value, static)
        return combined.call_tagged(q, context_value, t_value)

    loss_axis = "batch" if multi_device else None
    loss_fn = _finetune_loss(apply_fn, loss_axis)
    optimizer = _new_optimizer(
        config,
        loss_fn,
        multi_device=multi_device,
        axis_name="batch",
    )
    fisher_signs = _make_fisher_signs(key, q_cold)
    batch = (
        q_cold,
        energy,
        context,
        jnp.asarray(t, dtype=q_cold.dtype),
        fisher_signs,
    )
    _configure_kfac()
    state = optimizer.init(params, key, batch)
    _assert_no_naive_full(optimizer, state)
    return KFACBundle(optimizer=optimizer, loss_fn=loss_fn, state=state)


def apply_router_kfac_step(
    bundle,
    model,
    q_cold,
    energy,
    context,
    *,
    t: float,
    key,
    momentum,
    learning_rate,
    damping,
    route_advantage,
    route_tau,
):
    _configure_kfac()
    params, static = _partition(model)
    batch = (
        q_cold,
        energy,
        context,
        jnp.asarray(t, dtype=q_cold.dtype),
        jnp.asarray(route_tau, dtype=q_cold.dtype),
        route_advantage,
        _make_fisher_signs(key, q_cold),
    )
    new_params, state, _stats = bundle.optimizer.step(
        params,
        bundle.state,
        key,
        batch=batch,
        momentum=jnp.asarray(momentum, dtype=jnp.float32),
        learning_rate=jnp.asarray(learning_rate, dtype=jnp.float32),
        damping=jnp.asarray(damping, dtype=jnp.float32),
    )
    return eqx.combine(new_params, static), bundle._replace(state=state)


def apply_finetune_kfac_step(
    bundle,
    model,
    q_cold,
    energy,
    context,
    *,
    t: float,
    key,
    momentum,
    learning_rate,
    damping,
):
    _configure_kfac()
    params, static = _partition(model)
    batch = (
        q_cold,
        energy,
        context,
        jnp.asarray(t, dtype=q_cold.dtype),
        _make_fisher_signs(key, q_cold),
    )
    new_params, state, _stats = bundle.optimizer.step(
        params,
        bundle.state,
        key,
        batch=batch,
        momentum=jnp.asarray(momentum, dtype=jnp.float32),
        learning_rate=jnp.asarray(learning_rate, dtype=jnp.float32),
        damping=jnp.asarray(damping, dtype=jnp.float32),
    )
    return eqx.combine(new_params, static), bundle._replace(state=state)


__all__ = [
    "KFACBundle",
    "apply_finetune_kfac_step",
    "apply_router_kfac_step",
    "init_finetune_kfac_state",
    "init_router_kfac_state",
]
