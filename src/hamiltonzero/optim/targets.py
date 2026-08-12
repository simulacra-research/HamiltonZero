# Copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import jax
import jax.numpy as jnp


_ROUTE_SAMPLES = 8


def _clip_one_channel(x, width: float):
    median = jnp.nanmedian(x, axis=1, keepdims=True)
    mean_ad = jnp.nanmean(jnp.abs(x - median), axis=1, keepdims=True)
    delta = jnp.asarray(width, dtype=x.dtype) * mean_ad
    return jnp.clip(x, median - delta, median + delta)


def _mad_clip_per_system(local_energies, width: float):
    re = jnp.real(local_energies)
    im = jnp.imag(local_energies)
    re_clipped = _clip_one_channel(re, width)
    im_clipped = _clip_one_channel(im, width)
    if jnp.iscomplexobj(local_energies):
        clipped = (re_clipped + 1j * im_clipped).astype(local_energies.dtype)
    else:
        clipped = re_clipped.astype(local_energies.dtype)
    return clipped


def process_route_targets(
    sampled_energy,
    baseline_energy,
    sigma1,
    baseline_weights,
    *,
    mad_width: float = 5.0,
):
    systems = int(sampled_energy.shape[0])
    if systems % _ROUTE_SAMPLES:
        raise ValueError("router targets require a system axis divisible by K=8")
    sigma = sigma1.astype(sampled_energy.real.dtype)
    sampled_normalized = sampled_energy / sigma[:, None]
    baseline_normalized = baseline_energy / sigma[:, None]
    sampled_clipped = _mad_clip_per_system(
        sampled_normalized,
        mad_width,
    )
    centered = sampled_clipped - jnp.mean(
        sampled_clipped,
        axis=1,
        keepdims=True,
    )
    variance = jnp.mean(
        centered.real**2 + centered.imag**2,
        axis=1,
        keepdims=True,
    )
    group_variance = jnp.mean(
        variance.reshape(systems // _ROUTE_SAMPLES, _ROUTE_SAMPLES),
        axis=1,
        keepdims=True,
    )
    group_std = jnp.sqrt(
        jnp.broadcast_to(
            group_variance,
            (systems // _ROUTE_SAMPLES, _ROUTE_SAMPLES),
        )
    ).reshape(systems, 1)
    scale = jnp.maximum(group_std, jnp.asarray(1.0, dtype=group_std.dtype))
    sampled_target = sampled_clipped / scale
    baseline_target = baseline_normalized / scale
    sampled_rewards = jnp.mean(sampled_target.real, axis=1).astype(jnp.float32)
    baseline_rewards = jnp.sum(
        baseline_weights * baseline_target.real,
        axis=1,
    ).astype(jnp.float32)
    grouped_baseline = baseline_rewards.reshape((-1, _ROUTE_SAMPLES))
    group_is_finite = jnp.all(
        jnp.isfinite(grouped_baseline),
        axis=1,
        keepdims=True,
    )
    baseline_rewards = jnp.where(
        group_is_finite,
        grouped_baseline,
        jnp.zeros_like(grouped_baseline),
    ).reshape(baseline_rewards.shape)
    reward_delta = sampled_rewards.reshape(
        (-1, _ROUTE_SAMPLES)
    ) - baseline_rewards.reshape((-1, _ROUTE_SAMPLES))
    advantage = (
        float(_ROUTE_SAMPLES)
        / float(_ROUTE_SAMPLES - 1)
        * (reward_delta - jnp.mean(reward_delta, axis=1, keepdims=True))
    )
    advantage = jax.lax.stop_gradient(
        advantage.reshape(sampled_rewards.shape).astype(jnp.float32)
    )
    return sampled_target, advantage


def process_finetune_targets(
    energy,
    sigma1,
    *,
    mad_width: float = 5.0,
):
    normalized = energy / sigma1.astype(energy.real.dtype)[:, None]
    clipped = _mad_clip_per_system(normalized, mad_width)
    centered = clipped - jnp.mean(clipped, axis=1, keepdims=True)
    std = jnp.sqrt(jnp.mean(centered.real**2 + centered.imag**2, axis=1))
    return clipped / jnp.maximum(std, 1.0)[:, None]


__all__ = ["process_finetune_targets", "process_route_targets"]
