# Copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import NamedTuple, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from hamiltonzero.inference import PreparedInference


class BasisSamplerState(NamedTuple):
    bits: jax.Array
    log_abs: jax.Array
    key: jax.Array
    accepted: jax.Array
    proposed: jax.Array


class Renyi2Result(NamedTuple):
    purity: float | None
    imaginary_mean: float | None
    standard_error: float | None
    renyi2_nats: float | None
    renyi2_bits: float | None
    resolved: bool
    failure_reasons: tuple[str, ...]
    naive_standard_error: float | None
    imaginary_standard_error: float | None
    imaginary_naive_standard_error: float | None
    integrated_autocorrelation_time_blocks: float | None
    integrated_autocorrelation_time_imaginary_blocks: float | None
    effective_blocks: float
    effective_imaginary_blocks: float
    largest_absolute_block_fraction: float | None
    renyi2_standard_error_nats: float | None
    renyi2_lower_3sigma_nats: float | None
    n_blocks: int
    mean_log_abs: float
    mean_phase: float
    block_log_abs: np.ndarray
    block_phase: np.ndarray
    swap_log_abs: np.ndarray
    swap_phase: np.ndarray
    valid_denominator: np.ndarray


def _geometry(prepared: PreparedInference) -> tuple[jax.Array, int]:
    if not isinstance(prepared, PreparedInference):
        raise TypeError("prepared must be a PreparedInference")
    route_host = np.asarray(jax.device_get(prepared.route), dtype=np.int32)
    if route_host.ndim != 1:
        raise ValueError("prepared route must have shape [N]")
    if not np.array_equal(np.sort(route_host), np.arange(route_host.size)):
        raise ValueError("prepared route must be a permutation")
    mask = np.asarray(jax.device_get(prepared._initial_context.mask), dtype=np.bool_)
    n_spins = int(np.sum(mask))
    if mask.shape != route_host.shape or not np.array_equal(
        mask, np.arange(mask.size) < n_spins
    ):
        raise ValueError("prepared physical sites must be a contiguous prefix")
    return jnp.asarray(prepared.route, dtype=jnp.int32), n_spins


def _as_bits(bits, *, n_spins: int) -> jax.Array:
    value = jnp.asarray(bits)
    if (
        value.ndim not in (2, 3)
        or any(size < 1 for size in value.shape[:-1])
        or value.shape[-1] != n_spins
    ):
        raise ValueError(
            f"bits must have shape [pairs, {n_spins}] or [blocks, pairs, {n_spins}]"
        )
    if value.dtype != jnp.bool_:
        invalid = jnp.any((value != 0) & (value != 1))
        if bool(jax.device_get(invalid)):
            raise ValueError("bits must contain only zero and one")
    return value.astype(jnp.bool_)


def _routed_corners(bits, route, *, n_spins: int):
    width = int(route.shape[0])
    padding = width - int(n_spins)
    if padding < 0:
        raise ValueError("physical spin count exceeds compiled width")
    full_bits = jnp.pad(bits, ((0, 0), (0, padding)), constant_values=False)
    routed = jnp.take(full_bits, route, axis=-1)
    up = jnp.logical_not(routed).astype(jnp.float32)
    down = routed.astype(jnp.float32)
    zeros = jnp.zeros_like(up)
    return jnp.stack((up, zeros, down, zeros), axis=-1)


def _basis_log_wavefunction(wavefunction, bits, route, *, n_spins: int):
    q = _routed_corners(bits, route, n_spins=n_spins)
    return wavefunction(q, None, 0.0)


def _basis_log_abs(wavefunction, bits, route, *, n_spins: int):
    log_abs, _phase = _basis_log_wavefunction(
        wavefunction, bits, route, n_spins=n_spins
    )
    return log_abs


def _metropolis_log_acceptance(current_log_abs, proposed_log_abs):
    current_finite = jnp.isfinite(current_log_abs)
    proposed_finite = jnp.isfinite(proposed_log_abs)
    log_ratio = 2.0 * (proposed_log_abs - current_log_abs)
    log_accept = jnp.minimum(jnp.zeros_like(log_ratio), log_ratio)
    both_zero = jnp.isneginf(current_log_abs) & jnp.isneginf(proposed_log_abs)
    recover_to_finite = jnp.logical_not(current_finite) & proposed_finite
    invalid_proposal = jnp.logical_not(proposed_finite) & jnp.logical_not(both_zero)
    log_accept = jnp.where(both_zero | recover_to_finite, 0.0, log_accept)
    return jnp.where(invalid_proposal, -jnp.inf, log_accept)


def _state_from_bits(key, wavefunction, bits, route, *, n_spins: int):
    log_abs = _basis_log_abs(wavefunction, bits, route, n_spins=n_spins)
    return BasisSamplerState(
        bits=bits,
        log_abs=log_abs,
        key=key,
        accepted=jnp.asarray(0, dtype=jnp.int32),
        proposed=jnp.asarray(0, dtype=jnp.int32),
    )


def _basis_step(state, wavefunction, route, *, n_spins: int):
    site_key, accept_key, next_key = jax.random.split(state.key, 3)
    batch_size = state.bits.shape[0]
    sites = jax.random.randint(site_key, (batch_size,), 0, n_spins, dtype=jnp.int32)
    rows = jnp.arange(batch_size)
    proposed_bits = state.bits.at[rows, sites].set(
        jnp.logical_not(state.bits[rows, sites])
    )
    proposed_log_abs = _basis_log_abs(
        wavefunction, proposed_bits, route, n_spins=n_spins
    )
    log_accept = _metropolis_log_acceptance(state.log_abs, proposed_log_abs)
    log_uniform = jnp.log(
        jax.random.uniform(accept_key, state.log_abs.shape, dtype=state.log_abs.dtype)
    )
    accept = log_uniform < log_accept
    return BasisSamplerState(
        bits=jnp.where(accept[:, None], proposed_bits, state.bits),
        log_abs=jnp.where(accept, proposed_log_abs, state.log_abs),
        key=next_key,
        accepted=state.accepted + jnp.sum(accept, dtype=jnp.int32),
        proposed=state.proposed + jnp.asarray(state.bits.shape[0], dtype=jnp.int32),
    )


@eqx.filter_jit
def _run_basis_steps(state, wavefunction, route, *, n_spins: int, n_steps: int):
    def one_step(carry, _):
        return _basis_step(carry, wavefunction, route, n_spins=n_spins), None

    state, _ = jax.lax.scan(one_step, state, xs=None, length=n_steps)
    return state


def _require_finite_state(state: BasisSamplerState) -> None:
    finite = np.asarray(jax.device_get(jnp.isfinite(state.log_abs)))
    if not np.all(finite):
        raise RuntimeError(
            "basis burn-in ended with a zero or non-finite wavefunction coefficient"
        )


def burn_in_basis(
    prepared: PreparedInference,
    key,
    *,
    batch_size: int = 256,
    burn_in: int = 1024,
):
    if int(batch_size) < 1:
        raise ValueError("batch_size must be positive")
    if int(burn_in) < 0:
        raise ValueError("burn_in must be non-negative")
    route, n_spins = _geometry(prepared)
    bits_key, state_key = jax.random.split(key)
    bits = jax.random.bernoulli(bits_key, shape=(int(batch_size), n_spins))
    state = _state_from_bits(
        state_key,
        prepared.wavefunction,
        bits,
        route,
        n_spins=n_spins,
    )
    state = _run_basis_steps(
        state,
        prepared.wavefunction,
        route,
        n_spins=n_spins,
        n_steps=int(burn_in),
    )
    jax.block_until_ready(state.log_abs)
    _require_finite_state(state)
    return state, state.bits


def step_basis(
    prepared: PreparedInference,
    state: BasisSamplerState,
    *,
    steps: int = 24,
):
    if not isinstance(state, BasisSamplerState):
        raise TypeError("state must be a BasisSamplerState")
    if int(steps) < 1:
        raise ValueError("steps must be positive")
    route, n_spins = _geometry(prepared)
    if state.bits.shape[-1] != n_spins:
        raise ValueError("basis state width does not match the prepared system")
    state = _run_basis_steps(
        state,
        prepared.wavefunction,
        route,
        n_spins=n_spins,
        n_steps=int(steps),
    )
    jax.block_until_ready(state.log_abs)
    _require_finite_state(state)
    return state, state.bits


def _subsystem_mask(
    subsystem: Sequence[int] | Sequence[bool] | np.ndarray,
    *,
    n_spins: int,
) -> jax.Array:
    value = np.asarray(subsystem)
    if value.ndim == 1 and value.size == 0:
        mask = np.zeros((n_spins,), dtype=np.bool_)
    elif value.dtype == np.bool_:
        if value.shape != (n_spins,):
            raise ValueError(f"boolean subsystem mask must have shape [{n_spins}]")
        mask = value
    else:
        if value.ndim != 1:
            raise ValueError("subsystem site indices must be one-dimensional")
        if not np.issubdtype(value.dtype, np.integer):
            raise TypeError("subsystem must contain integer sites or booleans")
        sites = value.astype(np.int64)
        if len(np.unique(sites)) != sites.size:
            raise ValueError("subsystem site indices must be unique")
        if np.any(sites < 0) or np.any(sites >= n_spins):
            raise ValueError("subsystem site index is out of range")
        mask = np.zeros((n_spins,), dtype=np.bool_)
        mask[sites] = True
    return jnp.asarray(mask)


@eqx.filter_jit
def _swap_log_ratios(
    wavefunction,
    replica_x,
    replica_y,
    route,
    region_mask,
    *,
    n_spins: int,
):
    swapped_x = jnp.where(region_mask[None, :], replica_y, replica_x)
    swapped_y = jnp.where(region_mask[None, :], replica_x, replica_y)
    denominator_x, phase_x = _basis_log_wavefunction(
        wavefunction, replica_x, route, n_spins=n_spins
    )
    denominator_y, phase_y = _basis_log_wavefunction(
        wavefunction, replica_y, route, n_spins=n_spins
    )
    numerator_x, numerator_phase_x = _basis_log_wavefunction(
        wavefunction, swapped_x, route, n_spins=n_spins
    )
    numerator_y, numerator_phase_y = _basis_log_wavefunction(
        wavefunction, swapped_y, route, n_spins=n_spins
    )
    valid_denominator = jnp.isfinite(denominator_x) & jnp.isfinite(denominator_y)
    log_abs = numerator_x + numerator_y - denominator_x - denominator_y
    phase = numerator_phase_x + numerator_phase_y - phase_x - phase_y
    phase = jnp.arctan2(jnp.sin(phase), jnp.cos(phase))
    log_abs = jnp.where(valid_denominator, log_abs, -jnp.inf)
    phase = jnp.where(valid_denominator, phase, 0.0)
    exact_identity = jnp.logical_or(
        jnp.all(jnp.logical_not(region_mask)), jnp.all(region_mask)
    )
    log_abs = jnp.where(exact_identity & valid_denominator, 0.0, log_abs)
    phase = jnp.where(exact_identity & valid_denominator, 0.0, phase)
    return log_abs, phase, valid_denominator


def _complex_mean_log_polar(log_abs, phase) -> tuple[float, float]:
    logs = np.asarray(log_abs, dtype=np.float64).reshape(-1)
    phases = np.asarray(phase, dtype=np.float64).reshape(-1)
    if logs.shape != phases.shape or logs.size == 0:
        raise ValueError("log_abs and phase must have matching nonempty shapes")
    if (
        np.any(np.isnan(logs))
        or np.any(np.isposinf(logs))
        or np.any(~np.isfinite(phases))
    ):
        raise ValueError("non-finite SWAP log-polar sample")
    finite = np.isfinite(logs)
    if not np.any(finite):
        return -math.inf, 0.0
    pivot = float(np.max(logs[finite]))
    scaled = np.zeros(logs.shape, dtype=np.complex128)
    scaled[finite] = np.exp(logs[finite] - pivot + 1j * phases[finite])
    scaled_mean = np.mean(scaled)
    magnitude = float(abs(scaled_mean))
    if magnitude == 0.0:
        return -math.inf, 0.0
    return pivot + math.log(magnitude), float(np.angle(scaled_mean))


def _log_polar_to_complex(log_abs: float, phase: float) -> complex:
    if log_abs == -math.inf:
        return 0.0j
    if not math.isfinite(log_abs) or not math.isfinite(phase):
        raise ValueError("log-polar scalar must be finite or exact zero")
    if log_abs > math.log(np.finfo(np.float64).max):
        raise OverflowError("complex SWAP block mean exceeds float64 range")
    return complex(math.exp(log_abs) * np.exp(1j * phase))


def _integrated_autocorrelation_time(values) -> float:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if x.size < 2:
        return 0.5
    x = x - np.mean(x)
    variance = float(np.dot(x, x) / x.size)
    if not math.isfinite(variance) or variance <= 0.0:
        return 0.5
    correlations = []
    for lag in range(1, x.size):
        covariance = float(np.dot(x[:-lag], x[lag:]) / x.size)
        correlations.append(covariance / variance)
    tau = 0.5
    previous_pair = math.inf
    for offset in range(0, len(correlations) - 1, 2):
        pair = correlations[offset] + correlations[offset + 1]
        if not math.isfinite(pair) or pair <= 0.0:
            break
        pair = min(pair, previous_pair)
        tau += pair
        previous_pair = pair
    return max(0.5, float(tau))


def _summarize_blocks(block_log_abs, block_phase):
    logs = np.asarray(block_log_abs, dtype=np.float64).reshape(-1)
    phases = np.asarray(block_phase, dtype=np.float64).reshape(-1)
    if logs.shape != phases.shape or logs.size == 0:
        raise ValueError("block log magnitudes and phases must match")
    try:
        blocks = np.asarray(
            [
                _log_polar_to_complex(float(value), float(angle))
                for value, angle in zip(logs, phases, strict=True)
            ],
            dtype=np.complex128,
        )
    except (OverflowError, ValueError):
        return {
            "n_blocks": int(logs.size),
            "purity": None,
            "naive_standard_error": None,
            "standard_error": None,
            "imaginary_mean": None,
            "imaginary_naive_standard_error": None,
            "imaginary_standard_error": None,
            "integrated_autocorrelation_time_blocks": None,
            "integrated_autocorrelation_time_imaginary_blocks": None,
            "effective_blocks": 0.0,
            "effective_imaginary_blocks": 0.0,
            "largest_absolute_block_fraction": None,
            "resolved": False,
            "failure_reasons": ("block_mean_float64_overflow_or_nonfinite",),
            "renyi2_nats": None,
            "renyi2_bits": None,
            "renyi2_standard_error_nats": None,
            "renyi2_lower_3sigma_nats": None,
        }
    n_blocks = int(blocks.size)
    real = blocks.real
    imaginary = blocks.imag
    purity = float(np.mean(real))
    imaginary_mean = float(np.mean(imaginary))
    naive_standard_error = (
        float(np.std(real, ddof=1) / math.sqrt(n_blocks)) if n_blocks > 1 else math.inf
    )
    imaginary_naive_standard_error = (
        float(np.std(imaginary, ddof=1) / math.sqrt(n_blocks))
        if n_blocks > 1
        else math.inf
    )
    tau = _integrated_autocorrelation_time(real)
    effective_blocks = float(n_blocks / (2.0 * tau))
    imaginary_tau = _integrated_autocorrelation_time(imaginary)
    effective_imaginary_blocks = float(n_blocks / (2.0 * imaginary_tau))
    standard_error = (
        float(np.std(real, ddof=1) / math.sqrt(effective_blocks))
        if n_blocks > 1
        else math.inf
    )
    imaginary_standard_error = (
        float(np.std(imaginary, ddof=1) / math.sqrt(effective_imaginary_blocks))
        if n_blocks > 1
        else math.inf
    )
    absolute_sum = float(np.sum(np.abs(blocks)))
    tail_fraction = (
        float(np.max(np.abs(blocks)) / absolute_sum) if absolute_sum > 0.0 else 1.0
    )
    failures = []
    if n_blocks < 16:
        failures.append("too_few_blocks")
    if not math.isfinite(purity) or not math.isfinite(standard_error):
        failures.append("nonfinite_real_estimate")
    elif purity <= 3.0 * standard_error:
        failures.append("purity_not_resolved_above_zero")
    if math.isfinite(imaginary_standard_error):
        if abs(imaginary_mean) > 3.0 * imaginary_standard_error:
            failures.append("imaginary_null_test_failed")
    else:
        failures.append("nonfinite_imaginary_uncertainty")
    if purity > 1.0:
        failures.append("purity_point_above_physical_upper_bound")
    if effective_blocks < 8.0:
        failures.append("insufficient_effective_blocks")
    if tail_fraction > 0.25:
        failures.append("single_block_tail_dominance")
    resolved = not failures
    entropy = -math.log(purity) if resolved else None
    entropy_standard_error = standard_error / purity if resolved else None
    purity_upper_3sigma = min(1.0, purity + 3.0 * standard_error) if resolved else None
    entropy_lower_3sigma = (
        max(0.0, -math.log(purity_upper_3sigma))
        if purity_upper_3sigma is not None
        else None
    )
    return {
        "n_blocks": n_blocks,
        "purity": purity,
        "naive_standard_error": naive_standard_error,
        "standard_error": standard_error,
        "imaginary_mean": imaginary_mean,
        "imaginary_naive_standard_error": imaginary_naive_standard_error,
        "imaginary_standard_error": imaginary_standard_error,
        "integrated_autocorrelation_time_blocks": tau,
        "integrated_autocorrelation_time_imaginary_blocks": imaginary_tau,
        "effective_blocks": effective_blocks,
        "effective_imaginary_blocks": effective_imaginary_blocks,
        "largest_absolute_block_fraction": tail_fraction,
        "resolved": resolved,
        "failure_reasons": tuple(failures),
        "renyi2_nats": entropy,
        "renyi2_bits": entropy / math.log(2.0) if entropy is not None else None,
        "renyi2_standard_error_nats": entropy_standard_error,
        "renyi2_lower_3sigma_nats": entropy_lower_3sigma,
    }


def _evaluate_swap(
    prepared,
    x,
    y,
    route,
    mask,
    *,
    n_spins: int,
    chunk_size: int,
):
    logs = []
    phases = []
    valid = []
    for start in range(0, x.shape[0], chunk_size):
        stop = min(start + chunk_size, x.shape[0])
        values = _swap_log_ratios(
            prepared.wavefunction,
            x[start:stop],
            y[start:stop],
            route,
            mask,
            n_spins=n_spins,
        )
        values = jax.device_get(values)
        logs.append(np.asarray(values[0]))
        phases.append(np.asarray(values[1]))
        valid.append(np.asarray(values[2]))
    return (
        np.concatenate(logs),
        np.concatenate(phases),
        np.concatenate(valid),
    )


def _result(log_abs, phase, valid) -> Renyi2Result:
    swap_log_abs = np.asarray(log_abs)
    swap_phase = np.asarray(phase)
    valid_denominator = np.asarray(valid, dtype=np.bool_)
    if (
        swap_log_abs.ndim != 2
        or swap_log_abs.shape != swap_phase.shape
        or swap_log_abs.shape != valid_denominator.shape
    ):
        raise ValueError("SWAP blocks must have aligned shape [blocks, pairs]")
    if not np.all(valid_denominator):
        raise RuntimeError("SWAP denominator contains a zero wavefunction coefficient")
    block_values = [
        _complex_mean_log_polar(logs, phases)
        for logs, phases in zip(swap_log_abs, swap_phase, strict=True)
    ]
    block_log_abs = np.asarray([value[0] for value in block_values], dtype=np.float64)
    block_phase = np.asarray([value[1] for value in block_values], dtype=np.float64)
    summary = _summarize_blocks(block_log_abs, block_phase)
    mean_log_abs, mean_phase = _complex_mean_log_polar(block_log_abs, block_phase)
    return Renyi2Result(
        purity=summary["purity"],
        imaginary_mean=summary["imaginary_mean"],
        standard_error=summary["standard_error"],
        renyi2_nats=summary["renyi2_nats"],
        renyi2_bits=summary["renyi2_bits"],
        resolved=summary["resolved"],
        failure_reasons=summary["failure_reasons"],
        naive_standard_error=summary["naive_standard_error"],
        imaginary_standard_error=summary["imaginary_standard_error"],
        imaginary_naive_standard_error=summary["imaginary_naive_standard_error"],
        integrated_autocorrelation_time_blocks=summary[
            "integrated_autocorrelation_time_blocks"
        ],
        integrated_autocorrelation_time_imaginary_blocks=summary[
            "integrated_autocorrelation_time_imaginary_blocks"
        ],
        effective_blocks=summary["effective_blocks"],
        effective_imaginary_blocks=summary["effective_imaginary_blocks"],
        largest_absolute_block_fraction=summary["largest_absolute_block_fraction"],
        renyi2_standard_error_nats=summary["renyi2_standard_error_nats"],
        renyi2_lower_3sigma_nats=summary["renyi2_lower_3sigma_nats"],
        n_blocks=summary["n_blocks"],
        mean_log_abs=mean_log_abs,
        mean_phase=mean_phase,
        block_log_abs=block_log_abs,
        block_phase=block_phase,
        swap_log_abs=swap_log_abs,
        swap_phase=swap_phase,
        valid_denominator=valid_denominator,
    )


def renyi2_purity(
    prepared: PreparedInference,
    replica_x,
    replica_y,
    subsystem: Sequence[int] | Sequence[bool] | np.ndarray,
    *,
    chunk_size: int = 256,
) -> Renyi2Result:
    if int(chunk_size) < 1:
        raise ValueError("chunk_size must be positive")
    route, n_spins = _geometry(prepared)
    x = _as_bits(replica_x, n_spins=n_spins)
    y = _as_bits(replica_y, n_spins=n_spins)
    if x.shape != y.shape:
        raise ValueError("replica batches must have the same shape")
    if x.ndim == 2:
        x = x[None, ...]
        y = y[None, ...]
    n_blocks, pairs_per_block = x.shape[:2]
    mask = _subsystem_mask(subsystem, n_spins=n_spins)
    values = _evaluate_swap(
        prepared,
        x.reshape((n_blocks * pairs_per_block, n_spins)),
        y.reshape((n_blocks * pairs_per_block, n_spins)),
        route,
        mask,
        n_spins=n_spins,
        chunk_size=int(chunk_size),
    )
    return _result(
        values[0].reshape((n_blocks, pairs_per_block)),
        values[1].reshape((n_blocks, pairs_per_block)),
        values[2].reshape((n_blocks, pairs_per_block)),
    )


def measure_renyi2(
    prepared: PreparedInference,
    replica_x: BasisSamplerState,
    replica_y: BasisSamplerState,
    subsystem: Sequence[int] | Sequence[bool] | np.ndarray,
    *,
    blocks: int = 16,
    samples_per_block: int = 1,
    steps_between: int = 24,
    chunk_size: int = 256,
):
    if not isinstance(replica_x, BasisSamplerState) or not isinstance(
        replica_y, BasisSamplerState
    ):
        raise TypeError("replicas must be BasisSamplerState values")
    if int(blocks) < 1:
        raise ValueError("blocks must be positive")
    if int(samples_per_block) < 1:
        raise ValueError("samples_per_block must be positive")
    if int(steps_between) < 1:
        raise ValueError("steps_between must be positive")
    if int(chunk_size) < 1:
        raise ValueError("chunk_size must be positive")
    route, n_spins = _geometry(prepared)
    if replica_x.bits.shape != replica_y.bits.shape:
        raise ValueError("replica states must have the same walker shape")
    if replica_x.bits.ndim != 2 or replica_x.bits.shape[-1] != n_spins:
        raise ValueError("basis state width does not match the prepared system")
    _require_finite_state(replica_x)
    _require_finite_state(replica_y)
    mask = _subsystem_mask(subsystem, n_spins=n_spins)
    block_logs = []
    block_phases = []
    block_valid = []
    for _ in range(int(blocks)):
        logs = []
        phases = []
        valid = []
        for _ in range(int(samples_per_block)):
            replica_x = _run_basis_steps(
                replica_x,
                prepared.wavefunction,
                route,
                n_spins=n_spins,
                n_steps=int(steps_between),
            )
            replica_y = _run_basis_steps(
                replica_y,
                prepared.wavefunction,
                route,
                n_spins=n_spins,
                n_steps=int(steps_between),
            )
            values = _evaluate_swap(
                prepared,
                replica_x.bits,
                replica_y.bits,
                route,
                mask,
                n_spins=n_spins,
                chunk_size=int(chunk_size),
            )
            logs.append(values[0])
            phases.append(values[1])
            valid.append(values[2])
        block_logs.append(np.concatenate(logs))
        block_phases.append(np.concatenate(phases))
        block_valid.append(np.concatenate(valid))
    result = _result(
        np.stack(block_logs),
        np.stack(block_phases),
        np.stack(block_valid),
    )
    return replica_x, replica_y, result


__all__ = [
    "BasisSamplerState",
    "Renyi2Result",
    "burn_in_basis",
    "measure_renyi2",
    "renyi2_purity",
    "step_basis",
]
