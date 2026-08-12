# Copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


P01_TWO_SIDED = 2.576


def standard_error_from_tailstd(
    tailstd: float,
    *,
    n_systems: int,
    batch_size: int,
) -> float:
    if tailstd is None or not math.isfinite(float(tailstd)):
        return float("nan")
    denominator = max(1, int(n_systems) * int(batch_size))
    return float(tailstd) / math.sqrt(denominator)


def welch_band_walkermean(
    tailstd_a: float,
    tailstd_b: float,
    *,
    n_systems_a: int,
    batch_size_a: int,
    n_systems_b: int,
    batch_size_b: int,
    z: float = P01_TWO_SIDED,
) -> float:
    se_a = standard_error_from_tailstd(
        tailstd_a,
        n_systems=n_systems_a,
        batch_size=batch_size_a,
    )
    se_b = standard_error_from_tailstd(
        tailstd_b,
        n_systems=n_systems_b,
        batch_size=batch_size_b,
    )
    if not (math.isfinite(se_a) and math.isfinite(se_b)):
        return float("nan")
    return float(z) * math.sqrt(se_a * se_a + se_b * se_b)


def select_winner_per_physical(
    R,
    tailstd,
    beam_logp,
    batch_per_candidate,
    *,
    z: float = P01_TWO_SIDED,
    ucb_z: float = 2.0,
):
    R = np.asarray(R, dtype=float)
    tailstd = np.asarray(tailstd, dtype=float)
    beam_logp = np.asarray(beam_logp, dtype=float)
    P, K = R.shape
    Bp = int(batch_per_candidate)
    winner_idx = np.zeros(P, dtype=np.int64)
    tie_mask = np.zeros((P, K), dtype=bool)
    bands = np.full((P, K), np.nan, dtype=float)
    se = np.array(
        [
            [
                standard_error_from_tailstd(
                    tailstd[p, k],
                    n_systems=1,
                    batch_size=Bp,
                )
                for k in range(K)
            ]
            for p in range(P)
        ],
        dtype=float,
    )
    reason: list[str] = []
    for p in range(P):
        Rp, sp, lp = R[p], tailstd[p], beam_logp[p]
        finite = np.isfinite(Rp)
        if not finite.any():
            w = int(np.argmax(lp))
            winner_idx[p] = w
            tie_mask[p, w] = True
            reason.append("energy_unavailable")
            continue
        istar = int(np.argmin(np.where(finite, Rp, np.inf)))
        for k in range(K):
            band = welch_band_walkermean(
                sp[k],
                sp[istar],
                n_systems_a=1,
                batch_size_a=Bp,
                n_systems_b=1,
                batch_size_b=Bp,
                z=z,
            )
            bands[p, k] = band
            if k == istar:
                tie_mask[p, k] = True
            elif finite[k] and math.isfinite(band) and abs(Rp[k] - Rp[istar]) < band:
                tie_mask[p, k] = True
        tie_idx = np.where(tie_mask[p])[0]
        se_tie = se[p, tie_idx]
        ucb = Rp[tie_idx] + ucb_z * np.where(np.isfinite(se_tie), se_tie, np.inf)
        w = int(tie_idx[int(np.argmin(ucb))]) if np.any(np.isfinite(ucb)) else istar
        winner_idx[p] = w
        reason.append("argmin_energy" if w == istar else "tie_broken_by_ucb")
    return winner_idx, tie_mask, reason, bands, se


@dataclass(frozen=True, slots=True)
class ChannelMetrics:
    mean: float
    standard_error: float
    walker_tail_std: float
    local_energy_std: float
    lag1_autocorrelation: float | None

    def as_dict(self) -> dict[str, float | None]:
        return {
            "mean": self.mean,
            "standard_error": self.standard_error,
            "walker_tail_std": self.walker_tail_std,
            "local_energy_std": self.local_energy_std,
            "lag1_autocorrelation": self.lag1_autocorrelation,
        }


class EnergyWindow:
    def __init__(self, steps: int, systems: int, batch_size: int):
        self.steps = int(steps)
        self.systems = int(systems)
        self.batch_size = int(batch_size)
        shape = (self.steps, self.systems, self.batch_size)
        self.values = {
            "total": np.empty(shape, dtype=np.float32),
            "exchange": np.empty(shape, dtype=np.float32),
            "field": np.empty(shape, dtype=np.float32),
        }
        self.sums = {
            "total": np.zeros((self.systems, self.batch_size), dtype=np.float32),
            "exchange": np.zeros((self.systems, self.batch_size), dtype=np.float32),
            "field": np.zeros((self.systems, self.batch_size), dtype=np.float32),
        }
        self.count = 0

    def push(self, total, exchange, field) -> None:
        if self.count >= self.steps:
            raise IndexError("energy window is full")
        for name, value in (
            ("total", total),
            ("exchange", exchange),
            ("field", field),
        ):
            array = np.asarray(value).real.astype(np.float32, copy=False)
            expected = (self.systems, self.batch_size)
            if array.shape != expected:
                raise ValueError(
                    f"{name} energy must have shape {expected}, got {array.shape}"
                )
            self.values[name][self.count] = array
            self.sums[name] += array
        self.count += 1

    def tail_means(self, channel: str) -> np.ndarray:
        if self.count < 1:
            raise RuntimeError("energy window is empty")
        return (self.sums[channel] / float(self.count)).astype(np.float32, copy=False)

    def tail_mean(self, channel: str, system: int) -> float:
        return float(np.mean(self.tail_means(channel)[system]))

    def tail_std(self, channel: str, system: int) -> float:
        per_walker = np.asarray(self.tail_means(channel)[system]).ravel()
        if per_walker.size < 2:
            return float("nan")
        return float(np.std(per_walker, ddof=1))

    def metrics(self, channel: str) -> ChannelMetrics:
        if self.systems != 1:
            raise ValueError("bare eval metrics require one physical system")
        samples = self.values[channel][: self.count, 0]
        per_walker = self.tail_means(channel)[0]
        mean = float(np.mean(per_walker))
        tailstd = (
            float(np.std(per_walker, ddof=1)) if per_walker.size >= 2 else float("nan")
        )
        se = standard_error_from_tailstd(
            tailstd,
            n_systems=1,
            batch_size=self.batch_size,
        )
        step_std = np.std(samples, axis=1)
        gap_sq = float(np.nanmean(step_std**2))
        local_std = (
            math.sqrt(gap_sq)
            if math.isfinite(gap_sq) and gap_sq >= 0.0
            else float("nan")
        )
        step_means = np.mean(samples, axis=1)
        finite = step_means[np.isfinite(step_means)]
        autocorrelation = None
        if len(finite) >= 20:
            centered = finite - np.mean(finite)
            denominator = float(np.sum(centered * centered))
            if denominator > 0.0:
                autocorrelation = float(
                    np.sum(centered[:-1] * centered[1:]) / denominator
                )
        return ChannelMetrics(
            mean=mean,
            standard_error=se,
            walker_tail_std=tailstd,
            local_energy_std=local_std,
            lag1_autocorrelation=autocorrelation,
        )
