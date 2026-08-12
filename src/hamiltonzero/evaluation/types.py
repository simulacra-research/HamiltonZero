# Copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .statistics import ChannelMetrics


EvalPath = Literal["ordinary", "contest", "large_n"]


@dataclass(frozen=True, slots=True)
class ContestCandidate:
    index: int
    route_log_probability: float
    energy: float
    standard_error: float
    walker_tail_std: float
    in_tie_set: bool

    def as_dict(self) -> dict[str, int | float | bool]:
        return {
            "index": self.index,
            "route_log_probability": self.route_log_probability,
            "energy": self.energy,
            "standard_error": self.standard_error,
            "walker_tail_std": self.walker_tail_std,
            "in_tie_set": self.in_tie_set,
        }


@dataclass(frozen=True, slots=True)
class ContestResult:
    winner: int
    reason: str
    candidates: tuple[ContestCandidate, ...]

    def as_dict(self) -> dict:
        return {
            "winner": self.winner,
            "reason": self.reason,
            "candidates": [candidate.as_dict() for candidate in self.candidates],
        }


@dataclass(frozen=True, slots=True)
class EvalMetric:
    step: int
    energy: float
    energy_std: float
    step_walltime: float
    walltime: float


@dataclass(frozen=True, slots=True)
class EvalResult:
    path: EvalPath
    route: tuple[int, ...]
    route_log_probability: float | None
    measurements: int
    walltime_seconds: float
    energy: ChannelMetrics
    channels: dict[str, ChannelMetrics]
    contest: ContestResult | None = None

    def as_dict(self) -> dict:
        result = {
            "measurements": self.measurements,
            "walltime_seconds": self.walltime_seconds,
            "energy": self.energy.mean,
            "energy_std": self.energy.local_energy_std,
            "channels": {name: metrics.mean for name, metrics in self.channels.items()},
        }
        if self.energy.lag1_autocorrelation is not None:
            result["energy_lag1_autocorrelation"] = self.energy.lag1_autocorrelation
        return result
