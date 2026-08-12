# Copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from hamiltonzero.config import EnergyConfig, EvalMCMCConfig, ModelConfig


@dataclass(frozen=True, slots=True)
class CanonicalContext:
    context: Any
    old_inverse: Any


@dataclass(frozen=True, slots=True)
class BeamCandidates:
    permutations: Any
    log_probabilities: Any


@dataclass(frozen=True, slots=True)
class LargeNCompilation:
    wavefunction: Any
    permutation: Any
    log_probability: Any


@dataclass(frozen=True, slots=True)
class MCMCPopulation:
    q: Any
    sigma: Any
    beta: Any


class EvalBackend(Protocol):
    def load_system(self, path: Path, energy: EnergyConfig) -> Any: ...

    def load_model(
        self,
        checkpoint: Path,
        config: ModelConfig,
        key: Any,
        context: Any,
        *,
        contextualizer_attention: str | None,
    ) -> Any: ...

    def canonicalize_context(self, context: Any) -> CanonicalContext: ...

    def embedded_route(self, model: Any) -> Any | None: ...

    def route_context(
        self,
        context: Any,
        permutation: Any,
        *,
        compact_custom_lap: bool,
    ) -> Any: ...

    def release_context(self, context: Any) -> None: ...

    def virtual_context(self, context: Any, permutations: Any) -> Any: ...

    def beam_candidates(
        self,
        model: Any,
        context: Any,
        *,
        beam_width: int,
        top_k: int,
        temperature: float,
    ) -> BeamCandidates: ...

    def compile_single(self, model: Any, routed_context: Any) -> Any: ...

    def compile_embedded(self, model: Any) -> Any: ...

    def compile_candidates(
        self,
        model: Any,
        canonical_context: Any,
        permutations: Any,
    ) -> Any: ...

    def select_candidate(self, wavefunctions: Any, winner: int) -> Any: ...

    def compile_large_n(
        self,
        model: Any,
        canonical_context: Any,
        *,
        sequence_shards: int,
        pair_tile_size: int,
        temperature: float,
    ) -> LargeNCompilation: ...

    def prepare_singular(
        self,
        model: Any,
        context: Any,
        state: Any,
    ) -> tuple[Any, Any, Any]: ...

    def initialize_mcmc(
        self,
        key: Any,
        model: Any,
        context: Any,
        config: EvalMCMCConfig,
    ) -> Any: ...

    def step_mcmc(
        self,
        state: Any,
        model: Any,
        context: Any,
        *,
        replica_steps: int,
        walker_chunk_size: int,
    ) -> Any: ...

    def adapt_mcmc(self, state: Any, config: EvalMCMCConfig) -> Any: ...

    def route_mcmc(self, state: Any, permutation: Any) -> Any: ...

    def mcmc_population(self, state: Any) -> MCMCPopulation: ...

    def replace_mcmc_population(
        self,
        state: Any,
        population: MCMCPopulation,
    ) -> Any: ...

    def cold_walkers(self, state: Any) -> Any: ...

    def custom_lap_energy(
        self,
        model: Any,
        context: Any,
        q: Any,
        config: EnergyConfig,
    ) -> tuple[Any, Any, Any, Any]: ...

    def block_until_ready(self, value: Any) -> None: ...
