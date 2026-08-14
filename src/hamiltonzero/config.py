# Copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeVar


AttentionImplementation = Literal["tuned", "einsum"]


@dataclass(frozen=True, slots=True)
class ModelConfig:
    d_e: int = 512
    d_o: int = 32
    d_c: int = 512
    d_r: int = 32
    n_heads: int = 16
    n_layers: int = 8
    rank: int = 32
    edge_channels: int = 96
    attention_qk_dim: int = 256
    attention_v_dim: int = 256
    merge_dim: int = 1536
    trunk_edge_node_context_dim: int = 256
    trunk_edge_hidden_dim: int = 128
    trunk_attention_bias_hidden_dim: int = 64
    trunk_ffn_hidden_dim: int = 2048
    trunk_two_hop_hidden_dim: int = 128
    tree_edge_node_context_dim: int = 256
    global_dim: int = 256
    merge_hypernet_rank: int = 256
    featurizer_bond_dim: int = 128
    featurizer_heads: int = 8
    featurizer_head_dim: int = 64
    featurizer_global_queries: int = 4
    featurizer_edge_hidden_dim: int = 192
    featurizer_zeeman_hidden_dim: int = 2048
    featurizer_global_hidden_dim: int = 16384
    featurizer_combine_hidden_dim: int = 12288
    featurizer_token_initial_scale: float = 0.02
    polar_group_norm_tau: float = 0.001
    polar_bond_hidden_dim: int = 256
    polar_bond_groups: int = 16
    polar_bond_group_dim: int = 16
    polar_zeeman_groups: int = 16
    polar_zeeman_group_dim: int = 16
    router_max_n: int = 128
    router_model_dim: int = 512
    router_heads: int = 16
    router_attention_dim: int = 128
    router_score_dim: int = 512
    router_candidate_dim: int = 1024
    router_summary_dim: int = 1024
    router_ffn_dim: int = 1024
    router_score_initial_scale: float = 0.0
    router_rope_base: float = 10000.0
    router_rope_scaling: float = 1.0
    router_tree_prefix_layers: int = 4
    router_tree_candidate_layers: int = 2
    router_tree_merge_dim: int = 1024
    router_tree_post_layers: int = 4
    router_context_layers: int = 2
    router_context_heads: int = 4
    router_context_attention_dim: int = 256
    router_context_edge_node_dim: int = 256
    level_edge_heads: int = 8
    level_edge_mlp_dim: int = 384
    level_edge_mlp_blocks: int = 3
    level_edge_ffn_dim: int = 1024
    level_edge_rope_base: float = 10000.0
    level_edge_rope_scaling: float = 1.0
    root_readout_edge_rank: int = 128
    ngpt_alpha_initial: float = 0.25
    ngpt_alpha_initial_fraction: float = 0.25
    ngpt_alpha_maximum: float = 0.8
    global_ladder_tap_dim: int = 256
    level_edge_bias_mlp_dim: int = 128
    level_edge_bias_mlp_blocks: int = 1
    merge_context_mlp_dim: int = 1024
    readout_context_layers: int = 2
    readout_context_heads: int = 8
    readout_context_attention_dim: int = 128
    readout_context_edge_node_dim: int = 256
    readout_context_summary_dim: int = 1024
    readout_context_mlp_dim: int = 2048
    readout_context_bias_dim: int = 32
    readout_context_edge_ffn_dim: int = 384
    readout_context_rope_base: float = 10000.0
    readout_context_rope_scaling: float = 1.0
    two_hop_channels: int = 64
    tree_fwl_channels: int = 128
    attention: AttentionImplementation = "tuned"


@dataclass(frozen=True, slots=True)
class MCMCConfig:
    batch_size: int = 512
    replicas: int = 8
    steps: int = 32
    burn_in: int = 256
    burn_in_replica_steps: int = 2
    walker_chunk_size: int | None = None
    initial_sigma: float = 0.3
    initial_haar_sites: int = 1
    sigma_scale: float = 1.1
    langevin_target_acceptance: float = 0.574
    haar_target_acceptance: float = 0.234
    beta_history_weight: float = 0.9
    adapt_every: int = 1
    reuse_mcmc: Path | None = None


@dataclass(frozen=True, slots=True)
class KFACConfig:
    learning_rate_numerator: float = 0.05
    learning_rate_offset: float = 5.0
    learning_rate_decay_steps: float = 5000.0
    curvature_ema: float = 0.995
    curvature_update_period: int = 2
    inverse_update_period: int = 2
    damping: float = 0.001
    minimum_damping: float = 0.0001
    norm_constraint: float = 0.001
    mad_clip_width: float = 5.0
    momentum: float = 0.0
    l2_regularization: float = 0.0


@dataclass(frozen=True, slots=True)
class RouterConfig:
    temperature: float = 1.0
    loss_weight: float = 1.0


@dataclass(frozen=True, slots=True)
class EnergyConfig:
    mu: float | None = None
    eps: float = 0.1
    chunk_size: int = 512


@dataclass(frozen=True, slots=True)
class TrainConfig:
    systems: Path
    output: Path
    steps: int
    seed: int = 777
    n_max: int = 64
    checkpoint: Path | None = None
    model: ModelConfig = field(default_factory=ModelConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    mcmc: MCMCConfig = field(default_factory=MCMCConfig)
    kfac: KFACConfig = field(default_factory=KFACConfig)
    energy: EnergyConfig = field(default_factory=EnergyConfig)


def _finetune_mcmc() -> MCMCConfig:
    return MCMCConfig(batch_size=256, replicas=8, steps=2, burn_in=256)


def _finetune_kfac() -> KFACConfig:
    return KFACConfig(
        learning_rate_numerator=0.002,
        learning_rate_offset=1.0,
        learning_rate_decay_steps=10000.0,
        curvature_ema=0.99,
        curvature_update_period=2,
        inverse_update_period=4,
        damping=0.001,
    )


def _finetune_energy() -> EnergyConfig:
    return EnergyConfig(mu=2.86)


@dataclass(frozen=True, slots=True)
class FineTuneConfig:
    system: Path
    checkpoint: Path
    output: Path
    steps: int = 10000
    seed: int = 777
    leaf_rank: int = 1536
    merge_rank: int = 1024
    route_temperature: float = 1.0
    model: ModelConfig = field(default_factory=lambda: ModelConfig(attention="einsum"))
    mcmc: MCMCConfig = field(default_factory=_finetune_mcmc)
    kfac: KFACConfig = field(default_factory=_finetune_kfac)
    energy: EnergyConfig = field(default_factory=_finetune_energy)


def _eval_mcmc() -> EvalMCMCConfig:
    return EvalMCMCConfig(
        batch_size=256,
        replicas=8,
        steps=24,
        burn_in=1024,
        walker_chunk_size=16,
    )


@dataclass(frozen=True, slots=True)
class EvalMCMCConfig:
    batch_size: int = 256
    replicas: int = 8
    steps: int = 24
    burn_in: int = 1024
    burn_in_replica_steps: int = 2
    walker_chunk_size: int = 16
    initial_sigma: float = 0.3
    initial_haar_sites: int = 1
    sigma_scale: float = 1.1
    langevin_target_acceptance: float = 0.574
    haar_target_acceptance: float = 0.234
    beta_history_weight: float = 0.9


@dataclass(frozen=True, slots=True)
class EvalConfig:
    system: Path
    checkpoint: Path
    output: Path
    seed: int = 777
    contest: bool = False
    large_n: bool = False
    measurements: int = 256
    contest_candidates: int = 8
    contest_beam_width: int = 8
    contest_preburn: int = 128
    contest_measurements: int = 128
    contest_se_multiplier: float = 2.0
    route_temperature: float = 4.0
    large_n_sequence_shards: int = 0
    large_n_pair_tile_size: int = 128
    contextualizer_attention: AttentionImplementation | None = None
    model: ModelConfig = field(default_factory=ModelConfig)
    mcmc: EvalMCMCConfig = field(default_factory=_eval_mcmc)
    energy: EnergyConfig = field(default_factory=EnergyConfig)

    def __post_init__(self) -> None:
        if self.contest and self.large_n:
            raise ValueError("contest and large_n are mutually exclusive")


Config = TrainConfig | FineTuneConfig | EvalConfig
T = TypeVar("T")


def _coerce(cls: type[T], values: dict[str, Any]) -> T:
    nested = {
        "model": ModelConfig,
        "router": RouterConfig,
        "mcmc": MCMCConfig,
        "kfac": KFACConfig,
        "energy": EnergyConfig,
    }
    if cls is EvalConfig:
        nested["mcmc"] = EvalMCMCConfig
    data = dict(values)
    fields_by_name = {item.name: item for item in dataclasses.fields(cls)}
    for name, nested_cls in nested.items():
        if name in data and isinstance(data[name], dict):
            item = fields_by_name.get(name)
            defaults: dict[str, Any] = {}
            if item is not None and item.default_factory is not dataclasses.MISSING:
                defaults = dataclasses.asdict(item.default_factory())
            nested_values = {**defaults, **data[name]}
            if name == "mcmc" and nested_values.get("reuse_mcmc") is not None:
                nested_values["reuse_mcmc"] = Path(nested_values["reuse_mcmc"])
            data[name] = nested_cls(**nested_values)
    path_fields = {"systems", "system", "checkpoint", "output"}
    for item in dataclasses.fields(cls):
        if item.name in path_fields and item.name in data:
            data[item.name] = Path(data[item.name])
    return cls(**data)


def load_config(path: str | Path, mode: Literal["train", "finetune", "eval"]) -> Config:
    values = json.loads(Path(path).read_text())
    cls = {"train": TrainConfig, "finetune": FineTuneConfig, "eval": EvalConfig}[mode]
    return _coerce(cls, values)


__all__ = [
    "AttentionImplementation",
    "EnergyConfig",
    "EvalConfig",
    "EvalMCMCConfig",
    "FineTuneConfig",
    "KFACConfig",
    "MCMCConfig",
    "ModelConfig",
    "RouterConfig",
    "TrainConfig",
    "load_config",
]
