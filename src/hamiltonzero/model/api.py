# Copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from .model import (
    SpinAnsatz,
    _normalize_leaf_carriers,
)
from .tree import (
    _balanced_subtree_mask,
    _quadrilinear_merge,
    _tagged_dense,
    _tagged_dense_no_bias,
    _tagged_rms_eqx_style,
    _tree_active_clock_depth,
    _tree_depth_count_features,
    _tree_sphere,
    edge_merge_masked,
)


normalize_leaf_carriers = _normalize_leaf_carriers
quadrilinear_merge = _quadrilinear_merge
tagged_dense_no_bias = _tagged_dense_no_bias
tree_sphere = _tree_sphere
balanced_subtree_mask = _balanced_subtree_mask
tree_active_clock_depth = _tree_active_clock_depth
tree_depth_count_features = _tree_depth_count_features
tagged_dense = _tagged_dense
tagged_rms_eqx_style = _tagged_rms_eqx_style


def _attention_name(value: str) -> str:
    if value == "tuned":
        return "mhsea_tuned"
    if value == "einsum":
        return "einsum"
    raise ValueError("attention must be 'tuned' or 'einsum'")


def build_model(config: Any, key, *, n_max: int) -> SpinAnsatz:
    attention = _attention_name(str(config.attention))
    model = SpinAnsatz(
        d_e=int(config.d_e),
        d_o=int(config.d_o),
        d_c=int(config.d_c),
        d_r=int(config.d_r),
        n_heads=int(config.n_heads),
        n_layers=int(config.n_layers),
        rank=int(config.rank),
        n_edge=int(config.edge_channels),
        d_e_attn=int(config.attention_qk_dim),
        d_c_attn=int(config.attention_v_dim),
        trunk_edge_node_ctx_dim=int(config.trunk_edge_node_context_dim),
        trunk_edge_hidden_dim=int(config.trunk_edge_hidden_dim),
        trunk_attn_bias_hidden_dim=int(config.trunk_attention_bias_hidden_dim),
        trunk_ffn_hidden_dim=int(config.trunk_ffn_hidden_dim),
        trunk_two_hop_hidden_dim=int(config.trunk_two_hop_hidden_dim),
        tree_edge_node_ctx_dim=int(config.tree_edge_node_context_dim),
        attn_impl=attention,
        global_d_g=int(config.global_dim),
        d_m_merge=int(config.merge_dim),
        merge_chain_hypernet_rank=int(config.merge_hypernet_rank),
        feat_d_bond=int(config.featurizer_bond_dim),
        feat_n_heads=int(config.featurizer_heads),
        feat_head_dim=int(config.featurizer_head_dim),
        feat_n_global_q=int(config.featurizer_global_queries),
        feat_edge_hidden_dim=int(config.featurizer_edge_hidden_dim),
        feat_zeeman_hidden_dim=int(config.featurizer_zeeman_hidden_dim),
        feat_global_hidden_dim=int(config.featurizer_global_hidden_dim),
        feat_combine_hidden_dim=int(config.featurizer_combine_hidden_dim),
        feat_token_initial_scale=float(config.featurizer_token_initial_scale),
        feat_d_edge=int(config.edge_channels),
        polar_group_norm_tau=float(config.polar_group_norm_tau),
        polar_group_norm_bond_hidden=int(config.polar_bond_hidden_dim),
        polar_group_norm_n_bond_groups=int(config.polar_bond_groups),
        polar_group_norm_d_bond_group=int(config.polar_bond_group_dim),
        polar_group_norm_n_zeeman_groups=int(config.polar_zeeman_groups),
        polar_group_norm_d_zeeman_group=int(config.polar_zeeman_group_dim),
        route_pointer_max_n=max(int(config.router_max_n), int(n_max)),
        route_pointer_d_model=int(config.router_model_dim),
        route_pointer_n_heads=int(config.router_heads),
        route_pointer_attn_dim=int(config.router_attention_dim),
        route_pointer_score_dim=int(config.router_score_dim),
        route_pointer_candidate_hidden=int(config.router_candidate_dim),
        route_pointer_summary_hidden=int(config.router_summary_dim),
        route_pointer_ffn_hidden=int(config.router_ffn_dim),
        route_pointer_score_init_scale=float(config.router_score_initial_scale),
        route_pointer_rope_base=float(config.router_rope_base),
        route_pointer_rope_scaling=float(config.router_rope_scaling),
        route_tree_prefix_layers=int(config.router_tree_prefix_layers),
        route_tree_prefix_candidate_layers=int(config.router_tree_candidate_layers),
        route_tree_prefix_merge_hidden=int(config.router_tree_merge_dim),
        route_tree_prefix_post_prefix_suffix_layers=int(config.router_tree_post_layers),
        route_contextualizer_layers=int(config.router_context_layers),
        route_contextualizer_n_heads=int(config.router_context_heads),
        route_contextualizer_attn_dim=int(config.router_context_attention_dim),
        route_contextualizer_edge_node_ctx_dim=int(config.router_context_edge_node_dim),
        level_edge_attn_n_heads=int(config.level_edge_heads),
        level_edge_attn_edge_mlp_hidden=int(config.level_edge_mlp_dim),
        level_edge_attn_edge_mlp_n_blocks=int(config.level_edge_mlp_blocks),
        level_edge_attn_ffn_d_hidden=int(config.level_edge_ffn_dim),
        level_edge_attn_rope_base=float(config.level_edge_rope_base),
        level_edge_attn_rope_scaling=float(config.level_edge_rope_scaling),
        root_readout_edge_rank=int(config.root_readout_edge_rank),
        ngpt_alpha_initial=float(config.ngpt_alpha_initial),
        ngpt_alpha_initial_fraction=float(config.ngpt_alpha_initial_fraction),
        ngpt_alpha_maximum=float(config.ngpt_alpha_maximum),
        global_ladder_tap_dim=int(config.global_ladder_tap_dim),
        level_edge_attn_bias_mlp_hidden=int(config.level_edge_bias_mlp_dim),
        level_edge_attn_bias_mlp_n_blocks=int(config.level_edge_bias_mlp_blocks),
        merge_c_mlp_hidden=int(config.merge_context_mlp_dim),
        readout_leaf_context_layers=int(config.readout_context_layers),
        readout_leaf_context_n_heads=int(config.readout_context_heads),
        readout_leaf_context_attn_dim=int(config.readout_context_attention_dim),
        readout_leaf_context_edge_node_ctx_dim=int(
            config.readout_context_edge_node_dim
        ),
        readout_leaf_context_summary_hidden=int(config.readout_context_summary_dim),
        readout_leaf_context_mlp_hidden=int(config.readout_context_mlp_dim),
        readout_leaf_context_bias_hidden=int(config.readout_context_bias_dim),
        readout_leaf_context_edge_ffn_hidden=int(config.readout_context_edge_ffn_dim),
        readout_leaf_context_rope_base=float(config.readout_context_rope_base),
        readout_leaf_context_rope_scaling=float(config.readout_context_rope_scaling),
        two_hop_channels=int(config.two_hop_channels),
        tree_edge_fwl_channels=int(config.tree_fwl_channels),
        key=key,
    )
    return model


__all__ = [
    "SpinAnsatz",
    "balanced_subtree_mask",
    "build_model",
    "edge_merge_masked",
    "normalize_leaf_carriers",
    "quadrilinear_merge",
    "tagged_dense",
    "tagged_dense_no_bias",
    "tagged_rms_eqx_style",
    "tree_active_clock_depth",
    "tree_depth_count_features",
    "tree_sphere",
]
