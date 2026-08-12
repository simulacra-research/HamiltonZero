# Copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from typing import NamedTuple
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray
from .context import SpinContext
from .featurizer import SystemFeaturizer
from .odd_ops import BiasFreeLinear
from .readout_leaf_context import PhysicalReadoutContext, RouterContext
from .route_pointer import TreePrefixPointerMHSEA
from .tree import (
    LeafBuilder,
    MergeOp,
    RootReadout,
    balanced_tree_reduce_masked_scan as balanced_tree_reduce_masked,
)
from .trunk import Trunk


class PerSystemInvariants(NamedTuple):
    g_emb: Float[Array, "d_g"]
    e_leaf: Float[Array, "n d_e"]
    edge_leaf: Float[Array, "n n d_edge"]
    g_stream: Float[Array, "d_global_feat"]


def _normalize_leaf_carriers(u_all):
    eps = 1e-30
    u32 = u_all.astype(jnp.float32)
    rms = jnp.sqrt(jnp.mean(u32 * u32, axis=-1, keepdims=True) + eps)
    u_all = u_all / rms.astype(u_all.dtype)
    log_rms = jnp.log(rms)[..., 0]
    return (u_all, log_rms)


def _shallow_replace(obj, **changes):
    cls = type(obj)
    new_obj = object.__new__(cls)
    new_obj.__dict__.update(obj.__dict__)
    for k, v in changes.items():
        object.__setattr__(new_obj, k, v)
    return new_obj


class SpinAnsatz(eqx.Module):
    featurizer: SystemFeaturizer
    trunk: Trunk
    leaf: LeafBuilder
    merge: MergeOp
    readout: RootReadout
    readout_leaf_context: PhysicalReadoutContext
    route_contextualizer: RouterContext
    gladder_post: "EdgeRowColGlobalUpdate"
    gladder_to_gemb_w: Float[Array, "d_global_feat d_g"]
    gladder_to_gemb_b: Float[Array, "d_g"]
    gladder_gemb_ln_s: Float[Array, "d_g"]
    gladder_tree_pool: "GDescriptorPool"
    gladder_tree_update: "TreeGlobalUpdate"
    gladder_tree_proj_w: Float[Array, "d_global_feat d_g"]
    gladder_tree_proj_b: Float[Array, "d_g"]
    gladder_root_proj_w: Float[Array, "d_global_feat d_g"]
    gladder_root_proj_b: Float[Array, "d_g"]
    gladder_root_ln_s: Float[Array, "d_g"]
    gladder_fork_phys: "EdgeRowColGlobalUpdate"
    gladder_fork_route: "EdgeRowColGlobalUpdate"
    route_decoder: TreePrefixPointerMHSEA
    q_to_odd: BiasFreeLinear

    def __init__(
        self,
        *,
        d_e: int,
        d_o: int,
        d_c: int,
        d_r: int,
        n_heads: int,
        n_layers: int,
        rank: int,
        n_edge: int,
        d_e_attn: int,
        d_c_attn: int,
        trunk_edge_node_ctx_dim: int,
        trunk_edge_hidden_dim: int,
        trunk_attn_bias_hidden_dim: int,
        trunk_ffn_hidden_dim: int,
        trunk_two_hop_hidden_dim: int,
        tree_edge_node_ctx_dim: int,
        global_d_g: int,
        d_m_merge: int,
        merge_chain_hypernet_rank: int,
        feat_d_bond: int,
        feat_n_heads: int,
        feat_head_dim: int,
        feat_n_global_q: int,
        feat_edge_hidden_dim: int,
        feat_zeeman_hidden_dim: int,
        feat_global_hidden_dim: int,
        feat_combine_hidden_dim: int,
        feat_token_initial_scale: float,
        feat_d_edge: int,
        polar_group_norm_tau: float,
        polar_group_norm_bond_hidden: int,
        polar_group_norm_n_bond_groups: int,
        polar_group_norm_d_bond_group: int,
        polar_group_norm_n_zeeman_groups: int,
        polar_group_norm_d_zeeman_group: int,
        route_pointer_max_n: int,
        route_pointer_d_model: int,
        route_pointer_n_heads: int,
        route_pointer_attn_dim: int,
        route_pointer_score_dim: int,
        route_pointer_candidate_hidden: int,
        route_pointer_summary_hidden: int,
        route_pointer_ffn_hidden: int,
        route_pointer_score_init_scale: float,
        route_pointer_rope_base: float,
        route_pointer_rope_scaling: float,
        route_tree_prefix_layers: int,
        route_tree_prefix_candidate_layers: int,
        route_tree_prefix_merge_hidden: int,
        route_tree_prefix_post_prefix_suffix_layers: int,
        route_contextualizer_layers: int,
        route_contextualizer_n_heads: int,
        route_contextualizer_attn_dim: int,
        route_contextualizer_edge_node_ctx_dim: int,
        level_edge_attn_n_heads: int,
        level_edge_attn_edge_mlp_hidden: int,
        level_edge_attn_edge_mlp_n_blocks: int,
        level_edge_attn_ffn_d_hidden: int,
        level_edge_attn_rope_base: float,
        level_edge_attn_rope_scaling: float,
        root_readout_edge_rank: int,
        ngpt_alpha_initial: float,
        ngpt_alpha_initial_fraction: float,
        ngpt_alpha_maximum: float,
        global_ladder_tap_dim: int,
        level_edge_attn_bias_mlp_hidden: int,
        level_edge_attn_bias_mlp_n_blocks: int,
        merge_c_mlp_hidden: int,
        readout_leaf_context_layers: int,
        readout_leaf_context_n_heads: int,
        readout_leaf_context_attn_dim: int,
        readout_leaf_context_edge_node_ctx_dim: int,
        readout_leaf_context_summary_hidden: int,
        readout_leaf_context_mlp_hidden: int,
        readout_leaf_context_bias_hidden: int,
        readout_leaf_context_edge_ffn_hidden: int,
        readout_leaf_context_rope_base: float,
        readout_leaf_context_rope_scaling: float,
        two_hop_channels: int,
        tree_edge_fwl_channels: int,
        attn_impl: str,
        key: PRNGKeyArray,
    ):
        from .odd_ops import bounded_gain_logit

        alpha_init = bounded_gain_logit(
            ngpt_alpha_initial,
            max_gain=ngpt_alpha_maximum,
            init_fraction=ngpt_alpha_initial_fraction,
        )
        k_feat, k_tr, k_lf, k_mg, k_ro, k_ge, k_route, k_extras = jax.random.split(
            key, 8
        )
        k_leaf_ctx = jax.random.fold_in(k_route, 85897159)
        self.featurizer = SystemFeaturizer(
            key=k_feat,
            d_bond=feat_d_bond,
            n_heads=feat_n_heads,
            head_dim=feat_head_dim,
            n_global_q=feat_n_global_q,
            d_edge=feat_d_edge,
            d_hidden_edge=feat_edge_hidden_dim,
            polar_group_norm_tau=polar_group_norm_tau,
            polar_group_norm_bond_hidden=polar_group_norm_bond_hidden,
            polar_group_norm_n_bond_groups=polar_group_norm_n_bond_groups,
            polar_group_norm_d_bond_group=polar_group_norm_d_bond_group,
            polar_group_norm_n_zeeman_groups=polar_group_norm_n_zeeman_groups,
            polar_group_norm_d_zeeman_group=polar_group_norm_d_zeeman_group,
            zeeman_hidden_dim=feat_zeeman_hidden_dim,
            global_hidden_dim=feat_global_hidden_dim,
            combine_hidden_dim=feat_combine_hidden_dim,
            token_initial_scale=feat_token_initial_scale,
        )
        d_local = feat_n_heads * feat_head_dim
        d_global_feat = feat_n_global_q * feat_n_heads * feat_head_dim
        self.trunk = Trunk(
            d_e=d_e,
            n_heads=n_heads,
            n_layers=n_layers,
            n_edge=n_edge,
            d_local_in=d_local,
            d_edge_in=feat_d_edge,
            key=k_tr,
            gladder_d_g=d_global_feat,
            global_tap_dim=global_ladder_tap_dim,
            attn_impl=attn_impl,
            attn_dim=d_e_attn,
            attn_bias_hidden_dim=trunk_attn_bias_hidden_dim,
            ffn_hidden_dim=trunk_ffn_hidden_dim,
            edge_hidden_dim=trunk_edge_hidden_dim,
            edge_node_ctx_dim=trunk_edge_node_ctx_dim,
            two_hop_channels=two_hop_channels,
            two_hop_hidden_dim=trunk_two_hop_hidden_dim,
        )
        self.q_to_odd = BiasFreeLinear(4, d_o, key=jax.random.fold_in(k_tr, 2430463726))
        tree_d_c = d_c
        tree_d_r = d_r
        self.leaf = LeafBuilder(
            d_e=d_e,
            d_o=d_o,
            d_c=tree_d_c,
            d_r=tree_d_r,
            rank=rank,
            key=k_lf,
            d_g=global_d_g,
            leaf_hypernet_rank=merge_chain_hypernet_rank,
            d_m_merge=d_m_merge,
        )
        self.merge = MergeOp(
            d_r=tree_d_r,
            d_c=tree_d_c,
            key=k_mg,
            d_g=global_d_g,
            alpha_init=alpha_init,
            alpha_max=ngpt_alpha_maximum,
            d_m_merge=d_m_merge,
            merge_output_hypernet_rank=merge_chain_hypernet_rank,
            level_edge_attn_d_edge=feat_d_edge,
            level_edge_attn_n_heads=level_edge_attn_n_heads,
            level_edge_attn_attn_dim=d_c_attn,
            tree_edge_node_ctx_dim=tree_edge_node_ctx_dim,
            level_edge_attn_attn_impl=attn_impl,
            level_edge_attn_edge_mlp_hidden=level_edge_attn_edge_mlp_hidden,
            level_edge_attn_edge_mlp_n_blocks=level_edge_attn_edge_mlp_n_blocks,
            level_edge_attn_ffn_d_hidden=level_edge_attn_ffn_d_hidden,
            level_edge_attn_max_n=int(route_pointer_max_n),
            level_edge_attn_rope_base=float(level_edge_attn_rope_base),
            level_edge_attn_rope_scaling=float(level_edge_attn_rope_scaling),
            tree_edge_fwl_channels=tree_edge_fwl_channels,
            level_edge_attn_bias_mlp_hidden=level_edge_attn_bias_mlp_hidden,
            level_edge_attn_bias_mlp_n_blocks=level_edge_attn_bias_mlp_n_blocks,
            merge_c_mlp_hidden=merge_c_mlp_hidden,
        )
        self.readout = RootReadout(
            d_r=tree_d_r,
            key=k_ro,
            d_m_merge=d_m_merge,
            d_edge=feat_d_edge,
            edge_rank=root_readout_edge_rank,
            d_g=global_d_g,
            d_c=tree_d_c,
        )
        self.readout_leaf_context = PhysicalReadoutContext(
            d_e=d_e,
            d_edge=n_edge,
            n_layers=int(readout_leaf_context_layers),
            n_heads=int(readout_leaf_context_n_heads),
            summary_hidden=readout_leaf_context_summary_hidden,
            mlp_hidden=readout_leaf_context_mlp_hidden,
            bias_hidden=readout_leaf_context_bias_hidden,
            edge_ffn_hidden=readout_leaf_context_edge_ffn_hidden,
            attn_dim=readout_leaf_context_attn_dim,
            edge_node_ctx_dim=readout_leaf_context_edge_node_ctx_dim,
            attn_impl=attn_impl,
            rope_base=float(readout_leaf_context_rope_base),
            rope_scaling=float(readout_leaf_context_rope_scaling),
            gladder_d_g=d_global_feat,
            global_tap_dim=global_ladder_tap_dim,
            key=k_leaf_ctx,
        )
        self.route_contextualizer = RouterContext(
            d_e=d_e,
            d_edge=n_edge,
            n_layers=int(route_contextualizer_layers),
            n_heads=int(route_contextualizer_n_heads),
            mlp_hidden=readout_leaf_context_mlp_hidden,
            bias_hidden=readout_leaf_context_bias_hidden,
            edge_ffn_hidden=readout_leaf_context_edge_ffn_hidden,
            attn_dim=route_contextualizer_attn_dim,
            edge_node_ctx_dim=route_contextualizer_edge_node_ctx_dim,
            attn_impl=attn_impl,
            gladder_d_g=d_global_feat,
            global_tap_dim=global_ladder_tap_dim,
            key=jax.random.fold_in(k_route, 2802764542),
        )
        from .global_ladder import (
            EdgeRowColGlobalUpdate,
            GDescriptorPool,
            TreeGlobalUpdate,
        )

        _k_gl = jax.random.split(jax.random.fold_in(key, 25005), 2)
        self.gladder_post = EdgeRowColGlobalUpdate(
            d_global_feat,
            feat_d_edge,
            key=_k_gl[0],
            tag="gladder.post_trunk",
            tap_dim=global_ladder_tap_dim,
        )
        self.gladder_to_gemb_w = jax.random.normal(
            _k_gl[1], (d_global_feat, global_d_g)
        ) * d_global_feat ** (-0.5)
        self.gladder_to_gemb_b = jnp.zeros((global_d_g,))
        self.gladder_gemb_ln_s = jnp.ones((global_d_g,))
        _k_gt = jax.random.split(jax.random.fold_in(key, 25006), 3)
        self.gladder_tree_pool = GDescriptorPool(
            d_global_feat, tree_d_c, key=_k_gt[0], tag="gladder.tree.pool"
        )
        self.gladder_tree_update = TreeGlobalUpdate(
            d_global_feat,
            self.gladder_tree_pool.d_out,
            key=_k_gt[1],
            tag="gladder.tree.upd",
            tap_dim=global_ladder_tap_dim,
            alpha_init=alpha_init,
            alpha_max=ngpt_alpha_maximum,
        )
        self.gladder_tree_proj_w = jax.random.normal(
            _k_gt[2], (d_global_feat, global_d_g)
        ) * d_global_feat ** (-0.5)
        self.gladder_tree_proj_b = jnp.zeros((global_d_g,))
        _k_rp = jax.random.fold_in(key, 25010)
        self.gladder_root_proj_w = jax.random.normal(
            _k_rp, (d_global_feat, global_d_g)
        ) * d_global_feat ** (-0.5)
        self.gladder_root_proj_b = jnp.zeros((global_d_g,))
        self.gladder_root_ln_s = jnp.ones((global_d_g,))
        _k_gf = jax.random.split(jax.random.fold_in(key, 25008), 2)
        self.gladder_fork_phys = EdgeRowColGlobalUpdate(
            d_global_feat,
            feat_d_edge,
            key=_k_gf[0],
            tag="gladder.fork_phys",
            tap_dim=global_ladder_tap_dim,
        )
        self.gladder_fork_route = EdgeRowColGlobalUpdate(
            d_global_feat,
            feat_d_edge,
            key=_k_gf[1],
            tag="gladder.fork_route",
            tap_dim=global_ladder_tap_dim,
        )
        d_global_route = d_global_feat
        route_d_model = int(route_pointer_d_model)
        route_n_heads = int(route_pointer_n_heads)
        self.route_decoder = TreePrefixPointerMHSEA(
            d_in=d_e,
            d_edge=n_edge,
            d_global=d_global_route,
            d_model=route_d_model,
            n_heads=route_n_heads,
            attention_dim=route_pointer_attn_dim,
            pointer_score_dim=route_pointer_score_dim,
            candidate_hidden=route_pointer_candidate_hidden,
            summary_hidden=route_pointer_summary_hidden,
            ffn_hidden=route_pointer_ffn_hidden,
            global_tap_dim=global_ladder_tap_dim,
            alpha_init=alpha_init,
            alpha_max=ngpt_alpha_maximum,
            max_n=int(route_pointer_max_n),
            score_init_scale=float(route_pointer_score_init_scale),
            route_tree_prefix_layers=int(route_tree_prefix_layers),
            route_tree_prefix_candidate_layers=int(route_tree_prefix_candidate_layers),
            route_tree_prefix_merge_hidden=route_tree_prefix_merge_hidden,
            route_tree_prefix_post_prefix_suffix_layers=int(
                route_tree_prefix_post_prefix_suffix_layers
            ),
            route_decoder_attn_impl=attn_impl,
            rope_base=float(route_pointer_rope_base),
            rope_scaling=float(route_pointer_rope_scaling),
            key=k_route,
        )

    def _gladder_g_stream(self, edge, mask, g_in):
        return self.gladder_post(g_in.astype(edge.dtype), edge, mask)

    def _gladder_tree_refs(self):
        return (
            self.gladder_tree_pool,
            self.gladder_tree_update,
            self.gladder_tree_proj_w,
            self.gladder_tree_proj_b,
        )

    def _gladder_project(self, g_stream):
        from .tree import _tagged_dense, _tagged_rms_eqx_style

        structural_active = jnp.asarray(True)
        out = _tagged_dense(
            self.gladder_to_gemb_w,
            self.gladder_to_gemb_b,
            g_stream,
            tag_id="gladder.to_gemb",
            pathway="even",
            kfac_structural_mask=structural_active,
            kfac_scan_shared=False,
            kfac_repeat_ndim=0,
        )
        return _tagged_rms_eqx_style(
            self.gladder_gemb_ln_s,
            out,
            tag_id="gladder.gemb_ln",
            pathway="even",
            kfac_structural_mask=structural_active,
            kfac_scan_shared=False,
            kfac_repeat_ndim=0,
        )

    def _gladder_root_project(self, g_final):
        from .tree import _tagged_dense, _tagged_rms_eqx_style

        structural_active = jnp.asarray(True)
        out = _tagged_dense(
            self.gladder_root_proj_w,
            self.gladder_root_proj_b,
            g_final,
            tag_id="gladder.root_proj",
            pathway="even",
            kfac_structural_mask=structural_active,
            kfac_scan_shared=False,
            kfac_repeat_ndim=0,
        )
        return _tagged_rms_eqx_style(
            self.gladder_root_ln_s,
            out,
            tag_id="gladder.root_ln",
            pathway="even",
            kfac_structural_mask=structural_active,
            kfac_scan_shared=False,
            kfac_repeat_ndim=0,
        )

    def _contextualize_leaf_even_with_edge_g(self, e, edge, mask, bmask, g):
        return self.readout_leaf_context.with_edge(e, edge, mask, bmask, g=g)

    def route_features(
        self, ctx: SpinContext
    ) -> tuple[
        Float[Array, "n d_e"], Float[Array, "n n d_edge"], Float[Array, "d_global_feat"]
    ]:
        edge_feat, local_feat, global_feat = self.featurizer(
            ctx.J_double_prime, ctx.mask, ctx.h_prime
        )
        from .tree import _tree_sphere

        g_trunk = _tree_sphere(global_feat.astype(local_feat.dtype))
        e, edge, g_trunk = self.trunk(ctx, edge_feat, local_feat, g_trunk)
        g_route = self.gladder_post(g_trunk, edge, ctx.mask)
        e, edge, g_route = self.route_contextualizer.with_edge(
            e, edge, ctx.mask, ctx.bmask, g=g_route
        )
        g_route = self.gladder_fork_route(g_route, edge, ctx.bmask)
        return (e, edge, g_route.astype(e.dtype))

    def call_with_route_logprob(
        self,
        q: Float[Array, "n 4"],
        ctx: SpinContext,
        t: Float[Array, ""] | float = 0.0,
        *,
        tau: float = 1.0,
    ) -> tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
        t_val = jnp.asarray(t, dtype=q.dtype)
        first_orbit_ids = (
            ctx.route_quotient_node_key,
            ctx.route_quotient_edge_key,
            ctx.needs_fwl2,
        )
        edge_feat, local_feat, global_feat = self.featurizer(
            ctx.J_double_prime, ctx.mask, ctx.h_prime
        )
        from .tree import _tree_sphere

        g_trunk = _tree_sphere(global_feat.astype(local_feat.dtype))
        e, edge_route, g_trunk = self.trunk(ctx, edge_feat, local_feat, g_trunk)
        z = self.q_to_odd(
            q,
            pathway="odd",
            kfac_structural_mask=ctx.mask,
            kfac_repeat_ndim=1,
            kfac_context_primal_reused_over_walkers=False,
        )
        g_post = self.gladder_post(g_trunk, edge_route, ctx.mask)
        e_route, edge_route_for_policy, g_route = self.route_contextualizer.with_edge(
            e, edge_route, ctx.mask, ctx.bmask, g=g_post
        )
        g_route = self.gladder_fork_route(g_route, edge_route_for_policy, ctx.bmask)
        route_logp = self.route_decoder.logprob_identity(
            e_route,
            edge_route_for_policy,
            ctx.bmask,
            global_feat=g_route.astype(e.dtype),
            tau=tau,
            real_mask=ctx.mask,
            first_orbit_ids=first_orbit_ids,
        )
        e_leaf, edge_leaf, g_stream = self.readout_leaf_context.with_edge(
            e, edge_route, ctx.mask, ctx.bmask, g=g_post
        )
        g_stream = self.gladder_fork_phys(g_stream, edge_leaf, ctx.bmask)
        g_emb = self._gladder_project(g_stream)
        re, im = self._forward_leaf_to_readout(
            q=q,
            ctx=ctx,
            t=t_val,
            z=z,
            e_leaf=e_leaf,
            g_emb=g_emb,
            edge_leaf=edge_leaf,
            g_stream=g_stream,
        )
        return (re, im, route_logp)

    def compute_per_system_invariants(
        self, ctx: SpinContext, t: Float[Array, ""] | float = 0.0
    ) -> PerSystemInvariants:
        del t
        edge_feat, local_feat, global_feat = self.featurizer(
            ctx.J_double_prime, ctx.mask, ctx.h_prime
        )
        from .tree import _tree_sphere

        g_trunk = _tree_sphere(global_feat.astype(local_feat.dtype))
        e, edge_trunk, g_trunk = self.trunk(ctx, edge_feat, local_feat, g_trunk)
        g_stream = self.gladder_post(g_trunk, edge_trunk, ctx.mask)
        e_leaf, edge_leaf, g_stream = self.readout_leaf_context.with_edge(
            e, edge_trunk, ctx.mask, ctx.bmask, g=g_stream
        )
        g_stream = self.gladder_fork_phys(g_stream, edge_leaf, ctx.bmask)
        g_emb = self._gladder_project(g_stream)
        return PerSystemInvariants(
            g_emb=g_emb, e_leaf=e_leaf, edge_leaf=edge_leaf, g_stream=g_stream
        )

    def __call__(
        self,
        q: Float[Array, "n 4"],
        ctx: SpinContext,
        t: Float[Array, ""] | float = 0.0,
    ) -> tuple[Float[Array, ""], Float[Array, ""]]:
        t_val = jnp.asarray(t, dtype=q.dtype)
        edge_feat, local_feat, global_feat = self.featurizer(
            ctx.J_double_prime, ctx.mask, ctx.h_prime
        )
        from .tree import _tree_sphere

        g_trunk = _tree_sphere(global_feat.astype(local_feat.dtype))
        e, edge_trunk, g_trunk = self.trunk(ctx, edge_feat, local_feat, g_trunk)
        z = self.q_to_odd(
            q,
            pathway="odd",
            kfac_structural_mask=ctx.mask,
            kfac_repeat_ndim=1,
            kfac_context_primal_reused_over_walkers=False,
        )
        g_stream = self.gladder_post(g_trunk, edge_trunk, ctx.mask)
        e_leaf, edge_leaf, g_stream = self.readout_leaf_context.with_edge(
            e, edge_trunk, ctx.mask, ctx.bmask, g=g_stream
        )
        g_stream = self.gladder_fork_phys(g_stream, edge_leaf, ctx.bmask)
        g_emb = self._gladder_project(g_stream)
        return self._forward_leaf_to_readout(
            q=q,
            ctx=ctx,
            t=t_val,
            z=z,
            e_leaf=e_leaf,
            g_emb=g_emb,
            edge_leaf=edge_leaf,
            g_stream=g_stream,
        )

    def forward_with_precomputed(
        self,
        q: Float[Array, "n 4"],
        ctx: SpinContext,
        t: Float[Array, ""] | float = 0.0,
        *,
        precomputed: PerSystemInvariants,
    ) -> tuple[Float[Array, ""], Float[Array, ""]]:
        t_val = jnp.asarray(t, dtype=q.dtype)
        z = self.q_to_odd(
            q,
            pathway="odd",
            kfac_structural_mask=ctx.mask,
            kfac_repeat_ndim=1,
            kfac_context_primal_reused_over_walkers=False,
        )
        return self._forward_leaf_to_readout(
            q=q,
            ctx=ctx,
            t=t_val,
            z=z,
            e_leaf=precomputed.e_leaf,
            g_emb=precomputed.g_emb,
            edge_leaf=precomputed.edge_leaf,
            g_stream=precomputed.g_stream,
        )

    def _forward_leaf_to_readout(
        self, *, q, ctx, t, z, e_leaf, g_emb, edge_leaf, g_stream
    ):
        del q, t
        c_all, u_all, s_all = self.leaf(
            e_leaf,
            z,
            g_emb=g_emb,
            kfac_structural_mask=ctx.bmask,
            kfac_odd_structural_mask=ctx.mask,
        )
        u_all, log_rms = _normalize_leaf_carriers(u_all)
        s_all = s_all + log_rms.astype(s_all.dtype)
        mask = ctx.mask.astype(c_all.dtype)
        edges = edge_leaf.astype(c_all.dtype)
        reduced = balanced_tree_reduce_masked(
            c_all,
            u_all,
            s_all,
            mask,
            self.merge,
            g_emb,
            edges_init=edges,
            gladder=self._gladder_tree_refs(),
            g_stream0=g_stream,
        )
        reduced, final_stream = (reduced[:-1], reduced[-1])
        c_root, u_root, s_root, _, edge_root = reduced
        g_emb = self._gladder_root_project(final_stream)
        return self.readout(
            u_root,
            s_root,
            e_root=edge_root,
            g_emb=g_emb,
            c_root=c_root,
            kfac_structural_mask=jnp.any(ctx.bmask.astype(bool)),
        )
