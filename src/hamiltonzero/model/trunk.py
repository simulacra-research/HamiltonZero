# Copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray
from .context import SpinContext
from .fused_silu import fused_silu
from .odd_ops import BiasFreeLinear, Linear, MLP, UnnormalizedMLP, _RMS


class MultiHeadEvenAttention(eqx.Module):
    W_QKV: BiasFreeLinear
    W_O: BiasFreeLinear
    bias_mlp: UnnormalizedMLP
    ln_edge: _RMS
    n_heads: int = eqx.field(static=True)
    n_heads_kernel: int = eqx.field(static=True)
    d_head: int = eqx.field(static=True)
    d_attn: int = eqx.field(static=True)
    attn_impl: str = eqx.field(static=True)

    def __init__(
        self,
        d_e: int,
        n_heads: int,
        n_edge: int,
        *,
        key: PRNGKeyArray,
        attn_impl: str,
        n_layers: int,
        attn_dim: int,
        bias_hidden_dim: int,
    ):
        if n_heads < 1:
            raise ValueError(f"n_heads must be >= 1, got {n_heads}")
        d_attn = int(attn_dim)
        if d_attn < 1:
            raise ValueError(f"attn_dim must be positive or None, got {attn_dim}")
        if d_attn % n_heads != 0:
            raise ValueError(
                f"attention inner width must be divisible by n_heads: attn_dim={d_attn}, n_heads={n_heads}"
            )
        if attn_impl not in ("einsum", "mhsea_tuned"):
            raise ValueError("attn_impl must be 'einsum' or 'mhsea_tuned'")
        k_qkv, k_o, k_b, k_ln_edge = jax.random.split(key, 4)
        del k_ln_edge
        d_head = d_attn // n_heads
        n_heads_kernel = 2 * n_heads
        d_qkv_out = n_heads_kernel * d_head
        self.W_QKV = BiasFreeLinear(d_e, 3 * d_qkv_out, key=k_qkv)
        self.W_O = BiasFreeLinear(d_attn, d_e, key=k_o)
        self.bias_mlp = UnnormalizedMLP(
            n_edge,
            bias_hidden_dim,
            n_heads_kernel,
            key=k_b,
            n_blocks=1,
            inner_gain=float(n_layers) ** (-0.5),
        )
        self.ln_edge = _RMS(n_edge)
        self.n_heads = n_heads
        self.n_heads_kernel = n_heads_kernel
        self.d_head = d_head
        self.d_attn = d_attn
        self.attn_impl = attn_impl

    def __call__(
        self,
        e: Float[Array, "n d_e"],
        edge: Float[Array, "n n n_edge"],
        mask: Int[Array, "n"],
    ) -> Float[Array, "n d_e"]:
        n = e.shape[0]
        node_structural_mask = mask.astype(bool)
        pair_structural_mask = (
            node_structural_mask[:, None] & node_structural_mask[None, :]
        )
        qkv = self.W_QKV(
            e,
            pathway="even",
            kfac_structural_mask=node_structural_mask,
            kfac_repeat_ndim=1,
            kfac_context_primal_reused_over_walkers=True,
        ).reshape(n, 3, self.n_heads_kernel, self.d_head)
        Q, K, V = (qkv[:, 0], qkv[:, 1], qkv[:, 2])
        edge_pre = self.ln_edge(
            edge,
            pathway="even",
            kfac_structural_mask=pair_structural_mask,
            kfac_repeat_ndim=2,
            kfac_context_primal_reused_over_walkers=True,
        )
        coup_bias = self.bias_mlp(
            edge_pre,
            pathway="even",
            kfac_structural_mask=pair_structural_mask,
            kfac_repeat_ndim=2,
            kfac_context_primal_reused_over_walkers=True,
        )
        coup_bias = coup_bias / jnp.sqrt(
            jnp.asarray(self.d_head, dtype=coup_bias.dtype)
        )
        from .pallas_attention import (
            mhsea_tuned_edge_attention,
            reference_edge_attention,
        )

        if self.attn_impl == "einsum":
            out = reference_edge_attention(Q, K, V, coup_bias, mask)
        else:
            d_head_padded = max(16, self.d_head)
            pad_amount = d_head_padded - self.d_head
            scale = jnp.sqrt(jnp.asarray(d_head_padded / self.d_head, dtype=Q.dtype))
            Q_pad = jnp.concatenate(
                [
                    Q * scale,
                    jnp.zeros((n, self.n_heads_kernel, pad_amount), dtype=Q.dtype),
                ],
                axis=-1,
            )
            K_pad = jnp.concatenate(
                [K, jnp.zeros((n, self.n_heads_kernel, pad_amount), dtype=K.dtype)],
                axis=-1,
            )
            V_pad = jnp.concatenate(
                [V, jnp.zeros((n, self.n_heads_kernel, pad_amount), dtype=V.dtype)],
                axis=-1,
            )
            out = mhsea_tuned_edge_attention(Q_pad, K_pad, V_pad, coup_bias, mask)
            out = out[..., : self.d_head]
        gate_heads = out[:, : self.n_heads, :]
        value_heads = out[:, self.n_heads :, :]
        out = jax.nn.sigmoid(gate_heads) * value_heads
        out = out.reshape(n, -1)
        return self.W_O(
            out,
            pathway="even",
            kfac_structural_mask=node_structural_mask,
            kfac_repeat_ndim=1,
            kfac_context_primal_reused_over_walkers=True,
        )


class EvenFFN(eqx.Module):
    l1: Linear
    l2: Linear

    def __init__(self, d_e: int, d_hidden: int, *, key: PRNGKeyArray):
        k1, k2 = jax.random.split(key, 2)
        self.l1 = Linear(d_e, d_hidden, key=k1)
        self.l2 = Linear(d_hidden, d_e, key=k2)

    def __call__(
        self, e: Float[Array, "... d_e"], *, kfac_structural_mask=None
    ) -> Float[Array, "... d_e"]:
        kwargs = dict(
            kfac_structural_mask=kfac_structural_mask,
            kfac_repeat_ndim=1,
            kfac_context_primal_reused_over_walkers=True,
        )
        return self.l2(
            fused_silu(self.l1(e, pathway="even", **kwargs)), pathway="even", **kwargs
        )


class EdgeUpdateContextAware(eqx.Module):
    ln_edge: _RMS
    ln_even: _RMS
    node_ctx_proj: BiasFreeLinear
    ffn: MLP
    psi_L_in: Linear
    psi_L_out: Linear
    psi_R_in: Linear
    psi_R_out: Linear
    ln_path: _RMS
    two_hop_channels: int = eqx.field(static=True, default=0)
    edge_node_ctx_dim: int = eqx.field(static=True, default=0)

    def __init__(
        self,
        d_e: int,
        n_edge: int,
        *,
        key: PRNGKeyArray,
        d_hidden: int,
        n_layers: int,
        edge_node_ctx_dim: int,
        two_hop_channels: int,
        two_hop_hidden_dim: int,
    ):
        node_ctx_dim = int(edge_node_ctx_dim)
        if node_ctx_dim < 1:
            raise ValueError(
                f"edge_node_ctx_dim must be positive, got {edge_node_ctx_dim}"
            )
        self.ln_edge = _RMS(n_edge)
        self.ln_even = _RMS(d_e)
        self.node_ctx_proj = BiasFreeLinear(
            d_e, node_ctx_dim, key=jax.random.fold_in(key, 60782)
        )
        d_pair = n_edge + 2 * node_ctx_dim
        d_in = d_pair + two_hop_channels
        (
            k_ffn,
            k_psi_L_in,
            k_psi_L_out,
            k_psi_L_gate,
            k_psi_R_in,
            k_psi_R_out,
            k_psi_R_gate,
        ) = jax.random.split(key, 7)
        del k_psi_L_gate, k_psi_R_gate
        self.ffn = MLP(
            d_in,
            d_hidden,
            n_edge,
            key=k_ffn,
            n_blocks=1,
            inner_gain=float(n_layers) ** (-0.5),
        )
        self.edge_node_ctx_dim = int(node_ctx_dim)
        self.psi_L_in = Linear(d_pair, two_hop_hidden_dim, key=k_psi_L_in)
        self.psi_L_out = Linear(two_hop_hidden_dim, two_hop_channels, key=k_psi_L_out)
        self.psi_R_in = Linear(d_pair, two_hop_hidden_dim, key=k_psi_R_in)
        self.psi_R_out = Linear(two_hop_hidden_dim, two_hop_channels, key=k_psi_R_out)
        self.ln_path = _RMS(two_hop_channels)
        self.two_hop_channels = int(two_hop_channels)

    def __call__(
        self,
        edge: Float[Array, "n n n_edge"],
        even: Float[Array, "n d_e"],
        mask: Int[Array, "n"] | Float[Array, "n"],
    ) -> Float[Array, "n n n_edge"]:
        n = even.shape[0]
        node_structural_mask = mask.astype(bool)
        pair_structural_mask = (
            node_structural_mask[:, None] & node_structural_mask[None, :]
        )
        node_kfac = dict(
            kfac_structural_mask=node_structural_mask,
            kfac_repeat_ndim=1,
            kfac_context_primal_reused_over_walkers=True,
        )
        pair_kfac = dict(
            kfac_structural_mask=pair_structural_mask,
            kfac_repeat_ndim=2,
            kfac_context_primal_reused_over_walkers=True,
        )
        edge_ln = self.ln_edge(edge, pathway="even", **pair_kfac)
        even_ln = self.ln_even(even, pathway="even", **node_kfac)
        even_ctx = self.node_ctx_proj(even_ln, pathway="even", **node_kfac)
        d_ctx = even_ctx.shape[-1]
        even_i_b = jnp.broadcast_to(even_ctx[:, None, :], (n, n, d_ctx))
        even_j_b = jnp.broadcast_to(even_ctx[None, :, :], (n, n, d_ctx))
        pair_ij = jnp.concatenate([edge_ln, even_i_b, even_j_b], axis=-1)
        A = self._psi_apply(
            pair_ij,
            self.psi_L_in,
            self.psi_L_out,
            kfac_structural_mask=pair_structural_mask,
        )
        B = self._psi_apply(
            pair_ij,
            self.psi_R_in,
            self.psi_R_out,
            kfac_structural_mask=pair_structural_mask,
        )
        m = mask.astype(A.dtype)
        A = A * (m[:, None, None] * m[None, :, None])
        B = B * m[None, :, None]
        n_eff = jnp.maximum(jnp.sum(m), 1.0).astype(A.dtype)
        P = jnp.einsum("ikc,kjc->ijc", A, B) / jnp.sqrt(n_eff)
        p_ij = self.ln_path(P, pathway="even", **pair_kfac)
        cat = jnp.concatenate([pair_ij, p_ij], axis=-1)
        return self.ffn(cat, pathway="even", **pair_kfac)

    def _psi_apply(
        self,
        pair_ij: Float[Array, "n n d_pair"],
        l_in: Linear,
        l_out: Linear,
        *,
        kfac_structural_mask=None,
    ) -> Float[Array, "n n C"]:
        kfac_kwargs = dict(
            kfac_structural_mask=kfac_structural_mask,
            kfac_repeat_ndim=2,
            kfac_context_primal_reused_over_walkers=True,
        )
        hidden = fused_silu(l_in(pair_ij, pathway="even", **kfac_kwargs))
        return l_out(hidden, pathway="even", **kfac_kwargs)


class TransformerBlock(eqx.Module):
    edge_update_ctx: EdgeUpdateContextAware
    ln_attn: _RMS
    attn: MultiHeadEvenAttention
    ln_ffn: _RMS
    ffn: EvenFFN
    g_pool: "GDescriptorPool"
    g_update: "ResidualGlobalUpdate"
    g_ffn_proj_w: Float[Array, "d_gstream d_e"]
    residual_gain: float = eqx.field(static=True)

    def __init__(
        self,
        d_e: int,
        n_heads: int,
        n_edge: int,
        gladder_d_g: int,
        *,
        key: PRNGKeyArray,
        global_tap_dim: int,
        n_layers: int,
        attn_impl: str,
        attn_dim: int,
        attn_bias_hidden_dim: int,
        ffn_hidden_dim: int,
        edge_hidden_dim: int,
        edge_node_ctx_dim: int,
        two_hop_channels: int,
        two_hop_hidden_dim: int,
    ):
        k_e, k_a, k_f, k_o = jax.random.split(key, 4)
        del k_o
        self.edge_update_ctx = EdgeUpdateContextAware(
            d_e=d_e,
            n_edge=n_edge,
            key=k_e,
            d_hidden=edge_hidden_dim,
            n_layers=n_layers,
            edge_node_ctx_dim=edge_node_ctx_dim,
            two_hop_channels=two_hop_channels,
            two_hop_hidden_dim=two_hop_hidden_dim,
        )
        self.ln_attn = _RMS(d_e)
        self.attn = MultiHeadEvenAttention(
            d_e,
            n_heads,
            n_edge,
            key=k_a,
            attn_impl=attn_impl,
            n_layers=n_layers,
            attn_dim=attn_dim,
            bias_hidden_dim=attn_bias_hidden_dim,
        )
        self.ln_ffn = _RMS(d_e)
        self.ffn = EvenFFN(d_e, ffn_hidden_dim, key=k_f)
        from .global_ladder import GDescriptorPool, ResidualGlobalUpdate

        k_global = jax.random.split(jax.random.fold_in(key, 25009), 3)
        self.g_pool = GDescriptorPool(
            gladder_d_g, d_e, key=k_global[0], tag="gladder.trunk.pool"
        )
        self.g_update = ResidualGlobalUpdate(
            gladder_d_g,
            self.g_pool.d_out,
            key=k_global[1],
            tap_dim=global_tap_dim,
            tag="gladder.trunk.upd",
            residual_gain=float(n_layers) ** (-0.5),
        )
        self.g_ffn_proj_w = jax.random.normal(
            k_global[2], (gladder_d_g, d_e)
        ) * gladder_d_g ** (-0.5)
        self.residual_gain = float(n_layers) ** (-0.5)

    def _even_edge_step(
        self,
        e: Float[Array, "n d_e"],
        edge: Float[Array, "n n n_edge"],
        mask: Int[Array, "n"],
        g: Float[Array, "d_gstream"],
    ):
        node_structural_mask = mask.astype(bool)
        node_kfac = dict(
            kfac_structural_mask=node_structural_mask,
            kfac_repeat_ndim=1,
            kfac_context_primal_reused_over_walkers=True,
        )
        edge_update = self.residual_gain * self.edge_update_ctx(edge, e, mask)
        edge = edge + edge_update
        e_pre = self.ln_attn(e, pathway="even", **node_kfac)
        e = e + self.residual_gain * self.attn(e_pre, edge, mask)
        e_pre = self.ln_ffn(e, pathway="even", **node_kfac)
        from .tree import _tagged_dense_no_bias

        gg = _tagged_dense_no_bias(
            self.g_ffn_proj_w,
            g,
            tag_id="gladder.trunk.fproj",
            pathway="even",
            kfac_structural_mask=jnp.any(mask.astype(bool)),
            kfac_scan_shared=False,
            kfac_repeat_ndim=0,
            kfac_context_primal_reused_over_walkers=True,
        )
        e_pre = e_pre + gg[None, :].astype(e_pre.dtype)
        e = e + self.residual_gain * self.ffn(
            e_pre, kfac_structural_mask=node_structural_mask
        )
        system_active = jnp.any(mask.astype(bool))
        pooled = self.g_pool(
            g,
            e,
            mask,
            kfac_structural_mask=mask,
            kfac_update_mask=system_active,
            kfac_scan_shared=False,
            kfac_repeat_ndim=1,
            kfac_context_primal_reused_over_walkers=True,
        )
        g = self.g_update(
            g,
            pooled,
            kfac_structural_mask=system_active,
            kfac_scan_shared=False,
            kfac_context_primal_reused_over_walkers=True,
        )
        return (e, edge, g)

    def __call__(
        self,
        e: Float[Array, "n d_e"],
        edge: Float[Array, "n n n_edge"],
        mask: Int[Array, "n"],
        g: Float[Array, "d_gstream"],
    ):
        return self._even_edge_step(e, edge, mask, g=g)


class Trunk(eqx.Module):
    blocks: TransformerBlock

    def __init__(
        self,
        d_e: int,
        n_heads: int,
        n_layers: int,
        n_edge: int,
        d_local_in: int,
        d_edge_in: int,
        *,
        key: PRNGKeyArray,
        gladder_d_g: int,
        global_tap_dim: int,
        attn_impl: str,
        attn_dim: int,
        attn_bias_hidden_dim: int,
        ffn_hidden_dim: int,
        edge_hidden_dim: int,
        edge_node_ctx_dim: int,
        two_hop_channels: int,
        two_hop_hidden_dim: int,
    ):
        if d_local_in != d_e:
            raise ValueError(
                f"Trunk requires d_local_in (= feat_n_heads*feat_head_dim = {d_local_in}) == d_e (= {d_e}). Adjust featurizer config so the widths match."
            )
        if d_edge_in != n_edge:
            raise ValueError(
                f"Trunk requires d_edge_in (= feat_d_edge = {d_edge_in}) == n_edge (= {n_edge})."
            )
        k_odd, k_blocks = jax.random.split(key, 2)
        del k_odd
        block_keys = jax.random.split(k_blocks, n_layers)

        def make_block(k: PRNGKeyArray) -> TransformerBlock:
            return TransformerBlock(
                d_e,
                n_heads,
                n_edge,
                gladder_d_g,
                key=k,
                global_tap_dim=global_tap_dim,
                n_layers=n_layers,
                attn_impl=attn_impl,
                attn_dim=attn_dim,
                attn_bias_hidden_dim=attn_bias_hidden_dim,
                ffn_hidden_dim=ffn_hidden_dim,
                edge_hidden_dim=edge_hidden_dim,
                edge_node_ctx_dim=edge_node_ctx_dim,
                two_hop_channels=two_hop_channels,
                two_hop_hidden_dim=two_hop_hidden_dim,
            )

        block_list = [make_block(k) for k in block_keys]
        dynamic_static = [eqx.partition(block, eqx.is_array) for block in block_list]
        dynamic = [part for part, _ in dynamic_static]
        _, static_template = dynamic_static[0]
        stacked_dynamic = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *dynamic)
        self.blocks = eqx.combine(stacked_dynamic, static_template)

    def __call__(
        self,
        ctx: SpinContext,
        edge_feat: Float[Array, "n n d_edge_in"],
        local_feat: Float[Array, "n d_local_in"],
        g: Float[Array, "d_gstream"],
    ):
        e = local_feat.astype(jnp.float32)
        edge = edge_feat.astype(jnp.float32)
        dynamic, static = eqx.partition(self.blocks, eqx.is_array)

        def scan_body(carry, layer_dynamic):
            e_carry, edge_carry, g_carry = carry
            block = eqx.combine(layer_dynamic, static)
            e_carry, edge_carry, g_carry = block(
                e_carry,
                edge_carry,
                ctx.mask,
                g_carry,
            )
            return (e_carry, edge_carry, g_carry), None

        (e, edge, g), _ = jax.lax.scan(
            scan_body,
            (e, edge, g),
            dynamic,
        )
        return (e, edge, g)
