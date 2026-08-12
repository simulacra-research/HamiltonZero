# Copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray
from hamiltonzero.model.odd_ops import BiasFreeLinear, Linear, _RMS
from hamiltonzero.model.fused_silu import fused_silu


class GroupLayerNorm(eqx.Module):
    weight: Float[Array, "n_groups d_group"]
    eps: float = eqx.field(static=True, default=1e-05)
    _use_id: str = eqx.field(static=True, default="")
    n_groups: int = eqx.field(static=True)
    d_group: int = eqx.field(static=True)

    def __init__(self, n_groups: int, d_group: int, eps: float = 1e-05):
        self.weight = jnp.ones((n_groups, d_group))
        self.eps = float(eps)
        self._use_id = ""
        self.n_groups = int(n_groups)
        self.d_group = int(d_group)

    def __call__(
        self,
        x: Float[Array, "... n_groups d_group"],
        *,
        pathway: str | None = None,
        kfac_structural_mask=None,
        kfac_repeat_ndim: int = 0,
        kfac_context_primal_reused_over_walkers: bool = False,
    ) -> Float[Array, "... n_groups d_group"]:
        if x.shape[-2] != self.n_groups or x.shape[-1] != self.d_group:
            raise ValueError(
                f"GroupLayerNorm expected trailing shape ({self.n_groups}, {self.d_group}), got {x.shape[-2:]}."
            )
        if pathway is None:
            pathway = "even"
        from hamiltonzero.model.fp32 import compute_dtype as _compute_dtype
        from hamiltonzero.model.tree import _kfac_name_kw

        out_cdtype = (
            _compute_dtype() if pathway in ("even", "hypernet_eside") else jnp.float32
        )
        stats_dtype = jnp.promote_types(jnp.float32, x.dtype)
        x_hi = x.astype(stats_dtype) if x.dtype != stats_dtype else x
        ms = jnp.mean(x_hi * x_hi, axis=-1, keepdims=True)
        normalized_hi = x_hi * jax.lax.rsqrt(ms + self.eps)
        normalized = (
            normalized_hi.astype(out_cdtype)
            if normalized_hi.dtype != out_cdtype
            else normalized_hi
        )
        w = (
            self.weight.astype(out_cdtype)
            if self.weight.dtype != out_cdtype
            else self.weight
        )
        y = normalized * w
        from hamiltonzero.optim.blocks import (
            register_structural_trailing_stacked_scale_and_shift,
        )

        return register_structural_trailing_stacked_scale_and_shift(
            y,
            normalized,
            kfac_structural_mask,
            self.weight,
            repeat_ndim=kfac_repeat_ndim,
            context_primal_reused_over_walkers=kfac_context_primal_reused_over_walkers,
            **_kfac_name_kw(self._use_id),
        )


class SystemFeaturizer(eqx.Module):
    bond_group_w1: Linear
    bond_group_w2: Linear
    bond_group_post: Linear
    bond_group_out: Linear
    bond_group_ln: GroupLayerNorm
    Q_col: Float[Array, "n_heads head_dim"]
    K_col: BiasFreeLinear
    V_col: BiasFreeLinear
    ln_bond: _RMS
    Q_row: Float[Array, "n_global_q n_heads head_dim"]
    K_row: BiasFreeLinear
    V_row: BiasFreeLinear
    ln_local_pre_row: _RMS
    ln_edge_cond: _RMS
    edge_cond_w1: Linear
    edge_cond_w2: Linear
    ln_edge_global: _RMS
    edge_global_film: Linear
    edge_residual_proj: BiasFreeLinear
    ln_edge_out: _RMS
    ln_local_out: _RMS
    ln_global_out: _RMS
    zeeman_w1: Linear
    zeeman_group_proj: Linear
    zeeman_group_ln: GroupLayerNorm
    zeeman_group_post: Linear
    zeeman_w2: Linear
    Q_row2: Float[Array, "n_global_q n_heads head_dim"]
    K_row2: BiasFreeLinear
    V_row2: BiasFreeLinear
    ln_local_pre_row2: _RMS
    ln_g_edge: _RMS
    ln_g_zee: _RMS
    global_w1: Linear
    global_w2: Linear
    ln_c1: _RMS
    ln_c2: _RMS
    ln_c3: _RMS
    combine_w1: Linear
    combine_w2: Linear
    tok_bond_gln: Float[Array, "gb db"]
    tok_zeeman_gln: Float[Array, "gz dz"]
    tok_bond_key: Float[Array, "d_bond"]
    tok_field_row: Float[Array, "d_local"]
    tok_field_global: Float[Array, "d_global"]
    tok_field_combine: Float[Array, "d_local"]
    _use_id_Q_col: str = eqx.field(static=True, default="")
    _use_id_Q_row: str = eqx.field(static=True, default="")
    _use_id_Q_row2: str = eqx.field(static=True, default="")
    _use_id_tok_bond_gln: str = eqx.field(static=True, default="")
    _use_id_tok_zeeman_gln: str = eqx.field(static=True, default="")
    _use_id_tok_bond_key: str = eqx.field(static=True, default="")
    _use_id_tok_field_row: str = eqx.field(static=True, default="")
    _use_id_tok_field_global: str = eqx.field(static=True, default="")
    _use_id_tok_field_combine: str = eqx.field(static=True, default="")
    d_bond: int = eqx.field(static=True)
    d_local: int = eqx.field(static=True)
    d_global: int = eqx.field(static=True)
    d_edge: int = eqx.field(static=True)
    n_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    n_global_q: int = eqx.field(static=True)
    d_hidden_edge: int = eqx.field(static=True)
    polar_group_norm_tau: float = eqx.field(static=True, default=0.001)
    polar_group_norm_bond_hidden: int = eqx.field(static=True, default=128)
    polar_group_norm_n_bond_groups: int = eqx.field(static=True, default=16)
    polar_group_norm_d_bond_group: int = eqx.field(static=True, default=16)
    polar_group_norm_n_zeeman_groups: int = eqx.field(static=True, default=16)
    polar_group_norm_d_zeeman_group: int = eqx.field(static=True, default=16)

    def __init__(
        self,
        *,
        key: PRNGKeyArray,
        d_bond: int,
        n_heads: int,
        head_dim: int,
        n_global_q: int,
        d_edge: int,
        d_hidden_edge: int,
        polar_group_norm_tau: float,
        polar_group_norm_bond_hidden: int,
        polar_group_norm_n_bond_groups: int,
        polar_group_norm_d_bond_group: int,
        polar_group_norm_n_zeeman_groups: int,
        polar_group_norm_d_zeeman_group: int,
        zeeman_hidden_dim: int,
        global_hidden_dim: int,
        combine_hidden_dim: int,
        token_initial_scale: float,
    ):
        d_local = n_heads * head_dim
        d_global = n_global_q * n_heads * head_dim
        n_h_kernel = 2 * n_heads
        d_local_kernel = n_h_kernel * head_dim
        edge_cond_in = d_bond + 2 * d_local
        self.polar_group_norm_tau = float(polar_group_norm_tau)
        self.polar_group_norm_bond_hidden = int(polar_group_norm_bond_hidden)
        self.polar_group_norm_n_bond_groups = int(polar_group_norm_n_bond_groups)
        self.polar_group_norm_d_bond_group = int(polar_group_norm_d_bond_group)
        self.polar_group_norm_n_zeeman_groups = int(polar_group_norm_n_zeeman_groups)
        self.polar_group_norm_d_zeeman_group = int(polar_group_norm_d_zeeman_group)
        keys = jax.random.split(key, 25)
        bond_group_dim = polar_group_norm_n_bond_groups * polar_group_norm_d_bond_group
        self.bond_group_w1 = Linear(20, polar_group_norm_bond_hidden, key=keys[0])
        self.bond_group_w2 = Linear(
            polar_group_norm_bond_hidden, bond_group_dim, key=keys[1]
        )
        self.bond_group_post = Linear(
            bond_group_dim, polar_group_norm_bond_hidden, key=keys[2]
        )
        self.bond_group_ln = GroupLayerNorm(
            polar_group_norm_n_bond_groups, polar_group_norm_d_bond_group
        )
        self.bond_group_out = Linear(polar_group_norm_bond_hidden, d_bond, key=keys[3])
        self.Q_col = jax.random.normal(keys[4], (n_h_kernel, head_dim)) * head_dim ** (
            -0.5
        )
        self.K_col = BiasFreeLinear(d_bond, d_local_kernel, key=keys[5])
        self.V_col = BiasFreeLinear(d_bond, d_local_kernel, key=keys[6])
        self.ln_bond = _RMS(d_bond)
        self.Q_row = jax.random.normal(
            keys[7], (n_global_q, n_h_kernel, head_dim)
        ) * head_dim ** (-0.5)
        self.K_row = BiasFreeLinear(d_local, d_local_kernel, key=keys[8])
        self.V_row = BiasFreeLinear(d_local, d_local_kernel, key=keys[9])
        self.ln_local_pre_row = _RMS(d_local)
        self.ln_edge_cond = _RMS(edge_cond_in)
        self.edge_cond_w1 = Linear(edge_cond_in, d_hidden_edge, key=keys[10])
        self.edge_cond_w2 = Linear(d_hidden_edge, d_edge, key=keys[11])
        self.ln_edge_global = _RMS(d_global)
        self.edge_global_film = Linear(d_global, 2 * d_edge, key=keys[12])
        self.edge_residual_proj = BiasFreeLinear(d_bond, d_edge, key=keys[13])
        zeeman_group_dim = (
            polar_group_norm_n_zeeman_groups * polar_group_norm_d_zeeman_group
        )
        self.zeeman_w1 = Linear(7, zeeman_hidden_dim, key=keys[14])
        self.zeeman_group_proj = Linear(
            zeeman_hidden_dim, zeeman_group_dim, key=keys[15]
        )
        self.zeeman_group_ln = GroupLayerNorm(
            polar_group_norm_n_zeeman_groups, polar_group_norm_d_zeeman_group
        )
        self.zeeman_group_post = Linear(
            zeeman_group_dim, zeeman_hidden_dim, key=keys[16]
        )
        self.zeeman_w2 = Linear(zeeman_hidden_dim, d_local, key=keys[17])
        self.Q_row2 = jax.random.normal(
            keys[18], (n_global_q, n_h_kernel, head_dim)
        ) * head_dim ** (-0.5)
        self.K_row2 = BiasFreeLinear(d_local, d_local_kernel, key=keys[19])
        self.V_row2 = BiasFreeLinear(d_local, d_local_kernel, key=keys[20])
        self.ln_local_pre_row2 = _RMS(d_local)
        self.ln_g_edge = _RMS(d_global)
        self.ln_g_zee = _RMS(d_global)
        self.global_w1 = Linear(2 * d_global + 8, global_hidden_dim, key=keys[21])
        self.global_w2 = Linear(global_hidden_dim, d_global, key=keys[22])
        combine_in = 2 * d_local + d_global
        self.ln_c1 = _RMS(d_local)
        self.ln_c2 = _RMS(d_local)
        self.ln_c3 = _RMS(d_global)
        self.combine_w1 = Linear(combine_in, combine_hidden_dim, key=keys[23])
        self.combine_w2 = Linear(combine_hidden_dim, d_local, key=keys[24])
        self.ln_edge_out = _RMS(d_edge)
        self.ln_local_out = _RMS(d_local)
        self.ln_global_out = _RMS(d_global)
        tok_keys = jax.random.split(jax.random.fold_in(key, 7389448), 6)
        self.tok_bond_gln = token_initial_scale * jax.random.normal(
            tok_keys[0], (polar_group_norm_n_bond_groups, polar_group_norm_d_bond_group)
        )
        self.tok_zeeman_gln = token_initial_scale * jax.random.normal(
            tok_keys[1],
            (polar_group_norm_n_zeeman_groups, polar_group_norm_d_zeeman_group),
        )
        self.tok_bond_key = token_initial_scale * jax.random.normal(
            tok_keys[2], (d_bond,)
        )
        self.tok_field_row = token_initial_scale * jax.random.normal(
            tok_keys[3], (d_local,)
        )
        self.tok_field_global = token_initial_scale * jax.random.normal(
            tok_keys[4], (d_global,)
        )
        self.tok_field_combine = token_initial_scale * jax.random.normal(
            tok_keys[5], (d_local,)
        )
        self.d_bond = d_bond
        self.d_local = d_local
        self.d_global = d_global
        self.d_edge = d_edge
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_global_q = n_global_q
        self.d_hidden_edge = d_hidden_edge

    def _polar_split(
        self, x: Float[Array, "... d"]
    ) -> tuple[Float[Array, "... 1"], Float[Array, "... d"]]:
        x_f32 = x.astype(jnp.float32)
        sq = jnp.sum(x_f32 * x_f32, axis=-1, keepdims=True)
        tau = jnp.asarray(self.polar_group_norm_tau, dtype=jnp.float32)
        r = jnp.sqrt(sq).astype(x.dtype)
        direction = (x_f32 * jax.lax.rsqrt(sq + tau * tau)).astype(x.dtype)
        return (r, direction)

    def _bond_input(
        self, J_double_prime: Float[Array, "n n 10"]
    ) -> Float[Array, "n n 20"]:
        J9 = J_double_prime[..., :9]
        eye = J_double_prime[..., 9:]
        rJ, uJ = self._polar_split(J9)
        return jnp.concatenate([J9, rJ, uJ, eye], axis=-1)

    def _zeeman_input(self, h_prime: Float[Array, "n 3"]) -> Float[Array, "n d_in"]:
        rh, uh = self._polar_split(h_prime)
        return jnp.concatenate([h_prime, rh, uh], axis=-1)

    def _embed_bonds(
        self,
        J_double_prime: Float[Array, "n n 10"],
        *,
        pathway: str,
        structural_mask: Float[Array, "n n"] | None = None,
    ) -> Float[Array, "n n d_bond"]:
        if structural_mask is None:
            structural_mask = jnp.ones(J_double_prime.shape[:-1], dtype=bool)
        z = fused_silu(
            self._dense_structural(
                self.bond_group_w1,
                self._bond_input(J_double_prime),
                structural_mask,
                repeat_ndim=2,
                pathway=pathway,
            )
        )
        z = self._dense_structural(
            self.bond_group_w2, z, structural_mask, repeat_ndim=2, pathway=pathway
        )
        pair_shape = J_double_prime.shape[:-1]
        z = z.reshape(
            *pair_shape,
            self.polar_group_norm_n_bond_groups,
            self.polar_group_norm_d_bond_group,
        )
        z = self.bond_group_ln(
            z,
            pathway=pathway,
            kfac_structural_mask=structural_mask,
            kfac_repeat_ndim=2,
            kfac_context_primal_reused_over_walkers=True,
        )
        from hamiltonzero.optim.spin_blocks import register_small_full

        present = jnp.any(J_double_prime[..., :9] != 0, axis=-1) | (
            J_double_prime[..., 9] > 0.5
        )
        tok = register_small_full(
            self.tok_bond_gln, tag_id=self._use_id_tok_bond_gln
        ).astype(z.dtype)
        z = jnp.where(present[..., None, None], z, tok)
        z = z.reshape(*pair_shape, -1)
        z = fused_silu(
            self._dense_structural(
                self.bond_group_post, z, structural_mask, repeat_ndim=2, pathway=pathway
            )
        )
        return fused_silu(
            self._dense_structural(
                self.bond_group_out, z, structural_mask, repeat_ndim=2, pathway=pathway
            )
        )

    def _embed_zeeman(
        self,
        h_prime: Float[Array, "n 3"],
        *,
        pathway: str,
        structural_mask: Float[Array, "n"] | None = None,
    ) -> Float[Array, "n d_local"]:
        if structural_mask is None:
            structural_mask = jnp.ones(h_prime.shape[:-1], dtype=bool)
        z = fused_silu(
            self._dense_structural(
                self.zeeman_w1,
                self._zeeman_input(h_prime),
                structural_mask,
                repeat_ndim=1,
                pathway=pathway,
            )
        )
        z = self._dense_structural(
            self.zeeman_group_proj, z, structural_mask, repeat_ndim=1, pathway=pathway
        )
        n = h_prime.shape[0]
        z = z.reshape(
            n,
            self.polar_group_norm_n_zeeman_groups,
            self.polar_group_norm_d_zeeman_group,
        )
        z = self.zeeman_group_ln(
            z,
            pathway=pathway,
            kfac_structural_mask=structural_mask,
            kfac_repeat_ndim=1,
            kfac_context_primal_reused_over_walkers=True,
        )
        from hamiltonzero.optim.spin_blocks import register_small_full

        present = jnp.any(h_prime != 0, axis=-1)
        tok = register_small_full(
            self.tok_zeeman_gln, tag_id=self._use_id_tok_zeeman_gln
        ).astype(z.dtype)
        z = jnp.where(present[:, None, None], z, tok)
        z = z.reshape(n, -1)
        z = fused_silu(
            self._dense_structural(
                self.zeeman_group_post,
                z,
                structural_mask,
                repeat_ndim=1,
                pathway=pathway,
            )
        )
        return self._dense_structural(
            self.zeeman_w2, z, structural_mask, repeat_ndim=1, pathway=pathway
        )

    def _bond_norm_gated(
        self,
        bond_emb: Float[Array, "n n d_bond"],
        J_double_prime: Float[Array, "n n 10"],
        *,
        pathway: str,
        structural_mask: Float[Array, "n n"] | None = None,
    ) -> Float[Array, "n n d_bond"]:
        from hamiltonzero.optim.spin_blocks import register_small_full

        present = jnp.any(J_double_prime[..., :9] != 0, axis=-1) | (
            J_double_prime[..., 9] > 0.5
        )
        if structural_mask is None:
            structural_mask = jnp.ones(present.shape, dtype=bool)
        tok = register_small_full(
            self.tok_bond_key, tag_id=self._use_id_tok_bond_key
        ).astype(bond_emb.dtype)
        return jnp.where(
            present[..., None],
            self._norm_structural(
                self.ln_bond, bond_emb, structural_mask, repeat_ndim=2, pathway=pathway
            ),
            tok,
        )

    @staticmethod
    def _dense_structural(lin, x, structural_mask, *, repeat_ndim: int, pathway: str):
        return lin(
            x,
            pathway=pathway,
            kfac_structural_mask=structural_mask,
            kfac_repeat_ndim=repeat_ndim,
            kfac_context_primal_reused_over_walkers=True,
        )

    @staticmethod
    def _norm_structural(norm, x, structural_mask, *, repeat_ndim: int, pathway: str):
        return norm(
            x,
            pathway=pathway,
            kfac_structural_mask=structural_mask,
            kfac_repeat_ndim=repeat_ndim,
            kfac_context_primal_reused_over_walkers=True,
        )

    def _dense_bare(
        self, lin: Linear, x: Float[Array, "..."], *, pathway: str
    ) -> Float[Array, "..."]:
        from hamiltonzero.model.tree import _tagged_dense

        return _tagged_dense(
            lin.weight,
            lin.bias,
            x,
            tag_id=getattr(lin, "_use_id", ""),
            pathway=pathway,
            kfac_structural_mask=jnp.asarray(True),
            kfac_scan_shared=False,
            kfac_repeat_ndim=0,
            kfac_context_primal_reused_over_walkers=True,
        )

    @staticmethod
    def _eval_tiles(n: int, tile_size: int):
        tile_size = int(tile_size)
        if tile_size < 1:
            raise ValueError(f"tile_size must be positive, got {tile_size}")
        return tuple((slice(j, min(j + tile_size, n)) for j in range(0, n, tile_size)))

    def eval_embed_local_rows(
        self,
        J_double_prime_rows: Float[Array, "r n 10"],
        row_mask: Float[Array, "r"],
        mask: Float[Array, "n"],
        *,
        tile_size: int = 128,
    ) -> tuple[Float[Array, "r n d_bond"], Float[Array, "r d_local"]]:
        if J_double_prime_rows.ndim != 3 or J_double_prime_rows.shape[-1] != 10:
            raise ValueError(
                f"J_double_prime_rows must have shape [R,N,10], got {J_double_prime_rows.shape}"
            )
        r, n = J_double_prime_rows.shape[:2]
        if row_mask.shape != (r,) or mask.shape != (n,):
            raise ValueError(
                f"row_mask/mask must match J rows/columns, got {row_mask.shape}, {mask.shape}, {J_double_prime_rows.shape}"
            )
        from hamiltonzero.model.fp32 import compute_dtype as _compute_dtype
        from hamiltonzero.optim.spin_blocks import register_small_full

        PW = "even"
        dtype = _compute_dtype()
        mr = row_mask.astype(dtype)
        mc = mask.astype(dtype)
        pair_mask = mr[:, None] * mc[None, :]
        bond_tiles = []
        tiles = self._eval_tiles(n, tile_size)
        for sl in tiles:
            pm = pair_mask[:, sl]
            bond_tile = self._embed_bonds(
                J_double_prime_rows[:, sl], pathway=PW, structural_mask=pm
            )
            bond_tiles.append(bond_tile * pm[..., None])
        bond_emb = jnp.concatenate(bond_tiles, axis=1)
        n_h_kernel = 2 * self.n_heads
        scale = self.head_dim ** (-0.5)
        Q_col = register_small_full(self.Q_col, tag_id=self._use_id_Q_col).astype(dtype)
        score_max = jnp.full((r, n_h_kernel), -jnp.inf, dtype=dtype)
        for sl in tiles:
            pm = pair_mask[:, sl]
            bn = self._bond_norm_gated(
                bond_emb[:, sl],
                J_double_prime_rows[:, sl],
                pathway=PW,
                structural_mask=pm,
            )
            k_tile = self._dense_structural(
                self.K_col, bn, pm, repeat_ndim=2, pathway=PW
            ).reshape(r, sl.stop - sl.start, n_h_kernel, self.head_dim)
            scores = jnp.einsum("hd,ijhd->ijh", Q_col, k_tile) * scale
            scores = jnp.where(
                pm[..., None] > 0,
                scores,
                jnp.asarray(-1000000000.0, dtype=scores.dtype),
            )
            score_max = jnp.maximum(score_max, jnp.max(scores, axis=1))
        denom = jnp.zeros((r, n_h_kernel), dtype=dtype)
        numer = jnp.zeros((r, n_h_kernel, self.head_dim), dtype=dtype)
        for sl in tiles:
            width = sl.stop - sl.start
            pm = pair_mask[:, sl]
            bn = self._bond_norm_gated(
                bond_emb[:, sl],
                J_double_prime_rows[:, sl],
                pathway=PW,
                structural_mask=pm,
            )
            k_tile = self._dense_structural(
                self.K_col, bn, pm, repeat_ndim=2, pathway=PW
            ).reshape(r, width, n_h_kernel, self.head_dim)
            scores = jnp.einsum("hd,ijhd->ijh", Q_col, k_tile) * scale
            scores = jnp.where(
                pm[..., None] > 0,
                scores,
                jnp.asarray(-1000000000.0, dtype=scores.dtype),
            )
            weight = jnp.exp(scores - score_max[:, None, :])
            v_tile = self._dense_structural(
                self.V_col, bn, pm, repeat_ndim=2, pathway=PW
            ).reshape(r, width, n_h_kernel, self.head_dim)
            denom = denom + jnp.sum(weight, axis=1)
            numer = numer + jnp.einsum("ijh,ijhd->ihd", weight, v_tile)
        col_out = numer / jnp.maximum(
            denom[..., None], jnp.asarray(1e-30, dtype=numer.dtype)
        )
        gate = col_out[:, : self.n_heads, :]
        val = col_out[:, self.n_heads :, :]
        local_desc_rows = (jax.nn.sigmoid(gate) * val).reshape(r, self.d_local) * mr[
            :, None
        ]
        return (bond_emb, local_desc_rows)

    def eval_jh_stats_rows(
        self,
        J_double_prime_rows: Float[Array, "r n 10"],
        row_mask: Float[Array, "r"],
        mask: Float[Array, "n"],
        *,
        row_indices: Int[Array, "r"] | None = None,
    ) -> tuple[Float[Array, ""], Float[Array, ""]]:
        r, n = J_double_prime_rows.shape[:2]
        if row_indices is None:
            if r != n:
                raise ValueError("row_indices is required when R != N")
            row_indices = jnp.arange(n, dtype=jnp.int32)
        row_indices = jnp.asarray(row_indices, dtype=jnp.int32)
        col_indices = jnp.arange(n, dtype=jnp.int32)
        active = (
            row_mask[:, None].astype(J_double_prime_rows.dtype)
            * mask[None, :].astype(J_double_prime_rows.dtype)
            * (row_indices[:, None] != col_indices[None, :]).astype(
                J_double_prime_rows.dtype
            )
        )
        norm2 = jnp.sum(jnp.square(J_double_prime_rows[..., :9]), axis=-1)
        return (jnp.sum(norm2 * active), jnp.sum(active))

    def eval_edge_rows(
        self,
        *,
        bond_emb_rows: Float[Array, "r n d_bond"],
        local_rows: Float[Array, "r d_local"],
        local_final_all: Float[Array, "n d_local"],
        global_feat: Float[Array, "d_global"],
        row_indices: Int[Array, "r"],
        mask: Float[Array, "n"],
        tile_size: int = 128,
    ) -> Float[Array, "r n d_edge"]:
        PW = "even"
        r, n = bond_emb_rows.shape[:2]
        row_indices = jnp.asarray(row_indices, dtype=jnp.int32)
        if row_indices.shape != (r,):
            raise ValueError("row_indices must have shape [R]")
        if local_rows.shape != (r, self.d_local):
            raise ValueError("local_rows must have shape [R,d_local]")
        if local_final_all.shape != (n, self.d_local):
            raise ValueError("local_final_all must have shape [N,d_local]")
        if mask.shape != (n,):
            raise ValueError("mask must have shape [N]")
        from hamiltonzero.model.fp32 import compute_dtype as _compute_dtype

        dtype = _compute_dtype()
        m = mask.astype(dtype)
        mr = m[row_indices]
        pair_mask = mr[:, None] * m[None, :]
        global_mask = jnp.asarray(True)
        film_in = self._norm_structural(
            self.ln_edge_global, global_feat, global_mask, repeat_ndim=0, pathway=PW
        )
        film = self._dense_bare(self.edge_global_film, film_in, pathway=PW)
        gamma, beta = jnp.split(film, 2, axis=-1)
        gamma = 0.1 * jnp.tanh(gamma)
        beta = 0.1 * beta
        edge_tiles = []
        for sl in self._eval_tiles(n, tile_size):
            pm = pair_mask[:, sl]
            bond = bond_emb_rows[:, sl]
            width = sl.stop - sl.start
            li = jnp.broadcast_to(local_rows[:, None, :], (r, width, self.d_local))
            lj = jnp.broadcast_to(
                local_final_all[sl][None, :, :], (r, width, self.d_local)
            )
            edge_in = jnp.concatenate([bond, li, lj], axis=-1)
            edge_in = self._norm_structural(
                self.ln_edge_cond, edge_in, pm, repeat_ndim=2, pathway=PW
            )
            core = self._dense_structural(
                self.edge_cond_w2,
                fused_silu(
                    self._dense_structural(
                        self.edge_cond_w1, edge_in, pm, repeat_ndim=2, pathway=PW
                    )
                ),
                pm,
                repeat_ndim=2,
                pathway=PW,
            )
            update_edge = core * (1.0 + gamma[None, None, :]) + beta[None, None, :]
            residual = self._dense_structural(
                self.edge_residual_proj, bond, pm, repeat_ndim=2, pathway=PW
            )
            edge_tile = (
                self._norm_structural(
                    self.ln_edge_out,
                    residual + update_edge,
                    pm,
                    repeat_ndim=2,
                    pathway=PW,
                )
                * pm[..., None]
            )
            edge_tiles.append(edge_tile)
        return jnp.concatenate(edge_tiles, axis=1)

    def eval_finalize_local_rows(
        self,
        *,
        J_double_prime_rows: Float[Array, "r n 10"],
        local_desc_rows: Float[Array, "r d_local"],
        local_desc_all: Float[Array, "n d_local"],
        row_indices: Int[Array, "r"],
        mask: Float[Array, "n"],
        h_prime: Float[Array, "n 3"],
        jh_stats: tuple[Float[Array, ""], Float[Array, ""]] | None = None,
    ) -> tuple[Float[Array, "r d_local"], Float[Array, "d_global"]]:
        from hamiltonzero.model.fp32 import compute_dtype as _compute_dtype
        from hamiltonzero.optim.spin_blocks import register_small_full

        PW = "even"
        dtype = _compute_dtype()
        r, n = J_double_prime_rows.shape[:2]
        row_indices = jnp.asarray(row_indices, dtype=jnp.int32)
        if row_indices.shape != (r,):
            raise ValueError("row_indices must have shape [R]")
        if local_desc_rows.shape != (r, self.d_local):
            raise ValueError("local_desc_rows has the wrong shape")
        if local_desc_all.shape != (n, self.d_local) or mask.shape != (n,):
            raise ValueError("local_desc_all/mask must have global width N")
        m = mask.astype(dtype)
        mr = m[row_indices]
        n_h_kernel = 2 * self.n_heads
        scale = self.head_dim ** (-0.5)
        global_mask = jnp.asarray(True)

        def row_descriptor(x, ln_pre, Q_arr, K_lin, V_lin, q_use_id, present=None):
            x_norm = self._norm_structural(ln_pre, x, m, repeat_ndim=1, pathway=PW)
            if present is not None:
                tok_row = register_small_full(
                    self.tok_field_row, tag_id=self._use_id_tok_field_row
                ).astype(x_norm.dtype)
                x_norm = jnp.where(present[:, None], x_norm, tok_row)
            qr = register_small_full(Q_arr, tag_id=q_use_id).astype(dtype)
            kr = self._dense_structural(
                K_lin, x_norm, m, repeat_ndim=1, pathway=PW
            ).reshape(n, n_h_kernel, self.head_dim)
            vr = self._dense_structural(
                V_lin, x_norm, m, repeat_ndim=1, pathway=PW
            ).reshape(n, n_h_kernel, self.head_dim)
            sc = jnp.einsum("qhd,ihd->qhi", qr, kr) * scale
            sc = jnp.where(
                m[None, None, :] > 0, sc, jnp.asarray(-1000000000.0, dtype=sc.dtype)
            )
            a = jax.nn.softmax(sc, axis=-1)
            out = jnp.einsum("qhi,ihd->qhd", a, vr)
            out = jax.nn.sigmoid(out[:, : self.n_heads, :]) * out[:, self.n_heads :, :]
            return out.reshape(self.d_global)

        h_prime = h_prime.astype(dtype)
        local_prime_all = (
            self._embed_zeeman(h_prime, pathway=PW, structural_mask=m) * m[:, None]
        )
        local_prime_rows = local_prime_all[row_indices]
        h_present = jnp.any(h_prime != 0, axis=-1)
        g_edge = row_descriptor(
            local_desc_all,
            self.ln_local_pre_row,
            self.Q_row,
            self.K_row,
            self.V_row,
            self._use_id_Q_row,
        )
        g_zee = row_descriptor(
            local_prime_all,
            self.ln_local_pre_row2,
            self.Q_row2,
            self.K_row2,
            self.V_row2,
            self._use_id_Q_row2,
            present=h_present,
        )
        if jh_stats is None:
            if r != n:
                raise ValueError(
                    "row-sharded featurization requires the psum result from eval_jh_stats_rows"
                )
            jh_stats = self.eval_jh_stats_rows(
                J_double_prime_rows, mr, m, row_indices=row_indices
            )
        sum_j2, count_j = jh_stats
        s_j2 = sum_j2 / jnp.maximum(count_j, 1.0)
        s_h2 = jnp.sum(jnp.sum(jnp.square(h_prime), axis=-1) * m) / jnp.maximum(
            jnp.sum(m), 1.0
        )

        def safe_sqrt(x):
            ok = x > 0
            return jnp.where(ok, jnp.sqrt(jnp.where(ok, x, 1.0)), 0.0)

        rj = safe_sqrt(s_j2)
        rh = safe_sqrt(s_h2)
        ok = rj + rh > 0
        theta = jnp.arctan2(jnp.where(ok, rh, 0.0), jnp.where(ok, rj, 1.0))
        jh_features = jnp.stack(
            [
                jnp.log1p(rj),
                jnp.log1p(rh),
                jnp.sin(2.0 * theta),
                jnp.cos(2.0 * theta),
                jnp.sin(4.0 * theta),
                jnp.cos(4.0 * theta),
                jnp.sin(8.0 * theta),
                jnp.cos(8.0 * theta),
            ]
        ).astype(dtype)
        global_cat = jnp.concatenate(
            [
                self._norm_structural(
                    self.ln_g_edge, g_edge, global_mask, repeat_ndim=0, pathway=PW
                ),
                jnp.where(
                    jnp.any(h_present),
                    self._norm_structural(
                        self.ln_g_zee, g_zee, global_mask, repeat_ndim=0, pathway=PW
                    ),
                    register_small_full(
                        self.tok_field_global, tag_id=self._use_id_tok_field_global
                    ).astype(dtype),
                ),
                jh_features,
            ],
            axis=-1,
        )
        global_raw = self._dense_bare(
            self.global_w2,
            fused_silu(self._dense_bare(self.global_w1, global_cat, pathway=PW)),
            pathway=PW,
        )
        global_feat = self._norm_structural(
            self.ln_global_out, global_raw, global_mask, repeat_ndim=0, pathway=PW
        )
        combine = jnp.concatenate(
            [
                jnp.where(
                    h_present[row_indices, None],
                    self._norm_structural(
                        self.ln_c1, local_prime_rows, mr, repeat_ndim=1, pathway=PW
                    ),
                    register_small_full(
                        self.tok_field_combine, tag_id=self._use_id_tok_field_combine
                    ).astype(dtype),
                ),
                self._norm_structural(
                    self.ln_c2, local_desc_rows, mr, repeat_ndim=1, pathway=PW
                ),
                jnp.broadcast_to(
                    self._norm_structural(
                        self.ln_c3, global_feat, global_mask, repeat_ndim=0, pathway=PW
                    ),
                    (r, self.d_global),
                ),
            ],
            axis=-1,
        )
        update = self._dense_structural(
            self.combine_w2,
            fused_silu(
                self._dense_structural(
                    self.combine_w1, combine, mr, repeat_ndim=1, pathway=PW
                )
            ),
            mr,
            repeat_ndim=1,
            pathway=PW,
        )
        local_rows = (
            self._norm_structural(
                self.ln_local_out,
                local_prime_rows + update,
                mr,
                repeat_ndim=1,
                pathway=PW,
            )
            * mr[:, None]
        )
        return (local_rows, global_feat)

    def eval_streamed(
        self,
        J_double_prime: Float[Array, "n n 10"],
        mask: Float[Array, "n"],
        h_prime: Float[Array, "n 3"],
        *,
        tile_size: int = 128,
    ):
        n = J_double_prime.shape[0]
        idx = jnp.arange(n, dtype=jnp.int32)
        bond, local_desc = self.eval_embed_local_rows(
            J_double_prime, mask, mask, tile_size=tile_size
        )
        stats = self.eval_jh_stats_rows(J_double_prime, mask, mask, row_indices=idx)
        local, global_feat = self.eval_finalize_local_rows(
            J_double_prime_rows=J_double_prime,
            local_desc_rows=local_desc,
            local_desc_all=local_desc,
            row_indices=idx,
            mask=mask,
            h_prime=h_prime,
            jh_stats=stats,
        )
        edge = self.eval_edge_rows(
            bond_emb_rows=bond,
            local_rows=local,
            local_final_all=local,
            global_feat=global_feat,
            row_indices=idx,
            mask=mask,
            tile_size=tile_size,
        )
        return (edge, local, global_feat)

    def __call__(
        self,
        J_double_prime: Float[Array, "n n 10"],
        mask: Float[Array, "n"],
        h_prime: Float[Array, "n 3"],
    ) -> tuple[
        Float[Array, "n n d_edge"], Float[Array, "n d_local"], Float[Array, "d_global"]
    ]:
        PW = "even"
        n = J_double_prime.shape[0]
        from hamiltonzero.model.fp32 import compute_dtype as _compute_dtype

        dtype = _compute_dtype()
        m = mask.astype(dtype)
        pair_mask = m[:, None] * m[None, :]
        global_mask = jnp.asarray(True)
        head_dim = self.head_dim
        n_heads = self.n_heads
        n_global_q = self.n_global_q
        scale = head_dim ** (-0.5)
        bond_emb = self._embed_bonds(
            J_double_prime, pathway=PW, structural_mask=pair_mask
        )
        bond_emb = bond_emb * pair_mask[..., None]
        n_h_kernel = 2 * n_heads
        from hamiltonzero.optim.spin_blocks import register_small_full

        bond_norm = self._bond_norm_gated(
            bond_emb, J_double_prime, pathway=PW, structural_mask=pair_mask
        )
        Q_col = register_small_full(self.Q_col, tag_id=self._use_id_Q_col).astype(dtype)
        K = self._dense_structural(
            self.K_col, bond_norm, pair_mask, repeat_ndim=2, pathway=PW
        ).reshape(n, n, n_h_kernel, head_dim)
        V = self._dense_structural(
            self.V_col, bond_norm, pair_mask, repeat_ndim=2, pathway=PW
        ).reshape(n, n, n_h_kernel, head_dim)
        scores = jnp.einsum("hd,ijhd->ijh", Q_col, K) * scale
        scores = jnp.where(
            pair_mask[..., None] > 0,
            scores,
            jnp.asarray(-1000000000.0, dtype=scores.dtype),
        )
        attn = jax.nn.softmax(scores, axis=1)
        col_out = jnp.einsum("ijh,ijhd->ihd", attn, V)
        gate = col_out[:, :n_heads, :]
        val = col_out[:, n_heads:, :]
        col_out = jax.nn.sigmoid(gate) * val
        local_i_raw = col_out.reshape(n, self.d_local) * m[:, None]
        h_prime = h_prime.astype(dtype)

        def _row_descriptor(x, ln_pre, Q_arr, K_lin, V_lin, q_use_id, present=None):
            x_norm = self._norm_structural(ln_pre, x, m, repeat_ndim=1, pathway=PW)
            if present is not None:
                tok_row = register_small_full(
                    self.tok_field_row, tag_id=self._use_id_tok_field_row
                ).astype(x_norm.dtype)
                x_norm = jnp.where(present[:, None], x_norm, tok_row)
            Qr = register_small_full(Q_arr, tag_id=q_use_id).astype(dtype)
            Kr = self._dense_structural(
                K_lin, x_norm, m, repeat_ndim=1, pathway=PW
            ).reshape(n, n_h_kernel, head_dim)
            Vr = self._dense_structural(
                V_lin, x_norm, m, repeat_ndim=1, pathway=PW
            ).reshape(n, n_h_kernel, head_dim)
            sc = jnp.einsum("qhd,ihd->qhi", Qr, Kr) * scale
            sc = jnp.where(
                m[None, None, :] > 0, sc, jnp.asarray(-1000000000.0, dtype=sc.dtype)
            )
            a = jax.nn.softmax(sc, axis=-1)
            ro = jnp.einsum("qhi,ihd->qhd", a, Vr)
            ro = jax.nn.sigmoid(ro[:, :n_heads, :]) * ro[:, n_heads:, :]
            return ro.reshape(self.d_global)

        local_desc_i = local_i_raw
        local_i_prime = self._embed_zeeman(h_prime, pathway=PW, structural_mask=m)
        local_i_prime = local_i_prime * m[:, None]
        h_present = jnp.any(h_prime != 0, axis=-1)
        g_edge = _row_descriptor(
            local_desc_i,
            self.ln_local_pre_row,
            self.Q_row,
            self.K_row,
            self.V_row,
            self._use_id_Q_row,
        )
        g_zee = _row_descriptor(
            local_i_prime,
            self.ln_local_pre_row2,
            self.Q_row2,
            self.K_row2,
            self.V_row2,
            self._use_id_Q_row2,
            present=h_present,
        )
        off_diag = pair_mask * (1.0 - jnp.eye(n, dtype=dtype))
        bond_magnitude2 = jnp.sum(
            jnp.square(J_double_prime[..., :9].astype(dtype)), axis=-1
        )
        mean_j2 = jnp.sum(bond_magnitude2 * off_diag) / jnp.maximum(
            jnp.sum(off_diag), 1.0
        )
        mean_h2 = jnp.sum(jnp.sum(jnp.square(h_prime), axis=-1) * m) / jnp.maximum(
            jnp.sum(m), 1.0
        )

        def _safe_sqrt(x):
            ok = x > 0
            return jnp.where(ok, jnp.sqrt(jnp.where(ok, x, 1.0)), 0.0)

        rj = _safe_sqrt(mean_j2)
        rh = _safe_sqrt(mean_h2)
        nonzero = rj + rh > 0
        theta = jnp.arctan2(jnp.where(nonzero, rh, 0.0), jnp.where(nonzero, rj, 1.0))
        jh_features = jnp.stack(
            [
                jnp.log1p(rj),
                jnp.log1p(rh),
                jnp.sin(2.0 * theta),
                jnp.cos(2.0 * theta),
                jnp.sin(4.0 * theta),
                jnp.cos(4.0 * theta),
                jnp.sin(8.0 * theta),
                jnp.cos(8.0 * theta),
            ]
        ).astype(dtype)
        global_cat = jnp.concatenate(
            [
                self._norm_structural(
                    self.ln_g_edge, g_edge, global_mask, repeat_ndim=0, pathway=PW
                ),
                jnp.where(
                    jnp.any(h_present),
                    self._norm_structural(
                        self.ln_g_zee, g_zee, global_mask, repeat_ndim=0, pathway=PW
                    ),
                    register_small_full(
                        self.tok_field_global, tag_id=self._use_id_tok_field_global
                    ).astype(dtype),
                ),
                jh_features,
            ],
            axis=-1,
        )
        global_raw = self._dense_bare(
            self.global_w2,
            fused_silu(self._dense_bare(self.global_w1, global_cat, pathway=PW)),
            pathway=PW,
        )
        global_feat = self._norm_structural(
            self.ln_global_out, global_raw, global_mask, repeat_ndim=0, pathway=PW
        )
        combine_in = jnp.concatenate(
            [
                jnp.where(
                    h_present[:, None],
                    self._norm_structural(
                        self.ln_c1, local_i_prime, m, repeat_ndim=1, pathway=PW
                    ),
                    register_small_full(
                        self.tok_field_combine, tag_id=self._use_id_tok_field_combine
                    ).astype(dtype),
                ),
                self._norm_structural(
                    self.ln_c2, local_desc_i, m, repeat_ndim=1, pathway=PW
                ),
                jnp.broadcast_to(
                    self._norm_structural(
                        self.ln_c3, global_feat, global_mask, repeat_ndim=0, pathway=PW
                    )[None, :],
                    (n, self.d_global),
                ),
            ],
            axis=-1,
        )
        local_update = self._dense_structural(
            self.combine_w2,
            fused_silu(
                self._dense_structural(
                    self.combine_w1, combine_in, m, repeat_ndim=1, pathway=PW
                )
            ),
            m,
            repeat_ndim=1,
            pathway=PW,
        )
        local_i = self._norm_structural(
            self.ln_local_out,
            local_i_prime + local_update,
            m,
            repeat_ndim=1,
            pathway=PW,
        )
        local_i = local_i * m[:, None]
        local_i_for_i = jnp.broadcast_to(local_i[:, None, :], (n, n, self.d_local))
        local_i_for_j = jnp.broadcast_to(local_i[None, :, :], (n, n, self.d_local))
        edge_cond_in = jnp.concatenate(
            [bond_emb, local_i_for_i, local_i_for_j], axis=-1
        )
        edge_cond_norm = self._norm_structural(
            self.ln_edge_cond, edge_cond_in, pair_mask, repeat_ndim=2, pathway=PW
        )
        edge_core = self._dense_structural(
            self.edge_cond_w2,
            fused_silu(
                self._dense_structural(
                    self.edge_cond_w1,
                    edge_cond_norm,
                    pair_mask,
                    repeat_ndim=2,
                    pathway=PW,
                )
            ),
            pair_mask,
            repeat_ndim=2,
            pathway=PW,
        )
        ln_global_for_film = self._norm_structural(
            self.ln_edge_global, global_feat, global_mask, repeat_ndim=0, pathway=PW
        )
        film = self._dense_bare(self.edge_global_film, ln_global_for_film, pathway=PW)
        gamma, beta = jnp.split(film, 2, axis=-1)
        gamma = 0.1 * jnp.tanh(gamma)
        beta = 0.1 * beta
        edge_update = edge_core * (1.0 + gamma[None, None, :]) + beta[None, None, :]
        bond_emb_proj = self._dense_structural(
            self.edge_residual_proj, bond_emb, pair_mask, repeat_ndim=2, pathway=PW
        )
        edge_ij = bond_emb_proj + edge_update
        edge_ij = self._norm_structural(
            self.ln_edge_out, edge_ij, pair_mask, repeat_ndim=2, pathway=PW
        )
        edge_ij = edge_ij * pair_mask[..., None]
        return (edge_ij, local_i, global_feat)
