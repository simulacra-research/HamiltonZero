# Copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from hamiltonzero.model import (
    normalize_leaf_carriers,
    quadrilinear_merge,
    tagged_dense_no_bias,
)
from hamiltonzero.optim import register_scale_and_shift

from .execute import (
    _compiled_quadrilinear_merge,
    _factorized_apply,
    _gate_reference,
    _single,
)
from .types import EMPTY, MERGE, OPCODE_DTYPE, CompiledTree, SharedKernel, level_widths


def _scale_tag(y_flat, x_flat, scale_param, *, tag_id: str):
    return register_scale_and_shift(
        y_flat,
        x_flat,
        scale=scale_param,
        tag_id=tag_id,
    )


class CompiledFinetuneWaveFunction(eqx.Module):
    kernel: SharedKernel
    leaf_h: Array
    merge_h: Array
    readout_h: Array
    perm: Array
    inv_perm: Array
    leaf_real: Array
    opcodes: Array
    n_sites: int = eqx.field(static=True)
    r_leaf: int = eqx.field(static=True)
    r_merge: int = eqx.field(static=True)

    def leaf_h_rows(self) -> Array:
        return self.leaf_h.reshape(self.n_sites, self.r_leaf)

    def merge_h_level(self, level: int, width: int) -> Array:
        return self.merge_h[level, : width * self.r_merge].reshape(width, self.r_merge)

    @classmethod
    def from_compiled(cls, kernel: SharedKernel, tree: CompiledTree):
        leaf_h = _single(tree.leaf_h, "leaf conditioner")
        readout_h = _single(tree.readout_h, "readout conditioner")
        n_sites = int(tree.perm.shape[-1])
        widths = level_widths(n_sites)
        w_max = widths[0]
        r_leaf = int(leaf_h.shape[-1])
        r_merge = int(tree.merge_h[0].shape[-1])
        merge_rows = []
        opcode_rows = []
        for width, h_l, ops_l in zip(widths, tree.merge_h, tree.opcodes, strict=True):
            if h_l.shape[-2] != width or ops_l.shape[-1] != width:
                raise ValueError(
                    f"level width mismatch: expected {width}, got "
                    f"{h_l.shape[-2]}/{ops_l.shape[-1]}"
                )
            pad = w_max - width
            merge_rows.append(
                jnp.pad(h_l, ((0, pad), (0, 0))).reshape(-1) if pad else h_l.reshape(-1)
            )
            opcode_rows.append(
                jnp.pad(ops_l, (0, pad), constant_values=EMPTY) if pad else ops_l
            )
        return cls(
            kernel=kernel,
            leaf_h=leaf_h.reshape(-1),
            merge_h=jnp.stack(merge_rows),
            readout_h=readout_h,
            perm=tree.perm,
            inv_perm=tree.inv_perm,
            leaf_real=tree.leaf_real,
            opcodes=jnp.stack(opcode_rows).astype(OPCODE_DTYPE),
            n_sites=n_sites,
            r_leaf=r_leaf,
            r_merge=r_merge,
        )

    def as_compiled_tree(self) -> CompiledTree:
        widths = level_widths(self.n_sites)
        return CompiledTree(
            perm=self.perm,
            inv_perm=self.inv_perm,
            leaf_real=self.leaf_real,
            leaf_h=(self.leaf_h_rows(),),
            leaf_combiner_h=(),
            merge_h=tuple(
                self.merge_h_level(i, width) for i, width in enumerate(widths)
            ),
            opcodes=tuple(self.opcodes[i, :width] for i, width in enumerate(widths)),
            readout_h=(self.readout_h,),
            readout_combiner_h=(),
        )

    def route_q(self, q: Array) -> Array:
        return jnp.take(q, self.perm, axis=-2)

    def param_counts(self) -> dict:
        kernel_leaves = {
            "q_to_odd": self.kernel.q_to_odd.weight.size,
            "leaf_V": self.kernel.leaf_factors[0].V.size,
            "leaf_U": self.kernel.leaf_factors[0].U.size,
            "merge_T": self.kernel.merge_T.size,
            "merge_V": self.kernel.merge_factors[0].V.size,
            "merge_U": self.kernel.merge_factors[0].U.size,
            "readout_V": self.kernel.readout_factors[0].V.size,
            "readout_U": self.kernel.readout_factors[0].U.size,
        }
        tree_leaves = {
            "leaf_h": self.leaf_h.size,
            "merge_h": self.merge_h.size,
            "readout_h": self.readout_h.size,
        }
        return {
            "kernel": kernel_leaves,
            "tree": tree_leaves,
            "kernel_total": sum(kernel_leaves.values()),
            "tree_total": sum(tree_leaves.values()),
            "total": sum(kernel_leaves.values()) + sum(tree_leaves.values()),
        }

    def __call__(self, q: Array, ctx: Any = None, t: Any = 0.0):
        del ctx, t
        return self._forward_plain(q)

    def _forward_plain(self, q: Array):
        kernel = self.kernel
        odd_dtype = jnp.float32
        q_c = q if q.dtype == odd_dtype else q.astype(odd_dtype)
        weight = kernel.q_to_odd.weight
        weight = weight if weight.dtype == odd_dtype else weight.astype(odd_dtype)
        z = q_c @ weight
        u_raw = _factorized_apply(kernel.leaf_factors[0], self.leaf_h_rows(), z)
        u, log_rms = normalize_leaf_carriers(u_raw)
        s = jnp.zeros(u.shape[:-1], dtype=odd_dtype) + log_rms.astype(odd_dtype)
        widths = level_widths(self.n_sites)
        for level, width in enumerate(widths):
            h_level = self.merge_h_level(level, width)
            opcodes = self.opcodes[level, :width]
            u_left, u_right = u[..., 0::2, :], u[..., 1::2, :]
            s_left, s_right = s[..., 0::2], s[..., 1::2]
            raw = _compiled_quadrilinear_merge(kernel.merge_T, u_left, u_right)
            out = raw + _factorized_apply(kernel.merge_factors[0], h_level, raw)
            scale = jnp.sqrt(jnp.mean(out * out, axis=-1) + kernel.merge_eps)
            candidate_u = out / scale[..., None]
            candidate_s = s_left + s_right + jnp.log(scale)
            u = _gate_reference(
                candidate_u, u_left, u_right, opcodes, feature_axis=True
            )
            s = _gate_reference(
                candidate_s, s_left, s_right, opcodes, feature_axis=False
            )
        return self._readout(
            _factorized_apply(kernel.readout_factors[0], self.readout_h, u[..., 0, :]),
            s[..., 0],
        )

    def _readout(self, psi: Array, s_root: Array):
        psi_re, psi_im = psi[..., 0], psi[..., 1]
        log_abs = 0.5 * jnp.log(psi_re * psi_re + psi_im * psi_im) + s_root
        return log_abs, jnp.arctan2(psi_im, psi_re)

    def call_tagged(self, q: Array, ctx: Any = None, t: Any = 0.0):
        del ctx, t
        if q.ndim != 2:
            raise ValueError(
                f"call_tagged is per-walker: expected q [P, 4], got {q.shape}"
            )
        kernel = self.kernel
        odd_dtype = jnp.float32
        q_c = q if q.dtype == odd_dtype else q.astype(odd_dtype)
        real = self.leaf_real
        z = tagged_dense_no_bias(
            kernel.q_to_odd.weight,
            q_c,
            tag_id="compiled.q_to_odd",
            pathway="odd",
            kfac_structural_mask=real,
            kfac_scan_shared=False,
            kfac_repeat_ndim=1,
        )
        leaf = kernel.leaf_factors[0]
        vz = tagged_dense_no_bias(
            leaf.V,
            z,
            tag_id="compiled.leaf.V",
            pathway="odd",
            kfac_structural_mask=real,
            kfac_scan_shared=False,
            kfac_repeat_ndim=1,
        )
        vz_flat = vz.reshape(-1)
        mixed = _scale_tag(
            vz_flat * self.leaf_h,
            vz_flat,
            self.leaf_h,
            tag_id="compiled.leaf.h",
        ).reshape(vz.shape)
        u_raw = tagged_dense_no_bias(
            leaf.U,
            mixed,
            tag_id="compiled.leaf.U",
            pathway="odd",
            kfac_structural_mask=real,
            kfac_scan_shared=False,
            kfac_repeat_ndim=1,
        )
        u, log_rms = normalize_leaf_carriers(u_raw)
        s = jnp.zeros(u.shape[:-1], dtype=odd_dtype) + log_rms.astype(odd_dtype)
        merge = kernel.merge_factors[0]
        merge_T = kernel.merge_T
        merge_eps = kernel.merge_eps

        def level_body(carry, xs):
            u_buffer, s_buffer = carry
            h_level, opcodes = xs
            u_left, u_right = u_buffer[0::2], u_buffer[1::2]
            s_left, s_right = s_buffer[0::2], s_buffer[1::2]
            merge_rows = opcodes == MERGE
            raw = quadrilinear_merge(
                merge_T,
                u_left,
                u_right,
                tag_id="compiled.merge.T",
                pathway="odd",
                kfac_structural_mask=merge_rows,
                kfac_scan_shared=True,
                kfac_repeat_ndim=1,
            )
            vx = tagged_dense_no_bias(
                merge.V,
                raw,
                tag_id="compiled.merge.V",
                pathway="odd",
                kfac_structural_mask=merge_rows,
                kfac_scan_shared=True,
                kfac_repeat_ndim=1,
            )
            vx_flat = vx.reshape(-1)
            mixed_level = _scale_tag(
                vx_flat * h_level,
                vx_flat,
                h_level,
                tag_id="compiled.merge.h",
            ).reshape(vx.shape)
            correction = tagged_dense_no_bias(
                merge.U,
                mixed_level,
                tag_id="compiled.merge.U",
                pathway="odd",
                kfac_structural_mask=merge_rows,
                kfac_scan_shared=True,
                kfac_repeat_ndim=1,
            )
            out = raw + correction
            scale = jnp.sqrt(jnp.mean(out * out, axis=-1) + merge_eps)
            candidate_u = out / scale[..., None]
            candidate_s = s_left + s_right + jnp.log(scale)
            u_next = _gate_reference(
                candidate_u, u_left, u_right, opcodes, feature_axis=True
            )
            s_next = _gate_reference(
                candidate_s, s_left, s_right, opcodes, feature_axis=False
            )
            return (
                jnp.concatenate([u_next, jnp.zeros_like(u_next)], axis=0),
                jnp.concatenate([s_next, jnp.zeros_like(s_next)], axis=0),
            ), None

        (u_buffer, s_buffer), _ = jax.lax.scan(
            level_body,
            (u, s),
            (self.merge_h, self.opcodes),
        )
        readout = kernel.readout_factors[0]
        vr = tagged_dense_no_bias(
            readout.V,
            u_buffer[0],
            tag_id="compiled.readout.V",
            pathway="odd",
        )
        mixed_readout = _scale_tag(
            vr * self.readout_h,
            vr,
            self.readout_h,
            tag_id="compiled.readout.h",
        )
        psi = tagged_dense_no_bias(
            readout.U,
            mixed_readout,
            tag_id="compiled.readout.U",
            pathway="odd",
        )
        return self._readout(psi, s_buffer[0])


def _expand_stage(V, U, h_2d, new_rank: int, key):
    old_rank = V.shape[-1]
    if new_rank < old_rank:
        raise ValueError(f"cannot shrink rank {old_rank} -> {new_rank}")
    if new_rank == old_rank:
        return V, U, h_2d
    extra = new_rank - old_rank
    key_u, key_h = jax.random.split(key)
    v_new = jnp.zeros((*V.shape[:-1], extra), dtype=V.dtype)
    u_new = jnp.std(U) * jax.random.normal(key_u, (extra, *U.shape[1:]), dtype=U.dtype)
    h_new = jnp.std(h_2d) * jax.random.normal(
        key_h, (*h_2d.shape[:-1], extra), dtype=h_2d.dtype
    )
    return (
        jnp.concatenate([V, v_new], axis=-1),
        jnp.concatenate([U, u_new], axis=0),
        jnp.concatenate([h_2d, h_new], axis=-1),
    )


def expand_rank(
    model: CompiledFinetuneWaveFunction,
    *,
    leaf_rank: int,
    merge_rank: int,
    key,
) -> CompiledFinetuneWaveFunction:
    key_leaf, key_merge = jax.random.split(jnp.asarray(key), 3)[:2]
    kernel = model.kernel
    leaf = kernel.leaf_factors[0]
    merge = kernel.merge_factors[0]
    n_sites = model.n_sites
    max_width = n_sites // 2
    n_levels = model.merge_h.shape[0]
    leaf_h = model.leaf_h.reshape(n_sites, model.r_leaf)
    merge_h = model.merge_h.reshape(n_levels, max_width, model.r_merge)
    readout_h = model.readout_h
    r_leaf, r_merge = model.r_leaf, model.r_merge
    V, U, leaf_h = _expand_stage(leaf.V, leaf.U, leaf_h, int(leaf_rank), key_leaf)
    leaf = eqx.tree_at(lambda factor: (factor.V, factor.U), leaf, (V, U))
    r_leaf = int(leaf_rank)
    V, U, merge_h = _expand_stage(merge.V, merge.U, merge_h, int(merge_rank), key_merge)
    merge = eqx.tree_at(lambda factor: (factor.V, factor.U), merge, (V, U))
    r_merge = int(merge_rank)
    kernel = eqx.tree_at(
        lambda value: (
            value.leaf_factors,
            value.merge_factors,
        ),
        kernel,
        ((leaf,), (merge,)),
    )
    return CompiledFinetuneWaveFunction(
        kernel=kernel,
        leaf_h=leaf_h.reshape(-1),
        merge_h=merge_h.reshape(n_levels, -1),
        readout_h=readout_h,
        perm=model.perm,
        inv_perm=model.inv_perm,
        leaf_real=model.leaf_real,
        opcodes=model.opcodes,
        n_sites=n_sites,
        r_leaf=r_leaf,
        r_merge=r_merge,
    )


def compile_finetune_model(
    eager_model,
    ctx_row,
    *,
    leaf_rank: int,
    merge_rank: int,
    physical_perm,
    key,
) -> CompiledFinetuneWaveFunction:
    from .tree import compile_physical_tree_reference
    from .trunk import bind_shared_kernel, compile_shared_trunk

    shared_trunk = compile_shared_trunk(eager_model, ctx_row)
    n_sites = int(shared_trunk.real_mask.shape[-1])
    identity = jnp.arange(n_sites, dtype=jnp.int32)
    tree = compile_physical_tree_reference(eager_model, shared_trunk, identity)
    model = CompiledFinetuneWaveFunction.from_compiled(
        bind_shared_kernel(eager_model), tree
    )
    model = expand_rank(
        model,
        leaf_rank=leaf_rank,
        merge_rank=merge_rank,
        key=key,
    )
    physical_perm = jnp.asarray(physical_perm, dtype=jnp.int32)
    if physical_perm.shape != (n_sites,):
        raise ValueError(
            f"physical_perm must have shape {(n_sites,)}, got {physical_perm.shape}"
        )
    model = eqx.tree_at(
        lambda value: (value.perm, value.inv_perm),
        model,
        (
            physical_perm,
            jnp.argsort(physical_perm).astype(jnp.int32),
        ),
    )
    return model


def build_finetune_template_model(
    eager_model,
    n_sites: int,
    *,
    leaf_rank: int,
    merge_rank: int,
) -> CompiledFinetuneWaveFunction:
    from .trunk import bind_shared_kernel

    kernel = bind_shared_kernel(eager_model)
    widths = level_widths(int(n_sites))
    r_leaf = int(kernel.leaf_factors[0].V.shape[-1])
    r_merge = int(kernel.merge_factors[0].V.shape[-1])
    r_readout = int(kernel.readout_factors[0].V.shape[-1])
    tree = CompiledTree(
        perm=jnp.arange(n_sites, dtype=jnp.int32),
        inv_perm=jnp.arange(n_sites, dtype=jnp.int32),
        leaf_real=jnp.ones((n_sites,), dtype=jnp.bool_),
        leaf_h=(jnp.ones((n_sites, r_leaf), dtype=jnp.float32),),
        leaf_combiner_h=(),
        merge_h=tuple(
            jnp.ones((width, r_merge), dtype=jnp.float32) for width in widths
        ),
        opcodes=tuple(
            jnp.full((width,), MERGE, dtype=OPCODE_DTYPE) for width in widths
        ),
        readout_h=(jnp.ones((r_readout,), dtype=jnp.float32),),
        readout_combiner_h=(),
    )
    model = CompiledFinetuneWaveFunction.from_compiled(kernel, tree)
    return expand_rank(
        model,
        leaf_rank=leaf_rank,
        merge_rank=merge_rank,
        key=jax.random.PRNGKey(0),
    )
