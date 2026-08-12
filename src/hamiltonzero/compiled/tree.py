# Copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from hamiltonzero.model import (
    edge_merge_masked,
    tagged_dense,
    tagged_rms_eqx_style,
    tree_active_clock_depth,
    tree_depth_count_features,
    tree_sphere,
)

from .types import (
    CARRY_LEFT,
    CARRY_RIGHT,
    EMPTY,
    MERGE,
    CompiledTree,
    LadderProjectionKernel,
    PhysicalCompilerKernel,
)


def bind_physical_compiler_kernel(model: Any) -> PhysicalCompilerKernel:
    leaf = eqx.tree_at(lambda x: (x.P_u.V, x.P_u.U), model.leaf, (None, None))
    merge = eqx.tree_at(
        lambda x: (x.T, x.output_hypernet.V, x.output_hypernet.U),
        model.merge,
        (None, None, None),
    )
    readout = eqx.tree_at(
        lambda x: (x.output_hypernet.V, x.output_hypernet.U),
        model.readout,
        (None, None),
    )
    return PhysicalCompilerKernel(
        contextualizer=model.readout_leaf_context,
        global_fork=model.gladder_fork_phys,
        leaf=leaf,
        merge=merge,
        readout=readout,
        leaf_projection=LadderProjectionKernel(
            model.gladder_to_gemb_w,
            model.gladder_to_gemb_b,
            model.gladder_gemb_ln_s,
        ),
        tree_pool=model.gladder_tree_pool,
        tree_update=model.gladder_tree_update,
        tree_projection_weight=model.gladder_tree_proj_w,
        tree_projection_bias=model.gladder_tree_proj_b,
        root_projection=LadderProjectionKernel(
            model.gladder_root_proj_w,
            model.gladder_root_proj_b,
            model.gladder_root_ln_s,
        ),
    )


def _project_global(
    projection: LadderProjectionKernel,
    value,
    *,
    dense_tag: str,
    norm_tag: str,
):
    structural_active = jnp.asarray(True)
    out = tagged_dense(
        projection.weight,
        projection.bias,
        value,
        tag_id=dense_tag,
        pathway="even",
        kfac_structural_mask=structural_active,
        kfac_scan_shared=False,
        kfac_repeat_ndim=0,
    )
    return tagged_rms_eqx_style(
        projection.norm_scale,
        out,
        tag_id=norm_tag,
        pathway="even",
        kfac_structural_mask=structural_active,
        kfac_scan_shared=False,
        kfac_repeat_ndim=0,
    )


def compile_context_only_reduction(
    *,
    merge,
    c_leaf,
    leaf_real,
    g_emb,
    edges,
    structural_mask,
    gladder,
    n_total=None,
    clock_depth=None,
    initial_counts=None,
    level_offset: int = 0,
    feature_n_levels=None,
):
    c = jnp.asarray(c_leaf)
    m = jnp.asarray(leaf_real, dtype=c.dtype)
    if c.shape[0] != m.shape[0]:
        raise ValueError("c_leaf and leaf_real widths differ")
    n = c.shape[0]
    if n == 0 or n & (n - 1):
        raise ValueError(f"context-only width must be a power of two, got {n}")
    k = jnp.asarray(structural_mask, dtype=c.dtype)
    if k.shape != m.shape:
        raise ValueError("structural_mask and leaf_real widths differ")
    n_total = jnp.sum(m) if n_total is None else n_total
    feature_n_levels = (
        tree_active_clock_depth(m) if feature_n_levels is None else feature_n_levels
    )
    clock_depth = tree_active_clock_depth(k) if clock_depth is None else clock_depth
    counts = m if initial_counts is None else jnp.asarray(initial_counts, dtype=m.dtype)
    if counts.shape != m.shape:
        raise ValueError("initial_counts and leaf_real widths differ")
    candidates = []
    carried = []
    depth_levels = []
    opcode_levels = []
    g_curr = g_emb
    e_curr = edges
    e_curr = tree_sphere(e_curr)
    level = int(level_offset)
    while c.shape[0] > 1:
        c_a, c_b = c[0::2], c[1::2]
        m_a, m_b = m[0::2], m[1::2]
        k_a, k_b = k[0::2], k[1::2]
        pair_count = c_a.shape[0]
        pair_idx = jnp.arange(pair_count, dtype=jnp.int32)
        both_struct = k_a * k_b
        pair_base = jnp.maximum(
            jnp.sum((k_a + k_b - k_a * k_b).astype(jnp.int32)),
            jnp.asarray(2, dtype=jnp.int32),
        )
        cnt_a, cnt_b = counts[0::2], counts[1::2]
        depth = tree_depth_count_features(
            cnt_a, cnt_b, n_total, level, feature_n_levels, c.dtype
        )
        counts = cnt_a + cnt_b
        depth_levels.append(depth)
        d_edge = e_curr.shape[-1]
        e_pairs = e_curr.reshape(pair_count, 2, pair_count, 2, d_edge)
        sibling_lr = e_pairs[pair_idx, 0, pair_idx, 1]
        sibling_rl = e_pairs[pair_idx, 1, pair_idx, 0]
        level_active = jnp.any(both_struct.astype(bool))
        g_level = g_curr
        g_level = tagged_dense(
            gladder[2],
            gladder[3],
            g_curr,
            tag_id="gladder.tree.proj",
            pathway="even",
            kfac_structural_mask=level_active,
            kfac_scan_shared=False,
            kfac_repeat_ndim=0,
        )

        def candidate_one(ca, cb, elr, erl, dep, pidx, struct_active):
            return merge.context_candidate(
                ca,
                cb,
                g_level,
                sibling_edge_lr=elr,
                sibling_edge_rl=erl,
                level_idx=jnp.int32(level),
                pair_idx=pidx,
                pair_base=pair_base,
                clock_depth=clock_depth,
                depth_feats=dep,
                kfac_structural_mask=struct_active,
                kfac_g_structural_mask=level_active,
                kfac_scan_shared=False,
            )

        candidate = jax.vmap(candidate_one)(
            c_a,
            c_b,
            sibling_lr,
            sibling_rl,
            depth,
            pair_idx,
            both_struct,
        )
        candidates.append(candidate)
        gate_m_a, gate_m_b = k_a, k_b
        gate_both = gate_m_a * gate_m_b
        gate_a = gate_m_a * (1.0 - gate_m_b)
        gate_b = (1.0 - gate_m_a) * gate_m_b
        c = (
            gate_both[:, None] * candidate
            + gate_a[:, None] * c_a
            + gate_b[:, None] * c_b
        )
        m = m_a + m_b - m_a * m_b
        k = k_a + k_b - k_a * k_b
        opcode_levels.append(
            jnp.where(
                m_a.astype(jnp.bool_),
                jnp.where(m_b.astype(jnp.bool_), MERGE, CARRY_LEFT),
                jnp.where(m_b.astype(jnp.bool_), CARRY_RIGHT, EMPTY),
            ).astype(jnp.uint8)
        )
        attn_mask = both_struct
        d_edge = e_curr.shape[-1]
        e_blocks = e_curr.reshape(pair_count, 2, pair_count, 2, d_edge)
        e00, e01 = e_blocks[:, 0, :, 0], e_blocks[:, 0, :, 1]
        e10, e11 = e_blocks[:, 1, :, 0], e_blocks[:, 1, :, 1]

        def edge_row(e0, e1, e2, e3, ma, mb, ka, kb, ca, cb):
            return jax.vmap(
                lambda x0, x1, x2, x3, mqa, mqb, kqa, kqb, cqa, cqb: edge_merge_masked(
                    x0,
                    x1,
                    x2,
                    x3,
                    ma,
                    mb,
                    mqa,
                    mqb,
                    ca,
                    cb,
                    cqa,
                    cqb,
                    merge.edge_merge,
                    k_2i=ka,
                    k_2i1=kb,
                    k_2j=kqa,
                    k_2j1=kqb,
                    kfac_scan_shared=False,
                )[0]
            )(e0, e1, e2, e3, m_a, m_b, k_a, k_b, c_a, c_b)

        e_new = jax.vmap(edge_row)(e00, e01, e10, e11, m_a, m_b, k_a, k_b, c_a, c_b)
        e_new = merge.tree_edge_fwl.apply_residual(
            e_new, c, attn_mask, kfac_scan_shared=False
        )
        edge_keep = (both_struct[:, None] * both_struct[None, :]).astype(bool)
        e_curr = jnp.where(edge_keep[..., None], e_new, e00)
        e_curr = jnp.where(edge_keep[..., None], tree_sphere(e_curr), e00)
        c_skip = c
        c = merge.level_edge_attn(
            c,
            e_curr,
            attn_mask,
            level_idx=jnp.int32(level),
            kfac_scan_shared=False,
        )
        c = jnp.where(attn_mask.astype(bool)[:, None], tree_sphere(c), c_skip)
        carried.append(c)
        level_mask = k
        update_active = jnp.any(attn_mask.astype(bool))
        pool_structural_mask = level_mask.astype(c.dtype) * update_active.astype(
            c.dtype
        )
        pooled = gladder[0](
            g_curr,
            c,
            level_mask.astype(c.dtype),
            kfac_structural_mask=pool_structural_mask,
            kfac_update_mask=update_active,
            kfac_scan_shared=False,
            kfac_repeat_ndim=1,
        )
        g_curr = gladder[1](
            g_curr,
            pooled,
            update_mask=update_active,
            kfac_structural_mask=update_active,
            kfac_scan_shared=False,
        )
        level += 1
    e_root = e_curr[0, 0]
    merge_h = tuple(
        compile_merge_h(
            merge,
            candidate,
            depth_levels[i],
        )
        for i, candidate in enumerate(candidates)
    )
    return {
        "c_candidate": tuple(candidates),
        "c_carried": tuple(carried),
        "depth_features": tuple(depth_levels),
        "merge_h": merge_h,
        "opcodes": tuple(opcode_levels),
        "c_root": c[0],
        "e_root": e_root,
        "g_final": g_curr,
    }


def project_conditioner(context, hypernet):
    return jnp.matmul(context, hypernet.W_h)


def leaf_context(leaf_builder, e_leaf, g_emb):
    g_broadcast = jnp.broadcast_to(g_emb, e_leaf.shape[:-1] + g_emb.shape)
    return jnp.concatenate((e_leaf, g_broadcast), axis=-1)


def compile_target_leaf_h(leaf_builder, e_leaf, g_emb):
    context = leaf_context(leaf_builder, e_leaf, g_emb)
    return (project_conditioner(context, leaf_builder.P_u),)


def merge_context(merge, c_p_candidate, depth_features):
    return jnp.concatenate(
        (c_p_candidate, depth_features.astype(c_p_candidate.dtype)), axis=-1
    )


def compile_merge_h(merge, c_p_candidate, depth_features):
    return project_conditioner(
        merge_context(merge, c_p_candidate, depth_features), merge.output_hypernet
    )


def readout_context(readout, e_root, c_root, g_emb):
    e_norm = readout.ln_e(e_root, pathway="even")
    return jnp.concatenate(
        (e_norm, c_root.astype(e_norm.dtype), g_emb.astype(e_norm.dtype)), axis=-1
    )


def compile_target_readout_h(readout, e_root, c_root, g_emb):
    context = readout_context(readout, e_root, c_root, g_emb)
    return (project_conditioner(context, readout.output_hypernet),)


def classify_merge_opcodes(leaf_real):
    active = jnp.asarray(leaf_real, dtype=jnp.bool_)
    n = active.shape[0]
    if n == 0 or n & (n - 1):
        raise ValueError(f"leaf_real width must be a nonzero power of two, got {n}")
    levels = []
    while active.shape[0] > 1:
        left = active[0::2]
        right = active[1::2]
        opcode = jnp.where(
            left,
            jnp.where(right, MERGE, CARRY_LEFT),
            jnp.where(right, CARRY_RIGHT, EMPTY),
        ).astype(jnp.uint8)
        levels.append(opcode)
        active = left | right
    return tuple(levels)


def assemble_compiled_tree(*, perm, leaf_real, boundaries) -> CompiledTree:
    perm = jnp.asarray(perm, dtype=jnp.int32)
    if perm.ndim != 1:
        raise ValueError(f"perm must be rank one, got shape {perm.shape}")
    leaf_real = jnp.asarray(leaf_real, dtype=jnp.bool_)
    if leaf_real.shape != perm.shape:
        raise ValueError(
            f"leaf_real shape {leaf_real.shape} must match perm {perm.shape}"
        )
    inv_perm = jnp.argsort(perm).astype(jnp.int32)
    return CompiledTree(
        perm=perm,
        inv_perm=inv_perm,
        leaf_real=leaf_real,
        leaf_h=tuple(boundaries["leaf_h"]),
        leaf_combiner_h=tuple(boundaries["leaf_combiner_h"]),
        merge_h=tuple(boundaries["merge_h"]),
        opcodes=tuple(boundaries["opcodes"]),
        readout_h=tuple(boundaries["readout_h"]),
        readout_combiner_h=tuple(boundaries["readout_combiner_h"]),
    )


def compile_physical_tree_from_reduced_state(
    kernel: PhysicalCompilerKernel,
    *,
    perm,
    leaf_real,
    leaf_h,
    c_reduced,
    edge_reduced,
    real_reduced,
    structural_reduced,
    counts_reduced,
    g_reduced,
    early_merge_h=(),
    early_opcodes=(),
    full_structural_mask=None,
) -> CompiledTree:
    perm = jnp.asarray(perm, dtype=jnp.int32)
    leaf_real = jnp.asarray(leaf_real)
    if perm.ndim != 1 or leaf_real.shape != perm.shape:
        raise ValueError("perm and leaf_real must be matching rank-one arrays")
    early_merge_h = tuple(early_merge_h)
    early_opcodes = tuple(early_opcodes)
    if len(early_merge_h) != len(early_opcodes):
        raise ValueError("early merge_h/opcode level counts differ")
    if full_structural_mask is None:
        full_structural_mask = leaf_real
    level_offset = len(early_merge_h)
    reduced = compile_context_only_reduction(
        merge=kernel.merge,
        c_leaf=c_reduced,
        leaf_real=real_reduced,
        g_emb=g_reduced,
        edges=edge_reduced,
        structural_mask=structural_reduced,
        n_total=jnp.sum(leaf_real),
        clock_depth=tree_active_clock_depth(jnp.asarray(full_structural_mask)),
        gladder=(
            kernel.tree_pool,
            kernel.tree_update,
            kernel.tree_projection_weight,
            kernel.tree_projection_bias,
        ),
        initial_counts=counts_reduced,
        level_offset=level_offset,
        feature_n_levels=tree_active_clock_depth(leaf_real),
    )
    readout_g_emb = _project_global(
        kernel.root_projection,
        reduced["g_final"],
        dense_tag="gladder.root_proj",
        norm_tag="gladder.root_ln",
    )
    boundaries = {
        "leaf_h": tuple(leaf_h),
        "leaf_combiner_h": (),
        "merge_h": early_merge_h + tuple(reduced["merge_h"]),
        "opcodes": early_opcodes + tuple(reduced["opcodes"]),
        "readout_h": compile_target_readout_h(
            kernel.readout,
            reduced["e_root"],
            reduced["c_root"],
            readout_g_emb,
        ),
        "readout_combiner_h": (),
    }
    return assemble_compiled_tree(
        perm=perm,
        leaf_real=leaf_real,
        boundaries=boundaries,
    )


def compile_physical_tree_from_shared_trunk(
    kernel: PhysicalCompilerKernel,
    shared_trunk,
    perm,
) -> CompiledTree:
    perm = jnp.asarray(perm, dtype=jnp.int32)
    if perm.ndim != 1 or perm.shape != shared_trunk.real_mask.shape:
        raise ValueError("perm must be rank one and match the shared trunk site width")
    node = shared_trunk.node_raw[perm]
    edge = shared_trunk.edge_raw[perm][:, perm]
    leaf_real = shared_trunk.real_mask[perm]
    structural_mask = shared_trunk.balanced_mask
    e_leaf, edge_leaf, g_stream = kernel.contextualizer.with_edge(
        node,
        edge,
        leaf_real,
        structural_mask,
        g=shared_trunk.global_stream,
    )
    g_stream = kernel.global_fork(g_stream, edge_leaf, structural_mask)
    leaf_g_emb = _project_global(
        kernel.leaf_projection,
        g_stream,
        dense_tag="gladder.to_gemb",
        norm_tag="gladder.gemb_ln",
    )
    c_leaf = tree_sphere(kernel.leaf.P_c(e_leaf, pathway="even"))
    reduced = compile_context_only_reduction(
        merge=kernel.merge,
        c_leaf=c_leaf,
        leaf_real=leaf_real,
        g_emb=g_stream,
        edges=edge_leaf,
        structural_mask=structural_mask,
        gladder=(
            kernel.tree_pool,
            kernel.tree_update,
            kernel.tree_projection_weight,
            kernel.tree_projection_bias,
        ),
    )
    readout_g_emb = _project_global(
        kernel.root_projection,
        reduced["g_final"],
        dense_tag="gladder.root_proj",
        norm_tag="gladder.root_ln",
    )
    boundaries = {
        "leaf_h": compile_target_leaf_h(
            kernel.leaf,
            e_leaf,
            leaf_g_emb,
        ),
        "leaf_combiner_h": (),
        "merge_h": reduced["merge_h"],
        "opcodes": reduced["opcodes"],
        "readout_h": compile_target_readout_h(
            kernel.readout,
            reduced["e_root"],
            reduced["c_root"],
            readout_g_emb,
        ),
        "readout_combiner_h": (),
    }
    return assemble_compiled_tree(
        perm=perm,
        leaf_real=leaf_real,
        boundaries=boundaries,
    )


def compile_physical_tree_reference(
    model,
    shared_trunk,
    perm,
) -> CompiledTree:
    if model.gladder_post is None or model.gladder_fork_phys is None:
        raise ValueError(
            "reference physical compiler requires the target global ladder"
        )
    perm = jnp.asarray(perm, dtype=jnp.int32)
    if perm.ndim != 1 or perm.shape != shared_trunk.real_mask.shape:
        raise ValueError("perm must be rank one and match the shared trunk site width")
    node = shared_trunk.node_raw[perm]
    edge = shared_trunk.edge_raw[perm][:, perm]
    leaf_real = shared_trunk.real_mask[perm]
    structural_mask = shared_trunk.balanced_mask
    e_leaf, edge_leaf, g_stream = model._contextualize_leaf_even_with_edge_g(
        node,
        edge,
        leaf_real,
        structural_mask,
        shared_trunk.global_stream,
    )
    g_stream = model.gladder_fork_phys(g_stream, edge_leaf, structural_mask)
    leaf_g_emb = model._gladder_project(g_stream)
    c_leaf = tree_sphere(model.leaf.P_c(e_leaf, pathway="even"))
    reduced = compile_context_only_reduction(
        merge=model.merge,
        c_leaf=c_leaf,
        leaf_real=leaf_real,
        g_emb=g_stream,
        edges=edge_leaf,
        structural_mask=structural_mask,
        gladder=model._gladder_tree_refs(),
    )
    readout_g_emb = model._gladder_root_project(reduced["g_final"])
    boundaries = {
        "leaf_h": compile_target_leaf_h(model.leaf, e_leaf, leaf_g_emb),
        "leaf_combiner_h": (),
        "merge_h": reduced["merge_h"],
        "opcodes": reduced["opcodes"],
        "readout_h": compile_target_readout_h(
            model.readout,
            reduced["e_root"],
            reduced["c_root"],
            readout_g_emb,
        ),
        "readout_combiner_h": (),
    }
    return assemble_compiled_tree(
        perm=perm,
        leaf_real=leaf_real,
        boundaries=boundaries,
    )
