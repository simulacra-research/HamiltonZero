# Copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int


def _edge_relation_tags(mask: Array, bmask: Array) -> Int[Array, "n n"]:
    n = mask.shape[0]
    idx = jnp.arange(n, dtype=jnp.int32)
    ii = idx[:, None]
    jj = idx[None, :]
    real_i = mask.astype(bool)[:, None]
    real_j = mask.astype(bool)[None, :]
    ctx_i = bmask.astype(bool)[:, None]
    ctx_j = bmask.astype(bool)[None, :]
    ctx_pair = ctx_i & ctx_j

    real_real = real_i & real_j
    empty_i = (~real_i) & ctx_i
    empty_j = (~real_j) & ctx_j
    self_pair = ii == jj

    tag = jnp.zeros((n, n), dtype=jnp.int32)
    tag = jnp.where(real_real & self_pair, 1, tag)
    tag = jnp.where(empty_i & empty_j & self_pair, 2, tag)
    tag = jnp.where(empty_i & empty_j & (~self_pair), 3, tag)
    tag = jnp.where(empty_i & real_j, 4, tag)
    tag = jnp.where(real_i & empty_j, 5, tag)
    return jnp.where(ctx_pair, tag, jnp.asarray(-1, dtype=jnp.int32))


def _quantized_hash(x: Array, coeff: Array, tol: float) -> Int[Array, "..."]:
    q = jnp.rint(
        jnp.real(x).astype(jnp.float32) / jnp.asarray(tol, dtype=jnp.float32)
    ).astype(jnp.int32)
    return jnp.sum(q * coeff.astype(jnp.int32), axis=-1).astype(jnp.int32)


def _canonical_hermitian_j(J_full: Float[Array, "n n 3 3"]) -> Float[Array, "n n 3 3"]:
    return 0.5 * (J_full + jnp.transpose(J_full, (1, 0, 3, 2)))


def _j_pair_hash(
    J_full: Float[Array, "n n 3 3"],
    *,
    tol: float,
) -> Int[Array, "n n"]:
    n = J_full.shape[0]
    coeff = jnp.asarray(
        [
            1_000_003,
            1_009_003,
            1_021_009,
            1_033_013,
            1_049_009,
            1_061_009,
            1_073_003,
            1_087_009,
            1_093_013,
            1_109_009,
            1_117_001,
            1_123_003,
            1_129_009,
            1_151_003,
            1_159_013,
            1_171_009,
            1_181_003,
            1_187_009,
        ],
        dtype=jnp.int32,
    )
    J_key = _canonical_hermitian_j(J_full)
    pair = jnp.concatenate(
        [
            J_key.reshape(n, n, 9),
            jnp.transpose(J_key, (1, 0, 2, 3)).reshape(n, n, 9),
        ],
        axis=-1,
    )
    return _quantized_hash(pair, coeff, tol)


def _j_diag_hash(
    J_full: Float[Array, "n n 3 3"],
    *,
    tol: float,
) -> Int[Array, "n"]:

    n = J_full.shape[0]
    coeff = jnp.asarray(
        [
            1_000_003,
            1_009_003,
            1_021_009,
            1_033_013,
            1_049_009,
            1_061_009,
            1_073_003,
            1_087_009,
            1_093_013,
        ],
        dtype=jnp.int32,
    )
    J_key = _canonical_hermitian_j(J_full)
    idx = jnp.arange(n)
    diag = J_key[idx, idx].reshape(n, 9)
    return _quantized_hash(diag, coeff, tol)


def route_quotient_keys(
    J_full: Float[Array, "n n 3 3"],
    h: Float[Array, "n 3"],
    mask: Int[Array, "n"] | Array,
    bmask: Int[Array, "n"] | Array,
    *,
    tol: float = 1e-6,
) -> tuple[Int[Array, "n"], Int[Array, "n n"]]:

    real = mask.astype(bool)
    context = bmask.astype(bool)
    h_coeff = jnp.asarray([1_000_003, 1_009_003, 1_021_009], dtype=jnp.int32)
    h_key = _quantized_hash(h, h_coeff, tol)

    node_key = jnp.where(
        real,
        h_key
        + _j_diag_hash(J_full, tol=tol) * jnp.asarray(131_063, dtype=jnp.int32)
        + jnp.asarray(17_071, dtype=jnp.int32),
        jnp.asarray(-313_037, dtype=jnp.int32),
    )
    node_key = jnp.where(context, node_key, jnp.asarray(-1, dtype=jnp.int32))

    edge_hash = _j_pair_hash(J_full, tol=tol)
    tags = _edge_relation_tags(mask, bmask)
    edge_hash = jnp.where(tags == 1, jnp.asarray(0, dtype=jnp.int32), edge_hash)
    edge_key = edge_hash + tags * jnp.asarray(131_071, dtype=jnp.int32)
    edge_key = jnp.where(context[:, None] & context[None, :], edge_key, 0)
    return node_key, edge_key


def conditional_orbit_ids_from_keys(
    node_key: Int[Array, "n"],
    edge_key: Int[Array, "n n"],
    valid_mask: Int[Array, "n"] | Array,
    context_mask: Int[Array, "n"] | Array,
    prefix_ids: Int[Array, "n"],
    prefix_len,
    *,
    max_rounds: int | None = None,
) -> Int[Array, "n"]:

    n = node_key.shape[0]
    if max_rounds is None:
        max_rounds = n
    idx = jnp.arange(n, dtype=jnp.int32)
    prefix_len_i = jnp.asarray(prefix_len, dtype=jnp.int32)
    context = context_mask.astype(bool)
    valid = valid_mask.astype(bool) & context

    prefix_active = (idx < prefix_len_i) & context[prefix_ids]
    prefix_pos = jnp.max(
        jnp.where(
            prefix_active[:, None] & (prefix_ids[:, None] == idx[None, :]),
            idx[:, None],
            jnp.asarray(-1, dtype=jnp.int32),
        ),
        axis=0,
    )
    is_prefix = prefix_pos >= 0
    valid = valid & (~is_prefix)
    prefix_color = jnp.asarray(
        2_000_000_000, dtype=jnp.int32
    ) - prefix_pos * jnp.asarray(1_000_003, dtype=jnp.int32)
    colors0 = jnp.where(is_prefix, prefix_color, node_key)
    colors0 = jnp.where(context, colors0, jnp.asarray(-1, dtype=jnp.int32))
    big = jnp.asarray(n + 1, dtype=jnp.int32)

    def body(colors, _):
        pair_key = edge_key + colors[None, :] * jnp.asarray(1_310_719, dtype=jnp.int32)
        pair_key = jnp.where(
            context[None, :], pair_key, jnp.asarray(0, dtype=jnp.int32)
        )
        sorted_keys = jnp.sort(pair_key, axis=1)
        same_sig = (
            context[:, None]
            & context[None, :]
            & (colors[:, None] == colors[None, :])
            & jnp.all(sorted_keys[:, None, :] == sorted_keys[None, :, :], axis=-1)
        )
        new_colors = jnp.min(jnp.where(same_sig, idx[None, :], big), axis=1)
        new_colors = jnp.where(is_prefix, prefix_color, new_colors)
        return jnp.where(context, new_colors, jnp.asarray(-1, dtype=jnp.int32)), None

    colors, _ = jax.lax.scan(body, colors0, None, length=int(max_rounds))
    same_valid = valid[:, None] & valid[None, :] & (colors[:, None] == colors[None, :])
    reps = jnp.min(jnp.where(same_valid, idx[None, :], big), axis=1)
    return jnp.where(valid, reps, jnp.asarray(-1, dtype=jnp.int32))


def conditional_orbit_pair_ids_from_keys(
    node_key: Int[Array, "n"],
    edge_key: Int[Array, "n n"],
    valid_mask: Int[Array, "n"] | Array,
    context_mask: Int[Array, "n"] | Array,
    prefix_ids: Int[Array, "n"],
    prefix_len,
) -> Int[Array, "n"]:

    n = int(node_key.shape[0])
    idx = jnp.arange(n, dtype=jnp.int32)
    pidx = jnp.arange(n * n, dtype=jnp.int32)
    nn = jnp.asarray(n * n, dtype=jnp.int32)
    big = jnp.asarray(n * n, dtype=jnp.int32)
    prefix_len_i = jnp.asarray(prefix_len, dtype=jnp.int32)
    context = context_mask.astype(bool)
    valid = valid_mask.astype(bool) & context
    prefix_active = (idx < prefix_len_i) & context[prefix_ids]
    prefix_pos = jnp.max(
        jnp.where(
            prefix_active[:, None] & (prefix_ids[:, None] == idx[None, :]),
            idx[:, None],
            jnp.asarray(-1, dtype=jnp.int32),
        ),
        axis=0,
    )
    is_prefix = prefix_pos >= 0
    valid = valid & (~is_prefix)
    prefix_color = jnp.asarray(
        2_000_000_000, dtype=jnp.int32
    ) - prefix_pos * jnp.asarray(1_000_003, dtype=jnp.int32)
    node_colors = jnp.where(is_prefix, prefix_color, node_key)
    node_colors = jnp.where(context, node_colors, jnp.asarray(-1, dtype=jnp.int32))
    cc = (context[:, None] & context[None, :]).reshape(-1)

    def _canon(components):

        same = cc[:, None] & cc[None, :]
        for s in components:
            sf = s.reshape(-1)
            same = same & (sf[:, None] == sf[None, :])
        reps = jnp.min(jnp.where(same, pidx[None, :], big), axis=1)
        return jnp.where(cc, reps, jnp.asarray(-1, dtype=jnp.int32)).reshape(n, n)

    ni = jnp.broadcast_to(node_colors[:, None], (n, n))
    nj = jnp.broadcast_to(node_colors[None, :], (n, n))
    pc0 = _canon([ni, nj, edge_key, jnp.transpose(edge_key)])

    base_a = jnp.asarray(1_000_003, dtype=jnp.int32)
    base_b = jnp.asarray(1_300_021, dtype=jnp.int32)
    if n > 1:
        pow_a = jnp.concatenate(
            [
                jnp.ones((1,), jnp.int32),
                jnp.cumprod(jnp.full((n - 1,), base_a, jnp.int32)),
            ]
        )
        pow_b = jnp.concatenate(
            [
                jnp.ones((1,), jnp.int32),
                jnp.cumprod(jnp.full((n - 1,), base_b, jnp.int32)),
            ]
        )
    else:
        pow_a = jnp.ones((1,), jnp.int32)
        pow_b = jnp.ones((1,), jnp.int32)
    ctx_k = context[None, None, :]
    off = jnp.asarray(7, dtype=jnp.int32)
    neutral = jnp.asarray(-1, dtype=jnp.int32)

    def _body(pc, _):
        a = pc[:, None, :]
        b = jnp.transpose(pc)[None, :, :]
        code = jnp.where(ctx_k, a * nn + b, neutral)
        sc = jnp.sort(code, axis=-1)

        ph_a = jnp.sum((sc + off) * pow_a[None, None, :], axis=-1)
        ph_b = jnp.sum((sc + off) * pow_b[None, None, :], axis=-1)
        return _canon([pc, ph_a, ph_b]), None

    pc, _ = jax.lax.scan(_body, pc0, None, length=n)
    diag = jnp.diagonal(pc)
    same_valid = valid[:, None] & valid[None, :] & (diag[:, None] == diag[None, :])
    reps = jnp.min(jnp.where(same_valid, idx[None, :], big), axis=1)
    return jnp.where(valid, reps, jnp.asarray(-1, dtype=jnp.int32))


_WL1_BREAKER_PATTERN = re.compile(r"wl1|srg|paley|shrikhande|rook", re.IGNORECASE)
_WL1_BREAKER_CATEGORY = "13_wl1_breaking"


def _tag_forces_fwl2(
    *,
    category: str | None = None,
    tag: str | None = None,
    topology_class: str | None = None,
    j_class: str | None = None,
) -> bool:
    if category is not None and str(category) == _WL1_BREAKER_CATEGORY:
        return True
    for value in (tag, topology_class, j_class):
        if value is not None and _WL1_BREAKER_PATTERN.search(str(value)):
            return True
    return False


def system_needs_fwl2(
    J_full,
    h,
    n_spins: int,
    *,
    category: str | None = None,
    tag: str | None = None,
    topology_class: str | None = None,
    j_class: str | None = None,
    tol: float = 1e-6,
) -> bool:
    if _tag_forces_fwl2(
        category=category,
        tag=tag,
        topology_class=topology_class,
        j_class=j_class,
    ):
        return True

    n = int(n_spins)
    J_arr = jnp.asarray(np.asarray(J_full, dtype=np.float64))
    h_arr = jnp.asarray(np.asarray(h, dtype=np.float64))
    mask = jnp.ones((n,), dtype=jnp.int32)
    bmask = jnp.ones((n,), dtype=jnp.int32)
    idx = jnp.arange(n, dtype=jnp.int32)
    prefix_len = jnp.int32(0)

    node_key, edge_key = route_quotient_keys(J_arr, h_arr, mask, bmask, tol=tol)
    wl1 = conditional_orbit_ids_from_keys(
        node_key, edge_key, mask, bmask, idx, prefix_len
    )
    fwl2 = conditional_orbit_pair_ids_from_keys(
        node_key, edge_key, mask, bmask, idx, prefix_len
    )
    return bool(not jnp.array_equal(wl1, fwl2))


__all__ = [
    "conditional_orbit_ids_from_keys",
    "conditional_orbit_pair_ids_from_keys",
    "route_quotient_keys",
    "system_needs_fwl2",
]
