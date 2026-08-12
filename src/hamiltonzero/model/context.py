# Copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int


def _balanced_mask(mask: Int[Array, "n"]) -> Int[Array, "n"]:
    width = int(mask.shape[-1])
    n_real = jnp.sum(mask.astype(jnp.int32))
    max_power = max(1, (width - 1).bit_length())
    powers = 2 ** jnp.arange(max_power + 1, dtype=jnp.int32)
    sentinel = jnp.asarray(1 << 30, dtype=jnp.int32)
    next_power = jnp.min(jnp.where(powers >= jnp.maximum(n_real, 1), powers, sentinel))
    return (jnp.arange(width, dtype=jnp.int32) < next_power).astype(jnp.int32)


_EPS_ABC = jnp.asarray(
    [
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
        [[0.0, 0.0, -1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    ],
    dtype=jnp.float32,
)


def _real_dtype(dtype):
    return jnp.real(jnp.zeros((), dtype)).dtype


def _host_eigvalsh(x):
    out_dtype = _real_dtype(x.dtype)
    out_shape = jax.ShapeDtypeStruct(x.shape[:-1], out_dtype)

    def callback(a):
        values = np.linalg.eigvalsh(np.asarray(a))
        return values.astype(np.dtype(out_dtype))

    return jax.pure_callback(
        callback,
        out_shape,
        x,
        vmap_method="sequential",
    )


def _compute_J_double_prime_batched(
    J_full: Float[Array, "s n n 3 3"],
    h: Float[Array, "s n 3"],
    mask: Int[Array, "s n"],
) -> tuple[Float[Array, "s n n 10"], Float[Array, "s"]]:
    n = J_full.shape[1]
    n_systems = J_full.shape[0]
    dtype = J_full.dtype

    J_filled = (J_full + jnp.conj(jnp.transpose(J_full, (0, 2, 1, 4, 3)))) / 2.0

    eps_abc = _EPS_ABC.astype(_real_dtype(dtype))
    M_h = jnp.einsum("abc,sic->siab", eps_abc, h.astype(eps_abc.dtype)) * 0.5
    M_h = M_h * mask.astype(M_h.dtype)[:, :, None, None]
    diagonal = jnp.arange(n)

    complex_dtype = jnp.result_type(J_filled.dtype, jnp.complex64)
    M_h_diagonal = (
        jnp.zeros(J_filled.shape, dtype=complex_dtype)
        .at[:, diagonal, diagonal]
        .set((2j * M_h).astype(complex_dtype))
    )
    J_for_norm = J_filled.astype(complex_dtype) + M_h_diagonal
    J_matrix = jnp.transpose(J_for_norm, (0, 1, 3, 2, 4)).reshape(
        n_systems, 3 * n, 3 * n
    )
    eigh_epsilon = jnp.asarray(1e-6, dtype=_real_dtype(J_matrix.dtype))
    eye_3n = jnp.eye(3 * n, dtype=J_matrix.dtype)
    eigenvalues = _host_eigvalsh(J_matrix + eigh_epsilon * eye_3n[None]) - eigh_epsilon
    s_norm = jnp.maximum(
        jnp.max(jnp.abs(eigenvalues), axis=-1),
        jnp.asarray(1e-12, eigenvalues.dtype),
    ).astype(dtype)

    J_normalized = J_filled / s_norm[:, None, None, None, None]
    J_normalized = J_normalized.at[:, diagonal, diagonal, :, :].set(0.0)
    J_flat = jnp.real(J_normalized).reshape(n_systems, n, n, 9).astype(dtype)
    identity_column = jnp.broadcast_to(
        jnp.eye(n, dtype=dtype)[None, ..., None],
        (n_systems, n, n, 1),
    )
    return jnp.concatenate([J_flat, identity_column], axis=-1), s_norm


def _compute_J_double_prime(
    J_full: Float[Array, "n n 3 3"],
    h: Float[Array, "n 3"],
    mask: Int[Array, "n"],
) -> tuple[Float[Array, "n n 10"], Float[Array, ""]]:
    J_double_prime, s_norm = _compute_J_double_prime_batched(
        J_full[None], h[None], mask[None]
    )
    return J_double_prime[0], s_norm[0]


class SpinContext(eqx.Module):
    mask: Int[Array, "n"]
    bmask: Int[Array, "n"]
    J_double_prime: Float[Array, "n n 10"]
    s_norm: Float[Array, ""]
    h_prime: Float[Array, "n 3"]
    route_quotient_node_key: Int[Array, "n"]
    route_quotient_edge_key: Int[Array, "n n"]
    needs_fwl2: Array
    route_perm: Int[Array, "n"]

    def __init__(
        self,
        J_full: Float[Array, "n n 3 3"],
        h: Float[Array, "n 3"],
        mask: Int[Array, "n"],
        *,
        needs_fwl2: Array | bool,
    ) -> None:
        from .route_quotient import route_quotient_keys

        self.mask = mask.astype(jnp.int32)
        self.bmask = _balanced_mask(self.mask)
        self.J_double_prime, self.s_norm = _compute_J_double_prime(J_full, h, self.mask)
        self.h_prime = h / jnp.real(self.s_norm).astype(h.dtype)
        (
            self.route_quotient_node_key,
            self.route_quotient_edge_key,
        ) = route_quotient_keys(J_full, h, self.mask, self.bmask)
        self.needs_fwl2 = jnp.asarray(needs_fwl2, dtype=jnp.bool_)
        self.route_perm = jnp.arange(self.mask.shape[0], dtype=jnp.int32)

    @classmethod
    def from_precomputed(
        cls,
        *,
        mask,
        bmask,
        J_double_prime,
        s_norm,
        h_prime,
        route_quotient_node_key,
        route_quotient_edge_key,
        needs_fwl2,
        route_perm,
    ) -> "SpinContext":
        self = object.__new__(cls)
        fields = {
            "mask": mask,
            "bmask": bmask,
            "J_double_prime": J_double_prime,
            "s_norm": s_norm,
            "h_prime": h_prime,
            "route_quotient_node_key": route_quotient_node_key,
            "route_quotient_edge_key": route_quotient_edge_key,
            "needs_fwl2": needs_fwl2,
            "route_perm": route_perm,
        }
        for name, value in fields.items():
            dtype = (
                jnp.bool_
                if name == "needs_fwl2"
                else jnp.int32
                if name
                in {
                    "mask",
                    "bmask",
                    "route_quotient_node_key",
                    "route_quotient_edge_key",
                    "route_perm",
                }
                else None
            )
            object.__setattr__(self, name, jnp.asarray(value, dtype=dtype))
        return self

    @property
    def n_sites(self) -> int:
        return int(self.mask.shape[0])


class MultiSystemContext(eqx.Module):
    mask: Int[Array, "s n"]
    bmask: Int[Array, "s n"]
    J_double_prime: Float[Array, "s n n 10"]
    s_norm: Float[Array, "s"]
    h_prime: Float[Array, "s n 3"]
    route_quotient_node_key: Int[Array, "s n"]
    route_quotient_edge_key: Int[Array, "s n n"]
    needs_fwl2: Array
    route_perm: Int[Array, "s n"]

    def __init__(
        self,
        J_full: Float[Array, "s n n 3 3"],
        h: Float[Array, "s n 3"],
        mask: Int[Array, "s n"],
        *,
        needs_fwl2: Array | bool,
    ) -> None:
        from .route_quotient import route_quotient_keys

        self.mask = mask.astype(jnp.int32)
        self.bmask = jax.vmap(_balanced_mask)(self.mask)
        self.J_double_prime, self.s_norm = _compute_J_double_prime_batched(
            J_full, h, self.mask
        )
        self.h_prime = h / jnp.real(self.s_norm).astype(h.dtype)[:, None, None]
        n_systems = self.mask.shape[0]
        (
            self.route_quotient_node_key,
            self.route_quotient_edge_key,
        ) = jax.jit(jax.vmap(route_quotient_keys))(J_full, h, self.mask, self.bmask)
        self.needs_fwl2 = jnp.broadcast_to(
            jnp.asarray(needs_fwl2, dtype=jnp.bool_),
            (n_systems,),
        )
        self.route_perm = jnp.broadcast_to(
            jnp.arange(self.mask.shape[1], dtype=jnp.int32)[None, :],
            self.mask.shape,
        )

    @classmethod
    def from_precomputed(
        cls,
        *,
        mask,
        bmask,
        J_double_prime,
        s_norm,
        h_prime,
        route_quotient_node_key,
        route_quotient_edge_key,
        needs_fwl2,
        route_perm,
    ) -> "MultiSystemContext":
        self = object.__new__(cls)
        fields = {
            "mask": mask,
            "bmask": bmask,
            "J_double_prime": J_double_prime,
            "s_norm": s_norm,
            "h_prime": h_prime,
            "route_quotient_node_key": route_quotient_node_key,
            "route_quotient_edge_key": route_quotient_edge_key,
            "needs_fwl2": needs_fwl2,
            "route_perm": route_perm,
        }
        for name, value in fields.items():
            dtype = (
                jnp.bool_
                if name == "needs_fwl2"
                else jnp.int32
                if name
                in {
                    "mask",
                    "bmask",
                    "route_quotient_node_key",
                    "route_quotient_edge_key",
                    "route_perm",
                }
                else None
            )
            object.__setattr__(self, name, jnp.asarray(value, dtype=dtype))
        return self

    @classmethod
    def from_single(cls, context: SpinContext) -> "MultiSystemContext":
        return cls.from_precomputed(
            mask=context.mask[None],
            bmask=context.bmask[None],
            J_double_prime=context.J_double_prime[None],
            s_norm=context.s_norm[None],
            h_prime=context.h_prime[None],
            route_quotient_node_key=context.route_quotient_node_key[None],
            route_quotient_edge_key=context.route_quotient_edge_key[None],
            needs_fwl2=context.needs_fwl2[None],
            route_perm=context.route_perm[None],
        )

    @classmethod
    def stack(cls, contexts: list[SpinContext]) -> "MultiSystemContext":
        if not contexts:
            raise ValueError("MultiSystemContext.stack requires a context")
        widths = [int(context.mask.shape[0]) for context in contexts]
        n_max = max(widths)

        def pad_sites(value, n):
            return (
                value
                if n == n_max
                else jnp.pad(value, ((0, n_max - n),) + ((0, 0),) * (value.ndim - 1))
            )

        def pad_pairs(value, n):
            padding = n_max - n
            return (
                value
                if padding == 0
                else jnp.pad(
                    value,
                    ((0, padding), (0, padding)) + ((0, 0),) * (value.ndim - 2),
                )
            )

        mask = jnp.stack(
            [
                pad_sites(context.mask, width)
                for context, width in zip(contexts, widths, strict=True)
            ]
        )
        bmask = jax.vmap(_balanced_mask)(mask)
        J_double_prime = jnp.stack(
            [
                pad_pairs(context.J_double_prime, width)
                for context, width in zip(contexts, widths, strict=True)
            ]
        )
        diagonal = jnp.arange(n_max, dtype=jnp.int32)
        J_double_prime = J_double_prime.at[:, diagonal, diagonal, 9].set(1.0)
        s_norm = jnp.stack([context.s_norm for context in contexts])
        h_prime = jnp.stack(
            [
                pad_sites(context.h_prime, width)
                for context, width in zip(contexts, widths, strict=True)
            ]
        )

        edge_shapes = {
            tuple(context.route_quotient_edge_key.shape) for context in contexts
        }
        compact_edges = all(shape == (0, 0) for shape in edge_shapes)
        full_edges = all(
            shape == (width, width)
            for shape, width in zip(
                [context.route_quotient_edge_key.shape for context in contexts],
                widths,
                strict=True,
            )
        )
        if not (compact_edges or full_edges):
            raise ValueError("cannot stack mixed quotient edge carriers")
        route_edge = (
            jnp.zeros((len(contexts), 0, 0), dtype=jnp.int32)
            if compact_edges
            else jnp.stack(
                [
                    pad_pairs(context.route_quotient_edge_key, width)
                    for context, width in zip(contexts, widths, strict=True)
                ]
            )
        )

        def pad_node_key(value, n):
            padding = n_max - n
            return (
                value
                if padding == 0
                else jnp.pad(value, ((0, padding),), constant_values=-1)
            )

        def pad_perm(value, n):
            if n == n_max:
                return value
            return jnp.concatenate([value, jnp.arange(n, n_max, dtype=value.dtype)])

        return cls.from_precomputed(
            mask=mask,
            bmask=bmask,
            J_double_prime=J_double_prime,
            s_norm=s_norm,
            h_prime=h_prime,
            route_quotient_node_key=jnp.stack(
                [
                    pad_node_key(context.route_quotient_node_key, width)
                    for context, width in zip(contexts, widths, strict=True)
                ]
            ),
            route_quotient_edge_key=route_edge,
            needs_fwl2=jnp.stack([context.needs_fwl2 for context in contexts]),
            route_perm=jnp.stack(
                [
                    pad_perm(context.route_perm, width)
                    for context, width in zip(contexts, widths, strict=True)
                ]
            ),
        )

    def select(self, system_id: int) -> SpinContext:
        return SpinContext.from_precomputed(
            mask=self.mask[system_id],
            bmask=self.bmask[system_id],
            J_double_prime=self.J_double_prime[system_id],
            s_norm=self.s_norm[system_id],
            h_prime=self.h_prime[system_id],
            route_quotient_node_key=self.route_quotient_node_key[system_id],
            route_quotient_edge_key=self.route_quotient_edge_key[system_id],
            needs_fwl2=self.needs_fwl2[system_id],
            route_perm=self.route_perm[system_id],
        )

    @property
    def n_systems(self) -> int:
        return int(self.mask.shape[0])

    @property
    def n_sites(self) -> int:
        return int(self.mask.shape[1])


__all__ = [
    "MultiSystemContext",
    "SpinContext",
]
