# Copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable, Iterable, Sequence

import networkx as nx
import numpy as np


_PHI = 0.5 * (1.0 + np.sqrt(5.0))


def _exchange_matrix(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0:
        return np.eye(3, dtype=np.float32) * array
    if array.shape == (3,):
        return np.diag(array)
    if array.shape == (3, 3):
        return array
    raise ValueError("exchange must be a scalar, length-3 vector, or 3x3 matrix")


def _field_vector(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0:
        return np.asarray((0.0, 0.0, float(array)), dtype=np.float32)
    if array.shape == (3,):
        return array
    raise ValueError("field must be a scalar z-field or length-3 vector")


def _pair_alpha(exchange: np.ndarray) -> float:
    if not np.any(exchange):
        return 0.0
    left, singular_values, right_t = np.linalg.svd(exchange)
    operator_norm = float(singular_values[0])
    nuclear_norm = float(singular_values.sum())
    trace_component = float(np.trace(exchange) / 3.0)
    trace_residual = exchange - trace_component * np.eye(3, dtype=exchange.dtype)
    trace_singular_values = np.linalg.svd(trace_residual, compute_uv=False)
    rotation = left @ right_t
    polar_component = float(np.trace(rotation.T @ exchange) / 3.0)
    polar_residual = exchange - polar_component * rotation
    polar_singular_values = np.linalg.svd(polar_residual, compute_uv=False)
    full_bound = min(
        (_PHI / 2.0) * operator_norm,
        0.5 * nuclear_norm,
        0.75 * abs(trace_component) + (_PHI / 2.0) * float(trace_singular_values[0]),
        0.75 * abs(trace_component) + 0.5 * float(trace_singular_values.sum()),
        0.75 * abs(polar_component) + (_PHI / 2.0) * float(polar_singular_values[0]),
        0.75 * abs(polar_component) + 0.5 * float(polar_singular_values.sum()),
    )
    symmetric = 0.5 * (exchange + exchange.T)
    antisymmetric = exchange - symmetric
    scale = max(float(np.linalg.norm(exchange)), 1e-9)
    if float(np.linalg.norm(antisymmetric)) > 1e-9 * scale:
        return full_bound
    eigenvalues = np.linalg.eigvalsh(symmetric)
    if not (np.all(eigenvalues >= -1e-12) or np.all(eigenvalues <= 1e-12)):
        return full_bound
    return min(full_bound, 0.5 * operator_norm)


def _mu_safe(exchange: np.ndarray, field: np.ndarray) -> float:
    row_bound = np.zeros(exchange.shape[0], dtype=np.float64)
    for left in range(exchange.shape[0]):
        for right in range(left + 1, exchange.shape[0]):
            pair_bound = _pair_alpha(exchange[left, right])
            row_bound[left] += pair_bound
            row_bound[right] += pair_bound
    field_bound = np.linalg.norm(field, axis=-1)
    return float(np.max(row_bound + (2.0 / 3.0) * field_bound, initial=0.0))


@dataclass(frozen=True, slots=True)
class SpinHamiltonian:
    _coupling: np.ndarray
    _J: np.ndarray
    _h: np.ndarray
    nodes: tuple[Hashable, ...]
    _mu: float | None = None
    _needs_fwl2: bool | None = None
    _category: str | None = None
    _tag: str | None = None
    _topology_class: str | None = None
    _j_class: str | None = None

    def __post_init__(self) -> None:
        coupling = np.asarray(self._coupling, dtype=np.float32)
        exchange = np.asarray(self._J, dtype=np.float32)
        field = np.asarray(self._h, dtype=np.float32)
        n_spins = len(self.nodes)
        if coupling.shape != (n_spins, n_spins):
            raise ValueError("coupling must have shape [N,N]")
        if exchange.shape != (n_spins, n_spins, 3, 3):
            raise ValueError("J must have shape [N,N,3,3]")
        if field.shape != (n_spins, 3):
            raise ValueError("h must have shape [N,3]")
        if not np.array_equal(coupling, coupling.T):
            raise ValueError("coupling must be symmetric")
        if np.any(np.diag(coupling)) or np.any(
            exchange[np.arange(n_spins), np.arange(n_spins)]
        ):
            raise ValueError("self-couplings are not supported")
        if not np.allclose(exchange, exchange.transpose(1, 0, 3, 2), atol=0.0):
            raise ValueError("J[j,i] must equal transpose(J[i,j])")
        mu = None if self._mu is None else float(self._mu)
        if mu is not None and (not np.isfinite(mu) or mu < 0.0):
            raise ValueError("mu must be finite and non-negative")
        object.__setattr__(self, "_coupling", coupling)
        object.__setattr__(self, "_J", exchange)
        object.__setattr__(self, "_h", field)
        object.__setattr__(self, "_mu", mu)
        object.__setattr__(
            self,
            "_needs_fwl2",
            None if self._needs_fwl2 is None else bool(self._needs_fwl2),
        )

    @classmethod
    def from_networkx(
        cls,
        graph: nx.Graph,
        *,
        J: Any = 1.0,
        h: Any = 0.0,
        edge_attribute: str = "J",
        node_attribute: str = "h",
        nodes: Iterable[Hashable] | None = None,
        mu: float | None = None,
    ) -> "SpinHamiltonian":
        if graph.is_directed() or graph.is_multigraph():
            raise TypeError("expected a simple undirected NetworkX graph")
        order = tuple(graph.nodes if nodes is None else nodes)
        if len(order) != graph.number_of_nodes() or set(order) != set(graph.nodes):
            raise ValueError("nodes must contain every graph node exactly once")
        indices = {node: index for index, node in enumerate(order)}
        n_spins = len(order)
        coupling = np.zeros((n_spins, n_spins), dtype=np.float32)
        exchange = np.zeros((n_spins, n_spins, 3, 3), dtype=np.float32)
        field = np.zeros((n_spins, 3), dtype=np.float32)
        for node, index in indices.items():
            public_field = graph.nodes[node].get(node_attribute, h)
            field[index] = -_field_vector(public_field)
        for left_node, right_node, attributes in graph.edges(data=True):
            left = indices[left_node]
            right = indices[right_node]
            public_exchange = _exchange_matrix(attributes.get(edge_attribute, J))
            coupling[left, right] = coupling[right, left] = 1.0
            exchange[left, right] = -0.5 * public_exchange
            exchange[right, left] = -0.5 * public_exchange.T
        return cls(
            coupling,
            exchange,
            field,
            order,
            mu,
            graph.graph.get("needs_fwl2"),
            graph.graph.get("category"),
            graph.graph.get("tag"),
            graph.graph.get("topology_class"),
            graph.graph.get("j_class"),
        )

    @classmethod
    def from_arrays(
        cls,
        J: Any,
        h: Any | None = None,
        *,
        coupling: Any | None = None,
        nodes: Sequence[Hashable] | None = None,
        mu: float | None = None,
    ) -> "SpinHamiltonian":
        exchange = np.asarray(J, dtype=np.float32)
        if exchange.ndim == 2 and exchange.shape[0] == exchange.shape[1]:
            exchange = (
                exchange[:, :, None, None] * np.eye(3, dtype=np.float32)[None, None]
            )
        elif exchange.ndim == 3 and exchange.shape[-1] == 3:
            promoted = np.zeros((*exchange.shape[:2], 3, 3), dtype=np.float32)
            diagonal = np.arange(3)
            promoted[:, :, diagonal, diagonal] = exchange
            exchange = promoted
        if exchange.ndim != 4 or exchange.shape[-2:] != (3, 3):
            raise ValueError("J must have shape [N,N,3] or [N,N,3,3]")
        n_spins = exchange.shape[0]
        if exchange.shape[1] != n_spins:
            raise ValueError("J site axes must be square")
        public_field = np.zeros((n_spins, 3), dtype=np.float32)
        if h is not None:
            h_array = np.asarray(h, dtype=np.float32)
            if h_array.ndim == 0:
                public_field[:, 2] = h_array
            elif h_array.shape == (3,):
                public_field[:] = h_array
            elif h_array.shape == (n_spins,):
                public_field[:, 2] = h_array
            elif h_array.shape == (n_spins, 3):
                public_field = h_array
            else:
                raise ValueError("h must be scalar, [3], [N], or [N,3]")
        if coupling is None:
            coupling_array = np.any(exchange != 0.0, axis=(-1, -2)).astype(np.float32)
        else:
            coupling_array = np.asarray(coupling, dtype=np.float32)
        node_order = tuple(range(n_spins)) if nodes is None else tuple(nodes)
        return cls(
            coupling_array,
            -0.5 * exchange,
            -public_field,
            node_order,
            mu,
        )

    @property
    def n_spins(self) -> int:
        return len(self.nodes)

    @property
    def coupling(self) -> np.ndarray:
        return self._coupling.copy()

    @property
    def J(self) -> np.ndarray:
        return -2.0 * self._J.copy()

    @property
    def h(self) -> np.ndarray:
        return -self._h.copy()

    @property
    def mu(self) -> float:
        return _mu_safe(self._J, self._h) if self._mu is None else self._mu

    def model_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._coupling.copy(), self._J.copy(), self._h.copy()

    def to_dict(self) -> dict[str, Any]:
        result = {
            "convention": "textbook",
            "nodes": list(self.nodes),
            "coupling": self._coupling.tolist(),
            "J": self.J.tolist(),
            "h": self.h.tolist(),
            "mu": self.mu,
        }
        if self._needs_fwl2 is not None:
            result["needs_fwl2"] = self._needs_fwl2
        for name in ("category", "tag", "topology_class", "j_class"):
            value = getattr(self, f"_{name}")
            if value is not None:
                result[name] = value
        return result


__all__ = ["SpinHamiltonian"]
