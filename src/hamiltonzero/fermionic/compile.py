# Copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .algebra import fock_matrix
from .encoding import (
    DEFAULT_PIN_FIELD,
    binary_encoding,
    normalize_terms,
    onehot_encoding,
)
from .sectors import Sector, ground_sector, spin_resolve

AXIS = {"X": 0, "Y": 1, "Z": 2}


@dataclass(frozen=True)
class CompiledSector:
    pauli_terms: dict[str, float]
    n_qubits: int
    scalar_shift: float
    energy_scale: float
    reference_energy: float
    sector: Sector
    checks: dict = field(default_factory=dict)
    projection: np.ndarray | None = None

    @property
    def raw_model_reference(self) -> float:
        return (self.reference_energy - self.scalar_shift) / self.energy_scale

    def to_hartree(self, model_energy: float) -> float:
        return self.energy_scale * model_energy + self.scalar_shift

    def spin_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = self.n_qubits
        exchange = np.zeros((n, n, 3, 3))
        fields = np.zeros((n, 3))
        for word, value in self.pauli_terms.items():
            support = [(i, s) for i, s in enumerate(word) if s != "I"]
            if len(support) == 1:
                site, symbol = support[0]
                fields[site, AXIS[symbol]] += 2.0 * value
            else:
                (i, a), (j, b) = support
                exchange[i, j, AXIS[a], AXIS[b]] += 4.0 * value
                exchange[j, i, AXIS[b], AXIS[a]] += 4.0 * value
        coupling = np.any(exchange != 0.0, axis=(-1, -2)).astype(np.float32)
        return exchange, fields, coupling

    def spin_hamiltonian(self, nodes: tuple[str, ...] | None = None):
        from hamiltonzero import SpinHamiltonian

        exchange, fields, coupling = self.spin_arrays()
        if nodes is None:
            nodes = tuple(f"sector_{i}" for i in range(self.n_qubits))
        return SpinHamiltonian.from_arrays(
            exchange, fields, coupling=coupling, nodes=nodes, mu=None
        )


def compile_fermionic_sector(
    one_body: np.ndarray,
    two_body: np.ndarray,
    constant: float,
    n_electrons: int,
    *,
    spin: float | None = None,
    normalize: bool = True,
    pin_field: float = DEFAULT_PIN_FIELD,
) -> CompiledSector:
    matrix = fock_matrix(one_body, two_body, constant)
    n_modes = int(one_body.shape[0])
    sector, block = ground_sector(matrix, n_modes, n_electrons)
    reference = float(np.linalg.eigvalsh(block)[0])

    projection = None
    if spin is not None:
        projection, block = spin_resolve(matrix, sector, n_modes, spin=spin)
        resolved = float(np.linalg.eigvalsh(block)[0])
        if abs(resolved - reference) > 1e-9:
            raise ValueError(
                "spin-resolved block does not contain the sector ground"
            )
        sector = Sector(
            sector.n_alpha,
            sector.n_beta,
            sector.fock_states,
            spin_resolved=f"S={spin}",
            resolved_dimension=int(block.shape[0]),
        )

    if block.shape[0] <= 4:
        terms, shift, checks = binary_encoding(block, pin_field=pin_field)
        n_qubits = 4
    else:
        terms, shift, checks = onehot_encoding(block)
        n_qubits = int(block.shape[0])

    scale = 1.0
    if normalize:
        terms, scale = normalize_terms(terms)

    checks = dict(checks)
    checks["sector_dimension"] = int(block.shape[0])
    checks["spin_resolved"] = sector.spin_resolved
    return CompiledSector(
        pauli_terms=terms,
        n_qubits=n_qubits,
        scalar_shift=shift,
        energy_scale=scale,
        reference_energy=reference,
        sector=sector,
        checks=checks,
        projection=projection,
    )
