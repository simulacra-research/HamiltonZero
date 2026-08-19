# Copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .algebra import apply_string, fock_matrix


@dataclass(frozen=True)
class Sector:
    n_alpha: int | None
    n_beta: int | None
    fock_states: tuple[int, ...]
    spin_resolved: str | None = None
    resolved_dimension: int | None = None

    @property
    def dimension(self) -> int:
        if self.resolved_dimension is not None:
            return self.resolved_dimension
        return len(self.fock_states)


def number_sector_states(n_modes: int, n_alpha: int, n_beta: int) -> list[int]:
    alpha_mask = sum(1 << m for m in range(0, n_modes, 2))
    beta_mask = sum(1 << m for m in range(1, n_modes, 2))
    return [
        state
        for state in range(1 << n_modes)
        if bin(state & alpha_mask).count("1") == n_alpha
        and bin(state & beta_mask).count("1") == n_beta
    ]


def total_number_states(n_modes: int, n_electrons: int) -> list[int]:
    return [
        state
        for state in range(1 << n_modes)
        if bin(state).count("1") == n_electrons
    ]


def exact_reference(
    one_body: np.ndarray,
    two_body: np.ndarray,
    constant: float,
    n_electrons: int,
) -> float:
    matrix = fock_matrix(one_body, two_body, constant)
    states = total_number_states(int(one_body.shape[0]), n_electrons)
    if not states:
        raise ValueError("empty electron sector")
    return float(np.linalg.eigvalsh(matrix[np.ix_(states, states)])[0])


def ground_sector(
    matrix: np.ndarray, n_modes: int, n_electrons: int
) -> tuple[Sector, np.ndarray]:
    states = total_number_states(n_modes, n_electrons)
    if not states:
        raise ValueError("empty electron sector")
    block = matrix[np.ix_(states, states)]
    alpha_mask = sum(1 << m for m in range(0, n_modes, 2))
    n_alpha = np.array([bin(s & alpha_mask).count("1") for s in states])
    coupling = block[n_alpha[:, None] != n_alpha[None, :]]
    if coupling.size and float(np.abs(coupling).max()) > 1e-12:
        return Sector(None, None, tuple(states)), block
    best = None
    for split in range(n_electrons + 1):
        keep = [i for i, count in enumerate(n_alpha) if count == split]
        if not keep:
            continue
        sub = block[np.ix_(keep, keep)]
        low = float(np.linalg.eigvalsh(sub)[0])
        candidate = (low, abs(2 * split - n_electrons), split, keep, sub)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    _, _, split, keep, sub = best
    kept = tuple(states[i] for i in keep)
    return Sector(split, n_electrons - split, kept), sub


def s2_matrix(n_modes: int) -> np.ndarray:
    dimension = 1 << n_modes
    s_plus = np.zeros((dimension, dimension))
    for orbital in range(n_modes // 2):
        for column in range(dimension):
            step = apply_string(
                column, ((2 * orbital, True), (2 * orbital + 1, False))
            )
            if step is not None:
                s_plus[step[1], column] += step[0]
    s_z = np.zeros(dimension)
    for state in range(dimension):
        n_alpha = bin(state & sum(1 << m for m in range(0, n_modes, 2))).count("1")
        n_beta = bin(state & sum(1 << m for m in range(1, n_modes, 2))).count("1")
        s_z[state] = 0.5 * (n_alpha - n_beta)
    return s_plus.T @ s_plus + np.diag(s_z * (s_z + 1.0))


def spin_resolve(
    matrix: np.ndarray,
    sector: Sector,
    n_modes: int,
    spin: float = 0.0,
    tolerance: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    states = list(sector.fock_states)
    block = matrix[np.ix_(states, states)]
    s2_block = s2_matrix(n_modes)[np.ix_(states, states)]
    if np.linalg.norm(block @ s2_block - s2_block @ block) > 1e-9:
        raise ValueError("S^2 does not commute with H on this sector")
    values, vectors = np.linalg.eigh(s2_block)
    target = spin * (spin + 1.0)
    if np.abs(values - np.round(values, 6)).max() > tolerance:
        raise ValueError("S^2 spectrum is not integer-clustered")
    keep = np.abs(values - target) < 1e-6
    if not keep.any():
        raise ValueError(f"no S={spin} component in this sector")
    isometry = vectors[:, keep]
    reduced = isometry.T @ block @ isometry
    return isometry, 0.5 * (reduced + reduced.T)


def sector_dimension(
    n_orbitals: int, n_electrons: int, spin_z: float = 0.0
) -> int:
    from math import comb

    n_alpha = n_electrons / 2 + spin_z
    if abs(n_alpha - round(n_alpha)) > 1e-12:
        raise ValueError("n_electrons / 2 + spin_z must be an integer")
    n_alpha = int(round(n_alpha))
    if n_alpha < 0 or n_electrons - n_alpha < 0:
        raise ValueError("invalid spin projection for this electron count")
    return comb(n_orbitals, n_alpha) * comb(n_orbitals, n_electrons - n_alpha)
