# Copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools

import numpy as np

PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    "Y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
    "Z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
}


def weight(word: str) -> int:
    return sum(1 for symbol in word if symbol != "I")


def dense_fock_word(word: str) -> np.ndarray:
    matrix = np.eye(1, dtype=complex)
    for symbol in word:
        matrix = np.kron(PAULI[symbol], matrix)
    return matrix


def dense_site_word(word: str) -> np.ndarray:
    matrix = np.eye(1, dtype=complex)
    for symbol in word:
        matrix = np.kron(matrix, PAULI[symbol])
    return matrix


def apply_ladder(state: int, mode: int, dagger: bool) -> tuple[int, int] | None:
    occupied = (state >> mode) & 1
    if dagger == bool(occupied):
        return None
    lower = state & ((1 << mode) - 1)
    sign = -1 if bin(lower).count("1") % 2 else 1
    return sign, state ^ (1 << mode)


def apply_string(
    state: int, operators: tuple[tuple[int, bool], ...]
) -> tuple[int, int] | None:
    sign = 1
    for mode, dagger in reversed(operators):
        step = apply_ladder(state, mode, dagger)
        if step is None:
            return None
        factor, state = step
        sign *= factor
    return sign, state


def spinorb_from_spatial(
    one_body: np.ndarray, two_body: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    n = 2 * int(one_body.shape[0])
    spin = np.arange(n) % 2
    orb = np.arange(n) // 2
    one = one_body[np.ix_(orb, orb)] * (spin[:, None] == spin[None, :])
    two = two_body[np.ix_(orb, orb, orb, orb)] * (
        (spin[:, None, None, None] == spin[None, None, None, :])
        & (spin[None, :, None, None] == spin[None, None, :, None])
    )
    return one, two


def fock_matrix(
    one_body: np.ndarray, two_body: np.ndarray, constant: float
) -> np.ndarray:
    n_modes = int(one_body.shape[0])
    dimension = 1 << n_modes
    matrix = constant * np.eye(dimension, dtype=complex)
    ones = list(zip(*np.nonzero(np.abs(one_body) > 1e-14), strict=True))
    twos = list(zip(*np.nonzero(np.abs(two_body) > 1e-14), strict=True))
    for column in range(dimension):
        for p, q in ones:
            step = apply_string(column, ((int(p), True), (int(q), False)))
            if step is not None:
                matrix[step[1], column] += step[0] * one_body[p, q]
        for p, q, r, s in twos:
            step = apply_string(
                column,
                ((int(p), True), (int(q), True), (int(r), False), (int(s), False)),
            )
            if step is not None:
                matrix[step[1], column] += step[0] * two_body[p, q, r, s]
    if np.linalg.norm(matrix - matrix.conj().T) > 1e-9:
        raise ValueError("fermionic input is not Hermitian")
    hermitian = 0.5 * (matrix + matrix.conj().T)
    if float(np.abs(hermitian.imag).max()) > 1e-12:
        raise ValueError(
            "complex-valued Hamiltonians are not supported; supply real integrals"
        )
    return hermitian.real


def car_residual(n_modes: int) -> float:
    dimension = 1 << n_modes
    residual = 0.0
    for p in range(n_modes):
        for q in range(n_modes):
            a_p = np.zeros((dimension, dimension))
            a_q = np.zeros((dimension, dimension))
            adag_q = np.zeros((dimension, dimension))
            for column in range(dimension):
                for matrix, mode, dagger in (
                    (a_p, p, False), (a_q, q, False), (adag_q, q, True),
                ):
                    step = apply_ladder(column, mode, dagger)
                    if step is not None:
                        matrix[step[1], column] += step[0]
            anti = a_p @ a_q + a_q @ a_p
            mixed = a_p @ adag_q + adag_q @ a_p
            expected = np.eye(dimension) if p == q else 0.0
            residual = max(
                residual,
                float(np.abs(anti).max()),
                float(np.abs(mixed - expected).max()),
            )
    return residual


def pauli_expansion(
    matrix: np.ndarray, n_qubits: int, tolerance: float = 1e-11
) -> dict[str, float]:
    output: dict[str, float] = {}
    dimension = 1 << n_qubits
    for symbols in itertools.product("IXYZ", repeat=n_qubits):
        word = "".join(symbols)
        coefficient = complex(
            np.trace(dense_site_word(word).conj().T @ matrix) / dimension
        )
        if abs(coefficient.imag) > 1e-10:
            raise ValueError(f"non-real coefficient on {word}")
        if abs(coefficient.real) > tolerance:
            output[word] = float(coefficient.real)
    return output
