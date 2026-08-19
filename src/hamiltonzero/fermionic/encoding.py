# Copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools

import numpy as np

from .algebra import dense_site_word, pauli_expansion, weight

DEFAULT_PIN_FIELD = 0.25
LIFT_MARGIN = 2.0


def _zword(sites: set[int], n_qubits: int) -> str:
    return "".join("Z" if i in sites else "I" for i in range(n_qubits))


def binary_encoding(
    block: np.ndarray,
    pin_field: float = DEFAULT_PIN_FIELD,
) -> tuple[dict[str, float], float, dict]:
    d = int(block.shape[0])
    if d > 4:
        raise ValueError("binary encoding requires d <= 4")
    target = float(np.linalg.eigvalsh(block)[0])
    logical = np.zeros((4, 4))
    logical[:d, :d] = block
    lift = float(np.linalg.eigvalsh(block)[-1]) + LIFT_MARGIN
    for k in range(d, 4):
        logical[k, k] = lift
    terms = pauli_expansion(logical, 2)
    identity = terms.pop("II", 0.0)
    compiled = {word + "II": value for word, value in terms.items()}
    compiled["IIZI"] = -pin_field
    compiled["IIIZ"] = -pin_field
    shift = identity + 2.0 * pin_field
    matrix = sum(v * dense_site_word(w) for w, v in compiled.items())
    spectrum = np.linalg.eigvalsh(matrix)
    bias = float(spectrum[0]) + shift - target
    if abs(bias) > 1e-10:
        raise AssertionError(f"binary encoding bias {bias}")
    checks = {
        "encoding": "binary",
        "bias_hartree": bias,
        "gap_hartree": float(spectrum[1] - spectrum[0]),
        "pin_field_hartree": pin_field,
        "lifted_states": 4 - d,
    }
    return compiled, shift, checks


def _block_minimum(tilde: np.ndarray, excitations: int) -> float:
    d = int(tilde.shape[0])
    states = list(itertools.combinations(range(d), excitations))
    if not states:
        return 0.0
    index = {s: i for i, s in enumerate(states)}
    matrix = np.zeros((len(states), len(states)))
    for s, i in index.items():
        matrix[i, i] = sum(tilde[k, k] for k in s)
        occupied = set(s)
        for k in s:
            for l in range(d):
                if l in occupied:
                    continue
                t = tilde[min(k, l), max(k, l)]
                if abs(t) < 1e-14:
                    continue
                target = tuple(sorted(occupied - {k} | {l}))
                matrix[index[target], i] += t
    return float(np.linalg.eigvalsh(matrix)[0])


def onehot_encoding(
    block: np.ndarray,
    penalty_margin: float = 1.0,
) -> tuple[dict[str, float], float, dict]:
    d = int(block.shape[0])
    center = float(np.trace(block) / d)
    tilde = block - center * np.eye(d)
    target = float(np.linalg.eigvalsh(tilde)[0])

    off_max = float(np.abs(tilde - np.diag(np.diag(tilde))).max())
    diag_min = float(np.diag(tilde).min())
    requirements: dict[int, float] = {}
    exact_minima: dict[int, float] = {}
    for excitations in (0, 2, 3):
        if excitations > d:
            continue
        exact_minima[excitations] = _block_minimum(tilde, excitations)
        requirements[excitations] = (
            target + penalty_margin - exact_minima[excitations]
        ) / (excitations - 1) ** 2
    for excitations in range(4, d + 1):
        gershgorin = excitations * diag_min - excitations * (
            d - excitations
        ) * off_max
        requirements[excitations] = (target + penalty_margin - gershgorin) / (
            excitations - 1
        ) ** 2
    lam = max(1.0, max(requirements.values()))

    terms: dict[str, float] = {}

    def add(word: str, value: float) -> None:
        if abs(value) > 1e-11:
            terms[word] = terms.get(word, 0.0) + value

    for k in range(d):
        for l in range(k + 1, d):
            t = float(tilde[k, l])
            if abs(t) > 1e-11:
                for axis in "XY":
                    add(
                        "".join(axis if i in (k, l) else "I" for i in range(d)),
                        t / 2.0,
                    )
    diag_constant = 0.0
    for k in range(d):
        add(_zword({k}, d), -float(tilde[k, k]) / 2.0)
        diag_constant += float(tilde[k, k]) / 2.0

    def diag_product(a: dict, b: dict) -> dict:
        out: dict[tuple, float] = {}
        for sa, ca in a.items():
            for sb, cb in b.items():
                key = tuple(sorted(set(sa) ^ set(sb)))
                out[key] = out.get(key, 0.0) + ca * cb
        return out

    n_minus_1: dict[tuple, float] = {(): d / 2.0 - 1.0}
    for k in range(d):
        n_minus_1[(k,)] = -0.5
    squared = diag_product(n_minus_1, n_minus_1)
    penalty_identity = lam * squared.pop(())
    for sites, value in squared.items():
        add(_zword(set(sites), d), lam * value)

    shift = center + diag_constant + penalty_identity
    if max(map(weight, terms)) > 2:
        raise AssertionError("one-hot encoding produced weight > 2")

    rebuilt = np.zeros((d, d))
    for word, value in terms.items():
        support = [(i, s) for i, s in enumerate(word) if s != "I"]
        if all(s == "Z" for _, s in support):
            for k in range(d):
                sign = 1.0
                for i, _ in support:
                    sign *= -1.0 if i == k else 1.0
                rebuilt[k, k] += value * sign
        else:
            (i, _), (j, _) = support
            rebuilt[i, j] += value
            rebuilt[j, i] += value
    if not np.allclose(rebuilt + (shift - center) * np.eye(d), tilde, atol=1e-9):
        raise AssertionError("one-hot N=1 block reconstruction failed")

    checks: dict = {
        "encoding": "one-hot",
        "lambda_hartree": lam,
        "penalty_margin_hartree": penalty_margin,
    }
    if d <= 12:
        matrix = sum(v * dense_site_word(w) for w, v in terms.items())
        spectrum = np.linalg.eigvalsh(matrix)
        bias = float(spectrum[0]) + shift - (target + center)
        if abs(bias) > 1e-9:
            raise AssertionError(f"one-hot encoding bias {bias}")
        checks["bias_hartree"] = bias
        checks["gap_hartree"] = float(spectrum[1] - spectrum[0])
    else:
        margins = {
            n: exact_minima[n] + lam * (n - 1) ** 2 - target for n in exact_minima
        }
        if min(margins.values()) < 0.99 * penalty_margin:
            raise AssertionError(f"penalty margins too small: {margins}")
        checks["explicit_block_margins_hartree"] = {
            str(n): v for n, v in margins.items()
        }
    return terms, shift, checks


def normalize_terms(
    terms: dict[str, float]
) -> tuple[dict[str, float], float]:
    scale = max(1.0, max(abs(value) for value in terms.values()))
    return {word: value / scale for word, value in terms.items()}, scale
