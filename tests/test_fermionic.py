# Copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hamiltonzero.fermionic import (
    binary_encoding,
    compile_fermionic_sector,
    exact_reference,
    fock_matrix,
    s2_matrix,
    sector_dimension,
    spinorb_from_spatial,
    total_number_states,
)

H2O = json.loads(
    (Path(__file__).resolve().parent.parent / "examples" / "h2o_sto3g_cas22.json")
    .read_text()
)
H2O_CASCI = float(H2O["casci_energy"])


def h2o_case():
    one, two = spinorb_from_spatial(
        np.array(H2O["one_body_spatial"]), np.array(H2O["two_body_spatial"])
    )
    return one, two, float(H2O["constant"])


def random_number_conserving(n_modes: int, seed: int, n_terms: int = 24):
    rng = np.random.default_rng(seed)
    one = rng.normal(size=(n_modes, n_modes))
    one = 0.5 * (one + one.T)
    two = np.zeros((n_modes,) * 4)
    spins = np.arange(n_modes) % 2
    count = 0
    while count < n_terms:
        p, q, r, s = rng.integers(0, n_modes, size=4)
        if p == q or r == s:
            continue
        if sorted((spins[p], spins[q])) != sorted((spins[r], spins[s])):
            continue
        value = rng.normal()
        two[p, q, r, s] += value
        two[s, r, q, p] += value
        count += 1
    return one, two, float(rng.normal())


def test_h2o_reference_and_binary_sector():
    one, two, constant = h2o_case()
    assert abs(exact_reference(one, two, constant, 2) - H2O_CASCI) < 1e-9
    compiled = compile_fermionic_sector(one, two, constant, 2, spin=None)
    assert compiled.n_qubits == 4
    assert compiled.checks["encoding"] == "binary"
    assert abs(compiled.reference_energy - H2O_CASCI) < 1e-9
    assert abs(compiled.checks["bias_hartree"]) < 1e-10
    assert abs(
        compiled.to_hartree(compiled.raw_model_reference) - compiled.reference_energy
    ) < 1e-9


def test_h2o_spin_resolution_shrinks_register():
    one, two, constant = h2o_case()
    compiled = compile_fermionic_sector(one, two, constant, 2, spin=0.0)
    assert compiled.checks["sector_dimension"] == 3
    assert compiled.sector.dimension == 3
    assert abs(compiled.reference_energy - H2O_CASCI) < 1e-9


def brute_reference(one, two, constant, n_electrons):
    matrix = fock_matrix(one, two, constant)
    states = total_number_states(int(one.shape[0]), n_electrons)
    return float(np.linalg.eigvalsh(matrix[np.ix_(states, states)])[0])


@pytest.mark.parametrize(
    "seed,n_modes,dimension", [(0, 6, 15), (1, 6, 15), (2, 6, 15), (3, 4, 6), (4, 4, 6)]
)
def test_random_number_conserving_zero_bias(seed, n_modes, dimension):
    one, two, constant = random_number_conserving(n_modes, seed)
    brute = brute_reference(one, two, constant, 2)
    assert abs(exact_reference(one, two, constant, 2) - brute) < 1e-9
    compiled = compile_fermionic_sector(one, two, constant, 2)
    assert compiled.checks["encoding"] == "one-hot"
    assert compiled.checks["sector_dimension"] == dimension
    assert compiled.sector.dimension == dimension
    assert abs(compiled.reference_energy - brute) < 1e-9
    assert abs(
        compiled.to_hartree(compiled.raw_model_reference) - compiled.reference_energy
    ) < 1e-9


def test_spin_flip_hopping_sector():
    one = np.array([[0.0, 1.0], [1.0, 0.0]])
    two = np.zeros((2, 2, 2, 2))
    assert abs(exact_reference(one, two, 0.0, 1) + 1.0) < 1e-12
    compiled = compile_fermionic_sector(one, two, 0.0, 1)
    assert compiled.checks["encoding"] == "binary"
    assert compiled.sector.n_alpha is None
    assert abs(compiled.reference_energy + 1.0) < 1e-12
    assert abs(compiled.checks["bias_hartree"]) < 1e-10


def test_complex_input_rejected():
    one = np.zeros((2, 2), dtype=complex)
    one[0, 1] = 1j
    one[1, 0] = -1j
    with pytest.raises(ValueError, match="complex"):
        fock_matrix(one, np.zeros((2, 2, 2, 2)), 0.0)


def test_traceless_binary_block():
    terms, shift, checks = binary_encoding(np.diag([-2.0, 0.0, 0.0, 2.0]))
    assert checks["encoding"] == "binary"
    assert abs(checks["bias_hartree"]) < 1e-10


def test_spin_request_failure_raises():
    one, two, constant = h2o_case()
    with pytest.raises(ValueError, match="no S=2.0 component"):
        compile_fermionic_sector(one, two, constant, 2, spin=2.0)


def test_invalid_spin_rejected():
    one, two, constant = h2o_case()
    for bad in (-1.0, 0.3):
        with pytest.raises(ValueError, match="half-integer"):
            compile_fermionic_sector(one, two, constant, 2, spin=bad)


def test_odd_mode_spin_operators_rejected():
    with pytest.raises(ValueError, match="even number of modes"):
        s2_matrix(3)


def test_spin_projection_recorded():
    one, two, constant = h2o_case()
    compiled = compile_fermionic_sector(one, two, constant, 2, spin=0.0)
    matrix = fock_matrix(one, two, constant)
    states = list(compiled.sector.fock_states)
    block = matrix[np.ix_(states, states)]
    assert compiled.projection.shape == (4, 3)
    projected = compiled.projection.T @ block @ compiled.projection
    assert abs(
        float(np.linalg.eigvalsh(projected)[0]) - compiled.reference_energy
    ) < 1e-9


def test_odd_electron_sector_dimension():
    with pytest.raises(ValueError):
        sector_dimension(3, 3)
    assert sector_dimension(3, 3, spin_z=0.5) == 9
    assert sector_dimension(4, 3, spin_z=0.5) == 24


def test_normalization_convention():
    one, two, constant = random_number_conserving(6, 5)
    unscaled = compile_fermionic_sector(
        one, two, constant, 2, spin=None, normalize=False
    )
    scaled = compile_fermionic_sector(one, two, constant, 2, spin=None)
    assert max(abs(v) for v in scaled.pauli_terms.values()) <= 1.0 + 1e-12
    assert abs(
        scaled.to_hartree(scaled.raw_model_reference)
        - unscaled.to_hartree(unscaled.raw_model_reference)
    ) < 1e-9


def test_sector_dimension_formula():
    from hamiltonzero.fermionic import number_sector_states

    assert sector_dimension(2, 2) == 4
    assert sector_dimension(4, 4) == 36
    assert len(number_sector_states(8, 2, 2)) == sector_dimension(4, 4)


def test_spin_arrays_conventions():
    one, two, constant = h2o_case()
    compiled = compile_fermionic_sector(one, two, constant, 2, spin=None)
    exchange, fields, coupling = compiled.spin_arrays()
    assert exchange.shape == (4, 4, 3, 3)
    assert np.allclose(exchange, np.swapaxes(np.swapaxes(exchange, 0, 1), 2, 3))
    assert coupling.max() == 1.0
