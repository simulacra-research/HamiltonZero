# Copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: Apache-2.0

from .algebra import (
    car_residual,
    dense_fock_word,
    dense_site_word,
    fock_matrix,
    pauli_expansion,
    spinorb_from_spatial,
    weight,
)
from .compile import AXIS, CompiledSector, compile_fermionic_sector
from .encoding import (
    binary_encoding,
    normalize_terms,
    onehot_encoding,
)
from .sectors import (
    Sector,
    exact_reference,
    ground_sector,
    number_sector_states,
    s2_matrix,
    sector_dimension,
    spin_resolve,
    total_number_states,
)

__all__ = [
    "AXIS",
    "CompiledSector",
    "Sector",
    "binary_encoding",
    "car_residual",
    "compile_fermionic_sector",
    "dense_fock_word",
    "dense_site_word",
    "exact_reference",
    "fock_matrix",
    "ground_sector",
    "normalize_terms",
    "number_sector_states",
    "onehot_encoding",
    "pauli_expansion",
    "s2_matrix",
    "sector_dimension",
    "spin_resolve",
    "spinorb_from_spatial",
    "total_number_states",
    "weight",
]
