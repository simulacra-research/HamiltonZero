# Copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import IntEnum
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array


class MergeOpcode(IntEnum):
    MERGE = 0
    CARRY_LEFT = 1
    CARRY_RIGHT = 2
    EMPTY = 3


MERGE = int(MergeOpcode.MERGE)
CARRY_LEFT = int(MergeOpcode.CARRY_LEFT)
CARRY_RIGHT = int(MergeOpcode.CARRY_RIGHT)
EMPTY = int(MergeOpcode.EMPTY)
OPCODE_DTYPE = jnp.uint8


class QSideLeafInput(eqx.Module):
    weight: Array


class FactorizedQSide(eqx.Module):
    V: Array
    U: Array


class SharedKernel(eqx.Module):
    q_to_odd: QSideLeafInput
    leaf_factors: tuple[FactorizedQSide, ...]
    leaf_combiner_factors: tuple[FactorizedQSide, ...]
    merge_T: Array
    merge_factors: tuple[FactorizedQSide, ...]
    readout_factors: tuple[FactorizedQSide, ...]
    merge_eps: float = eqx.field(static=True)


class TrunkCompilerKernel(eqx.Module):
    featurizer: Any
    trunk: Any
    shared_global: Any


class LadderProjectionKernel(eqx.Module):
    weight: Array
    bias: Array
    norm_scale: Array


class PhysicalCompilerKernel(eqx.Module):
    contextualizer: Any
    global_fork: Any
    leaf: Any
    merge: Any
    readout: Any
    leaf_projection: LadderProjectionKernel
    tree_pool: Any
    tree_update: Any
    tree_projection_weight: Array
    tree_projection_bias: Array
    root_projection: LadderProjectionKernel


class ModelHamiltonianArrays(eqx.Module):
    coupling: Array
    full_coupling: Array
    field: Array


class GraphInputs(eqx.Module):
    node: Array
    edge: Array


class QuotientInputs(eqx.Module):
    node_key: Array
    edge_key: Array


class EnergyInputs(eqx.Module):
    custom_lap_J_eff: Array
    custom_lap_radial_const: Array
    one_body_fields: tuple[Array, ...]


class EnergyMasks(eqx.Module):
    real: Array
    balanced: Array


class EnergyFrame(eqx.Module):
    custom_lap_J_eff: Array
    w_levels: tuple[Array, ...]
    custom_lap_radial_const: Array
    one_body_fields: tuple[Array, ...]
    masks: EnergyMasks


EnergyFrameBatch = EnergyFrame


class CanonicalHamiltonian(eqx.Module):
    model_coupling_fields: ModelHamiltonianArrays
    node_mask: Array
    balanced_mask: Array
    graph_inputs: GraphInputs
    quotient_inputs: QuotientInputs
    energy_inputs: EnergyInputs
    system_identity: Array


class SharedTrunk(eqx.Module):
    node_raw: Array
    edge_raw: Array
    global_raw: Array
    global_stream: Array
    real_mask: Array
    balanced_mask: Array


class CompiledTree(eqx.Module):
    perm: Array
    inv_perm: Array
    leaf_real: Array
    leaf_h: tuple[Array, ...]
    leaf_combiner_h: tuple[Array, ...]
    merge_h: tuple[Array, ...]
    opcodes: tuple[Array, ...]
    readout_h: tuple[Array, ...]
    readout_combiner_h: tuple[Array, ...]


CompiledTreeBatch = CompiledTree


class CompiledWaveFunction(eqx.Module):
    kernel: SharedKernel
    tree: CompiledTree

    def __call__(self, q_routed, _ctx=None, _t=0.0):
        from .execute import execute_wavefunction

        return execute_wavefunction(self.kernel, self.tree, q_routed)


class CompiledWaveFunctions(eqx.Module):
    kernel: SharedKernel
    trees: CompiledTreeBatch

    def __call__(self, q_routed, _ctx=None, _t=0.0):
        from .execute import execute_wavefunction

        return execute_wavefunction(self.kernel, self.trees, q_routed)


def level_widths(n_sites: int) -> tuple[int, ...]:
    if n_sites <= 0 or n_sites & (n_sites - 1):
        raise ValueError(f"n_sites must be a positive power of two, got {n_sites}")
    return tuple(n_sites >> level for level in range(1, n_sites.bit_length()))
