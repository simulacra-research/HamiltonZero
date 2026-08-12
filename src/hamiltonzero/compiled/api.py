# Copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import jax
import jax.numpy as jnp

from .tree import compile_physical_tree_reference
from .trunk import bind_shared_kernel, compile_shared_trunk
from .types import CompiledWaveFunction, CompiledWaveFunctions


def compile_wavefunction(model, context, perm) -> CompiledWaveFunction:
    trunk = compile_shared_trunk(model, context)
    tree = compile_physical_tree_reference(model, trunk, perm)
    return CompiledWaveFunction(kernel=bind_shared_kernel(model), tree=tree)


def compile_wavefunctions(model, context, perms) -> CompiledWaveFunctions:
    trunk = compile_shared_trunk(model, context)
    perms = jnp.asarray(perms, dtype=jnp.int32)
    trees = jax.vmap(lambda perm: compile_physical_tree_reference(model, trunk, perm))(
        perms
    )
    return CompiledWaveFunctions(kernel=bind_shared_kernel(model), trees=trees)


def select_compiled_wavefunction(
    candidates: CompiledWaveFunctions,
    winner,
) -> CompiledWaveFunction:
    tree = jax.tree_util.tree_map(lambda value: value[winner], candidates.trees)
    return CompiledWaveFunction(kernel=candidates.kernel, tree=tree)
