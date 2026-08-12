# Copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np

from hamiltonzero import (
    SpinHamiltonian,
    burn_in,
    burn_in_basis,
    energy,
    measure_renyi2,
    prepare,
    spin,
    step,
    step_basis,
)


graph = nx.path_graph(8)
nx.set_edge_attributes(graph, 1.0, "J")
system = SpinHamiltonian.from_networkx(graph)

route_key, mcmc_key, basis_x_key, basis_y_key = jax.random.split(
    jax.random.PRNGKey(0), 4
)
compiled, order = prepare(
    system,
    Path("weights/hamiltonzero_v1.eqx"),
    route_key,
)
state, q = burn_in(
    compiled,
    mcmc_key,
    batch_size=256,
    replicas=8,
    burn_in=1024,
    walker_chunk_size=16,
)

local_energy = energy(compiled, q)
local_spin = spin(compiled, q)
print("leaf_to_input", np.asarray(order.leaf_to_input))
print("input_to_leaf", np.asarray(order.input_to_leaf))
print("energy", float(jnp.mean(local_energy.total.real)))
print("energy_std", float(jnp.std(local_energy.total.real)))
print("spin", np.asarray(jnp.mean(local_spin.real, axis=0)))

state, q = step(compiled, state, steps=24, walker_chunk_size=16)

basis_x, bits_x = burn_in_basis(compiled, basis_x_key, batch_size=256, burn_in=1024)
basis_y, bits_y = burn_in_basis(compiled, basis_y_key, batch_size=256, burn_in=1024)
basis_x, bits_x = step_basis(compiled, basis_x, steps=24)
basis_y, bits_y = step_basis(compiled, basis_y, steps=24)
basis_x, basis_y, purity = measure_renyi2(
    compiled,
    basis_x,
    basis_y,
    subsystem=range(4),
    blocks=16,
    samples_per_block=1,
    steps_between=24,
)
print("purity", purity.purity)
print("purity_standard_error", purity.standard_error)
print("purity_resolved", purity.resolved, purity.failure_reasons)
print("renyi2_nats", purity.renyi2_nats)
