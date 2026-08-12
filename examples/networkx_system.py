# Copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import networkx as nx

from hamiltonzero import SpinHamiltonian
from hamiltonzero.data import save_system


graph = nx.path_graph(8)
nx.set_edge_attributes(graph, 1.0, "J")
nx.set_node_attributes(graph, 0.0, "h")
system = SpinHamiltonian.from_networkx(graph)
save_system(Path("outputs/systems/chain_8.json"), system)
