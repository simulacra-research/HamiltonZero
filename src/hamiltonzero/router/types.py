# Copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import equinox as eqx
from jaxtyping import Array


class RouterKernel(eqx.Module):
    contextualizer: object
    global_fork: object
    decoder: object


class RouterStatic(eqx.Module):
    node_input: Array
    node_projected: Array
    global_input: Array
    global_projected: Array
    raw_edge: Array
    initial_suffix: Array
    prefix_edge_messages: Array
    suffix_edge_messages: Array
    order_decay: Array
    virtual_decay: Array
    tree_pair_messages: Array
    static_bias_tables: tuple[Array, ...]
    quadratic_base_static: tuple[Array, ...]
    forked_global_static: tuple[Array, ...]
    quotient_node_key: Array
    quotient_edge_key: Array
    real_mask: Array
    routable_mask: Array
    needs_fwl2: Array


__all__ = ["RouterKernel", "RouterStatic"]
