# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Modifications copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: MIT

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

try:
    from jax.experimental.pallas import triton as plgpu
except ImportError:
    from jax.experimental.pallas import gpu as plgpu
from packaging.version import Version


def sum_columns(x: jax.Array) -> jax.Array:
    return x.astype(jnp.float32).sum(axis=1, keepdims=True, dtype=jnp.float32)


def get_query_block_spec(block_len: int, width: int):
    return pl.BlockSpec(
        index_map=lambda i, j, k: (i, j, k, 0),
        block_shape=(None, block_len, None, width),
    )


def get_key_value_block_spec(seq_len: int, width: int):
    return pl.BlockSpec(
        index_map=lambda i, _j, k: (i, 0, k, 0),
        block_shape=(None, seq_len, None, width),
    )


def get_mask_block_spec(seq_len: int):
    return pl.BlockSpec(index_map=lambda i, _j, _k: (i, 0), block_shape=(None, seq_len))


def get_lse_block_spec(block_len: int) -> pl.BlockSpec:
    return pl.BlockSpec(
        index_map=lambda i, j, k: (i, j, k), block_shape=(None, block_len, None)
    )


def create_grid(
    batch_len: int, seq_len: int, num_heads: int, q_block_len: int
) -> tuple[int, int, int]:
    return (batch_len, seq_len // q_block_len, num_heads)


def big_number() -> float:
    return jnp.float32(-10000.0)


def compiler_params(num_warps, num_stages):
    if Version(jax.__version__) >= Version("0.4.34"):
        if hasattr(plgpu, "CompilerParams"):
            return plgpu.CompilerParams(num_warps=num_warps, num_stages=num_stages)
        elif hasattr(plgpu, "TritonCompilerParams"):
            return plgpu.TritonCompilerParams(
                num_warps=num_warps, num_stages=num_stages
            )
    else:
        return dict(triton=dict(num_warps=num_warps, num_stages=num_stages))
