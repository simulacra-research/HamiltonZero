# Copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: Apache-2.0

import jax


def fused_silu(x):
    return jax.nn.silu(x)


__all__ = ["fused_silu"]
