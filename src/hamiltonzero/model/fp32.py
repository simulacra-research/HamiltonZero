# Copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp


def compute_dtype(*_args, **_kwargs):
    return jnp.float32


__all__ = ["compute_dtype"]
