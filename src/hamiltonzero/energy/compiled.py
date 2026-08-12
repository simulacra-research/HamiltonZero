# Copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from typing import Any

import jax

from hamiltonzero.energy.kernel import (
    _vmc_energy_custom_lap_finetune,
    _vmc_energy_custom_lap_prebuilt,
)


def vmc_energy_custom_lap_compiled(
    kernel: Any,
    tree: Any,
    energy_frame: Any,
    q_routed: jax.Array,
    *,
    chunk_size: int | None = 512,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    return _vmc_energy_custom_lap_prebuilt(
        kernel,
        tree,
        energy_frame,
        q_routed,
        chunk_size=chunk_size,
    )


def vmc_energy_custom_lap_finetune(
    model: Any,
    energy_frame: Any,
    q_routed: jax.Array,
    *,
    chunk_size: int | None = 512,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    from hamiltonzero.compiled.model import CompiledFinetuneWaveFunction

    if not isinstance(model, CompiledFinetuneWaveFunction):
        raise TypeError("fine-tune energy requires CompiledFinetuneWaveFunction")
    return _vmc_energy_custom_lap_finetune(
        model,
        energy_frame,
        q_routed,
        0.0,
        chunk_size=chunk_size,
    )


__all__ = [
    "vmc_energy_custom_lap_compiled",
    "vmc_energy_custom_lap_finetune",
]
