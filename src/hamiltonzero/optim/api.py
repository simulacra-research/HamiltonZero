# Copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import kfac_jax

from .production import (
    KFACBundle,
    apply_finetune_kfac_step,
    apply_router_kfac_step,
    init_finetune_kfac_state,
    init_router_kfac_state,
)
from .targets import process_finetune_targets, process_route_targets


def learning_rate(config, step: int) -> float:
    return float(config.learning_rate_numerator) / (
        float(config.learning_rate_offset)
        + float(step) / float(config.learning_rate_decay_steps)
    )


def register_scale_and_shift(y, x, scale, tag_id: str):
    from hamiltonzero.model.tree import _kfac_name_kw

    return kfac_jax.register_scale_and_shift(
        y,
        x,
        scale=scale,
        shift=None,
        **_kfac_name_kw(tag_id),
    )


__all__ = [
    "KFACBundle",
    "apply_finetune_kfac_step",
    "apply_router_kfac_step",
    "init_finetune_kfac_state",
    "init_router_kfac_state",
    "learning_rate",
    "process_finetune_targets",
    "process_route_targets",
    "register_scale_and_shift",
]
