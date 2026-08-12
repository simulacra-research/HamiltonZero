# Copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import jax


def snis_mode_baseline(
    total,
    candidate_log_p,
    sampled_log_p,
):
    if total.shape != candidate_log_p.shape:
        raise ValueError("baseline energy and log-density shapes differ")
    if sampled_log_p.shape != total.shape:
        raise ValueError("sampled log-density must match baseline energy")
    weights = jax.nn.softmax(
        candidate_log_p.astype(total.real.dtype)
        - sampled_log_p.astype(total.real.dtype),
        axis=-1,
    )
    return weights


__all__ = ["snis_mode_baseline"]
