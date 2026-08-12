# Copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

from hamiltonzero.config import EvalConfig
from hamiltonzero.evaluation import EvalBackend, EvalMetric, EvalResult, evaluate


def _strict_json(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _strict_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json(item) for item in value]
    return value


def run(config: EvalConfig, backend: EvalBackend | None = None) -> EvalResult:
    if backend is None:
        from hamiltonzero.evaluation.runtime import build_eval_backend

        backend = build_eval_backend()
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    metrics = output / "eval.metrics.jsonl"
    metrics.write_text("")

    def write_metric(metric: EvalMetric) -> None:
        with metrics.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    _strict_json(dataclasses.asdict(metric)),
                    separators=(",", ":"),
                )
                + "\n"
            )

    result = evaluate(config, backend, metric_sink=write_metric)
    destination = output / "eval.json"
    temporary = output / "eval.json.tmp"
    temporary.write_text(
        json.dumps(_strict_json(result.as_dict()), indent=2, sort_keys=True) + "\n"
    )
    temporary.replace(destination)
    return result


__all__ = ["run"]
