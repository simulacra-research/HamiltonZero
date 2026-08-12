# Copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, TypeVar

import equinox as eqx


T = TypeVar("T")
CheckpointKind = Literal["router", "compiled_finetune"]


def _metadata_path(path: str | Path) -> Path:
    source = Path(path)
    return source.with_name(source.name + ".json")


def save_model(
    path: str | Path,
    model: object,
    *,
    kind: CheckpointKind | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(destination, model)
    if kind is not None:
        payload = {"kind": kind, **(metadata or {})}
        _metadata_path(destination).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )


def load_model(path: str | Path, template: T) -> T:
    return eqx.tree_deserialise_leaves(Path(path), template)


def load_model_metadata(path: str | Path) -> dict[str, Any] | None:
    source = _metadata_path(path)
    return json.loads(source.read_text()) if source.exists() else None


def save_mcmc(path: str | Path, state: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(destination, state)


def load_mcmc(path: str | Path, template: T) -> T:
    return eqx.tree_deserialise_leaves(Path(path), template)


__all__ = [
    "CheckpointKind",
    "load_mcmc",
    "load_model",
    "load_model_metadata",
    "save_mcmc",
    "save_model",
]
