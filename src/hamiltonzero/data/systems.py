# Copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from hamiltonzero.hamiltonian import SpinHamiltonian, _exchange_matrix


def _load_payload(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if source.suffix == ".jsonl":
        records = []
        with source.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"JSONL record {line_number} must be an object")
                records.append(record)
        return {"systems": records}
    payload = json.loads(source.read_text())
    if not isinstance(payload, dict):
        raise ValueError("Hamiltonian dataset must be a JSON object")
    return payload


def _from_sparse(spec: dict[str, Any]) -> SpinHamiltonian:
    n_sites = int(spec.get("n_sites", spec.get("n_spins", 0)))
    if n_sites <= 0:
        raise ValueError("sparse Hamiltonian requires a positive n_sites")
    exchange = np.zeros((n_sites, n_sites, 3, 3), dtype=np.float32)
    coupling = np.zeros((n_sites, n_sites), dtype=np.float32)
    seen: set[tuple[int, int]] = set()
    for offset, term in enumerate(spec["exchange"]):
        if not isinstance(term, list) or len(term) != 3:
            raise ValueError(f"exchange term {offset} must be [i,j,J]")
        left, right, value = term
        if (
            not isinstance(left, int)
            or isinstance(left, bool)
            or not isinstance(right, int)
            or isinstance(right, bool)
            or not 0 <= left < right < n_sites
        ):
            raise ValueError(
                f"exchange term {offset} must satisfy 0 <= i < j < n_sites"
            )
        pair = (left, right)
        if pair in seen:
            raise ValueError(f"duplicate exchange term for sites {pair}")
        seen.add(pair)
        matrix = _exchange_matrix(value)
        exchange[left, right] = matrix
        exchange[right, left] = matrix.T
        coupling[left, right] = coupling[right, left] = 1.0
    field = spec.get("field", spec.get("h", spec.get("h_field", 0.0)))
    return SpinHamiltonian.from_arrays(
        exchange,
        field,
        coupling=coupling,
        nodes=spec.get("nodes"),
        mu=spec.get("mu"),
    )


def _next_power_of_two(value: int) -> int:
    return 1 if value <= 1 else 1 << (value - 1).bit_length()


def _from_record(
    record: dict[str, Any],
    *,
    needs_fwl2: bool | None = None,
) -> SpinHamiltonian:
    outer = record
    spec = record.get("spec", record)
    convention = spec.get("convention", "textbook")
    if convention != "textbook":
        raise ValueError("public Hamiltonian JSON must use convention='textbook'")
    if "exchange" in spec:
        system = _from_sparse(spec)
    else:
        system = SpinHamiltonian.from_arrays(
            spec["J"],
            spec.get("h", spec.get("h_field", 0.0)),
            coupling=spec.get("coupling"),
            nodes=spec.get("nodes"),
            mu=spec.get("mu"),
        )
    if needs_fwl2 is None:
        needs_fwl2 = outer.get("needs_fwl2", spec.get("needs_fwl2"))
    metadata = {
        name: outer.get(name, spec.get(name))
        for name in ("category", "tag", "topology_class", "j_class")
    }
    return replace(
        system,
        _needs_fwl2=needs_fwl2,
        _category=metadata["category"],
        _tag=metadata["tag"],
        _topology_class=metadata["topology_class"],
        _j_class=metadata["j_class"],
    )


def load_system(path: str | Path) -> SpinHamiltonian:
    payload = _load_payload(path)
    if "systems" in payload:
        systems = payload["systems"]
        if len(systems) != 1:
            raise ValueError("load_system requires exactly one system")
        dispatch = payload.get("needs_fwl2", payload.get("dataset_needs_fwl2"))
        if dispatch is not None:
            if not isinstance(dispatch, list) or len(dispatch) != 1:
                raise ValueError("needs_fwl2 sidecar must align with systems")
            return _from_record(systems[0], needs_fwl2=bool(dispatch[0]))
        return _from_record(systems[0])
    return _from_record(payload)


def load_systems(path: str | Path) -> list[SpinHamiltonian]:
    payload = _load_payload(path)
    records = payload.get("systems", [payload])
    dispatch = (
        payload.get("needs_fwl2", payload.get("dataset_needs_fwl2"))
        if "systems" in payload
        else None
    )
    if dispatch is None and isinstance(payload.get("per_system"), list):
        derived = payload["per_system"]
        if len(derived) == len(records) and all(
            isinstance(value, dict) and "needs_fwl2" in value for value in derived
        ):
            dispatch = [value["needs_fwl2"] for value in derived]
    if dispatch is not None:
        if not isinstance(dispatch, list) or len(dispatch) != len(records):
            raise ValueError("needs_fwl2 sidecar must align with systems")
        return [
            _from_record(record, needs_fwl2=bool(value))
            for record, value in zip(records, dispatch, strict=True)
        ]
    return [_from_record(record) for record in records]


def save_system(path: str | Path, system: SpinHamiltonian) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(system.to_dict(), indent=2) + "\n")


def padded_model_arrays(
    system: SpinHamiltonian,
    n_max: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    width = _next_power_of_two(system.n_spins) if n_max is None else int(n_max)
    if width < system.n_spins:
        raise ValueError("n_max cannot be smaller than the system")
    if width <= 0 or width & (width - 1):
        raise ValueError("n_max must be a positive power of two")
    coupling, exchange, field = system.model_arrays()
    padding = width - system.n_spins
    coupling = np.pad(coupling, ((0, padding), (0, padding)))
    exchange = np.pad(exchange, ((0, padding), (0, padding), (0, 0), (0, 0)))
    field = np.pad(field, ((0, padding), (0, 0)))
    mask = np.zeros((width,), dtype=np.int32)
    mask[: system.n_spins] = 1
    return coupling, exchange, field, mask


def _context_arrays(system: SpinHamiltonian, n_max: int | None):
    import jax.numpy as jnp

    from hamiltonzero.model.route_quotient import system_needs_fwl2

    _coupling, exchange, field, mask = padded_model_arrays(system, n_max)
    needs_fwl2 = system._needs_fwl2
    if needs_fwl2 is None:
        _, physical_exchange, physical_field = system.model_arrays()
        needs_fwl2 = system_needs_fwl2(
            physical_exchange,
            physical_field,
            system.n_spins,
            category=system._category,
            tag=system._tag,
            topology_class=system._topology_class,
            j_class=system._j_class,
        )
    return (
        jnp.asarray(exchange),
        jnp.asarray(field),
        jnp.asarray(mask),
        needs_fwl2,
    )


def build_context(
    system: SpinHamiltonian,
    n_max: int | None = None,
):
    from hamiltonzero.model import SpinContext

    exchange, field, mask, needs_fwl2 = _context_arrays(system, n_max)
    return SpinContext(
        J_full=exchange,
        h=field,
        mask=mask,
        needs_fwl2=needs_fwl2,
    )


def build_context_and_energy(
    system: SpinHamiltonian,
    n_max: int | None = None,
    mu: float | None = None,
    eps: float = 0.1,
):
    from hamiltonzero.energy.frame import build_energy_inputs
    from hamiltonzero.model import SpinContext

    exchange, field, mask, needs_fwl2 = _context_arrays(system, n_max)
    context = SpinContext(
        J_full=exchange,
        h=field,
        mask=mask,
        needs_fwl2=needs_fwl2,
    )
    energy_inputs = build_energy_inputs(
        exchange,
        field,
        mask,
        system.mu if mu is None else mu,
        eps,
    )
    return context, energy_inputs


def build_multi_context(
    systems: Iterable[SpinHamiltonian],
    n_max: int,
):
    from hamiltonzero.model import MultiSystemContext

    system_list = list(systems)
    contexts = [build_context(system, n_max=n_max) for system in system_list]
    return MultiSystemContext.stack(contexts)


__all__ = [
    "build_context",
    "build_context_and_energy",
    "build_multi_context",
    "load_system",
    "load_systems",
    "padded_model_arrays",
    "save_system",
]
