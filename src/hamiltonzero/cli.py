# Copyright (c) 2026 Simulacra Research Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from .config import load_config


def _write_metric(path: Path, metric) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(dataclasses.asdict(metric), separators=(",", ":")) + "\n"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hamiltonzero")
    commands = parser.add_subparsers(dest="mode", required=True)
    for mode in ("train", "finetune"):
        command = commands.add_parser(mode)
        command.add_argument("config", type=Path)
        command.add_argument("--reuse-mcmc", type=Path)
    evaluate = commands.add_parser("eval")
    evaluate.add_argument("config", type=Path)
    pathway = evaluate.add_mutually_exclusive_group()
    pathway.add_argument("--contest", action="store_true")
    pathway.add_argument("--large-n", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    config = load_config(args.config, args.mode)
    if args.mode == "eval":
        from .modes.eval import run

        if args.contest or args.large_n:
            config = dataclasses.replace(
                config,
                contest=bool(args.contest),
                large_n=bool(args.large_n),
            )
        run(config)
        return
    if args.reuse_mcmc is not None:
        config = dataclasses.replace(
            config,
            mcmc=dataclasses.replace(config.mcmc, reuse_mcmc=args.reuse_mcmc),
        )
    metrics_path = config.output.with_name(config.output.name + ".metrics.jsonl")
    sink = lambda metric: _write_metric(metrics_path, metric)
    if args.mode == "train":
        from .modes.train import run_train

        run_train(config, metric_sink=sink)
    else:
        from .modes.finetune import run_finetune

        run_finetune(config, metric_sink=sink)


if __name__ == "__main__":
    main()
