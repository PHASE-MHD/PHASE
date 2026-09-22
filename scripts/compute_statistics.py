#!/usr/bin/env python3
"""Compute PHASE normalization statistics from training data only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from phase.preprocessing import (
    diffusion_statistics,
    diffusion_statistics_by_re,
    multi_re_magnetic_p99,
    save_npz_statistics,
    trajectory_statistics,
)


SPLIT_MODES = ("single_re_seed42", "multi_re_seed_plus_index")


def add_split_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train-size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-mode", choices=SPLIT_MODES, default=SPLIT_MODES[0])
    parser.add_argument("--re-index", type=int, default=0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    trajectory = commands.add_parser("trajectory")
    trajectory.add_argument("--input", required=True)
    trajectory.add_argument("--output", required=True)
    add_split_arguments(trajectory)
    trajectory.add_argument("--sub-t", type=int, default=1)
    trajectory.add_argument("--sub-x", type=int, default=1)
    trajectory.add_argument("--time-stop-index", type=int)
    trajectory.add_argument("--chunk-size", type=int, default=8)

    diffusion = commands.add_parser("diffusion")
    diffusion.add_argument("--input", required=True)
    diffusion.add_argument("--output-prefix", required=True)
    diffusion.add_argument(
        "--prediction-mode", choices=("direct", "residual"), required=True
    )
    diffusion.add_argument("--chunk-size", type=int, default=8)

    per_re = commands.add_parser("diffusion-per-re")
    per_re.add_argument("--input", required=True)
    per_re.add_argument("--output-dir", required=True)
    per_re.add_argument("--prediction-mode", choices=("direct", "residual"), default="residual")
    per_re.add_argument("--chunk-size", type=int, default=8)

    multi = commands.add_parser("multi-re-p99")
    multi.add_argument("--data-root", required=True)
    multi.add_argument("--output", required=True)
    multi.add_argument("--re-values", nargs="+", type=float, required=True)
    multi.add_argument("--directory-template", default="mhd_Re{re}_N1000")
    multi.add_argument("--data-file", default="mhd_data_4channel.npy")
    add_split_arguments(multi)
    multi.set_defaults(split_mode="multi_re_seed_plus_index")
    multi.add_argument("--sub-t", type=int, default=1)
    multi.add_argument("--sub-x", type=int, default=1)
    multi.add_argument("--time-stop-index", type=int)
    multi.add_argument("--bins", type=int, default=20_000)
    multi.add_argument("--chunk-size", type=int, default=20)
    multi.add_argument("--percentile", type=float, default=99.0)
    return parser


def _re_label(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "trajectory":
        stats = trajectory_statistics(
            args.input,
            train_size=args.train_size,
            seed=args.seed,
            re_index=args.re_index,
            split_mode=args.split_mode,
            sub_t=args.sub_t,
            sub_x=args.sub_x,
            time_stop_index=args.time_stop_index,
            chunk_size=args.chunk_size,
        )
        output = save_npz_statistics(args.output, stats)
        print(f"Wrote {output}")
        return

    if args.command == "diffusion":
        stats = diffusion_statistics(
            args.input,
            prediction_mode=args.prediction_mode,
            chunk_size=args.chunk_size,
        )
        prefix = Path(args.output_prefix)
        input_path = save_npz_statistics(
            prefix.with_name(prefix.name + "_inputs"), stats["inputs"]
        )
        target_path = save_npz_statistics(
            prefix.with_name(prefix.name + "_targets"), stats["targets"]
        )
        print(f"Wrote {input_path}")
        print(f"Wrote {target_path}")
        return

    if args.command == "diffusion-per-re":
        results = diffusion_statistics_by_re(
            args.input, prediction_mode=args.prediction_mode, chunk_size=args.chunk_size
        )
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        target_kind = (
            "residual_targets"
            if args.prediction_mode == "residual"
            else "targets"
        )
        for re_value, stats in results.items():
            label = _re_label(re_value)
            input_path = save_npz_statistics(
                output_dir / f"Re{label}_inputs_stats", stats["inputs"]
            )
            target_path = save_npz_statistics(
                output_dir / f"Re{label}_{target_kind}_stats", stats["targets"]
            )
            print(f"Wrote {input_path}")
            print(f"Wrote {target_path}")
        return

    root = Path(args.data_root)
    paths = {}
    for value in args.re_values:
        directory = args.directory_template.format(
            re=_re_label(value), re_float=float(value)
        )
        paths[value] = root / directory / args.data_file
    stats = multi_re_magnetic_p99(
        paths,
        train_size=args.train_size,
        seed=args.seed,
        split_mode=args.split_mode,
        sub_t=args.sub_t,
        sub_x=args.sub_x,
        time_stop_index=args.time_stop_index,
        bins=args.bins,
        chunk_size=args.chunk_size,
        percentile=args.percentile,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as stream:
        json.dump(stats, stream, indent=2)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
