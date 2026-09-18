#!/usr/bin/env python3
"""Evaluate one model checkpoint on held-out test trajectories only."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from phase.evaluation.inference import build_test_inference
from phase.evaluation.metrics import evaluate_records
from phase.evaluation.reporting import sha256_file, write_evaluation_report
from phase.utils import load_config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--problem", choices=("turbulence", "kh"), required=True)
    parser.add_argument("--re", type=float, required=True, help="Evaluation Re=Rm.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-sample-steps", type=int, default=32)
    parser.add_argument("--diffusion-seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--lx", type=float, default=1.0)
    parser.add_argument("--ly", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    config = load_config(config_path)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable.")

    records, metadata = build_test_inference(
        config,
        checkpoint_path,
        target_re=args.re,
        device=device,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        diffusion_seed=args.diffusion_seed,
        num_sample_steps=args.num_sample_steps,
    )
    results = evaluate_records(records, problem=args.problem, lx=args.lx, ly=args.ly)
    metadata.update(
        {
            "problem": args.problem,
            "config": str(config_path),
            "config_sha256": sha256_file(config_path),
            "device": str(device),
            "domain": {"lx": args.lx, "ly": args.ly},
        }
    )
    paths = write_evaluation_report(args.output_dir, metadata, results)
    print(f"Evaluated {results['number_of_samples']} held-out test trajectories.")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
