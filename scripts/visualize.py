#!/usr/bin/env python3
"""Visualize one held-out test trajectory from a public PHASE model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from phase.evaluation.inference import build_test_inference
from phase.evaluation.physics import derive_fields
from phase.evaluation.reporting import sha256_file
from phase.utils import load_config
from phase.visualization import (
    advect_tracer,
    plot_field_comparison,
    plot_pdf_comparison,
    plot_spectrum_comparison,
    plot_tracer_comparison,
    resolve_time_indices,
)


PRODUCTS = {"fields", "spectra", "pdfs", "tracer"}


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--problem", choices=("turbulence", "kh"), required=True)
    parser.add_argument("--re", type=positive_float, required=True)
    parser.add_argument("--sample-id", type=nonnegative_int, required=True)
    parser.add_argument(
        "--products",
        nargs="+",
        choices=sorted(PRODUCTS),
        default=None,
        help="Defaults to fields/spectra/pdfs for turbulence and fields/tracer for KH.",
    )
    parser.add_argument(
        "--times",
        type=float,
        nargs="+",
        default=None,
        help="Physical output times; nearest stored frames are used. Default: final.",
    )
    parser.add_argument(
        "--time-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("START", "STOP"),
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf"),
        default=("png", "pdf"),
    )
    parser.add_argument("--dpi", type=positive_int, default=240)
    parser.add_argument("--num-workers", type=nonnegative_int, default=0)
    parser.add_argument("--num-sample-steps", type=positive_int, default=32)
    parser.add_argument("--diffusion-seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--lx", type=positive_float, default=1.0)
    parser.add_argument("--ly", type=positive_float, default=1.0)
    parser.add_argument("--tracer-delta", type=positive_float, default=0.05)
    parser.add_argument("--tracer-diffusivity", type=nonnegative_float, default=None)
    return parser.parse_args()


def _time_range(config: dict, problem: str, override) -> tuple[float, float]:
    if override is not None:
        start, stop = override
    else:
        dataset = config.get("dataset_params", {})
        values = dataset.get("source_time_range", dataset.get("t_range"))
        if values is None:
            values = (0.0, 5.0 if problem == "kh" else 1.0)
        start, stop = values
    start, stop = float(start), float(stop)
    if stop <= start:
        raise ValueError(f"Time range must satisfy STOP > START, got {(start, stop)}.")
    return start, stop


def _products(problem: str, requested) -> list[str]:
    products = list(requested or (
        ("fields", "tracer") if problem == "kh" else ("fields", "spectra", "pdfs")
    ))
    if problem == "kh" and any(name in {"spectra", "pdfs"} for name in products):
        raise ValueError("KH visualization excludes turbulence spectra and PDFs.")
    if problem == "turbulence" and "tracer" in products:
        raise ValueError("Passive-tracer visualization is KH-only.")
    return products


def _tag(time_value: float) -> str:
    return f"t{time_value:07.3f}".replace("-", "m").replace(".", "p")


def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    products = _products(args.problem, args.products)
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
        max_samples=1,
        diffusion_seed=args.diffusion_seed,
        num_sample_steps=args.num_sample_steps,
        sample_ids={args.sample_id},
    )
    try:
        record = next(records)
    except StopIteration as error:
        raise ValueError(
            f"Sample ID {args.sample_id} was not found in the held-out test "
            f"features for Re={args.re}."
        ) from error
    if record.sample_id != args.sample_id:
        raise RuntimeError("Inference returned a different sample than requested.")

    time_range = _time_range(config, args.problem, args.time_range)
    times, time_indices = resolve_time_indices(
        record.prediction.shape[1], time_range, args.times
    )
    formats = tuple(args.formats)
    outputs = []

    for index in time_indices:
        time_value = float(times[index])
        tag = _tag(time_value)
        common = {
            "prediction": record.prediction,
            "truth": record.truth,
            "representation": record.representation,
            "time_index": index,
            "time_value": time_value,
            "formats": formats,
            "dpi": args.dpi,
            "lx": args.lx,
            "ly": args.ly,
        }
        if "fields" in products:
            outputs.extend(
                plot_field_comparison(
                    **common,
                    output_stem=output_dir / "fields" / f"sample_{record.sample_id}_{tag}",
                )
            )
        if "spectra" in products:
            outputs.extend(
                plot_spectrum_comparison(
                    **common,
                    output_stem=output_dir / "spectra" / f"sample_{record.sample_id}_{tag}",
                )
            )
        if "pdfs" in products:
            outputs.extend(
                plot_pdf_comparison(
                    **common,
                    output_stem=output_dir / "pdfs" / f"sample_{record.sample_id}_{tag}",
                )
            )

    if "tracer" in products:
        fields_pred = derive_fields(
            record.prediction, record.representation, args.lx, args.ly
        )
        fields_true = derive_fields(
            record.truth, record.representation, args.lx, args.ly
        )
        diffusivity = (
            float(args.tracer_diffusivity)
            if args.tracer_diffusivity is not None
            else 1.0 / float(args.re)
        )
        tracer_pred = advect_tracer(
            fields_pred["ux"],
            fields_pred["uy"],
            times,
            delta=args.tracer_delta,
            diffusivity=diffusivity,
        )
        tracer_true = advect_tracer(
            fields_true["ux"],
            fields_true["uy"],
            times,
            delta=args.tracer_delta,
            diffusivity=diffusivity,
        )
        tracer_path = output_dir / "tracer" / f"sample_{record.sample_id}_tracer.npz"
        tracer_path.parent.mkdir(parents=True, exist_ok=True)
        import numpy as np

        np.savez_compressed(
            tracer_path,
            times=times,
            prediction=tracer_pred,
            truth=tracer_true,
        )
        outputs.append(tracer_path)
        for index in time_indices:
            time_value = float(times[index])
            outputs.extend(
                plot_tracer_comparison(
                    tracer_pred,
                    tracer_true,
                    time_index=index,
                    time_value=time_value,
                    output_stem=(
                        output_dir
                        / "tracer"
                        / f"sample_{record.sample_id}_{_tag(time_value)}"
                    ),
                    formats=formats,
                    dpi=args.dpi,
                )
            )

    metadata.update(
        {
            "problem": args.problem,
            "config": str(config_path),
            "config_sha256": sha256_file(config_path),
            "device": str(device),
            "sample_id": record.sample_id,
            "time_range": list(time_range),
            "time_indices": time_indices,
            "time_values": [float(times[index]) for index in time_indices],
            "products": products,
            "domain": {"lx": args.lx, "ly": args.ly},
            "tracer": (
                {
                    "method": "periodic semi-Lagrangian advection plus spectral diffusion",
                    "delta": args.tracer_delta,
                    "diffusivity": (
                        float(args.tracer_diffusivity)
                        if args.tracer_diffusivity is not None
                        else 1.0 / float(args.re)
                    ),
                    "post_processed": True,
                }
                if "tracer" in products
                else None
            ),
            "outputs": [
                {
                    "path": str(path.relative_to(output_dir)),
                    "sha256": sha256_file(path),
                }
                for path in outputs
            ],
        }
    )
    manifest = output_dir / "visualization_manifest.json"
    manifest.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"Visualized held-out test sample {record.sample_id} at Re={record.re}.")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
