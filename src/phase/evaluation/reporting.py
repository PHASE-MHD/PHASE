"""Stable machine-readable and human-readable evaluation reports."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


def sha256_file(path: str | Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _metric_lines(metrics: dict[str, float]) -> list[str]:
    sections = [
        (
            "RELATIVE L2 ERRORS",
            [f"rel_l2_{name}" for name in ("ux", "uy", "Bx", "By", "omega", "j")],
        ),
        (
            "MEAN SQUARED ERRORS",
            [f"mse_{name}" for name in ("ux", "uy", "Bx", "By", "omega", "j")],
        ),
        (
            "DIVERGENCE ERRORS",
            ["div_mse_u", "div_rms_u", "div_mse_B", "div_rms_B"],
        ),
        (
            "LOW-k SPECTRUM ERRORS",
            [f"spectrum_low_k_{name}" for name in ("u", "B", "omega", "j")],
        ),
        (
            "HIGH-k SPECTRUM ERRORS",
            [f"spectrum_high_k_{name}" for name in ("u", "B", "omega", "j")],
        ),
        (
            "PDF RELATIVE MAE",
            [
                f"pdf_relative_mae_{name}"
                for name in ("ux", "uy", "Bx", "By", "omega", "j")
            ],
        ),
        (
            "RELATIVE STANDARD-DEVIATION ERRORS",
            [
                f"relative_std_{name}"
                for name in ("ux", "uy", "Bx", "By", "omega", "j")
            ],
        ),
        (
            "ABSOLUTE KURTOSIS ERRORS",
            [
                f"absolute_kurtosis_{name}"
                for name in ("ux", "uy", "Bx", "By", "omega", "j")
            ],
        ),
    ]
    lines = []
    for title, names in sections:
        available = [name for name in names if name in metrics]
        if not available:
            continue
        lines.extend(["", title])
        lines.extend(f"{name:30s}: {metrics[name]:.8e}" for name in available)
    return lines


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_evaluation_report(
    output_dir: str | Path,
    metadata: dict,
    results: dict,
) -> dict[str, Path]:
    """Write JSON, aggregate CSV, per-sample CSV, and legacy-style text."""
    if metadata.get("split") != "test":
        raise ValueError("Public PHASE evaluation reports must use split='test'.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": metadata, "results": results}

    json_path = output_dir / "evaluate_error.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    aggregate_row = {
        "model": metadata["model"],
        "problem": metadata["problem"],
        "split": metadata["split"],
        "re": metadata["re"],
        "number_of_samples": results["number_of_samples"],
        **results["aggregate"],
    }
    aggregate_path = output_dir / "evaluate_error.csv"
    _write_csv(aggregate_path, [aggregate_row])
    per_sample_path = output_dir / "evaluate_error_per_sample.csv"
    _write_csv(per_sample_path, results["per_sample"])

    lines = [
        "PHASE HELD-OUT TEST EVALUATION",
        f"Model                    : {metadata['model']}",
        f"Problem                  : {metadata['problem']}",
        f"Representation           : {metadata['representation']}",
        f"Split                    : {metadata['split']}",
        f"Re=Rm                    : {metadata['re']}",
        f"Number of trajectories   : {results['number_of_samples']}",
        f"Sample ID source         : {metadata.get('sample_id_source', 'unspecified')}",
        f"Aggregation              : {results['aggregation']}",
        f"Configuration            : {metadata['config']}",
        f"Configuration SHA-256    : {metadata['config_sha256']}",
        f"Checkpoint               : {metadata['checkpoint']}",
        f"Checkpoint SHA-256       : {metadata['checkpoint_sha256']}",
        f"Checkpoint epoch         : {metadata.get('checkpoint_epoch')}",
        f"Diffusion seed           : {metadata.get('diffusion_seed', 'not applicable')}",
        f"Diffusion sampling steps : {metadata.get('num_sample_steps', 'not applicable')}",
        "Spectral derivatives     : periodic Fourier derivatives on physical fields",
    ]
    lines.extend(_metric_lines(results["aggregate"]))
    text_path = output_dir / "evaluate_error.txt"
    text_path.write_text("\n".join(lines) + "\n")
    return {
        "json": json_path,
        "aggregate_csv": aggregate_path,
        "per_sample_csv": per_sample_path,
        "text": text_path,
    }
