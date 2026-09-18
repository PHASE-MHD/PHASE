"""Model-independent metrics for physical MHD trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch

from .physics import (
    FIELD_NAMES,
    derive_fields,
    divergence_2d,
    scalar_spectrum,
    vector_spectrum,
)


EPS = 1.0e-30
LOW_K_BINS = 8
PDF_SPECS = {
    "ux": (60, (-2.0, 2.0)),
    "uy": (60, (-2.0, 2.0)),
    "Bx": (60, (-2.0, 2.0)),
    "By": (60, (-2.0, 2.0)),
    "omega": (90, (-3.0, 3.0)),
    "j": (90, (-3.0, 3.0)),
}


@dataclass(frozen=True)
class EvaluationRecord:
    """One denormalized prediction/DNS trajectory pair."""

    prediction: torch.Tensor
    truth: torch.Tensor
    representation: str
    sample_id: int
    re: float


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _relative_l2(prediction: torch.Tensor, truth: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm((prediction - truth).reshape(-1))
    denominator = torch.linalg.vector_norm(truth.reshape(-1))
    return float(numerator / (denominator + EPS))


def _distribution_errors(
    prediction: torch.Tensor, truth: torch.Tensor
) -> tuple[float, float]:
    pred = prediction.detach().double().reshape(-1)
    true = truth.detach().double().reshape(-1)
    pred_std = float(pred.std(unbiased=False))
    true_std = float(true.std(unbiased=False))
    std_error = abs(pred_std - true_std) / (true_std + EPS)

    def kurtosis(values: torch.Tensor, std: float) -> float:
        if std <= EPS:
            return 0.0
        centered = values - values.mean()
        return float(torch.mean((centered / std) ** 4))

    kurt_error = abs(kurtosis(pred, pred_std) - kurtosis(true, true_std))
    return float(std_error), float(kurt_error)


def _pdf_relative_mae(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    *,
    bins: int,
    value_range: tuple[float, float],
) -> float:
    pred = prediction.detach().cpu().numpy().ravel()
    true = truth.detach().cpu().numpy().ravel()
    true_hist, edges = np.histogram(
        true, bins=bins, range=value_range, density=True
    )
    pred_hist, _ = np.histogram(pred, bins=edges, density=True)
    mask = true_hist > 0
    if not np.any(mask):
        return 0.0
    return float(
        np.mean(np.abs(pred_hist[mask] - true_hist[mask]) / true_hist[mask])
    )


def _spectrum_errors(
    prediction: torch.Tensor, truth: torch.Tensor
) -> tuple[float, float]:
    split = min(LOW_K_BINS, prediction.numel())

    error = torch.abs(torch.log10((prediction + EPS) / (truth + EPS)))
    low = float(torch.mean(error[:split]))
    high = float(torch.mean(error[split:]))
    return low, high


def _frame_scales(fields: dict[str, torch.Tensor], time_index: int):
    ux = fields["ux"][time_index]
    uy = fields["uy"][time_index]
    bx = fields["Bx"][time_index]
    by = fields["By"][time_index]
    return {
        "ux": torch.sqrt(torch.mean(ux.square() + uy.square())) + EPS,
        "uy": torch.sqrt(torch.mean(ux.square() + uy.square())) + EPS,
        "Bx": torch.sqrt(torch.mean(bx.square() + by.square())) + EPS,
        "By": torch.sqrt(torch.mean(bx.square() + by.square())) + EPS,
        "omega": torch.sqrt(torch.mean(fields["omega"][time_index].square()))
        + EPS,
        "j": torch.sqrt(torch.mean(fields["j"][time_index].square())) + EPS,
    }


def evaluate_record(
    record: EvaluationRecord,
    *,
    problem: str,
    lx: float = 1.0,
    ly: float = 1.0,
) -> dict[str, float]:
    """Evaluate one physical trajectory with the locked problem convention."""
    if problem not in {"turbulence", "kh"}:
        raise ValueError("problem must be 'turbulence' or 'kh'.")
    if record.prediction.shape != record.truth.shape:
        raise ValueError(
            "Prediction and DNS shapes differ: "
            f"{tuple(record.prediction.shape)} != {tuple(record.truth.shape)}."
        )
    pred = derive_fields(record.prediction, record.representation, lx, ly)
    true = derive_fields(record.truth, record.representation, lx, ly)
    nt = pred["ux"].shape[0]
    values: dict[str, list[float]] = {}

    def add(name: str, value: float) -> None:
        values.setdefault(name, []).append(float(value))

    if problem == "kh":
        # Compare a complete trajectory before averaging over samples.
        result = {}
        for name in FIELD_NAMES:
            result[f"rel_l2_{name}"] = _relative_l2(pred[name], true[name])
            result[f"mse_{name}"] = float(
                torch.mean((pred[name] - true[name]) ** 2)
            )
        div_u = divergence_2d(pred["ux"], pred["uy"], lx, ly)
        div_b = divergence_2d(pred["Bx"], pred["By"], lx, ly)
        result["div_mse_u"] = float(torch.mean(div_u.square()))
        result["div_rms_u"] = float(torch.sqrt(torch.mean(div_u.square())))
        result["div_mse_B"] = float(torch.mean(div_b.square()))
        result["div_rms_B"] = float(torch.sqrt(torch.mean(div_b.square())))
        return result

    for time_index in range(nt):
        for name in FIELD_NAMES:
            pred_field = pred[name][time_index]
            true_field = true[name][time_index]
            add(f"rel_l2_{name}", _relative_l2(pred_field, true_field))
            add(f"mse_{name}", float(torch.mean((pred_field - true_field) ** 2)))

        div_u = divergence_2d(
            pred["ux"][time_index : time_index + 1],
            pred["uy"][time_index : time_index + 1],
            lx,
            ly,
        )[0]
        div_b = divergence_2d(
            pred["Bx"][time_index : time_index + 1],
            pred["By"][time_index : time_index + 1],
            lx,
            ly,
        )[0]
        add("div_mse_u", float(torch.mean(div_u.square())))
        add("div_rms_u", float(torch.sqrt(torch.mean(div_u.square()))))
        add("div_mse_B", float(torch.mean(div_b.square())))
        add("div_rms_B", float(torch.sqrt(torch.mean(div_b.square()))))

        scales = _frame_scales(true, time_index)
        for name in FIELD_NAMES:
            pred_field = pred[name][time_index]
            true_field = true[name][time_index]
            std_error, kurt_error = _distribution_errors(pred_field, true_field)
            add(f"relative_std_{name}", std_error)
            add(f"absolute_kurtosis_{name}", kurt_error)
            bins, value_range = PDF_SPECS[name]
            add(
                f"pdf_relative_mae_{name}",
                _pdf_relative_mae(
                    pred_field / scales[name],
                    true_field / scales[name],
                    bins=bins,
                    value_range=value_range,
                ),
            )

        spectra = {
            "u": (
                vector_spectrum(pred["ux"][time_index], pred["uy"][time_index]),
                vector_spectrum(true["ux"][time_index], true["uy"][time_index]),
            ),
            "B": (
                vector_spectrum(pred["Bx"][time_index], pred["By"][time_index]),
                vector_spectrum(true["Bx"][time_index], true["By"][time_index]),
            ),
            "omega": (
                scalar_spectrum(pred["omega"][time_index]),
                scalar_spectrum(true["omega"][time_index]),
            ),
            "j": (
                scalar_spectrum(pred["j"][time_index]),
                scalar_spectrum(true["j"][time_index]),
            ),
        }
        for name, (pred_spectrum, true_spectrum) in spectra.items():
            low, high = _spectrum_errors(pred_spectrum, true_spectrum)
            add(f"spectrum_low_k_{name}", low)
            add(f"spectrum_high_k_{name}", high)

    return {name: _mean(metric_values) for name, metric_values in values.items()}


def evaluate_records(
    records: Iterable[EvaluationRecord],
    *,
    problem: str,
    lx: float = 1.0,
    ly: float = 1.0,
) -> dict:
    """Evaluate trajectories with equal weighting over samples and time."""
    rows = []
    identities = set()
    for record in records:
        identity = (float(record.re), int(record.sample_id))
        if identity in identities:
            raise ValueError(f"Duplicate evaluation trajectory {identity}.")
        identities.add(identity)
        row = {
            "re": float(record.re),
            "sample_id": int(record.sample_id),
            **evaluate_record(record, problem=problem, lx=lx, ly=ly),
        }
        rows.append(row)
    if not rows:
        raise ValueError("No held-out test trajectories were evaluated.")

    metric_names = [key for key in rows[0] if key not in {"re", "sample_id"}]
    aggregate = {name: _mean([row[name] for row in rows]) for name in metric_names}
    by_re = {}
    for re_value in sorted({row["re"] for row in rows}):
        selected = [row for row in rows if row["re"] == re_value]
        by_re[str(int(re_value) if re_value.is_integer() else re_value)] = {
            "number_of_samples": len(selected),
            **{name: _mean([row[name] for row in selected]) for name in metric_names},
        }
    return {
        "aggregation": (
            "metric over each complete space-time trajectory, then mean over trajectories"
            if problem == "kh"
            else "mean over spatial metric, then time, then trajectories"
        ),
        "number_of_samples": len(rows),
        "aggregate": aggregate,
        "by_re": by_re,
        "per_sample": rows,
    }
