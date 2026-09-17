"""Train-split-only statistics for PHASE trajectories and diffusion features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


@dataclass
class ChannelMoments:
    """Streaming per-channel moments for channel-first chunks."""

    channels: int

    def __post_init__(self) -> None:
        self.count = 0
        self.total = np.zeros(self.channels, dtype=np.float64)
        self.total_sq = np.zeros(self.channels, dtype=np.float64)
        self.minimum = np.full(self.channels, np.inf, dtype=np.float64)
        self.maximum = np.full(self.channels, -np.inf, dtype=np.float64)

    def update(self, chunk: np.ndarray) -> None:
        if chunk.ndim < 2 or chunk.shape[1] != self.channels:
            raise ValueError(
                f"Expected channel-first chunk with {self.channels} channels, "
                f"got {chunk.shape}"
            )
        flat = np.asarray(chunk, dtype=np.float64).reshape(chunk.shape[0], self.channels, -1)
        values_per_channel = flat.shape[0] * flat.shape[2]
        self.count += values_per_channel
        self.total += flat.sum(axis=(0, 2), dtype=np.float64)
        self.total_sq += np.square(flat).sum(axis=(0, 2), dtype=np.float64)
        self.minimum = np.minimum(self.minimum, flat.min(axis=(0, 2)))
        self.maximum = np.maximum(self.maximum, flat.max(axis=(0, 2)))

    def result(self) -> dict[str, np.ndarray]:
        if self.count == 0:
            raise ValueError("No values were supplied")
        mean = self.total / self.count
        variance = np.maximum(self.total_sq / self.count - mean * mean, 0.0)
        return {
            "mean": mean,
            "std": np.sqrt(variance),
            "min_val": self.minimum,
            "max_val": self.maximum,
        }


def split_indices(
    num_samples: int,
    train_size: int,
    *,
    seed: int = 42,
    re_index: int = 0,
    mode: str = "single_re_seed42",
) -> np.ndarray:
    """Reproduce the single-Re or multi-Re training split exactly."""
    if train_size < 1 or train_size > num_samples:
        raise ValueError(
            f"train_size must be in [1,{num_samples}], got {train_size}"
        )
    if mode == "single_re_seed42":
        state = np.random.RandomState(seed)
        shuffled = state.permutation(num_samples)
    elif mode == "multi_re_seed_plus_index":
        shuffled = np.random.default_rng(seed + re_index).permutation(num_samples)
    else:
        raise ValueError(f"Unknown split mode: {mode}")
    return np.asarray(shuffled[:train_size], dtype=np.int64)


def trajectory_statistics(
    data_path: str | Path,
    *,
    train_size: int,
    seed: int = 42,
    re_index: int = 0,
    split_mode: str = "single_re_seed42",
    sub_t: int = 1,
    sub_x: int = 1,
    time_stop_index: int | None = None,
    chunk_size: int = 8,
) -> dict[str, np.ndarray]:
    """Compute channel statistics from only the selected training trajectories."""
    data = np.load(data_path, mmap_mode="r")
    if data.ndim != 5:
        raise ValueError(f"Expected (N,T,X,Y,C), got {data.shape}")
    indices = split_indices(
        data.shape[0],
        train_size,
        seed=seed,
        re_index=re_index,
        mode=split_mode,
    )
    accumulator = ChannelMoments(data.shape[-1])
    time_slice = slice(None, time_stop_index, sub_t)
    for start in range(0, len(indices), chunk_size):
        selected = indices[start : start + chunk_size]
        chunk = np.asarray(
            data[selected, time_slice, ::sub_x, ::sub_x, :], dtype=np.float64
        )
        accumulator.update(np.moveaxis(chunk, -1, 1))
    result = accumulator.result()
    result["train_indices"] = indices
    return result


def _load_diffusion_arrays(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    path = Path(path)
    if path.is_dir():
        return (
            np.load(path / "diff_inputs.npy", mmap_mode="r"),
            np.load(path / "diff_targets.npy", mmap_mode="r"),
        )
    loaded = np.load(path, allow_pickle=True)
    values = loaded.item() if isinstance(loaded, np.ndarray) else loaded
    return values["diff_inputs"], values["diff_targets"]


def diffusion_statistics(
    data_path: str | Path,
    *,
    prediction_mode: str = "direct",
    chunk_size: int = 8,
) -> dict[str, dict[str, np.ndarray]]:
    """Compute separate conditioner and clean-target diffusion statistics."""
    inputs, targets = _load_diffusion_arrays(data_path)
    if inputs.shape != targets.shape or inputs.ndim not in (4, 5):
        raise ValueError(
            f"Expected matching 4D/5D diffusion arrays, got {inputs.shape} and "
            f"{targets.shape}"
        )
    prediction_mode = prediction_mode.lower()
    if prediction_mode not in {"direct", "residual"}:
        raise ValueError("prediction_mode must be direct or residual")

    input_moments = ChannelMoments(inputs.shape[1])
    target_moments = ChannelMoments(targets.shape[1])
    for start in range(0, inputs.shape[0], chunk_size):
        stop = min(start + chunk_size, inputs.shape[0])
        x = np.asarray(inputs[start:stop], dtype=np.float64)
        y = np.asarray(targets[start:stop], dtype=np.float64)
        if prediction_mode == "residual":
            y = y - x
        input_moments.update(x)
        target_moments.update(y)
    return {"inputs": input_moments.result(), "targets": target_moments.result()}


def _histogram_abs_percentile(
    sources: Iterable[tuple[np.ndarray, np.ndarray]],
    *,
    channel: int,
    percentile: float,
    sub_t: int,
    sub_x: int,
    time_stop_index: int | None,
    bins: int,
    chunk_size: int,
) -> tuple[float, float, int]:
    sources = list(sources)
    time_slice = slice(None, time_stop_index, sub_t)
    abs_max = 0.0
    for array, indices in sources:
        for start in range(0, len(indices), chunk_size):
            selected = indices[start : start + chunk_size]
            values = np.asarray(
                array[selected, time_slice, ::sub_x, ::sub_x, channel],
                dtype=np.float32,
            )
            abs_max = max(abs_max, float(np.max(np.abs(values))))

    if abs_max == 0.0:
        total = sum(
            len(indices)
            * len(range(*time_slice.indices(array.shape[1])))
            * len(range(0, array.shape[2], sub_x))
            * len(range(0, array.shape[3], sub_x))
            for array, indices in sources
        )
        return 0.0, 0.0, total

    histogram = np.zeros(bins, dtype=np.int64)
    total = 0
    scale = bins / abs_max
    for array, indices in sources:
        for start in range(0, len(indices), chunk_size):
            selected = indices[start : start + chunk_size]
            values = np.abs(
                np.asarray(
                    array[selected, time_slice, ::sub_x, ::sub_x, channel],
                    dtype=np.float32,
                )
            ).ravel()
            bin_indices = np.floor(values * scale).astype(np.int64)
            np.minimum(bin_indices, bins - 1, out=bin_indices)
            histogram += np.bincount(bin_indices, minlength=bins)
            total += values.size

    target = percentile / 100.0 * total
    cumulative = np.cumsum(histogram)
    bin_index = int(np.searchsorted(cumulative, target, side="left"))
    bin_index = min(max(bin_index, 0), bins - 1)
    previous = cumulative[bin_index - 1] if bin_index else 0
    count = histogram[bin_index]
    left = abs_max * bin_index / bins
    right = abs_max * (bin_index + 1) / bins
    fraction = (target - previous) / count if count else 0.0
    estimate = left + np.clip(fraction, 0.0, 1.0) * (right - left)
    return float(estimate), abs_max, int(total)


def multi_re_magnetic_p99(
    paths_by_re: Mapping[float, str | Path],
    *,
    train_size: int,
    seed: int = 42,
    split_mode: str = "single_re_seed42",
    sub_t: int = 1,
    sub_x: int = 1,
    time_stop_index: int | None = None,
    bins: int = 20_000,
    chunk_size: int = 20,
    percentile: float = 99.0,
) -> dict:
    """Compute paired B scales per Re and globally from training samples only."""
    sources: list[tuple[np.ndarray, np.ndarray]] = []
    per_re = {}
    for re_index, (re_value, path) in enumerate(paths_by_re.items()):
        array = np.load(path, mmap_mode="r")
        if array.ndim != 5 or array.shape[-1] < 4:
            raise ValueError(f"Expected (N,T,X,Y,C>=4), got {array.shape}")
        indices = split_indices(
            array.shape[0],
            train_size,
            seed=seed,
            re_index=re_index,
            mode=split_mode,
        )
        source = [(array, indices)]
        bx = _histogram_abs_percentile(
            source,
            channel=2,
            percentile=percentile,
            sub_t=sub_t,
            sub_x=sub_x,
            time_stop_index=time_stop_index,
            bins=bins,
            chunk_size=chunk_size,
        )[0]
        by = _histogram_abs_percentile(
            source,
            channel=3,
            percentile=percentile,
            sub_t=sub_t,
            sub_x=sub_x,
            time_stop_index=time_stop_index,
            bins=bins,
            chunk_size=chunk_size,
        )[0]
        per_re[str(float(re_value))] = {
            "Bx_p99_abs": bx,
            "By_p99_abs": by,
            "chosen_train_p99_abs_B": max(bx, by),
            "train_indices": indices.tolist(),
        }
        sources.append((array, indices))

    global_bx = _histogram_abs_percentile(
        sources,
        channel=2,
        percentile=percentile,
        sub_t=sub_t,
        sub_x=sub_x,
        time_stop_index=time_stop_index,
        bins=bins,
        chunk_size=chunk_size,
    )[0]
    global_by = _histogram_abs_percentile(
        sources,
        channel=3,
        percentile=percentile,
        sub_t=sub_t,
        sub_x=sub_x,
        time_stop_index=time_stop_index,
        bins=bins,
        chunk_size=chunk_size,
    )[0]
    return {
        "global": {
            "Bx_p99_abs": global_bx,
            "By_p99_abs": global_by,
            "chosen_train_p99_abs_B": max(global_bx, global_by),
        },
        "per_re": per_re,
        "settings": {
            "train_size_per_re": train_size,
            "seed": seed,
            "split_mode": split_mode,
            "sub_t": sub_t,
            "sub_x": sub_x,
            "time_stop_index": time_stop_index,
            "bins": bins,
            "percentile": percentile,
        },
    }


def save_npz_statistics(path: str | Path, statistics: Mapping[str, np.ndarray]) -> Path:
    """Save array-valued statistics while excluding provenance-only indices."""
    path = Path(path)
    path = path if path.suffix == ".npz" else path.with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    values = {key: value for key, value in statistics.items() if key != "train_indices"}
    np.savez(path, **values)
    return path
