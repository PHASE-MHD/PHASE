"""Static validation for public PHASE experiment configurations."""

from __future__ import annotations

import math
import os
import re
from pathlib import Path

import yaml

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

SCOT_MODEL_TYPES = {
    "poseidon-mhd-finetune",
    "poseidon-mhd-re-finetune",
    "poseidon-mhd-re-input-finetune",
}
DIFFUSION_RECIPES = {
    "previous_dino",
    "phase_residual_single_re",
    "phase_residual_multi_re",
    "kh_phase_residual_single_re",
    "kh_phase_residual_multi_re",
}
PATH_KEYS = {
    "data_path", "data_root", "train_path", "val_path", "test_path",
    "stats_file", "inputs_stats_file", "targets_stats_file",
    "inputs_stats_template", "targets_stats_template", "checkpoint_path",
    "warm_start_checkpoint", "load_checkpoint",
}


class ConfigValidationError(ValueError):
    """Raised when a public experiment configuration is inconsistent."""


def unresolved_environment_variables(value):
    """Return sorted environment-variable placeholders left in a config."""
    found = set()

    def visit(item):
        if isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        elif isinstance(item, str):
            found.update(_ENV_PATTERN.findall(item))

    visit(value)
    return sorted(found)


def discover_config_paths(root):
    """Return every YAML recipe below a root directory."""
    root = Path(root)
    return sorted((*root.rglob("*.yaml"), *root.rglob("*.yml")))


def _expand_environment(value):
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    return value


def _require(mapping, keys, location, errors):
    if not isinstance(mapping, dict):
        errors.append(f"{location} must be a mapping")
        return
    for key in keys:
        if key not in mapping:
            errors.append(f"missing {location}.{key}")


def _positive(value, location, errors):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        errors.append(f"{location} must be numeric")
    elif not math.isfinite(float(value)) or value <= 0:
        errors.append(f"{location} must be finite and positive")


def _walk_paths(mapping, prefix=""):
    if not isinstance(mapping, dict):
        return
    for key, value in mapping.items():
        location = f"{prefix}.{key}" if prefix else key
        if key in PATH_KEYS and isinstance(value, str) and value:
            yield location, value
        if isinstance(value, dict):
            yield from _walk_paths(value, location)


def _multi_re_data_paths(dataset):
    root = dataset.get("data_root") or dataset.get("data_dir")
    re_values = dataset.get("re_values")
    if (
        not isinstance(root, str)
        or not root
        or not isinstance(re_values, (list, tuple))
        or not re_values
    ):
        return []
    data_file = dataset.get("data_file", "mhd_data_3channel.npy")
    n_value = dataset.get("n", dataset.get("N", 1000))
    template = dataset.get("data_dir_template")
    paths = []
    for re_value in re_values:
        if template:
            directory = template.format(re=re_value, Re=re_value)
            directory = Path(directory)
            if not directory.is_absolute():
                directory = Path(root) / directory
        else:
            directory = Path(root) / f"mhd_Re{re_value}_N{n_value}"
        paths.append(directory / data_file)
    return paths


def _required_sections_present(config, sections):
    return all(isinstance(config.get(name), dict) for name in sections)


def _mapping(config, name, errors, *, required=True):
    """Return a mapping section, recording a readable error when malformed."""
    value = config.get(name)
    if value is None:
        if required:
            errors.append(f"missing config.{name}")
        return None
    if not isinstance(value, dict):
        errors.append(f"config.{name} must be a mapping")
        return None
    return value


def _validate_common(config, errors):
    sections = {
        name: _mapping(config, name, errors)
        for name in (
            "model_params",
            "dataset_params",
            "optimizer_params",
            "train_params",
        )
    }
    if any(section is None for section in sections.values()):
        return
    model = sections["model_params"]
    optimizer = sections["optimizer_params"]
    train = sections["train_params"]
    _require(model, ("model_type",), "model_params", errors)
    _require(optimizer, ("optimizer_type", "lr"), "optimizer_params", errors)
    _require(train, ("epochs", "checkpoint_path"), "train_params", errors)
    if "lr" in optimizer:
        _positive(optimizer["lr"], "optimizer_params.lr", errors)
    if "epochs" in train:
        _positive(train["epochs"], "train_params.epochs", errors)


def _validate_operator(config, errors):
    model = config["model_params"]
    data = config["dataset_params"]
    loss = _mapping(config, "loss_params", errors)
    loaders = _mapping(config, "dataloader_params", errors)
    norm = _mapping(config, "normalization_params", errors)
    train = config["train_params"]
    if loss is None or loaders is None or norm is None:
        return
    _require(data, ("sub_t", "sub_x"), "dataset_params", errors)
    _require(
        loss,
        ("nu", "eta", "data_weight", "ic_weight", "pde_weight"),
        "loss_params",
        errors,
    )
    _require(loaders, ("train", "validation", "test"), "dataloader_params", errors)
    for name in ("nu", "eta"):
        if name in loss:
            _positive(loss[name], f"loss_params.{name}", errors)
    for name in ("sub_t", "sub_x"):
        if name in data:
            _positive(data[name], f"dataset_params.{name}", errors)
    if "train" in loaders:
        _require(
            loaders["train"], ("batch_size",), "dataloader_params.train", errors
        )
        if "batch_size" in loaders["train"]:
            _positive(
                loaders["train"]["batch_size"],
                "dataloader_params.train.batch_size",
                errors,
            )

    out_channels = model.get("out_channels")
    for key in ("input_norm", "output_norm"):
        values = norm.get(key)
        if out_channels in (3, 4) and values is not None:
            if not isinstance(values, (list, tuple)):
                errors.append(f"normalization_params.{key} must be a sequence")
            elif len(values) != out_channels:
                errors.append(
                    f"normalization_params.{key} length must match "
                    "model_params.out_channels"
                )

    for key in (
        "train_sample_fraction",
        "validation_sample_fraction",
        "test_sample_fraction",
    ):
        value = data.get(key)
        if value is not None and (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not 0.0 < float(value) <= 1.0
        ):
            errors.append(f"dataset_params.{key} must be in (0, 1]")

    if train.get("acceptance_run") is True:
        if train.get("epochs") != 10:
            errors.append("acceptance scOT recipes must use 10 epochs")
        fractions = [
            data.get("train_sample_fraction"),
            data.get("validation_sample_fraction"),
            data.get("test_sample_fraction"),
        ]
        if fractions != [0.2, 0.1, 0.1]:
            errors.append(
                "acceptance scOT recipes require sample fractions "
                "[0.2, 0.1, 0.1]"
            )

    re_values = data.get("re_values")
    rem_values = data.get("rem_values")
    if re_values is not None:
        if not isinstance(re_values, (list, tuple)):
            errors.append("dataset_params.re_values must be a sequence")
            re_values = None
        elif not re_values:
            errors.append("dataset_params.re_values must not be empty")
        if rem_values is not None and not isinstance(rem_values, (list, tuple)):
            errors.append("dataset_params.rem_values must be a sequence")
            rem_values = None
        if (
            re_values is not None
            and rem_values is not None
            and len(re_values) != len(rem_values)
        ):
            errors.append(
                "dataset_params.re_values and rem_values must have equal length"
            )
        if (
            re_values is not None
            and data.get("balanced_re_batches")
            and data.get("res_per_batch") != len(re_values)
        ):
            errors.append(
                "dataset_params.res_per_batch must equal the number of Re values"
            )


def _validate_diffusion(config, errors):
    model = config["model_params"]
    data = config["dataset_params"]
    norm = _mapping(config, "normalization_params", errors)
    train = config["train_params"]
    if norm is None:
        return
    _require(
        model, ("channels", "image_size", "num_sample_steps"), "model_params", errors
    )
    _require(
        data,
        ("train_path", "val_path", "test_path", "prediction_mode", "residual_target"),
        "dataset_params",
        errors,
    )
    for key in ("channels", "image_size", "num_sample_steps"):
        if key in model:
            _positive(model[key], f"model_params.{key}", errors)

    recipe = train.get("recipe")
    if recipe not in DIFFUSION_RECIPES:
        errors.append(f"unsupported diffusion train_params.recipe: {recipe!r}")
    expected_residual = recipe != "previous_dino"
    if bool(data.get("residual_target")) != expected_residual:
        errors.append(
            f"{recipe} requires dataset_params.residual_target={expected_residual}"
        )
    expected_mode = "residual" if expected_residual else "direct"
    if data.get("prediction_mode") != expected_mode:
        errors.append(
            f"{recipe} requires dataset_params.prediction_mode={expected_mode!r}"
        )
    if expected_residual:
        if model.get("channels") != 4:
            errors.append("PHASE residual diffusion requires model_params.channels=4")
        if not model.get("helmholtz_projection"):
            errors.append("PHASE residual diffusion requires Helmholtz projection")
        if model.get("projection_mode") != "full_field_residual":
            errors.append(
                "PHASE residual diffusion requires "
                "projection_mode='full_field_residual'"
            )
        if norm.get("channel_groups") != [[0, 1], [2, 3]]:
            errors.append(
                "PHASE residual diffusion requires paired velocity/magnetic groups"
            )
    elif model.get("helmholtz_projection"):
        errors.append("previous DINO must not enable PHASE Helmholtz projection")

    if train.get("acceptance_run") is True:
        if train.get("epochs") != 10:
            errors.append("acceptance diffusion recipes must use 10 epochs")
        fractions = [
            data.get("source_train_sample_fraction"),
            data.get("source_validation_sample_fraction"),
            data.get("source_test_sample_fraction"),
        ]
        if fractions != [0.2, 0.1, 0.1]:
            errors.append(
                "acceptance diffusion recipes require source fractions "
                "[0.2, 0.1, 0.1]"
            )

    for key in (
        "train_sample_fraction",
        "validation_sample_fraction",
        "test_sample_fraction",
    ):
        value = data.get(key)
        if value is not None and (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not 0.0 < float(value) <= 1.0
        ):
            errors.append(f"dataset_params.{key} must be in (0, 1]")

    if "source_time_range" in data:
        time_range = data["source_time_range"]
        if not isinstance(time_range, (list, tuple)) or len(time_range) != 2:
            errors.append("dataset_params.source_time_range must contain [start, stop]")
            return
        start, stop = time_range
        if not all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in time_range
        ):
            errors.append("dataset_params.source_time_range values must be numeric")
            return
        if not all(math.isfinite(float(value)) for value in time_range):
            errors.append("dataset_params.source_time_range values must be finite")
            return
        if stop <= start:
            errors.append("dataset_params.source_time_range stop must exceed start")
            return
        dt = data.get("source_output_dt")
        sub_t = data.get("source_sub_t")
        frames = data.get("frames_per_trajectory")
        if dt is None or sub_t is None or frames is None:
            errors.append(
                "source_time_range requires output_dt, sub_t, and "
                "frames_per_trajectory"
            )
        else:
            if (
                not isinstance(dt, (int, float))
                or isinstance(dt, bool)
                or not math.isfinite(float(dt))
                or dt <= 0
            ):
                errors.append(
                    "dataset_params.source_output_dt must be finite and positive"
                )
                return
            if not isinstance(sub_t, int) or isinstance(sub_t, bool) or sub_t <= 0:
                errors.append("dataset_params.source_sub_t must be a positive integer")
                return
            if not isinstance(frames, int) or isinstance(frames, bool) or frames <= 0:
                errors.append(
                    "dataset_params.frames_per_trajectory must be a positive integer"
                )
                return
            expected = int(round((stop - start) / (dt * sub_t))) + 1
            if frames != expected:
                errors.append(
                    f"dataset_params.frames_per_trajectory must be {expected}, "
                    f"got {frames}"
                )


def _validate_conditioner(config, errors):
    model = _mapping(config, "model_params", errors)
    data = _mapping(config, "dataset_params", errors)
    norm = _mapping(config, "normalization_params", errors)
    if model is None or data is None or norm is None:
        return
    _require(
        model,
        ("model_type", "in_channels", "out_channels", "num_fno_layers"),
        "model_params",
        errors,
    )
    _require(
        data,
        ("data_path", "train_size", "val_plus_test_size", "sub_t", "sub_x"),
        "dataset_params",
        errors,
    )
    if model.get("model_type") != "tfno":
        errors.append("conditioner configs currently support model_type='tfno' only")


def validate_config(config, *, check_paths=False):
    """Validate one expanded config and return a list of errors."""
    errors = []
    if not isinstance(config, dict):
        return ["config root must be a mapping"]
    config_type = config.get("config_type")
    if config_type == "conditioner":
        _validate_conditioner(config, errors)
    else:
        _validate_common(config, errors)
    if not isinstance(config.get("model_params"), dict):
        return errors
    model_type = config.get("model_params", {}).get("model_type")
    if config_type == "diffusion":
        if _required_sections_present(
            config,
            ("model_params", "dataset_params", "optimizer_params", "train_params"),
        ):
            _validate_diffusion(config, errors)
    elif config_type == "conditioner":
        pass
    elif model_type == "tfno" or model_type in SCOT_MODEL_TYPES:
        if _required_sections_present(
            config,
            (
                "model_params",
                "dataset_params",
                "optimizer_params",
                "train_params",
            ),
        ):
            _validate_operator(config, errors)
    else:
        errors.append(f"unsupported model_params.model_type: {model_type!r}")

    if check_paths:
        unresolved = unresolved_environment_variables(config)
        if unresolved:
            errors.append(
                "unresolved environment variables: " + ", ".join(unresolved)
            )
        else:
            for location, value in _walk_paths(config):
                candidates = [value]
                if "{re}" in value:
                    normalization = config.get("normalization_params")
                    re_values = (
                        normalization.get("re_values", [])
                        if isinstance(normalization, dict)
                        else []
                    )
                    if not re_values:
                        errors.append(
                            f"{location} contains {{re}} but no normalization "
                            "Re values are defined"
                        )
                        continue
                    candidates = [value.format(re=re_value) for re_value in re_values]
                for candidate in candidates:
                    path = Path(candidate).expanduser()
                    if location.endswith("checkpoint_path"):
                        if not path.parent.exists():
                            errors.append(
                                f"parent directory does not exist for {location}: "
                                f"{path.parent}"
                            )
                    elif not path.exists():
                        errors.append(
                            f"path does not exist for {location}: {path}"
                        )
            dataset = config.get("dataset_params")
            for path in _multi_re_data_paths(
                dataset if isinstance(dataset, dict) else {}
            ):
                if not path.expanduser().exists():
                    errors.append(f"multi-Re data file does not exist: {path}")
    return errors


def validate_config_file(path, *, check_paths=False, expand_environment=True):
    """Load and validate one YAML config without importing ML dependencies."""
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        return [f"could not load YAML config {path}: {exc}"]
    if expand_environment:
        config = _expand_environment(config)
    return validate_config(config, check_paths=check_paths)


def assert_valid_config(config, *, check_paths=False):
    """Raise one readable exception when validation fails."""
    errors = validate_config(config, check_paths=check_paths)
    if errors:
        raise ConfigValidationError(
            "Invalid PHASE config:\n- " + "\n- ".join(errors)
        )
