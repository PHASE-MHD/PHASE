import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler
from ..normalizations import create_normalization


def _format_re_for_stats_path(re_value):
    re_float = float(re_value)
    return str(int(re_float)) if re_float.is_integer() else str(re_float).replace('.', 'p')


def _load_per_re_stats(norm_params, kind):
    """Load a {Re: stats} map for per-Re normalization."""
    re_values = norm_params.get("re_values")
    if not re_values:
        raise ValueError("per_re_paired_minmax requires normalization_params.re_values")

    template_key = f"{kind}_stats_template"
    template = norm_params.get(template_key)
    if not template:
        stats_dir = norm_params.get("per_re_stats_dir")
        if not stats_dir:
            raise ValueError(
                f"per_re_paired_minmax requires {template_key} or per_re_stats_dir"
            )
        template = os.path.join(stats_dir, f"Re{{re}}_{kind}_stats.npz")

    stats_by_re = {}
    for re_value in re_values:
        re_label = _format_re_for_stats_path(re_value)
        stats_path = template.format(re=re_label, re_float=float(re_value), kind=kind)
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"Per-Re {kind} statistics file not found: {stats_path}")
        stats = np.load(stats_path)
        stats_by_re[float(re_value)] = {
            key: stats[key].tolist() if hasattr(stats[key], "tolist") else stats[key]
            for key in stats.files
        }
        print(f"Loaded Re={re_label} {kind} statistics from {stats_path}")
    return stats_by_re

class DiffusionDataset(Dataset):
    """Dataset for training diffusion models"""

    def __init__(
        self,
        data_path,
        config=None,
        input_normalizer=None,
        target_normalizer=None,
        split_name="train",
    ):
        """
        Args:
            data_path: Path to the data file
            config: Configuration for preprocessing
            input_normalizer: Normalizer for input data
            target_normalizer: Normalizer for target data
            split_name: Name of the split for logging ("train", "val", "test")
        """
        super().__init__()
        self.data_path = data_path
        self.config = config
        self.split_name = split_name
        self.input_normalizer = input_normalizer
        self.target_normalizer = target_normalizer
        dataset_params = (config or {}).get("dataset_params", {})
        self.return_metadata = bool(
            dataset_params.get("return_metadata", dataset_params.get("return_re", False))
        )
        self.prediction_mode = str(dataset_params.get("prediction_mode", "direct")).lower()
        self.residual_target = bool(
            dataset_params.get("residual_target", self.prediction_mode == "residual")
        )
        if self.residual_target:
            print(
                f"Using residual diffusion targets for {split_name}: "
                "target := DNS/high-fidelity - conditional input"
            )

        self.original_shape = None
        self.timesteps = None
        self.sample_indices = None

#        print(f"Loading {split_name} data from {data_path}")

        self.lazy_disk = os.path.isdir(data_path)
        if self.lazy_disk:
            self.inputs = np.load(os.path.join(data_path, "diff_inputs.npy"), mmap_mode="r")
            self.targets = np.load(os.path.join(data_path, "diff_targets.npy"), mmap_mode="r")
            re_path = os.path.join(data_path, "re.npy")
            sample_id_path = os.path.join(data_path, "sample_id.npy")
            self.re_per_sample = np.load(re_path, mmap_mode="r") if os.path.exists(re_path) else None
            self.sample_id_per_sample = (
                np.load(sample_id_path, mmap_mode="r") if os.path.exists(sample_id_path) else None
            )
            print(f"Using lazy directory-backed diffusion dataset: {data_path}")
        else:
            try:
                data = np.load(data_path, allow_pickle=True)
                data_dict = data.item() if isinstance(data, np.ndarray) else data
            except Exception:
                import pickle
                with open(data_path, "rb") as f:
                    data_dict = pickle.load(f)

            self.inputs = data_dict["diff_inputs"]
            self.targets = data_dict["diff_targets"]
            self.re_per_sample = data_dict.get("re", None)
            self.sample_id_per_sample = data_dict.get("sample_id", None)

        self.re_per_item = None
        self.sample_id_per_item = None
        self.num_items = None
        self.channel_indices = None
        self.out_channels = None

        self.original_shape = self.inputs.shape
        if len(self.original_shape) == 5:
            # For shape (nsamples, channels, timesteps, nx, ny)
            self.timesteps = self.original_shape[2]

        print(f"Original {split_name} input shape: {self.inputs.shape}")
        print(f"Original {split_name} target shape: {self.targets.shape}")
        if self.re_per_sample is not None:
            self.re_per_sample = np.asarray(self.re_per_sample, dtype=np.float32)
            print(
                f"Loaded Re metadata for {len(self.re_per_sample)} {split_name} simulations"
            )
        if self.sample_id_per_sample is not None:
            self.sample_id_per_sample = np.asarray(self.sample_id_per_sample)
        if self.timesteps:
            print(f"Timesteps per simulation: {self.timesteps}")

        self.apply_sample_fraction(dataset_params)

        if self.lazy_disk:
            if self.input_normalizer:
                print("Will apply input normalization lazily per item")
            if self.target_normalizer:
                print("Will apply target normalization lazily per item\n")
            if config:
                self.prepare_lazy_diffusion_input(config)
                self.prepare_metadata()
                if split_name == "train" and "model_params" in config:
                    self.calculate_sigma_data(config, split_name)
        else:
            if self.residual_target:
                self.targets = self.targets - self.inputs
                print(f"Converted {split_name} targets to residuals before normalization")

            # Apply normalization to all channels first
            if self.input_normalizer:
                inputs_tensor = torch.tensor(self.inputs, dtype=torch.float32)
                re_values = (
                    torch.as_tensor(self.re_per_sample, dtype=torch.float32)
                    if self.re_per_sample is not None
                    else None
                )

                try:
                    self.inputs = self.input_normalizer.normalize(inputs_tensor, re=re_values)
                except TypeError:
                    self.inputs = self.input_normalizer.normalize(inputs_tensor)
                print(f"Applied input normalization to all channels")

            if self.target_normalizer:
                targets_tensor = torch.tensor(self.targets, dtype=torch.float32)
                re_values = (
                    torch.as_tensor(self.re_per_sample, dtype=torch.float32)
                    if self.re_per_sample is not None
                    else None
                )

                try:
                    self.targets = self.target_normalizer.normalize(targets_tensor, re=re_values)
                except TypeError:
                    self.targets = self.target_normalizer.normalize(targets_tensor)
                print(f"Applied target normalization to all channels\n")

            # Keep the eager path tensor-based even when normalization is disabled.
            if not torch.is_tensor(self.inputs):
                self.inputs = torch.as_tensor(self.inputs, dtype=torch.float32)
            if not torch.is_tensor(self.targets):
                self.targets = torch.as_tensor(self.targets, dtype=torch.float32)

            # Then prepare data (reshape and select channels)
            if config:
                self.inputs, self.targets = self.prepare_diffusion_input(
                    self.inputs, self.targets, config
                )
                self.prepare_metadata()

                # Calculate sigma_data from normalized targets if this is the training set
                if split_name == "train" and "model_params" in config:
                    self.calculate_sigma_data(config, split_name)

    def prepare_lazy_diffusion_input(self, config):
        """Record channel/time reshape metadata for directory-backed datasets."""
        model_params = config.get("model_params", {})
        dataset_params = config.get("dataset_params", {})
        out_channels = model_params.get("channels", 3)
        channel_indices = dataset_params.get("channel_indices", None)

        if channel_indices is not None:
            if isinstance(channel_indices, int):
                assert out_channels == 1, (
                    f"Model expects {out_channels} channels but only one channel index {channel_indices} was provided"
                )
                channel_indices = [channel_indices]
            else:
                assert len(channel_indices) == out_channels, (
                    f"Model expects {out_channels} channels but {len(channel_indices)} channel indices were provided"
                )
                channel_indices = list(channel_indices)

        self.channel_indices = channel_indices
        self.out_channels = out_channels
        if len(self.original_shape) == 5:
            self.num_samples, self.in_channels, self.timesteps, self.nx, self.ny = self.original_shape
            self.num_items = self.num_samples * self.timesteps
        else:
            self.num_samples = self.original_shape[0]
            self.in_channels = self.original_shape[1]
            self.nx = self.original_shape[-2]
            self.ny = self.original_shape[-1]
            self.num_items = self.num_samples

        if self.channel_indices is None and out_channels < self.in_channels:
            self.channel_indices = list(range(out_channels))

        print(f"Preparing lazy data with output channels={out_channels}")
        if self.channel_indices is not None:
            print(f"Using channel indices {self.channel_indices}")
        print(f"Prepared {self.split_name} lazy item shape: ({self.num_items}, {out_channels}, {self.nx}, {self.ny})\n")

    def _normalize_single(self, tensor, normalizer, re=None):
        if normalizer is None:
            return tensor
        tensor_5d = tensor.unsqueeze(0).unsqueeze(2)
        try:
            normalized = normalizer.normalize(tensor_5d, re=re)
        except TypeError:
            normalized = normalizer.normalize(tensor_5d)
        return normalized.squeeze(0).squeeze(1)

    def prepare_metadata(self):
        """Repeat per-simulation metadata after flattening sample/time dimensions."""
        if self.re_per_sample is not None:
            if self.timesteps is None:
                self.re_per_item = torch.as_tensor(self.re_per_sample, dtype=torch.float32)
            else:
                self.re_per_item = torch.as_tensor(
                    np.repeat(self.re_per_sample, self.timesteps), dtype=torch.float32
                )
            print(f"Prepared Re metadata for {len(self.re_per_item)} {self.split_name} items")

        if self.sample_id_per_sample is not None:
            if self.timesteps is None:
                values = self.sample_id_per_sample
            else:
                values = np.repeat(self.sample_id_per_sample, self.timesteps)
            self.sample_id_per_item = torch.as_tensor(values, dtype=torch.long)

    def apply_sample_fraction(self, dataset_params):
        """Optionally keep a deterministic fraction of simulations before time flattening."""
        fraction = dataset_params.get(f"{self.split_name}_sample_fraction", None)
        if fraction is None:
            fraction = dataset_params.get("sample_fraction", None)
        if fraction is None:
            return

        fraction = float(fraction)
        if fraction <= 0.0 or fraction > 1.0:
            raise ValueError(f"sample_fraction must be in (0, 1], got {fraction}")
        if fraction >= 1.0:
            return

        num_samples = int(self.original_shape[0])
        rng = np.random.default_rng(int(dataset_params.get("seed", 42)))

        if self.re_per_sample is not None:
            selected = []
            for re_value in sorted(float(v) for v in np.unique(self.re_per_sample)):
                re_indices = np.flatnonzero(self.re_per_sample == np.float32(re_value))
                keep = max(1, int(round(len(re_indices) * fraction)))
                chosen = np.sort(rng.choice(re_indices, size=keep, replace=False))
                selected.append(chosen)
            self.sample_indices = np.sort(np.concatenate(selected)).astype(np.int64)
        else:
            keep = max(1, int(round(num_samples * fraction)))
            self.sample_indices = np.sort(
                rng.choice(np.arange(num_samples), size=keep, replace=False)
            ).astype(np.int64)

        if not self.lazy_disk:
            self.inputs = self.inputs[self.sample_indices]
            self.targets = self.targets[self.sample_indices]

        if self.re_per_sample is not None:
            self.re_per_sample = self.re_per_sample[self.sample_indices]
        if self.sample_id_per_sample is not None:
            self.sample_id_per_sample = self.sample_id_per_sample[self.sample_indices]

        self.original_shape = (
            (len(self.sample_indices),) + tuple(self.original_shape[1:])
        )
        print(
            f"Using {fraction:.3f} {self.split_name} sample fraction: "
            f"{len(self.sample_indices)}/{num_samples} simulations"
        )

    def calculate_sigma_data(self, config, split_name):
        """Calculate sigma_data from normalized training targets for the selected channels"""
        if not hasattr(self, "targets") or self.targets is None:
            return
        if getattr(self, "lazy_disk", False):
            configured = config.get("model_params", {}).get("sigma_data")
            if configured is not None:
                print(f"Keeping configured sigma_data={configured} for lazy {split_name} dataset\n")
                return configured
            config["model_params"]["sigma_data"] = 1.0
            print("Setting sigma_data=1.0 for lazy normalized training targets\n")
            return 1.0

        targets = self.targets

        channel_stds = []
        for c in range(targets.shape[1]):
            channel_std = targets[:, c].std().item()
            channel_stds.append(channel_std)

        # Use mean of all channel stds as sigma_data
        sigma_data = float(torch.tensor(channel_stds).mean().item())
        print(
            f"Setting sigma_data={sigma_data} based on mean of std from normalized {split_name} set targets\n"
        )

        config["model_params"]["sigma_data"] = sigma_data

        return sigma_data

    def prepare_diffusion_input(self, x, y, config):
        """Prepare input data for diffusion model by reshaping from
        (nsamples, c=3, t, x, y) to (nsamples*t, c={1,2,3}, x, y)

        Note: This function expects already normalized data
        """
        model_params = config.get("model_params", {})
        dataset_params = config.get("dataset_params", {})

        out_channels = model_params.get("channels", 3)

        channel_indices = dataset_params.get("channel_indices", None)

        # Assert proper relationship between out_channels and channel_indices
        if channel_indices is not None:
            if isinstance(channel_indices, int):
                assert out_channels == 1, (
                    f"Model expects {out_channels} channels but only one channel index {channel_indices} was provided"
                )
            else:
                assert len(channel_indices) == out_channels, (
                    f"Model expects {out_channels} channels but {len(channel_indices)} channel indices were provided"
                )

        # Get spatial dimensions from data
        nx = x.shape[3] if len(x.shape) > 3 else 128
        ny = x.shape[4] if len(x.shape) > 4 else 128

        print(f"Preparing data with output channels={out_channels}")
        if channel_indices is not None:
            print(f"Using channel indices {channel_indices}")

        if len(x.shape) == 5:  # (nsamples, c, t, nx, ny)
            nsamples, in_channels, timesteps, nx, ny = x.shape

            # Reshape to (nsamples*t, c, nx, ny)
            x_reshaped = x.permute(0, 2, 1, 3, 4).reshape(-1, in_channels, nx, ny)
            y_reshaped = y.permute(0, 2, 1, 3, 4).reshape(-1, in_channels, nx, ny)

            # Handle channel selection
            if channel_indices is not None:
                if isinstance(channel_indices, int):
                    channel_indices = [channel_indices]

                x_processed = x_reshaped[:, channel_indices, :, :]
                y_processed = y_reshaped[:, channel_indices, :, :]
            else:
                if out_channels < in_channels:
                    x_processed = x_reshaped[:, :out_channels, :, :]
                    y_processed = y_reshaped[:, :out_channels, :, :]
                else:
                    x_processed = x_reshaped
                    y_processed = y_reshaped
        else:
            # If already in correct format, just pass through
            x_processed = x
            y_processed = y

        print(f"Prepared {self.split_name} input shape: {x_processed.shape}")
        print(f"Prepared {self.split_name} target shape: {y_processed.shape}\n")

        return x_processed, y_processed

    def __len__(self):
        return self.num_items if self.num_items is not None else len(self.inputs)

    def _metadata_for_index(self, idx):
        if not self.return_metadata:
            return None
        metadata = {}
        if self.re_per_item is not None:
            re = self.re_per_item[idx].float()
            metadata["re"] = re
            metadata["rem"] = re.clone()
        if self.sample_id_per_item is not None:
            metadata["sample_id"] = self.sample_id_per_item[idx].long()
        return metadata if metadata else None

    def __getitem__(self, idx):
        metadata = self._metadata_for_index(idx)
        if getattr(self, "lazy_disk", False):
            if len(self.original_shape) == 5:
                sim_idx = idx // self.timesteps
                if self.sample_indices is not None:
                    sim_idx = int(self.sample_indices[sim_idx])
                time_idx = idx % self.timesteps
                x = torch.as_tensor(np.array(self.inputs[sim_idx, :, time_idx], copy=True), dtype=torch.float32)
                y = torch.as_tensor(np.array(self.targets[sim_idx, :, time_idx], copy=True), dtype=torch.float32)
            else:
                if self.sample_indices is not None:
                    idx = int(self.sample_indices[idx])
                x = torch.as_tensor(np.array(self.inputs[idx], copy=True), dtype=torch.float32)
                y = torch.as_tensor(np.array(self.targets[idx], copy=True), dtype=torch.float32)

            if self.residual_target:
                y = y - x

            re_value = metadata.get("re") if metadata is not None else None
            x = self._normalize_single(x, self.input_normalizer, re=re_value)
            y = self._normalize_single(y, self.target_normalizer, re=re_value)

            if self.channel_indices is not None:
                x = x[self.channel_indices]
                y = y[self.channel_indices]
            if metadata is not None:
                return x, y, metadata
            return x, y

        # Get input and target
        x = self.inputs[idx]
        y = self.targets[idx]

        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32)

        if metadata is not None:
            return x, y, metadata
        return x, y


class BalancedReBatchSampler(Sampler):
    """Batch sampler that keeps all configured Re values represented in each batch."""

    def __init__(self, dataset, batch_size, res_per_batch=None, shuffle=True, seed=42):
        if getattr(dataset, "re_per_item", None) is None:
            raise ValueError("BalancedReBatchSampler requires dataset.re_per_item metadata.")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        re_values = dataset.re_per_item.detach().cpu().numpy().astype(np.float32)
        unique_re = sorted(float(value) for value in np.unique(re_values))
        self.re_values = unique_re
        self.res_per_batch = min(int(res_per_batch or len(unique_re)), len(unique_re))
        self.indices_by_re = {
            re: np.flatnonzero(re_values == np.float32(re)).astype(np.int64)
            for re in unique_re
        }
        if self.res_per_batch <= 0 or self.batch_size < self.res_per_batch:
            raise ValueError(
                f"batch_size={self.batch_size} must be >= res_per_batch={self.res_per_batch}."
            )
        self.batches_per_epoch = int(np.ceil(len(dataset) / self.batch_size))
        self.epoch = 0

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        active_re = np.array(self.re_values, dtype=np.float32)
        pools = {}
        cursors = {}
        for re in self.re_values:
            values = self.indices_by_re[re].copy()
            if self.shuffle:
                rng.shuffle(values)
            pools[re] = values
            cursors[re] = 0

        for _batch_idx in range(self.batches_per_epoch):
            if self.shuffle:
                rng.shuffle(active_re)
            selected = [float(v) for v in active_re[: self.res_per_batch]]
            counts = [self.batch_size // self.res_per_batch] * self.res_per_batch
            for i in range(self.batch_size % self.res_per_batch):
                counts[i] += 1

            batch = []
            for re, count in zip(selected, counts):
                pool = pools[re]
                cursor = cursors[re]
                for _ in range(count):
                    if cursor >= len(pool):
                        cursor = 0
                        if self.shuffle:
                            rng.shuffle(pool)
                    batch.append(int(pool[cursor]))
                    cursor += 1
                cursors[re] = cursor
            if self.shuffle:
                rng.shuffle(batch)
            yield batch
        self.epoch += 1

    def __len__(self):
        return self.batches_per_epoch


def get_diffusion_dataloaders(config, num_workers=4):
    """Create dataloaders for diffusion model training

    Args:
        config: Configuration dictionary
        num_workers: Number of workers for data loading

    Returns:
        train_loader, val_loader, test_loader
    """
    input_normalizer = None
    target_normalizer = None

    if "normalization_params" in config:
        norm_params = config.get("normalization_params", {})
        norm_type = norm_params.get("type", "identity").lower()

        if norm_type == "per_re_paired_minmax":
            input_norm_params = norm_params.copy()
            input_norm_params["stats_by_re"] = _load_per_re_stats(norm_params, "inputs")
            input_normalizer = create_normalization({"normalization_params": input_norm_params})
            print("Created per-Re paired_minmax normalizer for inputs\n")

            target_norm_params = norm_params.copy()
            target_norm_params["stats_by_re"] = _load_per_re_stats(norm_params, "targets")
            target_normalizer = create_normalization({"normalization_params": target_norm_params})
            print("Created per-Re paired_minmax normalizer for targets\n")

            inputs_stats_file = None
            targets_stats_file = None
        else:
            inputs_stats_file = norm_params.get("inputs_stats_file")
            targets_stats_file = norm_params.get("targets_stats_file")

        # For backward compatibility, use the original stats_file if specific ones aren't provided
        if (
            norm_type != "per_re_paired_minmax"
            and not inputs_stats_file
            and not targets_stats_file
            and "stats_file" in norm_params
        ):
            inputs_stats_file = norm_params["stats_file"]
            targets_stats_file = norm_params["stats_file"]

        # Create input normalizer
        if inputs_stats_file and inputs_stats_file.endswith(".npz"):
            if not os.path.exists(inputs_stats_file):
                raise FileNotFoundError(
                    f"\nInput statistics file not found: {inputs_stats_file}"
                )

            print(f"\nLoading input normalization statistics from {inputs_stats_file}")
            input_stats = np.load(inputs_stats_file)

            input_norm_params = norm_params.copy()

            for key in input_stats.files:
                input_norm_params[key] = (
                    input_stats[key].tolist()
                    if hasattr(input_stats[key], "tolist")
                    else input_stats[key]
                )
                print(f"Input {key}: {input_stats[key]}")

            input_normalizer = create_normalization(
                {"normalization_params": input_norm_params}
            )
            print(f"Created {norm_type} normalizer for inputs\n")

        # Create target normalizer
        if targets_stats_file and targets_stats_file.endswith(".npz"):
            if not os.path.exists(targets_stats_file):
                raise FileNotFoundError(
                    f"\nTarget statistics file not found: {targets_stats_file}"
                )

            print(f"Loading target normalization statistics from {targets_stats_file}")
            target_stats = np.load(targets_stats_file)

            target_norm_params = norm_params.copy()

            for key in target_stats.files:
                target_norm_params[key] = (
                    target_stats[key].tolist()
                    if hasattr(target_stats[key], "tolist")
                    else target_stats[key]
                )
                print(f"Target {key}: {target_stats[key]}")

            target_normalizer = create_normalization(
                {"normalization_params": target_norm_params}
            )
            print(f"Created {norm_type} normalizer for targets\n")

        if not input_normalizer and not target_normalizer:
            print("\n***WARNING: No valid stats files found for normalization!***\n")

    dataset_params = config.get("dataset_params", {})
    train_path = dataset_params.get("train_path")
    val_path = dataset_params.get("val_path")
    test_path = dataset_params.get("test_path")

    if not train_path:
        raise ValueError("Train data path must be specified in config")

    train_batch_size = config.get("train_loader_params", {}).get("batch_size", 32)

    # Create datasets - create train first to calculate sigma_data
    print(f"Creating training dataset from {train_path}")
    train_dataset = DiffusionDataset(
        train_path,
        config=config,
        input_normalizer=input_normalizer,
        target_normalizer=target_normalizer,
        split_name="train",
    )

    val_dataset = None
    if val_path:
        print(f"Creating validation dataset from {val_path}")
        val_dataset = DiffusionDataset(
            val_path,
            config=config,
            input_normalizer=input_normalizer,
            target_normalizer=target_normalizer,
            split_name="validation",
        )

    test_dataset = None
    if test_path:
        print(f"Creating test dataset from {test_path}")
        test_dataset = DiffusionDataset(
            test_path,
            config=config,
            input_normalizer=input_normalizer,
            target_normalizer=target_normalizer,
            split_name="test",
        )

    # Create dataloaders with appropriate batch sizes
    balanced_re_batches = dataset_params.get("balanced_re_batches", False)
    if balanced_re_batches:
        train_sampler = BalancedReBatchSampler(
            train_dataset,
            batch_size=train_batch_size,
            res_per_batch=dataset_params.get("res_per_batch"),
            shuffle=True,
            seed=dataset_params.get("seed", 42),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
        )
        print(
            f"Using balanced Re batches with batch_size={train_batch_size}, "
            f"res_per_batch={train_sampler.res_per_batch}"
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
        )

    val_loader = None
    if val_dataset:
        # For validation, set batch size to timesteps
        val_batch_size = train_batch_size
        if val_dataset.timesteps:
            model_params = config.get("model_params", {})
            out_channels = model_params.get("channels", 3)
            val_batch_size = val_dataset.timesteps
            print(
                f"Using batch size {val_batch_size} for validation (= {val_dataset.timesteps} timesteps × {out_channels} channels)"
            )

        val_loader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,  # Full simulation in each batch if timesteps available
            shuffle=False,  # Important: keep temporal order
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
        )

    test_loader = None
    if test_dataset:
        # For test, set batch size to timesteps
        test_batch_size = train_batch_size
        if test_dataset.timesteps:
            model_params = config.get("model_params", {})
            out_channels = model_params.get("channels", 3)
            test_batch_size = test_dataset.timesteps
            print(
                f"Using batch size {test_batch_size} for test (= {test_dataset.timesteps} timesteps × {out_channels} channels)"
            )

        test_loader = DataLoader(
            test_dataset,
            batch_size=test_batch_size,  # Full simulation in each batch if timesteps available
            shuffle=False,  # Important: keep temporal order
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
        )

    return train_loader, val_loader, test_loader
