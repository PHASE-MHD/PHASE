import os
import math
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from ..normalizations import create_normalization


class MHDDataset(Dataset):
    def __init__(
        self,
        x_data,
        y_data,
        normalization_config=None,
        split="train",
        train_size=1000,
        val_plus_test_size=200,
        sample_fraction=1.0,
        seed=42,
        sub_t=1,
        sub_x=1,
        t_range=(0, 1.0),
        x_range=(0, 1.0),
        y_range=(0, 1.0),
    ):
        """
        Dataset for MHD simulation data with normalization support

        Args:
            x_data: Initial conditions with shape (num_samples, channels, 1, nx, ny)
            y_data: Full time evolution with shape (num_samples, channels, timesteps, nx, ny)
            normalization_config: Configuration dict for normalization
            split: 'train', 'val', or 'test'
            train_size: Size of the training set in samples
            seed: Random seed for reproducibility
            sub_t: Subsampling factor for time dimension
            sub_x: Subsampling factor for spatial dimensions
        """

        self.num_samples, self.channels, self.timesteps, self.nx, self.ny = y_data.shape
        self.x_data = x_data
        self.y_data = y_data

        self.sub_t = sub_t
        self.sub_x = sub_x

        # Create normalization module if specified
        self.normalizer = None
        if normalization_config:
            self.normalizer = create_normalization(normalization_config)

        np.random.seed(seed)

        # Create train/val/test indices
        indices = np.random.permutation(self.num_samples)
        val_size = val_plus_test_size // 2

        if split == "train":
            self.indices = indices[:train_size]
        elif split == "val":
            self.indices = indices[train_size : train_size + val_size]
        elif split == "test":
            self.indices = indices[train_size + val_size :]
        else:
            raise ValueError(
                f"Split {split} not recognized. Use 'train', 'val', or 'test'"
            )

        sample_fraction = float(sample_fraction)
        if sample_fraction <= 0.0 or sample_fraction > 1.0:
            raise ValueError("sample_fraction must be in (0, 1].")
        if sample_fraction < 1.0:
            original_count = len(self.indices)
            keep = max(1, int(round(original_count * sample_fraction)))
            self.indices = self.indices[:keep]
            print(f"Using {sample_fraction:.3f} {split} sample fraction: {keep}/{original_count}")

        print(f"Initialized {split} dataset with {len(self.indices)} samples")

        # Create coordinate system internally
        self.t = np.linspace(t_range[0], t_range[1], self.timesteps)[::sub_t]
        self.x = np.linspace(x_range[0], x_range[1], self.nx)[::sub_x]
        self.y = np.linspace(y_range[0], y_range[1], self.ny)[::sub_x]

        self.nt = len(self.t)
        self.nx = len(self.x)
        self.ny = len(self.y)

        # Precompute grid tensors (once, not per sample)
        t_tensor = torch.from_numpy(self.t.astype(np.float32))
        x_tensor = torch.from_numpy(self.x.astype(np.float32))
        y_tensor = torch.from_numpy(self.y.astype(np.float32))

        self.grid_t = t_tensor.reshape(self.nt, 1, 1, 1).repeat(1, self.nx, self.ny, 1)
        self.grid_x = x_tensor.reshape(1, self.nx, 1, 1).repeat(self.nt, 1, self.ny, 1)
        self.grid_y = y_tensor.reshape(1, 1, self.ny, 1).repeat(self.nt, self.nx, 1, 1)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """
        Returns:
            inputs: Model inputs with shape (in_dim, nt, nx, ny)
            outputs: Target outputs with shape (out_dim, nt, nx, ny)
        """
        sample_idx = self.indices[idx]

        sample_data = self.y_data[
            sample_idx, :, :: self.sub_t, :: self.sub_x, :: self.sub_x
        ]

        if self.normalizer:
            # Add batch dimension for normalization
            sample_data = sample_data.unsqueeze(0)

            # Make sure normalizer is on the same device as the data
            if hasattr(self.normalizer, "to"):
                # Get device of sample_data
                data_device = sample_data.device
                # Move normalizer to the same device
                self.normalizer = self.normalizer.to(data_device)

            sample_data = self.normalizer.normalize(sample_data)

            # Remove batch dimension
            sample_data = sample_data.squeeze(0)

        data = sample_data.float()

        # Initial condition handling - create data0 by repeating first timestep
        data0 = data[:, 0:1].repeat(1, self.nt, 1, 1)

        # Reshape to match grid tensors [nt, nx, ny, channels]
        data0 = data0.permute(1, 2, 3, 0)

        inputs = torch.cat(
            [self.grid_t, self.grid_x, self.grid_y, data0], dim=-1
        ).permute(3, 0, 1, 2)

        outputs = data.permute(0, 1, 2, 3)

        return inputs, outputs


def get_dataloaders(
    data_path,
    normalization_config=None,
    batch_size=1,
    num_workers=4,
    train_size=1000,
    val_plus_test_size=200,
    train_sample_fraction=1.0,
    validation_sample_fraction=1.0,
    test_sample_fraction=1.0,
    seed=42,
    sub_t=1,
    sub_x=1,
    t_range=(0, 1.0),
    x_range=(0, 1.0),
    y_range=(0, 1.0),
):
    """Create dataloaders for train, validation, and test sets with normalization support"""

    # Process normalization config and load NPZ stats if needed
    if normalization_config and "normalization_params" in normalization_config:
        norm_params = normalization_config["normalization_params"]

        if "stats_file" in norm_params and norm_params["stats_file"].endswith(".npz"):
            stats_file = norm_params["stats_file"]
            if not os.path.exists(stats_file):
                raise FileNotFoundError(f"Statistics file not found: {stats_file}")

            print(f"Loading normalization statistics from {stats_file}")

            stats = np.load(stats_file)

            # Create a copy of the config to avoid modifying the original
            normalization_config = {"normalization_params": norm_params.copy()}

            for key in stats.files:
                normalization_config["normalization_params"][key] = (
                    stats[key].tolist() if hasattr(stats[key], "tolist") else stats[key]
                )
                print(f"{key}: {stats[key]}")

        else:
            print(
                "\n***WARNING: If they exist, statistics provided in the main config file will be used! Make sure you are providing correct stats for the correct normalization option!***\n"
            )

        norm_type = norm_params.get("type", "unknown")
        print(f"Using {norm_type} normalization at dataset level\n")

    print(f"Loading data from {data_path}")
    print(f"This may take a few moments for large datasets...")

    data = np.load(data_path)
    print(f"Raw data shape: {data.shape}")

    # (num_samples, nt, nx, ny, channels) -> (num_samples, channels, nt, nx, ny)
    reshaped_data = torch.from_numpy(data).float().permute(0, 4, 1, 2, 3)
    print(f"Reshaped data: {reshaped_data.shape}")

    x_data = reshaped_data[:, :, 0:1]  # Shape: (num_samples, channels, 1, nx, ny)
    y_data = reshaped_data  # Shape: (num_samples, channels, timesteps, nx, ny)
    print(f"Initial conditions shape: {x_data.shape}")
    print(f"Full time evolution shape: {y_data.shape}")

    print(f"\nCreating dataloaders with train_size={train_size}, seed={seed}")
    print(f"Subsampling factors: sub_t={sub_t}, sub_x={sub_x}")
    print(f"Datasets using random seed: {seed} for each split")

    # Create datasets with the processed normalization config
    train_dataset = MHDDataset(
        x_data,
        y_data,
        normalization_config=normalization_config,
        split="train",
        train_size=train_size,
        val_plus_test_size=val_plus_test_size,
        sample_fraction=train_sample_fraction,
        seed=seed,
        sub_t=sub_t,
        sub_x=sub_x,
        t_range=t_range,
        x_range=x_range,
        y_range=y_range,
    )
    val_dataset = MHDDataset(
        x_data,
        y_data,
        normalization_config=normalization_config,
        split="val",
        train_size=train_size,
        val_plus_test_size=val_plus_test_size,
        sample_fraction=validation_sample_fraction,
        seed=seed,
        sub_t=sub_t,
        sub_x=sub_x,
        t_range=t_range,
        x_range=x_range,
        y_range=y_range,
    )
    test_dataset = MHDDataset(
        x_data,
        y_data,
        normalization_config=normalization_config,
        split="test",
        train_size=train_size,
        val_plus_test_size=val_plus_test_size,
        sample_fraction=test_sample_fraction,
        seed=seed,
        sub_t=sub_t,
        sub_x=sub_x,
        t_range=t_range,
        x_range=x_range,
        y_range=y_range,
    )

    # Important: Use persistent workers to avoid normalizer device issues
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),  # Keep workers alive between iterations
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    return train_loader, val_loader, test_loader
