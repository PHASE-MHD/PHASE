import torch
from torch.utils.data import Dataset
from ..normalizations import create_normalization
import numpy as np
import os


class EmbeddedMHDDataset(Dataset):
    """Dataset that applies coordinate embeddings on-the-fly and expands time dimension"""

    def __init__(
        self,
        data_tensor,
        target_tensor,
        nx,
        ny,
        normalization_config=None,
        model_config=None,
    ):
        self.data = data_tensor
        self.target = target_tensor

        # Pre-compute coordinate grids once (tiny memory footprint)
        self.x_grid = torch.linspace(0, 1, nx, dtype=torch.float32)
        self.y_grid = torch.linspace(0, 1, ny, dtype=torch.float32)

        self.normalizer = None
        if normalization_config:
            normalization_config = self._process_norm_config(normalization_config)
            self.normalizer = create_normalization(normalization_config)
            norm_type = normalization_config.get("normalization_params", {}).get(
                "type", "unknown"
            )
            print(f"Using {norm_type} normalization at dataset level")

        self.model_config = model_config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]  # [3, 1, nx, ny]
        y = self.target[idx]  # [3, nt, nx, ny]

        if self.normalizer:
            x, y = x.unsqueeze(0), y.unsqueeze(0)

            x = self.normalizer.normalize(x)
            y = self.normalizer.normalize(y)

            x, y = x.squeeze(0), y.squeeze(0)

        nx, ny = x.shape[2], x.shape[3]
        nt = y.shape[1]  # Number of timesteps from target

        x_expanded = x.expand(3, nt, nx, ny)  # [3, nt, nx, ny]

        x_coords = self.x_grid.view(1, 1, nx, 1).expand(1, nt, nx, ny)
        y_coords = self.y_grid.view(1, 1, 1, ny).expand(1, nt, nx, ny)

        t_values = torch.linspace(0, 1, nt, dtype=torch.float32)
        t_coords = t_values.view(1, nt, 1, 1).expand(1, nt, nx, ny)

        x_embedded = torch.cat(
            [
                t_coords,  # [1, nt, nx, ny]
                x_coords,  # [1, nt, nx, ny]
                y_coords,  # [1, nt, nx, ny]
                x_expanded,  # [3, nt, nx, ny]
            ],
            dim=0,
        )  # Result: [6, nt, nx, ny]

        if self.model_config:
            if not self.model_config["model_params"]["positional_embedding"]:
                # print("Adding positional embedding to the data...")  # Debug print
                return x_embedded, y
            # else:
            # print(
            #     f"Positional embedding type is: {self.model_config['model_params']['positional_embedding']}"
            # )  # Debug print
            return x_expanded, y
        else:
            print(
                "***WARNING: Model config not provided during dataset initialization! Returning embedded data anyway***"
            )
            return x_embedded, y

    def _process_norm_config(self, normalization_config):
        # Process normalization config and load NPZ stats if needed
        if normalization_config and "normalization_params" in normalization_config:
            norm_params = normalization_config["normalization_params"]

            if "stats_file" in norm_params and norm_params["stats_file"].endswith(
                ".npz"
            ):
                stats_file = norm_params["stats_file"]
                if not os.path.exists(stats_file):
                    raise FileNotFoundError(f"Statistics file not found: {stats_file}")

                print(f"Loading normalization statistics from {stats_file}")

                stats = np.load(stats_file)

                normalization_config = {"normalization_params": norm_params.copy()}

                for key in stats.files:
                    normalization_config["normalization_params"][key] = (
                        stats[key].tolist()
                        if hasattr(stats[key], "tolist")
                        else stats[key]
                    )
                    print(f"{key}: {stats[key]}")

            else:
                print(
                    "\n***WARNING: If they exist, statistics provided in the main config file will be used! Make sure you are providing correct stats for the correct normalization option!***\n"
                )

        return normalization_config
