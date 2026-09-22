# Documentation

These path-independent guides describe the public PHASE workflows:

- data layout and preparation: `data_format.md`, `preprocessing.md`;
- previous baselines: `previous_baseline_tfno.md`, `previous_baseline_dino.md`;
- three-channel scOT ablations: `scot_without_transfer_learning.md`,
  `scot_with_transfer_learning.md`, `naive_multi_regime.md`, and
  `gated_adapter_multi_regime.md`;
- four-channel decaying-turbulence models: `turbulence_single_re_scot.md`,
  `four_channel_hp_physics.md`, and `turbulence_residual_diffusion.md`;
- Kelvin--Helmholtz models: `kh_scot.md`, `kh_residual_diffusion.md`;
- held-out evaluation and figures: `evaluation.md`, `visualization.md`.

The YAML files under `configs/` are the source of truth for architecture,
normalization, optimizer, loss, split, and checkpoint settings. Model
checkpoints and training data are external artifacts; released checkpoint
metadata and downloads belong in the corresponding Hugging Face model cards.
