# Configurations

Canonical, path-independent YAML configurations will be organized by previous
baseline, ablation, physical problem, and single- versus multi-regime scope.

Machine-specific data, output, and checkpoint roots must not be committed.

Implemented ablation configs:

- `ablations/scot_without_tl/re1000.yaml`: three-channel scOT trained from
  scratch at `Re=Rm=1000`, with the reported batch size of 1.
