# Provenance

This directory records the legacy configs, selected checkpoint metadata, split
definitions, normalization statistics, and evaluation provenance for the
reported ablations and DT/KH model chains. It does not contain checkpoints or
large generated artifacts.

Implemented ablation provenance currently includes the previous tFNO/DINO
baselines, single-Re scOT without/with transfer learning, and naive
multi-regime input conditioning. Batch-level validation records are stored as
`batch_*.md`.
