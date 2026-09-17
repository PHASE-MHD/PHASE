# Batch 03: preprocessing

## Scope

This batch consolidates model-independent conversion and train-only statistics
from the legacy scripts. It does not copy, import, or execute any Dedalus
simulation generator.

Public interfaces:

- src/phase/preprocessing/conversion.py
- src/phase/preprocessing/statistics.py
- scripts/prepare_data.py
- scripts/compute_statistics.py

## Legacy source provenance

MHD-World commit: f6a13a6dee3a7a652e66abd8ea0ea18d4ac6cc72
MHD-World-new commit: 9657acc85cd765e7b9eb1dd79c8892dd07f1312d

| Legacy source | SHA-256 |
| --- | --- |
| MHD-World-new/scripts/convert_dedalus_h5_to_poseidon_bfield_npy.py | ebc2167e4c4bdb5ee23027ec968b2b549b8f59ed5c6cf5276be347c44e957f6b |
| MHD-World/scripts/convert_dedalus_h5_to_poseidon_npy.py | 4cd269d4aced25ecc86e6a78ca6b3ce98b333762ac6ae66cae56753cb1ea7015 |
| MHD-World/scripts/convert_poseidon_vecpot_to_bfield_npy.py | a4eebbe44658def5aa6908b2ab6d8d014a1af5f47b14d6e47cedb307abf8c28b |
| MHD-World-new/scripts/calculate_statistics.py | 52f53fa0e159fedfe1a1b366ac4de297def523ea909d6895f5e42934291cac43 |
| MHD-World-new/scripts/KH_multiRe/compute_KH_multiRe_bfield_norms.py | 025d08a1a6ebd36b38c74f01144620559c59a268d1e0019ea1fb336e3b2295ca |
| MHD-World-new/scripts/KH_multiRe/compute_KH_multiRe_bfield_norms_singleReCompatible.py | 23f8cb0954fced753c497ec53775450313da4dd5a6b14c83188e0ebec898c6c5 |

These working-tree hashes are authoritative for the inherited portions.

## Consolidation decisions

1. Duplicate HDF5 discovery and conversion logic is represented by one
   representation-selecting function.
2. Vector-potential conversion retains the periodic spectral convention
   Bx=dA/dy and By=-dA/dx.
3. Statistics are streaming/chunked and fitted to deterministic training
   indices only.
4. Single-Re and multi-Re split algorithms are both explicit because their
   historical random-number generators differ.
5. Diffusion conditioner and clean-target statistics remain separate; residual
   mode constructs DNS-conditioner before target fitting.
6. KH magnetic p99 uses a paired scale equal to the larger component-wise
   absolute p99 and records per-Re plus global values.
7. All cluster paths, fixed Re lists, and job-specific filenames were removed.

## Verification

Synthetic tests verify HDF5 task mapping, direct-B equivalence, the spectral
sign convention, deterministic train-only selection, residual statistics, and
paired multi-Re p99 output. All runtime assertions and CLI help checks passed
in the existing project container.
