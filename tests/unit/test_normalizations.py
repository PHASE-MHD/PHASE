"""Tests for paired and Reynolds-number-aware normalization."""

import math

import torch

from phase.normalizations import create_normalization
from phase.normalizations.min_max_norm import PerRePairedMinMaxNormalization


def test_physics_normalization_shares_vector_pair_scales():
    normalizer = create_normalization(
        {
            "normalization_params": {
                "type": "physics",
                "input_norm": [1.0, 3.0, 2.0, 4.0],
                "output_norm": [1.0, 3.0, 2.0, 4.0],
                "paired_input_groups": [[0, 1], [2, 3]],
                "paired_output_groups": [[0, 1], [2, 3]],
                "paired_mode": "rms",
            }
        }
    )
    expected = torch.tensor(
        [math.sqrt(5.0), math.sqrt(5.0), math.sqrt(10.0), math.sqrt(10.0)]
    )
    assert torch.allclose(normalizer.input_norm, expected)
    assert torch.allclose(normalizer.output_norm, expected)

    field = torch.randn(2, 4, 3, 5, 5)
    assert torch.allclose(
        normalizer.denormalize(normalizer.normalize(field)), field, atol=1.0e-6
    )


def test_per_re_minmax_interpolates_in_log_re():
    normalizer = PerRePairedMinMaxNormalization(
        stats_by_re={
            100.0: {
                "min_val": [0.0, 0.0, 0.0, 0.0],
                "max_val": [2.0, 4.0, 6.0, 8.0],
            },
            1000.0: {
                "min_val": [0.0, 0.0, 0.0, 0.0],
                "max_val": [4.0, 8.0, 12.0, 16.0],
            },
        },
        feature_range=(0.0, 1.0),
    )
    unseen_re = math.sqrt(100.0 * 1000.0)
    expected_max = torch.tensor([6.0, 6.0, 12.0, 12.0])
    values = expected_max.reshape(1, 4, 1, 1, 1)
    normalized = normalizer.normalize(values, re=torch.tensor([unseen_re]))
    assert torch.allclose(normalized, torch.ones_like(normalized), atol=1.0e-6)
    assert torch.allclose(
        normalizer.denormalize(normalized, re=torch.tensor([unseen_re])),
        values,
        atol=1.0e-6,
    )
