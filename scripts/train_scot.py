#!/usr/bin/env python3
"""Train a PHASE scOT ablation from a YAML configuration."""

from __future__ import annotations

import argparse

from phase.training import train_scot
from phase.utils import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train_scot(load_config(args.config))


if __name__ == "__main__":
    main()
