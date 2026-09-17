#!/usr/bin/env python3
"""Train the previous-study tFNO baseline from a YAML configuration."""

from __future__ import annotations

import argparse

from phase.training import train_tfno
from phase.utils import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train_tfno(load_config(args.config))


if __name__ == "__main__":
    main()
