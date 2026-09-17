#!/usr/bin/env python3
"""Train the previous-study full-field DINO baseline."""

import argparse

from phase.training import train_dino
from phase.utils import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train_dino(load_config(args.config))


if __name__ == "__main__":
    main()
