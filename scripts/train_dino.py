#!/usr/bin/env python3
"""Train a previous-study DINO or PHASE residual-diffusion recipe."""

import argparse

from phase.training import train_dino
from phase.utils import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume-checkpoint", default=None)
    args = parser.parse_args()
    train_dino(
        load_config(args.config), resume_checkpoint=args.resume_checkpoint
    )


if __name__ == "__main__":
    main()
