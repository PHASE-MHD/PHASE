#!/usr/bin/env python3
"""Convert MHD trajectories into PHASE NumPy arrays."""

from __future__ import annotations

import argparse

from phase.preprocessing import convert_dedalus_h5, convert_vector_potential_npy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    dedalus = commands.add_parser("dedalus", help="Convert Dedalus HDF5 outputs")
    dedalus.add_argument("--input-root", required=True)
    dedalus.add_argument("--output", required=True)
    dedalus.add_argument(
        "--representation",
        required=True,
        choices=("vector_potential", "direct_b"),
    )
    dedalus.add_argument("--max-sims", type=int)
    dedalus.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    dedalus.add_argument("--overwrite", action="store_true")

    vector = commands.add_parser(
        "vector-potential", help="Convert [ux,uy,A] NPY to [ux,uy,Bx,By]"
    )
    vector.add_argument("--input", required=True)
    vector.add_argument("--output", required=True)
    vector.add_argument("--lx", type=float, default=1.0)
    vector.add_argument("--ly", type=float, default=1.0)
    vector.add_argument("--max-sims", type=int)
    vector.add_argument("--chunk-size", type=int, default=8)
    vector.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    vector.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "dedalus":
        output = convert_dedalus_h5(
            args.input_root,
            args.output,
            representation=args.representation,
            max_sims=args.max_sims,
            dtype=args.dtype,
            overwrite=args.overwrite,
        )
    else:
        output = convert_vector_potential_npy(
            args.input,
            args.output,
            lx=args.lx,
            ly=args.ly,
            max_sims=args.max_sims,
            chunk_size=args.chunk_size,
            dtype=args.dtype,
            overwrite=args.overwrite,
        )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
