#!/usr/bin/env python3
"""Verify an external artifact against its documented SHA-256 checksum."""

import argparse
import hashlib
from pathlib import Path


def sha256(path, chunk_size=16 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("expected_sha256")
    args = parser.parse_args()
    actual = sha256(args.path)
    if actual.lower() != args.expected_sha256.lower():
        raise SystemExit(f"SHA-256 mismatch: expected {args.expected_sha256}, got {actual}")
    print(f"Verified {args.path}: {actual}")


if __name__ == "__main__":
    main()
