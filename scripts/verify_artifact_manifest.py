#!/usr/bin/env python3
"""Verify canonical external checkpoints and optionally emit test exports."""

import argparse
import hashlib
import os
import shlex
from pathlib import Path

import yaml


def sha256(path, chunk_size=16 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        manifest = yaml.safe_load(stream)
    if manifest.get("version") != 1 or not isinstance(manifest.get("artifacts"), list):
        raise ValueError("Unsupported or malformed checkpoint manifest.")
    return manifest


def select_artifacts(manifest, requested):
    artifacts = manifest["artifacts"]
    by_id = {artifact["id"]: artifact for artifact in artifacts}
    if len(by_id) != len(artifacts):
        raise ValueError("Checkpoint manifest contains duplicate artifact ids.")
    if not requested:
        return artifacts
    missing = sorted(set(requested) - set(by_id))
    if missing:
        raise ValueError(f"Unknown artifact ids: {', '.join(missing)}")
    return [by_id[artifact_id] for artifact_id in requested]


def resolve_path(artifact, root):
    override = os.environ.get(artifact["env_var"])
    if override:
        return Path(override).expanduser()
    if root is None:
        raise ValueError(
            f"Set {artifact['env_var']} or provide --artifact-root for {artifact['id']}."
        )
    return root / artifact["path_from_artifact_root"]


def verify(artifact, path, verify_hash=True):
    if not path.is_file():
        raise FileNotFoundError(f"{artifact['id']}: missing artifact: {path}")
    size = path.stat().st_size
    if size != artifact["bytes"]:
        raise ValueError(
            f"{artifact['id']}: byte-size mismatch: expected "
            f"{artifact['bytes']}, got {size}"
        )
    if verify_hash:
        actual = sha256(path)
        if actual.lower() != artifact["sha256"].lower():
            raise ValueError(
                f"{artifact['id']}: SHA-256 mismatch: expected "
                f"{artifact['sha256']}, got {actual}"
            )


def main():
    default_manifest = Path(__file__).parents[1] / "provenance/checkpoint_manifest.yaml"
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_ids", nargs="*")
    parser.add_argument("--manifest", type=Path, default=default_manifest)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--size-only", action="store_true")
    parser.add_argument("--emit-exports", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    root = args.artifact_root
    if root is None:
        root_value = os.environ.get(manifest["artifact_root_env"])
        root = Path(root_value).expanduser() if root_value else None

    resolved = []
    for artifact in select_artifacts(manifest, args.artifact_ids):
        path = resolve_path(artifact, root)
        verify(artifact, path, verify_hash=not args.size_only)
        resolved.append((artifact, path))
        if not args.emit_exports:
            mode = "size" if args.size_only else "SHA-256"
            print(f"Verified {artifact['id']} ({mode}): {path}")

    if args.emit_exports:
        for artifact, path in resolved:
            print(f"export {artifact['env_var']}={shlex.quote(str(path))}")


if __name__ == "__main__":
    main()
