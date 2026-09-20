#!/usr/bin/env python3
"""Verify canonical external checkpoints and optionally emit test exports."""

import argparse
import hashlib
import os
import re
import shlex
from pathlib import Path

import yaml


_ARTIFACT_FIELDS = {
    "id",
    "role",
    "env_var",
    "path_from_artifact_root",
    "bytes",
    "sha256",
    "epoch",
}
_ENV_VAR_PATTERN = re.compile(r"^[A-Z_][A-Z0-9_]*$")


def sha256(path, chunk_size=16 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        manifest = yaml.safe_load(stream)
    if not isinstance(manifest, dict):
        raise ValueError("Checkpoint manifest must be a mapping.")
    if manifest.get("version") != 1 or not isinstance(manifest.get("artifacts"), list):
        raise ValueError("Unsupported or malformed checkpoint manifest.")

    root_env = manifest.get("artifact_root_env")
    if not isinstance(root_env, str) or not _ENV_VAR_PATTERN.fullmatch(root_env):
        raise ValueError("artifact_root_env must be a valid environment variable name.")

    seen_ids = set()
    seen_env_vars = set()
    seen_paths = set()
    for index, artifact in enumerate(manifest["artifacts"]):
        if not isinstance(artifact, dict):
            raise ValueError(f"Artifact {index} must be a mapping.")
        missing = sorted(_ARTIFACT_FIELDS - set(artifact))
        if missing:
            raise ValueError(f"Artifact {index} is missing fields: {', '.join(missing)}")

        artifact_id = artifact["id"]
        env_var = artifact["env_var"]
        path_value = artifact["path_from_artifact_root"]
        if not isinstance(path_value, str) or not path_value:
            raise ValueError(
                f"Artifact {artifact_id} has an invalid path_from_artifact_root."
            )
        relative_path = Path(path_value)
        digest = artifact["sha256"]
        byte_size = artifact["bytes"]
        epoch = artifact["epoch"]

        if not isinstance(artifact_id, str) or not artifact_id:
            raise ValueError(f"Artifact {index} has an invalid id.")
        if artifact_id in seen_ids:
            raise ValueError(f"Duplicate artifact id: {artifact_id}")
        if not isinstance(artifact["role"], str) or not artifact["role"].strip():
            raise ValueError(f"Artifact {artifact_id} has an invalid role.")
        if not isinstance(env_var, str) or not _ENV_VAR_PATTERN.fullmatch(env_var):
            raise ValueError(f"Artifact {artifact_id} has an invalid env_var.")
        if env_var in seen_env_vars:
            raise ValueError(f"Duplicate artifact env_var: {env_var}")
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(
                f"Artifact {artifact_id} path must remain below the artifact root."
            )
        normalized_path = relative_path.as_posix()
        if normalized_path in seen_paths:
            raise ValueError(f"Duplicate artifact path: {normalized_path}")
        if (
            isinstance(byte_size, bool)
            or not isinstance(byte_size, int)
            or byte_size <= 0
        ):
            raise ValueError(f"Artifact {artifact_id} has an invalid byte size.")
        if not isinstance(digest, str) or not re.fullmatch(
            r"[0-9a-fA-F]{64}", digest
        ):
            raise ValueError(f"Artifact {artifact_id} has an invalid SHA-256 digest.")
        if epoch is not None and (
            isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0
        ):
            raise ValueError(f"Artifact {artifact_id} has an invalid epoch.")

        seen_ids.add(artifact_id)
        seen_env_vars.add(env_var)
        seen_paths.add(normalized_path)
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
