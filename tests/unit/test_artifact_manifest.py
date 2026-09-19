from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).parents[2]


def test_checkpoint_manifest_is_complete_and_unambiguous():
    path = ROOT / "provenance/checkpoint_manifest.yaml"
    manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    artifacts = manifest["artifacts"]
    ids = [artifact["id"] for artifact in artifacts]
    env_vars = [artifact["env_var"] for artifact in artifacts]
    assert manifest["version"] == 1
    assert len(artifacts) == 16
    assert len(ids) == len(set(ids))
    assert len(env_vars) == len(set(env_vars))
    assert "multi_re_phase_dt_epoch95" in ids
    assert "single_re_phase_dt_residual_epoch90" in ids
    assert "kh_multi_re_phase_t0_5_epoch95" in ids
    for artifact in artifacts:
        assert len(artifact["sha256"]) == 64
        int(artifact["sha256"], 16)
        assert artifact["bytes"] > 0
        assert not Path(artifact["path_from_artifact_root"]).is_absolute()


def test_manifest_verifier_checks_size_and_hash(tmp_path):
    from scripts.verify_artifact_manifest import verify

    artifact_path = tmp_path / "artifact.bin"
    artifact_path.write_bytes(b"phase")
    artifact = {
        "id": "fixture",
        "bytes": 5,
        "sha256": "02195b8e989603e4dfb45352b902e8dc34bff40bb228f3c26e56a6ddebdb85ad",
    }
    verify(artifact, artifact_path)
    artifact["bytes"] = 4
    with pytest.raises(ValueError, match="byte-size mismatch"):
        verify(artifact, artifact_path)
    artifact["bytes"] = 5
    artifact["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify(artifact, artifact_path)
