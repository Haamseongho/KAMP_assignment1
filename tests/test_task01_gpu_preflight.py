"""Synthetic file checks for the other-PC Task 01 GPU preflight."""

import hashlib
import json
from types import SimpleNamespace

import pytest

import task01_gpu_preflight as preflight


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fixture_dirs(tmp_path, monkeypatch):
    data_dir = tmp_path / "official"
    split_dir = tmp_path / "split"
    data_dir.mkdir()
    split_dir.mkdir()
    sources = {}
    for name in preflight.EXPECTED_CSV_SHA256:
        value = name.encode()
        (data_dir / name).write_bytes(value)
        sources[name] = digest(value)
    split_bytes = b"synthetic split; no competition rows\n"
    (split_dir / "split_manifest.csv").write_bytes(split_bytes)
    split_hash = digest(split_bytes)
    (split_dir / "run_manifest.json").write_text(json.dumps({"split_sha256": split_hash}))
    (split_dir / "execution.json").write_text(json.dumps({"status": "passed"}))
    monkeypatch.setattr(preflight, "EXPECTED_CSV_SHA256", sources)
    monkeypatch.setattr(preflight, "EXPECTED_SPLIT_SHA256", split_hash)
    monkeypatch.setattr(preflight.importlib.metadata, "version",
                        lambda name: preflight.EXPECTED_PACKAGES[name])
    monkeypatch.setattr(preflight.platform, "system", lambda: "Windows")
    monkeypatch.setattr(preflight.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        returncode=0, stdout="NVIDIA Test GPU, 999.0, 8192 MiB\n"))
    return data_dir, split_dir


def test_preflight_passes_only_as_technical_preflight(tmp_path, monkeypatch):
    data_dir, split_dir = fixture_dirs(tmp_path, monkeypatch)
    report = preflight.inspect(data_dir, split_dir)
    assert report["status"] == "passed_preflight_only"
    assert report["gpu_fit_verified"] is False
    assert report["independent_performance_verified"] is False
    assert len(report["source_files"]) == 4


def test_preflight_rejects_changed_official_csv(tmp_path, monkeypatch):
    data_dir, split_dir = fixture_dirs(tmp_path, monkeypatch)
    (data_dir / "moldset_labeled_rg3.csv").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Official CSV hash mismatch"):
        preflight.inspect(data_dir, split_dir)


def test_preflight_rejects_changed_split(tmp_path, monkeypatch):
    data_dir, split_dir = fixture_dirs(tmp_path, monkeypatch)
    (split_dir / "split_manifest.csv").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Frozen split hash mismatch"):
        preflight.inspect(data_dir, split_dir)


def test_preflight_rejects_missing_nvidia(tmp_path, monkeypatch):
    data_dir, split_dir = fixture_dirs(tmp_path, monkeypatch)
    monkeypatch.setattr(preflight.subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError()))
    with pytest.raises(ValueError, match="nvidia-smi"):
        preflight.inspect(data_dir, split_dir)
