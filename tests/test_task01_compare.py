"""Safety and leakage invariants for the task-01 comparison package."""

import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

import task01_compare as comparison


def test_frozen_development_has_no_holdout_or_fold_overlap():
    data, feature_matrix, _ = comparison.validate_frozen_data()
    assert len(data) == 1913
    assert data.feature_group.nunique() == 957
    assert feature_matrix.shape == (1913, 24)
    for fold in range(5):
        training = set(data.loc[data.outer_fold.ne(fold), "feature_group"])
        validation = set(data.loc[data.outer_fold.eq(fold), "feature_group"])
        assert not training.intersection(validation)
        assert data.loc[data.outer_fold.eq(fold), "PassOrFail"].sum() > 0


def test_gpu_refuses_missing_or_unverified_task04_receipt(tmp_path: Path):
    with pytest.raises(ValueError, match="GPU_BLOCKED"):
        comparison.check_gpu_receipt(None)
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"task04_results_recovered": False}))
    with pytest.raises(ValueError, match="GPU_BLOCKED"):
        comparison.check_gpu_receipt(receipt)


def test_gpu_refuses_incorrect_archive_hash(tmp_path: Path):
    archive = tmp_path / "task04.zip"
    archive.write_bytes(b"not a real archive")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"task04_results_recovered": True,
                                   "task04_zip": str(archive), "sha256": "0" * 64,
                                   "verified_at_utc": "2026-09-23T00:00:00Z",
                                   "verified_by": "test"}))
    with pytest.raises(ValueError, match="GPU_BLOCKED"):
        comparison.check_gpu_receipt(receipt)


def test_gpu_refuses_corrupt_zip_even_when_hash_matches(tmp_path: Path):
    archive = tmp_path / "task04.zip"
    archive.write_bytes(b"not a real archive")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"task04_results_recovered": True,
                                   "task04_zip": str(archive),
                                   "sha256": comparison.mg.sha256(archive),
                                   "verified_at_utc": "2026-09-23T00:00:00Z",
                                   "verified_by": "test"}))
    with pytest.raises(ValueError, match="GPU_BLOCKED"):
        comparison.check_gpu_receipt(receipt)


def test_local_gpu_opt_in_is_explicit_and_separate_from_legacy_receipt(tmp_path: Path):
    with pytest.raises(ValueError, match="GPU_BLOCKED"):
        comparison.check_gpu_authorization(None, False)
    local, receipt = comparison.check_gpu_authorization(None, True)
    assert local == {"mode": "user_controlled_local_gpu", "cost_and_power_acknowledged": True}
    assert receipt is None
    with pytest.raises(ValueError, match="choose"):
        comparison.check_gpu_authorization(tmp_path / "receipt.json", True)

    archive = tmp_path / "task04.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("result.txt", "TEST_ONLY")
    record = tmp_path / "receipt.json"
    record.write_text(json.dumps({"task04_results_recovered": True,
                                  "task04_zip": str(archive), "sha256": comparison.mg.sha256(archive),
                                  "verified_at_utc": "2026-09-23T00:00:00Z", "verified_by": "test"}))
    legacy, verified = comparison.check_gpu_authorization(record, False)
    assert legacy["mode"] == "legacy_task04_receipt"
    assert verified["sha256"] == comparison.mg.sha256(archive)


def test_gpu_runtime_accepts_windows_nvidia_but_not_missing_utility(monkeypatch):
    monkeypatch.setattr(comparison.platform, "system", lambda: "Windows")
    monkeypatch.setattr(comparison.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        returncode=0, stdout="NVIDIA Test GPU, 999.0, 8192 MiB\n"))
    runtime = comparison.check_gpu_runtime()
    assert runtime["platform"] == "Windows"
    assert "not_proof_of_gpu_training" in runtime["scope"]
    monkeypatch.setattr(comparison.subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError()))
    with pytest.raises(RuntimeError, match="GPU_BLOCKED"):
        comparison.check_gpu_runtime()


def test_gpu_backend_check_rejects_xgboost_cpu_fallback():
    def xgb_model(device):
        return SimpleNamespace(get_booster=lambda: SimpleNamespace(save_config=lambda: json.dumps({
            "learner": {"generic_param": {"device": device}}})))
    assert comparison.check_fitted_gpu_backend(xgb_model("cuda:0"), "xgb_shallow") == "cuda:0"
    with pytest.raises(RuntimeError, match="fitted on cpu"):
        comparison.check_fitted_gpu_backend(xgb_model("cpu"), "xgb_shallow")
    catboost = SimpleNamespace(get_all_params=lambda: {"task_type": "GPU"})
    assert comparison.check_fitted_gpu_backend(catboost, "catboost_ref") == "GPU"


def test_unweighted_diagnostic_score_range_and_finite():
    values = comparison.score(np.array([0, 1, 0, 1]), np.array([.1, .9, .3, .7]))
    assert values["average_precision"] == 1.0
    assert 0 < values["brier_unadjusted"] < 1
    with pytest.raises(ValueError, match="Invalid"):
        comparison.score(np.array([0, 1]), np.array([np.nan, .9]))
