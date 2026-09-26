"""Safety and leakage invariants for the task-01 comparison package."""

import json
from pathlib import Path

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


def test_unweighted_diagnostic_score_range_and_finite():
    values = comparison.score(np.array([0, 1, 0, 1]), np.array([.1, .9, .3, .7]))
    assert values["average_precision"] == 1.0
    assert 0 < values["brier_unadjusted"] < 1
    with pytest.raises(ValueError, match="Invalid"):
        comparison.score(np.array([0, 1]), np.array([np.nan, .9]))
