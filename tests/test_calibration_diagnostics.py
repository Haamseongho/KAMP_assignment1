"""Checks of development-only calibration; no simulated contest performance."""
import json

import numpy as np
import pandas as pd
import pytest

import calibration_diagnostics as review
import moldguard as mg


@pytest.fixture(scope="module")
def development():
    if mg.DATA_DIR is None or not (review.FROZEN / "split_manifest.csv").exists():
        pytest.skip("Official inputs and reviewed split are required")
    return review.load_development()[0]


def test_outer_development_membership_is_unchanged(development):
    assert len(development) == 1913
    assert development[mg.TARGET].sum() == 33
    assert development.feature_group.nunique() == 957
    frozen = pd.read_csv(review.FROZEN / "split_manifest.csv")
    excluded = set(frozen.loc[frozen.partition.eq("holdout"), "feature_group"])
    assert not set(development.feature_group) & excluded


def test_inner_folds_never_access_outer_validation_groups(development):
    for outer in range(5):
        fit = development.loc[development.outer_fold.ne(outer)].reset_index(drop=True)
        outer_groups = set(development.loc[development.outer_fold.eq(outer), "feature_group"])
        coverage = np.zeros(len(fit), dtype=int)
        for train, valid in review.grouped_cv(fit):
            training_groups = set(fit.iloc[train].feature_group)
            validation_groups = set(fit.iloc[valid].feature_group)
            assert not training_groups & validation_groups
            assert not (training_groups | validation_groups) & outer_groups
            assert len(train) + len(valid) == len(fit)
            coverage[valid] += 1
        assert np.all(coverage == 1)


def test_reliability_bin_edges_include_zero_and_one():
    # Scalar boundary fixtures test arithmetic only, not manufacturing performance.
    rows = review.reliability_rows(np.array([0, 1, 1]), np.array([0., .01, 1.]), "test", "test")
    assert sum(row["rows"] for row in rows) == 3
    assert rows[0]["rows"] == rows[1]["rows"] == rows[-1]["rows"] == 1
    assert any(row["mean_probability"] is None for row in rows)


def test_probability_loss_distinguishes_probability_from_rank():
    y = np.array([0, 1])
    optimistic = review.probability_metrics(y, np.array([.4, .8]))
    correct = review.probability_metrics(y, np.array([.1, .9]))
    assert optimistic["average_precision"] == correct["average_precision"]
    assert correct["brier"] < optimistic["brier"]
    assert correct["log_loss"] < optimistic["log_loss"]


def test_published_numeric_artifacts_and_model_are_untouched():
    path = mg.ROOT / "outputs/moldguard/reproducibility.json"
    if not path.exists():
        pytest.skip("Published reference is not installed")
    reference = json.loads(path.read_text())["artifact_hashes"]
    for name in ["experiment.json", "model.joblib", "unlabeled_priority.csv", "holdout_diagnostics.csv"]:
        relative = f"outputs/moldguard/{name}"
        assert mg.sha256(mg.ROOT / relative) == reference[relative]


def test_saved_review_contains_only_frozen_development_rows(development):
    folder = mg.ROOT / "outputs/calibration-review-20260922"
    if not folder.exists():
        pytest.skip("Run calibration_diagnostics.py before artifact-level checks")
    predictions = pd.read_csv(folder / "nested_oof_predictions.csv")
    split = pd.read_csv(folder / "nested_split_manifest.csv")
    assert set(zip(predictions.machine, predictions.source_row_id)) == set(
        zip(development.machine, development[mg.ID_COL]))
    assert len(predictions) == len(development)
    for outer in range(5):
        column = split[f"inner_fold_when_outer_{outer}"]
        assert column[split.outer_fold.eq(outer)].isna().all()
        assert column[split.outer_fold.ne(outer)].isin(range(5)).all()
        assert split.groupby("feature_group")[f"inner_fold_when_outer_{outer}"].nunique().max() == 1
    report = json.loads((folder / "calibration_review.json").read_text())
    assert report["old_holdout_evaluated"] is False
    assert report["published_model_replaced"] is False
    assert report["label_semantics_confirmed"] is False
    for model in ["raw", "sigmoid", "prior"]:
        scores = predictions[model].to_numpy()
        assert np.isfinite(scores).all() and ((0 <= scores) & (scores <= 1)).all()
        actual = review.probability_metrics(predictions.label_value.to_numpy(), scores)
        for name in ["brier", "log_loss", "average_precision", "mean_probability"]:
            assert actual[name] == pytest.approx(report["metrics"]["all"][model][name])


def test_error_slices_account_for_every_development_row():
    folder = mg.ROOT / "outputs/calibration-review-20260922"
    if not folder.exists():
        pytest.skip("Run calibration review before artifact-level checks")
    errors = pd.read_csv(folder / "conditional_errors.csv")
    for _, part in errors.groupby("condition"):
        assert part.rows.sum() == 1913
        assert part.positive_rows.sum() == 33
        assert (part[["tp", "fn", "fp", "tn"]].sum(axis=1) == part.rows).all()
        assert (part.tp + part.fn == part.positive_rows).all()
