"""TEST_ONLY fixtures check group-event invariants, not manufacturing performance."""

import numpy as np
import pandas as pd
import pytest

import moldguard as mg
from research_split import FEATURES
from task01_group_event_experiment import (
    event_matrix,
    fold_scores,
    policy_row,
    rg3_event_training,
)


def test_rg3_event_target_uses_training_groups_only_and_excludes_ids():
    # These invented rows test data plumbing only; they are not a KAMP dataset.
    rows = []
    for group, labels, value in (
        ("TEST_ONLY:fit_positive", (0, 1), 1.0),
        ("TEST_ONLY:fit_negative", (0, 0), 2.0),
        ("TEST_ONLY:validation", (1, 1), 3.0),
    ):
        for label in labels:
            rows.append({"machine": "rg3", "feature_group": group,
                         mg.ID_COL: len(rows) + 100, mg.TARGET: label,
                         **{feature: value for feature in FEATURES}})
    frame = pd.DataFrame(rows)
    training = frame.loc[frame.feature_group.ne("TEST_ONLY:validation")]
    event = rg3_event_training(training)

    assert event.feature_group.is_unique
    assert dict(zip(event.feature_group, event[mg.TARGET])) == {
        "TEST_ONLY:fit_positive": 1, "TEST_ONLY:fit_negative": 0}
    assert "TEST_ONLY:validation" not in set(event.feature_group)
    x = event_matrix(event)
    assert len(x) == 2
    assert not {mg.ID_COL, mg.TARGET, "feature_group", "outer_fold"} & set(x.columns)
    changed_ids = event.copy()
    changed_ids[mg.ID_COL] += 10_000
    pd.testing.assert_frame_equal(x, event_matrix(changed_ids))


def test_rg3_event_rejects_mismatched_features_within_a_group():
    rows = []
    for group, labels, value in (
        ("TEST_ONLY:conflict", (0, 1), 1.0),
        ("TEST_ONLY:negative", (0, 0), 2.0),
    ):
        for label in labels:
            rows.append({"machine": "rg3", "feature_group": group,
                         mg.ID_COL: len(rows), mg.TARGET: label,
                         **{feature: value for feature in FEATURES}})
    frame = pd.DataFrame(rows)
    frame.loc[1, FEATURES[0]] += 0.5
    with pytest.raises(ValueError, match="different measurements"):
        rg3_event_training(frame)


def test_fold_scores_rejects_exact_group_overlap_before_model_fit():
    frame = pd.DataFrame([{"machine": "rg3", "feature_group": "TEST_ONLY:shared",
                           mg.ID_COL: i, mg.TARGET: i,
                           **{feature: 1.0 for feature in FEATURES}}
                          for i in (0, 1)])
    with pytest.raises(ValueError, match="crosses a fit/validation boundary"):
        fold_scores(frame.iloc[[0]], frame.iloc[[1]], seed=20260922,
                    deadline=float("inf"))


def test_fixed_k192_accuracy_requires_seventeen_positives():
    # Exact cohort sizes exercise arithmetic and selection; scores are TEST_ONLY.
    machines = np.array(["cn7"] * 967 + ["rg3"] * 946)
    ids = np.r_[np.arange(967), np.arange(946)]
    score = np.r_[np.linspace(1, 0, 967), np.linspace(1, 0, 946)]
    y = np.zeros(1913, dtype=int)
    y[np.r_[np.arange(11), np.arange(100, 102),
             967 + np.arange(6), 967 + np.arange(100, 114)]] = 1
    base = pd.DataFrame({"machine": machines, "source_row_id": ids,
                         "feature_group": [f"TEST_ONLY:{i}" for i in range(1913)],
                         "label_value": y})
    seventeen = policy_row(base, score, "TEST_ONLY", 0)
    assert (seventeen["k"], seventeen["tp"], seventeen["fp"], seventeen["fn"], seventeen["tn"]) == (
        192, 17, 175, 16, 1705)
    assert seventeen["policy_derived_accuracy"] == pytest.approx(1722 / 1913)
    assert seventeen["policy_derived_accuracy"] >= 0.90

    base.loc[967 + 5, "label_value"] = 0
    base.loc[967 + 200, "label_value"] = 1
    sixteen = policy_row(base, score, "TEST_ONLY", 0)
    assert sixteen["tp"] == 16
    assert sixteen["policy_derived_accuracy"] == pytest.approx(1720 / 1913)
    assert sixteen["policy_derived_accuracy"] < 0.90
