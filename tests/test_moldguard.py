from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import moldguard as mg


def test_identical_measurements_never_cross_frozen_split():
    if mg.DATA_DIR is None:
        pytest.skip("Raw contest files are not installed")
    data, features, _ = mg.load_data(mg.DATA_DIR, True)
    data = mg.with_groups(data, features)
    development, holdout = mg.split_groups(data)
    assert not set(development) & set(holdout)
    assert data.feature_group.isin(development).sum() + data.feature_group.isin(holdout).sum() == len(data)
    assert mg.ID_COL not in mg.feature_matrix(data, features)
    assert mg.TARGET not in mg.feature_matrix(data, features)


def test_audit_finds_ambiguous_duplicates(tmp_path: Path):
    if mg.DATA_DIR is None:
        pytest.skip("Raw contest files are not installed")
    result = mg.audit(mg.DATA_DIR, tmp_path / "audit.json")
    assert result["positive_rows"] == 42
    assert result["ambiguous_groups"] == 36
    assert result["irreducible_row_errors_for_identical_features"] == 36
    assert result["unlabeled_rows"] == 71180


def test_per_machine_priority_and_ood_fallback(tmp_path: Path):
    data = pd.DataFrame({"machine": ["cn7"] * 19 + ["rg3"] * 21, mg.ID_COL: range(40)})
    scores = np.linspace(0.99, 0.01, len(data))
    ood = np.zeros(len(data), dtype=int)
    ood[0] = 3
    result = mg.priority_frame(data, scores, ood)
    assert result.groupby("machine").priority.apply(lambda s: s.eq("HIGH").sum()).to_dict() == {"cn7": 2, "rg3": 3}
    assert result.loc[result.ood_flag, "action"].tolist() == ["usual_inspection_ood"]
    assert result.loc[result.machine.eq("rg3"), "action"].eq("usual_inspection_unvalidated_machine").all()
    assert result.loc[result.machine.eq("cn7") & ~result.ood_flag, "action"].eq(
        "usual_inspection_label_unconfirmed"
    ).all()
    path = tmp_path / "priority.csv"
    result.to_csv(path, index=False)
    assert mg.check_priority(path, len(data), data)["ood_fallback"] == 1


def test_output_validation_rejects_duplicate_identifiers(tmp_path: Path):
    data = pd.DataFrame({"machine": ["cn7"] * 10, mg.ID_COL: range(10)})
    result = mg.priority_frame(data, np.linspace(1, 0, 10), np.zeros(10, dtype=int))
    result.loc[1, "source_row_id"] = result.loc[0, "source_row_id"]
    path = tmp_path / "bad.csv"
    result.to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        mg.check_priority(path, 10, data)


def test_output_validation_rejects_missing_ood_fallback(tmp_path: Path):
    data = pd.DataFrame({"machine": ["cn7"] * 10, mg.ID_COL: range(10)})
    result = mg.priority_frame(data, np.linspace(1, 0, 10), np.array([3] + [0] * 9))
    result.loc[0, "action"] = "inspect_first"
    path = tmp_path / "bad.csv"
    result.to_csv(path, index=False)
    with pytest.raises(ValueError, match="OOD, label, or machine fallback"):
        mg.check_priority(path, 10, data)


def test_output_validation_rejects_unvalidated_machine_action(tmp_path: Path):
    data = pd.DataFrame({"machine": ["rg3"] * 10, mg.ID_COL: range(10)})
    result = mg.priority_frame(data, np.linspace(1, 0, 10), np.zeros(10, dtype=int))
    result.loc[0, "action"] = "inspect_first"
    path = tmp_path / "bad.csv"
    result.to_csv(path, index=False)
    with pytest.raises(ValueError, match="fallback/priority action"):
        mg.check_priority(path, 10, data)


def test_output_validation_rejects_action_before_label_confirmation(tmp_path: Path):
    data = pd.DataFrame({"machine": ["cn7"] * 10, mg.ID_COL: range(10)})
    result = mg.priority_frame(data, np.linspace(1, 0, 10), np.zeros(10, dtype=int))
    result.loc[0, "action"] = "inspect_first"
    path = tmp_path / "bad.csv"
    result.to_csv(path, index=False)
    with pytest.raises(ValueError, match="label"):
        mg.check_priority(path, 10, data)
