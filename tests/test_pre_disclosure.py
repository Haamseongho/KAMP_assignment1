"""Safety checks that do not generate or use surrogate competition data."""

from argparse import Namespace
from pathlib import Path

import pytest

import kamp_prep


ROOT = Path(__file__).resolve().parents[1]


def test_unpublished_task_fields_remain_unset():
    config = kamp_prep.load_config(ROOT / "config" / "task.template.json")
    assert config["source_file"] is None
    assert config["target_column"] is None
    assert config["task_type"] is None
    assert config["metric"] is None
    assert config["split"]["strategy"] is None
    assert config["submission"]["expected_columns"] is None
    assert config["submission"]["prediction_columns"] is None


def test_baseline_refuses_unpublished_task(capsys):
    with pytest.raises(kamp_prep.PrepError, match="source_file"):
        kamp_prep.baseline(Namespace(config=ROOT / "config" / "task.template.json",
                                     allow_large=False))
    assert "baseline-" not in capsys.readouterr().out


def test_empty_intake_inventory_is_valid(tmp_path, capsys):
    kamp_prep.inventory(Namespace(directory=tmp_path, report=None))
    assert '"file_count": 0' in capsys.readouterr().out


def test_reports_are_not_overwritten(tmp_path):
    report = tmp_path / "report.json"
    kamp_prep.save_json(report, {"status": "first"})
    with pytest.raises(kamp_prep.PrepError, match="덮어쓰지"):
        kamp_prep.save_json(report, {"status": "second"})
