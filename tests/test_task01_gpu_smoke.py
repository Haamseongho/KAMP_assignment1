"""Failed or timed-out workers must never look like completed GPU checks."""

import json
import subprocess
from types import SimpleNamespace

import pytest

import task01_gpu_smoke as smoke


def test_smoke_timeout_is_failed_receipt(tmp_path, monkeypatch):
    output = tmp_path / "timeout"
    monkeypatch.setattr(smoke, "run_worker", lambda *a, **k: (_ for _ in ()).throw(
        subprocess.TimeoutExpired("synthetic worker", 1, output=b"partial fit log\n")))
    monkeypatch.setattr("research_runtime.git_info", lambda: {})
    with pytest.raises(RuntimeError, match="training process stopped"):
        smoke.run(output, 1, True)
    record = json.loads((output / "execution.json").read_text())
    assert record["status"] == "failed"
    assert not (output / "smoke.json").exists()
    assert "partial fit log" in (output / "run.log").read_text()


def test_failed_worker_cannot_pass(tmp_path, monkeypatch):
    output = tmp_path / "failed"
    monkeypatch.setattr(smoke, "run_worker", lambda *a, **k: SimpleNamespace(
        returncode=1, stdout="", stderr="GPU backend unavailable\n"))
    monkeypatch.setattr("research_runtime.git_info", lambda: {})
    with pytest.raises(RuntimeError, match="worker failed"):
        smoke.run(output, 120, True)
    assert json.loads((output / "execution.json").read_text())["status"] == "failed"


def test_smoke_requires_acknowledgement_before_creating_output(tmp_path):
    output = tmp_path / "unacknowledged"
    with pytest.raises(ValueError, match="local-gpu-ack"):
        smoke.run(output, 120, False)
    assert not output.exists()
