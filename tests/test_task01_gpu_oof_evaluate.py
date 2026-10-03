"""Frozen identity, policy arithmetic and aggregate-only GPU OOF handoff checks."""

import argparse
import json

import numpy as np
import pandas as pd
import pytest

import task01_gpu_oof_evaluate as evaluate


def sample_cohort():
    rows = []
    for machine in ("cn7", "rg3"):
        for row_id in range(1, 11):
            rows.append({"machine": machine, "source_row_id": row_id,
                         "label_value": int(row_id <= 2),
                         "feature_group": f"{machine}-{row_id}",
                         "outer_fold": (row_id - 1) % 5})
    return pd.DataFrame(rows)


def sample_oof(base, model="xgb_shallow", device="gpu", seeds=(20260922,)):
    rows = []
    for seed in seeds:
        frame = base.copy()
        frame["model"], frame["device"], frame["seed"] = model, device, seed
        frame["score"] = np.where(frame.source_row_id.eq(1), 0.9, 0.1)
        rows.append(frame)
    return pd.concat(rows, ignore_index=True).sample(frac=1, random_state=4).reset_index(drop=True)


def test_oof_alignment_and_same_budget_cpu_comparison():
    base = sample_cohort()
    gpu = sample_oof(base)
    rg3_negative = gpu.machine.eq("rg3") & gpu.source_row_id.eq(3)
    gpu.loc[rg3_negative, "score"] = 0.95
    gpu_scores = evaluate.validate_oof(gpu, base, {"device": "gpu", "models": ["xgb_shallow"],
                                                     "seeds": [20260922]}, "gpu")
    cpu_scores = evaluate.validate_oof(sample_oof(base, "logistic_ref", "cpu"), base,
                                       {"device": "cpu", "models": ["logistic_ref"],
                                        "seeds": [20260922]}, "cpu")
    gpu_summary, _ = evaluate.policy_summaries(base, gpu_scores, "gpu")
    cpu_summary, _ = evaluate.policy_summaries(base, cpu_scores, "cpu")
    overall = gpu_summary.loc[gpu_summary.machine.eq("all")].iloc[0]
    assert (overall.rows, overall.positives, overall.k) == (20, 4, 2)
    assert (overall.tp, overall.fp, overall.fn, overall.tn) == (1, 1, 3, 15)
    assert overall.recall == 0.25
    assert overall.precision == 0.5
    assert overall.f1_at_k == pytest.approx(1 / 3)
    assert overall.policy_derived_accuracy == 0.8
    delta = evaluate.compare_to_cpu(gpu_summary, cpu_summary)
    total_delta = delta.loc[delta.machine.eq("all")].iloc[0]
    assert total_delta.cpu_tp == 2
    assert total_delta.delta_tp == -1
    assert total_delta.delta_recall == -0.25


def test_tie_bounds_are_per_machine_then_summed_for_k_policy():
    base = sample_cohort()
    predictions = sample_oof(base)
    predictions.loc[predictions.machine.eq("rg3") & predictions.source_row_id.eq(3), "score"] = 0.9
    scores = evaluate.validate_oof(predictions, base, {"device": "gpu", "models": ["xgb_shallow"],
                                                       "seeds": [20260922]}, "gpu")
    summary, ties = evaluate.policy_summaries(base, scores, "gpu")
    rg3 = summary.loc[summary.machine.eq("rg3")].iloc[0]
    overall = summary.loc[summary.machine.eq("all")].iloc[0]
    assert (rg3.tp, rg3.tie_min_found, rg3.tie_max_found) == (1, 0, 1)
    assert (overall.tp, overall.tie_min_found, overall.tie_max_found) == (2, 1, 2)
    assert bool(overall.tie_sensitive)
    assert len(ties) == 2


@pytest.mark.parametrize("column,value,message", [
    ("label_value", 0, "label"),
    ("outer_fold", 4, "fold"),
    ("feature_group", "wrong-group", "group"),
    ("score", np.nan, "finite"),
    ("score", 1.1, "finite"),
    ("device", "cpu", "device"),
])
def test_oof_rejects_tampering(column, value, message):
    base = sample_cohort()
    oof = sample_oof(base)
    match = oof.machine.eq("cn7") & oof.source_row_id.eq(1)
    oof.loc[match, column] = value
    with pytest.raises(ValueError, match=message):
        evaluate.validate_oof(oof, base, {"device": "gpu", "models": ["xgb_shallow"],
                                               "seeds": [20260922]}, "gpu")


def test_oof_rejects_duplicate_and_missing_coverage():
    base = sample_cohort()
    oof = sample_oof(base)
    duplicate = oof.copy()
    duplicate.iloc[-1] = duplicate.iloc[0]
    for broken in (duplicate, oof.iloc[:-1]):
        with pytest.raises(ValueError, match="coverage"):
            evaluate.validate_oof(broken, base, {"device": "gpu", "models": ["xgb_shallow"],
                                                    "seeds": [20260922]}, "gpu")


def gpu_manifest():
    checks = [{"model": model, "seed": seed, "fold": fold,
               "configured_backend": "cuda:0" if model.startswith("xgb_") else "GPU"}
              for model in evaluate.GPU_MODELS
              for seed in (20260922, 20260923, 20260924) for fold in range(5)]
    return {"device": "gpu", "models": list(evaluate.GPU_MODELS),
            "seeds": [20260922, 20260923, 20260924],
            "gpu_runtime": {"nvidia_smi": "test"},
            "gpu_authorization": {"mode": "user_controlled_local_gpu",
                                  "cost_and_power_acknowledged": True},
            "gpu_receipt": None, "gpu_backend_checks": checks}


def test_gpu_evidence_requires_all_fitted_backends_and_accepts_both_authorizations(tmp_path):
    (tmp_path / "execution.json").write_text(json.dumps({"status": "passed"}))
    manifest = gpu_manifest()
    assert evaluate.validate_gpu_evidence(manifest, tmp_path)["fitted_backend_checks"] == 60
    legacy = {**manifest, "gpu_authorization": {"mode": "legacy_task04_receipt"},
              "gpu_receipt": {"sha256": "test"}}
    assert evaluate.validate_gpu_evidence(legacy, tmp_path)["authorization_mode"] == "legacy_task04_receipt"
    short = {**manifest, "gpu_backend_checks": manifest["gpu_backend_checks"][:-1]}
    with pytest.raises(ValueError, match="60 frozen"):
        evaluate.validate_gpu_evidence(short, tmp_path)
    fallback = {**manifest, "gpu_backend_checks": [dict(item) for item in manifest["gpu_backend_checks"]]}
    fallback["gpu_backend_checks"][0]["configured_backend"] = "cpu"
    with pytest.raises(ValueError, match="non-GPU"):
        evaluate.validate_gpu_evidence(fallback, tmp_path)
    wrong_models = {**manifest, "models": ["xgb_shallow"]}
    with pytest.raises(ValueError, match="60 frozen"):
        evaluate.validate_gpu_evidence(wrong_models, tmp_path)


def test_cli_writes_only_aggregate_artifacts(tmp_path, monkeypatch):
    base = sample_cohort()
    data = base.rename(columns={"source_row_id": "Unnamed: 0", "label_value": "PassOrFail"})
    sources = {"moldset_labeled_cn7.csv": {"sha256": "test-cn7"},
               "moldset_labeled_rg3.csv": {"sha256": "test-rg3"}}
    monkeypatch.setattr(evaluate.cd, "load_development", lambda *_: (data, [], sources))
    monkeypatch.setattr(evaluate, "validate_frozen_contract", lambda *_: None)
    monkeypatch.setattr(evaluate.research_verify, "verify", lambda *_: {"status": "passed"})
    split = tmp_path / "split"
    split.mkdir()
    (split / "split_manifest.csv").write_text("test split\n")
    seeds = (20260922, 20260923, 20260924)
    for folder, device in (("gpu", "gpu"), ("cpu", "cpu")):
        location = tmp_path / folder
        location.mkdir()
        models = evaluate.GPU_MODELS if device == "gpu" else ("logistic_ref",)
        manifest = gpu_manifest() if device == "gpu" else {
            "device": "cpu", "models": list(models), "seeds": list(seeds)}
        (location / "run_manifest.json").write_text(json.dumps(manifest))
        (location / "execution.json").write_text(json.dumps({"status": "passed"}))
        pd.concat([sample_oof(base, model, device, seeds) for model in models],
                  ignore_index=True).to_csv(location / "oof_predictions.csv", index=False)
    output = tmp_path / "new_evaluation"
    evaluate.run(argparse.Namespace(data_dir=tmp_path, split_dir=split, gpu_dir=tmp_path / "gpu",
                                    cpu_dir=tmp_path / "cpu", output_dir=output))
    assert {p.name for p in output.iterdir()} == {
        "execution.json", "run.log", "policy_summary.csv", "tie_audit.csv",
        "comparison_vs_cpu.csv", "decision.json"}
    decision = json.loads((output / "decision.json").read_text())
    assert decision["row_level_output_written"] is False
    assert decision["comparison_with_cpu_available"] is True
    assert len(pd.read_csv(output / "policy_summary.csv")) == 45
    assert decision["gpu_execution_evidence"]["fitted_backend_checks"] == 60
