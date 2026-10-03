"""Audit Task 01 GPU OOF scores at the frozen per-machine 10% inspection budget.

This reads completed comparison runs; it does not fit a model or infer that
numeric label 1 means a physical defect. All outputs are aggregate only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import calibration_diagnostics as cd
import moldguard as mg
import research_verify
from research_runtime import digest, execution_record, write_json
from upgrade_audit import metrics, tie_rows


IDENTITY = ["machine", "source_row_id", "label_value", "feature_group", "outer_fold"]
REQUIRED = set(IDENTITY) | {"model", "device", "seed", "score"}
MACHINES = ("cn7", "rg3")
FRACTION = 0.1
FROZEN_SPLIT_SHA256 = "e26a82c58d0e86952b8557c2e8542f816f64db8ed49d15a4a450ef7a86b5efc7"
FROZEN_LABELED_SHA256 = {
    "moldset_labeled_cn7.csv": "f870b0c259297f20d503f2e2739966553e5db5ef1deb0cc7513eefa4bbfa2753",
    "moldset_labeled_rg3.csv": "14aab21476eea02c823ccb00f82800040280d805dbce5a3eb7c11b176a9e5aa4",
}
GPU_MODELS = ("xgb_shallow", "xgb_regularized", "catboost_ref", "catboost_regularized")


def expected_identity(data: pd.DataFrame) -> pd.DataFrame:
    return data.rename(columns={mg.ID_COL: "source_row_id", mg.TARGET: "label_value"})[IDENTITY].copy()


def validate_frozen_contract(data: pd.DataFrame, sources: dict, split_sha256: str) -> None:
    if split_sha256 != FROZEN_SPLIT_SHA256:
        raise ValueError("Split SHA-256 differs from the frozen Task 01 development split")
    if {name: item.get("sha256") for name, item in sources.items()} != FROZEN_LABELED_SHA256:
        raise ValueError("Official labeled CSV SHA-256 differs from the frozen Task 01 sources")
    counts = data.groupby("machine")[mg.TARGET].agg(["size", "sum"])
    if (len(data) != 1913 or int(data[mg.TARGET].sum()) != 33
            or data.feature_group.nunique() != 957
            or counts.to_dict("index") != {"cn7": {"size": 967, "sum": 13},
                                           "rg3": {"size": 946, "sum": 20}}
            or set(data.outer_fold) != set(range(5))
            or data.groupby("feature_group").outer_fold.nunique().max() != 1):
        raise ValueError("Development rows, labels, groups or folds differ from the frozen cohort")


def validate_oof(predictions: pd.DataFrame, expected: pd.DataFrame,
                 record: dict, device: str) -> dict[tuple[str, int], np.ndarray]:
    """Return scores aligned to the official frozen development row order."""
    if not REQUIRED.issubset(predictions):
        raise ValueError(f"OOF columns missing: {sorted(REQUIRED - set(predictions))}")
    if record.get("device") != device or not predictions.device.eq(device).all():
        raise ValueError("OOF device differs from its completed run manifest")
    models, seeds = record.get("models"), record.get("seeds")
    if (not isinstance(models, list) or not models or len(models) != len(set(models))
            or not all(isinstance(m, str) and m for m in models)
            or not isinstance(seeds, list) or not seeds or len(seeds) != len(set(seeds))
            or not all(isinstance(s, int) for s in seeds)):
        raise ValueError("Invalid model/seed declaration in run manifest")
    if (predictions.duplicated(["model", "seed", "machine", "source_row_id"]).any()
            or len(predictions) != len(expected) * len(models) * len(seeds)):
        raise ValueError("Duplicate or incomplete OOF row coverage")
    pairs = set(zip(predictions.model, predictions.seed))
    if pairs != {(model, seed) for model in models for seed in seeds}:
        raise ValueError("OOF model/seed coverage differs from run manifest")
    if expected.duplicated(["machine", "source_row_id"]).any():
        raise ValueError("Official development identity is not unique")
    numeric = pd.to_numeric(predictions.score, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or np.any((numeric < 0) | (numeric > 1)):
        raise ValueError("OOF scores must be finite numbers in [0, 1]")
    predictions = predictions.copy()
    predictions["score"] = numeric
    keys = ["machine", "source_row_id"]
    expected_sorted = expected[IDENTITY].sort_values(keys).reset_index(drop=True)
    scores = {}
    for model in models:
        for seed in seeds:
            part = predictions.loc[predictions.model.eq(model) & predictions.seed.eq(seed)]
            if len(part) != len(expected):
                raise ValueError(f"Incomplete OOF coverage: {model}/{seed}")
            try:
                pd.testing.assert_frame_equal(part[IDENTITY].sort_values(keys).reset_index(drop=True),
                                              expected_sorted, check_dtype=False)
            except AssertionError as error:
                raise ValueError(f"OOF row key, label, group or fold mismatch: {model}/{seed}") from error
            aligned = expected[keys].merge(part[keys + ["score"]], on=keys, validate="one_to_one", sort=False)
            scores[model, seed] = aligned.score.to_numpy(dtype=float)
    return scores


def policy_summaries(base: pd.DataFrame, scores: dict[tuple[str, int], np.ndarray],
                     device: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score each model/seed once over pooled OOF, with fixed CN7/RG3 budgets."""
    rows, tie_details = [], []
    for (model, seed), values in scores.items():
        machine_ties = {}
        boundaries = tie_rows(base, values, model, seed)
        for machine in MACHINES:
            tie = next(row for row in boundaries
                       if row["machine"] == machine and row["fraction"] == FRACTION)
            machine_ties[machine] = tie
            tie_details.append({"device": device, **tie})
        for machine in ("all", *MACHINES):
            mask = np.ones(len(base), dtype=bool) if machine == "all" else base.machine.eq(machine).to_numpy()
            part, part_scores = base.loc[mask], values[mask]
            observed = metrics(part, part_scores, FRACTION, "P_MACHINE")
            tp, k, positive, n = observed["found"], observed["k"], observed["positives"], observed["rows"]
            fp, fn = k - tp, positive - tp
            tn = n - positive - fp
            relevant = list(machine_ties.values()) if machine == "all" else [machine_ties[machine]]
            minimum = sum(item["minimum_found"] for item in relevant)
            maximum = sum(item["maximum_found"] for item in relevant)
            rows.append({"device": device, "model": model, "seed": seed, "machine": machine,
                         "policy": "P_MACHINE", "fraction": FRACTION, **observed,
                         "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                         "f1_at_k": 2 * tp / (k + positive) if k + positive else None,
                         "policy_derived_accuracy": (tp + tn) / n if n else None,
                         "tie_min_found": minimum,
                         "tie_expected_found": sum(item["expected_found"] for item in relevant),
                         "tie_max_found": maximum,
                         "tie_sensitive": minimum != maximum})
    return pd.DataFrame(rows), pd.DataFrame(tie_details)


def compare_to_cpu(gpu: pd.DataFrame, cpu: pd.DataFrame) -> pd.DataFrame:
    baseline = cpu.loc[cpu.model.eq("logistic_ref")]
    if len(baseline) != 3 * gpu.seed.nunique():
        raise ValueError("CPU logistic_ref must cover all GPU seeds and three scopes")
    if set(baseline.seed) != set(gpu.seed):
        raise ValueError("CPU and GPU seeds differ")
    baseline = baseline.set_index(["seed", "machine"])
    rows = []
    measures = ("tp", "fp", "fn", "recall", "precision", "f1_at_k",
                "policy_derived_accuracy", "average_precision", "roc_auc")
    for row in gpu.itertuples(index=False):
        reference = baseline.loc[row.seed, row.machine]
        if (row.rows, row.groups, row.positives, row.k) != (
                reference.rows, reference.groups, reference.positives, reference.k):
            raise ValueError("CPU and GPU evaluation cohorts or budgets differ")
        item = {"model": row.model, "seed": row.seed, "machine": row.machine,
                "baseline_model": "logistic_ref", "policy": "P_MACHINE", "k": row.k}
        for name in measures:
            item[f"cpu_{name}"] = getattr(reference, name)
            item[f"gpu_{name}"] = getattr(row, name)
            item[f"delta_{name}"] = getattr(row, name) - getattr(reference, name)
        rows.append(item)
    return pd.DataFrame(rows)


def validate_gpu_evidence(record: dict, directory: Path) -> dict:
    """Check the input run's authorization and recorded fitted backends."""
    execution = directory / "execution.json"
    if (not execution.is_file()
            or json.loads(execution.read_text(encoding="utf-8")).get("status") != "passed"
            or not record.get("gpu_runtime")):
        raise ValueError("GPU execution or runtime receipt is incomplete")
    authorization = record.get("gpu_authorization") or {}
    mode = authorization.get("mode")
    local = mode == "user_controlled_local_gpu"
    legacy = mode == "legacy_task04_receipt" or (not mode and bool(record.get("gpu_receipt")))
    if local:
        if authorization.get("cost_and_power_acknowledged") is not True or record.get("gpu_receipt") is not None:
            raise ValueError("Local GPU acknowledgement is incomplete")
    elif legacy:
        if not record.get("gpu_receipt"):
            raise ValueError("Legacy task-04 recovery receipt is incomplete")
    else:
        raise ValueError("GPU authorization route is missing or unsupported")
    checks = record.get("gpu_backend_checks")
    if not isinstance(checks, list):
        raise ValueError("GPU fitted-backend checks are missing")
    expected = {(model, seed, fold) for model in GPU_MODELS
                for seed in (20260922, 20260923, 20260924) for fold in range(5)}
    observed = [(item.get("model"), item.get("seed"), item.get("fold")) for item in checks
                if isinstance(item, dict)]
    if (record.get("models") != list(GPU_MODELS)
            or len(checks) != 60 or len(observed) != 60
            or len(set(observed)) != 60 or set(observed) != expected):
        raise ValueError("GPU fitted-backend checks do not cover all 60 frozen model/seed/fold fits")
    for item in checks:
        backend = item.get("configured_backend")
        valid = (isinstance(backend, str) and backend.startswith("cuda")) if item["model"].startswith("xgb_") else backend == "GPU"
        if not valid:
            raise ValueError("A fitted model recorded a non-GPU backend")
    return {"authorization_mode": mode or "legacy_task04_receipt_pre_authorization_schema",
            "fitted_backend_checks": len(checks),
            "fitted_backend_coverage": "all_60_verified"}


def load_completed_run(directory: Path, data_dir: Path, split_dir: Path,
                       expected: pd.DataFrame, device: str) -> tuple[dict, dict, pd.DataFrame, pd.DataFrame]:
    verification = research_verify.verify(data_dir, split_dir, directory)
    record = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
    if record.get("seeds") != [20260922, 20260923, 20260924]:
        raise ValueError("Completed run does not cover the three frozen Task 01 seeds")
    if device == "gpu":
        validate_gpu_evidence(record, directory)
    predictions = pd.read_csv(directory / "oof_predictions.csv")
    scores = validate_oof(predictions, expected, record, device)
    summary, ties = policy_summaries(expected, scores, device)
    return verification, record, summary, ties


def run(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    for source in (args.gpu_dir, args.cpu_dir):
        if source is not None and (output == source.resolve() or source.resolve() in output.parents):
            raise ValueError("Evaluation output must be separate from input runs")
    with execution_record(output) as receipt:
        data, _, sources = cd.load_development(args.data_dir, args.split_dir)
        validate_frozen_contract(data, sources, digest(args.split_dir / "split_manifest.csv"))
        expected = expected_identity(data)
        gpu_check, gpu_record, gpu_summary, gpu_ties = load_completed_run(
            args.gpu_dir, args.data_dir, args.split_dir, expected, "gpu")
        if args.cpu_dir is not None:
            cpu_check, cpu_record, cpu_summary, cpu_ties = load_completed_run(
                args.cpu_dir, args.data_dir, args.split_dir, expected, "cpu")
            if set(gpu_record["seeds"]) != set(cpu_record["seeds"]):
                raise ValueError("CPU and GPU runs use different seeds")
            comparison = compare_to_cpu(gpu_summary, cpu_summary)
            summary = pd.concat([cpu_summary.loc[cpu_summary.model.eq("logistic_ref")], gpu_summary],
                                ignore_index=True)
            ties = pd.concat([cpu_ties.loc[cpu_ties.model.eq("logistic_ref")], gpu_ties],
                             ignore_index=True)
        else:
            cpu_check, cpu_record, comparison = None, None, None
            summary, ties = gpu_summary, gpu_ties
        summary.to_csv(output / "policy_summary.csv", index=False)
        ties.to_csv(output / "tie_audit.csv", index=False)
        if comparison is not None:
            comparison.to_csv(output / "comparison_vs_cpu.csv", index=False)
        all_gpu = gpu_summary.loc[gpu_summary.machine.eq("all")]
        observations = {}
        for model, part in all_gpu.groupby("model", sort=False):
            found = {str(row.seed): int(row.tp) for row in part.itertuples(index=False)}
            observations[model] = {
                "found_by_seed": found,
                "primary_seed_found": found.get("20260922"),
                "all_seeds_at_least_17_of_33": bool(part.tp.ge(17).all())
                if part.positives.eq(33).all() and part.k.eq(192).all() else None,
                "interpretation": "observed_development_oof_only_no_posthoc_model_adoption",
            }
        result = {
            "status": "development_oof_aggregate_evaluation_complete",
            "scope": "same_frozen_development_split_pooled_oof_no_independent_validation",
            "policy": "P_MACHINE top ceil(10%) per machine, CN7 97 + RG3 95 = K192 for the frozen cohort",
            "target": "17/33 numeric label-1 rows at K192 yields >=90% policy-derived accuracy",
            "split_sha256": digest(args.split_dir / "split_manifest.csv"),
            "gpu_input_manifest_sha256": digest(args.gpu_dir / "run_manifest.json"),
            "gpu_oof_sha256": digest(args.gpu_dir / "oof_predictions.csv"),
            "gpu_artifact_verification": gpu_check["status"],
            "cpu_input_manifest_sha256": digest(args.cpu_dir / "run_manifest.json") if args.cpu_dir else None,
            "cpu_oof_sha256": digest(args.cpu_dir / "oof_predictions.csv") if args.cpu_dir else None,
            "cpu_artifact_verification": cpu_check["status"] if cpu_check else None,
            "gpu_manifest_device_claim": gpu_record["device"],
            "gpu_execution_evidence": validate_gpu_evidence(gpu_record, args.gpu_dir),
            "gpu_hardware_receipt": "see input GPU run_manifest.json; this evaluator does not attest hardware use",
            "gpu_model_observations": observations,
            "comparison_with_cpu_available": comparison is not None,
            "label_1_physical_meaning": "unconfirmed",
            "policy_accuracy_and_f1_warning": "inspection selection treated as label 1; neither is official threshold F1 or field accuracy",
            "selection_warning": "comparing completed outer OOF models is post-hoc research, not nested selection or independent testing",
            "future_validation": "not_available_from_these_outputs",
            "row_level_output_written": False,
        }
        write_json(output / "decision.json", result)
        receipt["scope"] = result["scope"]
        receipt["aggregate_only"] = True
        print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--gpu-dir", type=Path, required=True)
    parser.add_argument("--cpu-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
