"""Development-only CPU test of RG3 group-event ranking on the frozen Task 01 split.

This experiment does not use the exposed historical holdout, change the operating
model, infer the physical meaning of label 1, or establish future performance.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import calibration_diagnostics as cd
import moldguard as mg
from model_upgrade import inner_folds, identity
from research_runtime import ROOT, digest, execution_record, write_json
from research_split import FEATURES
from upgrade_audit import comparison_rows, metrics, paired_bootstrap, select, tie_rows
from upgrade_models import fit_model, predict


SEEDS = (20260922, 20260923, 20260924)
BASELINE = {"kind": "logistic", "c": 0.1, "balanced": True}
NAMES = ("baseline", "rg3_group_logistic", "rg3_group_forest")
EVENT_SPECS = {
    "rg3_group_logistic": {"C": 0.1, "class_weight": "balanced", "max_iter": 2000},
    "rg3_group_forest": {"n_estimators": 200, "max_depth": 3,
                         "min_samples_leaf": 8, "max_features": "sqrt",
                         "class_weight": "balanced_subsample", "n_jobs": 1},
}


def check_deadline(deadline: float) -> None:
    if time.monotonic() > deadline:
        raise TimeoutError("Predeclared CPU experiment time limit reached")


def rg3_event_training(frame: pd.DataFrame) -> pd.DataFrame:
    """One RG3 training example per exact-feature group; no validation labels read."""
    rg3 = frame.loc[frame.machine.eq("rg3")]
    if rg3.empty:
        raise ValueError("RG3 training partition is empty")
    group = rg3.groupby("feature_group", sort=True)
    if group[list(FEATURES)].nunique(dropna=False).gt(1).any().any():
        raise ValueError("An exact-feature group contains different measurements")
    result = group.first().reset_index()
    result[mg.TARGET] = group[mg.TARGET].max().to_numpy(dtype=int)
    if set(result[mg.TARGET]) != {0, 1}:
        raise ValueError("RG3 group-event training needs both outcomes")
    if not result.feature_group.is_unique:
        raise ValueError("Duplicate group in event training table")
    return result


def event_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    # Reuse the frozen 23-feature model input; row IDs and group IDs never enter X.
    expected = set(FEATURES) | {"machine"}
    if not expected.issubset(frame) or not frame.machine.eq("rg3").all():
        raise ValueError("Invalid RG3 feature schema")
    x = mg.feature_matrix(frame, FEATURES)
    if not np.isfinite(x.to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite RG3 feature values")
    return x


def fit_event(frame: pd.DataFrame, name: str, seed: int):
    group = rg3_event_training(frame)
    x = event_matrix(group)
    y = group[mg.TARGET].to_numpy(dtype=int)
    if name == "rg3_group_logistic":
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=0.1, class_weight="balanced", max_iter=2000, random_state=seed))
    elif name == "rg3_group_forest":
        model = RandomForestClassifier(random_state=seed, **EVENT_SPECS[name])
    else:
        raise ValueError(f"Unknown event model: {name}")
    model.fit(x, y)
    if name == "rg3_group_logistic" and np.any(model[-1].n_iter_ >= model[-1].max_iter):
        raise RuntimeError("RG3 group-event logistic did not converge")
    return model


def fold_scores(training: pd.DataFrame, validation: pd.DataFrame,
                seed: int, deadline: float) -> dict[str, np.ndarray]:
    """Fit every transformation on this fit partition, then predict unseen groups."""
    if set(training.feature_group) & set(validation.feature_group):
        raise ValueError("Feature group crosses a fit/validation boundary")
    check_deadline(deadline)
    baseline = fit_model(training, BASELINE, seed)
    reference = predict(baseline, validation, BASELINE)
    result = {"baseline": reference}
    mask = validation.machine.eq("rg3").to_numpy()
    for name in ("rg3_group_logistic", "rg3_group_forest"):
        check_deadline(deadline)
        model = fit_event(training, name, seed)
        score = reference.copy()
        score[mask] = model.predict_proba(event_matrix(validation.loc[mask]))[:, 1]
        if not np.isfinite(score).all() or np.any((score < 0) | (score > 1)):
            raise ValueError(f"Invalid predictions from {name}")
        if not np.array_equal(score[~mask], reference[~mask]):
            raise ValueError("CN7 reference prediction changed")
        result[name] = score
    return result


def choose_on_inner(training: pd.DataFrame, seed: int, deadline: float):
    """Only the current outer-training rows may select the RG3 route."""
    scores = {name: np.full(len(training), np.nan) for name in NAMES}
    membership = np.full(len(training), -1, dtype=int)
    for inner, (fit, valid) in enumerate(inner_folds(training, seed)):
        membership[valid] = inner
        part = fold_scores(training.iloc[fit], training.iloc[valid], seed, deadline)
        for name in NAMES:
            scores[name][valid] = part[name]
    if np.any(membership < 0) or any(not np.isfinite(value).all() for value in scores.values()):
        raise ValueError("Incomplete inner out-of-fold prediction coverage")
    base = identity(training)
    rg3 = base.machine.eq("rg3").to_numpy()
    rows = [{"candidate": name, **metrics(base.loc[rg3], scores[name][rg3])} for name in NAMES]
    priority = {name: position for position, name in enumerate(NAMES)}
    winner = max(rows, key=lambda row: (
        row["found"], row["average_precision"], -priority[row["candidate"]]))["candidate"]
    for row in rows:
        row["selected"] = row["candidate"] == winner
    return winner, rows, membership


def policy_row(base: pd.DataFrame, scores: np.ndarray, model: str, seed: int) -> dict:
    observed = metrics(base, scores, 0.1, "P_MACHINE")
    tp, k, positive, n = observed["found"], observed["k"], observed["positives"], observed["rows"]
    fp, fn = k - tp, positive - tp
    tn = n - positive - fp
    return {"model": model, "seed": seed, "policy": "P_MACHINE", "fraction": 0.1,
            **observed, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "policy_derived_accuracy": (tp + tn) / n,
            "f1_at_k192": 2 * tp / (k + positive),
            "fixed_0_5_score_f1_diagnostic": float(f1_score(
                base.label_value.to_numpy(dtype=int), scores >= 0.5, zero_division=0)),
            "accuracy_interpretation": "selected_for_inspection_treated_as_label1_not_official_F1",
            "fixed_0_5_interpretation": "score_threshold_diagnostic_not_calibrated_probability_or_official_F1"}


def rg3_group_row(base: pd.DataFrame, scores: np.ndarray, model: str, seed: int) -> dict:
    chosen = select(base, scores, 0.1, "P_MACHINE")
    section = base.loc[base.machine.eq("rg3")].copy()
    section["selected"] = chosen[base.machine.eq("rg3").to_numpy()]
    section["score"] = scores[base.machine.eq("rg3").to_numpy()]
    groups = section.groupby("feature_group").agg(
        rows=("label_value", "size"), event=("label_value", "max"),
        selected_rows=("selected", "sum"), score=("score", "first"))
    spread = section.groupby("feature_group").score.agg(["min", "max"])
    maximum_spread = float((spread['max'] - spread['min']).max())
    # Windows BLAS can differ by one rounding bit for the same dot product.
    # This tolerance is far below a meaningful ranking difference.
    if maximum_spread > 1e-12:
        raise ValueError("The same RG3 feature group received different scores")
    positive = groups.event.eq(1)
    return {"model": model, "seed": seed, "groups": len(groups),
            "max_identical_input_score_spread": maximum_spread,
            "positive_groups": int(positive.sum()),
            "groups_with_any_selected_row": int(groups.selected_rows.gt(0).sum()),
            "positive_groups_with_any_selected_row": int((positive & groups.selected_rows.gt(0)).sum()),
            "positive_groups_with_all_rows_selected": int((positive & groups.selected_rows.eq(groups.rows)).sum()),
            "interpretation": "group_event_analysis_not_row_level_operational_sensitivity"}


def verify_historical_baseline(base: pd.DataFrame, scores: dict[int, dict[str, np.ndarray]]) -> dict:
    """Post-fit parity audit only; historical OOF never selects or trains a candidate."""
    path = ROOT / "outputs/task01_compare_20260923_cpu_v3/oof_predictions.csv"
    if not path.is_file():
        return {"status": "unavailable", "path": str(path), "used_for_selection": False}
    old = pd.read_csv(path)
    errors = {}
    for seed in SEEDS:
        historical = old.loc[old.model.eq("logistic_ref") & old.seed.eq(seed)]
        merged = base.merge(historical[["machine", "source_row_id", "label_value", "score"]],
                            on=["machine", "source_row_id", "label_value"],
                            how="left", validate="one_to_one", sort=False)
        if len(merged) != len(base) or merged.score.isna().any():
            raise ValueError("Historical baseline OOF row keys do not match")
        delta = float(np.max(np.abs(merged.score.to_numpy() - scores[seed]["baseline"])))
        if delta > 1e-12:
            raise ValueError(f"Historical baseline OOF parity failed for seed {seed}: {delta}")
        errors[str(seed)] = delta
    return {"status": "passed", "path": str(path), "sha256": digest(path),
            "max_absolute_error_by_seed": errors, "used_for_selection": False}


def run(args: argparse.Namespace) -> None:
    deadline = time.monotonic() + args.max_seconds
    with execution_record(args.output_dir) as receipt, threadpool_limits(limits=1):
        out = args.output_dir
        data, features, sources = cd.load_development(args.data_dir, args.split_dir)
        if features != list(FEATURES) or len(data) != 1913 or int(data[mg.TARGET].sum()) != 33:
            raise ValueError("Frozen development data differs from the predeclared cohort")
        if set(data.outer_fold) != set(range(5)):
            raise ValueError("Frozen outer folds 0..4 are required")
        if data.groupby("feature_group").outer_fold.nunique().max() != 1:
            raise ValueError("Exact-feature group crosses outer folds")
        if (len(data.loc[data.machine.eq("cn7")]) != 967
                or len(data.loc[data.machine.eq("rg3")]) != 946):
            raise ValueError("Frozen machine counts changed")
        plan = {
            "schema": "task01-group-event-plan-v1", "status": "frozen_before_first_model_fit",
            "scope": "local_cpu_development_only_no_exposed_holdout_or_remote_execution",
            "source_files": sources, "split_sha256": digest(args.split_dir / "split_manifest.csv"),
            "script_sha256": digest(Path(__file__)), "outer_folds": [0, 1, 2, 3, 4],
            "inner_folds": "group_stratified_min_3_fixed_seed_within_each_outer_train",
            "seeds": list(SEEDS), "baseline": BASELINE, "event_candidates": EVENT_SPECS,
            "group_event_definition": "RG3 outer/inner training groups only: max numeric label 1; one row per exact feature group",
            "feature_rule": "same 23 effective sensors plus constant RG3 machine bit; no row or group identifiers",
            "selection_rule": "inner RG3 found at machine top10pct; then RG3 AP; then baseline/logistic/forest order",
            "outer_policy": "P_MACHINE top ceil(10pct) per machine: CN7 97, RG3 95, total 192",
            "tie_rule": "score descending then machine then source_row_id; report min/expected/max boundary TP",
            "primary_seed": SEEDS[0], "numeric_target": "at least 17 of 33 label-1 rows at K192 = derived accuracy >=90pct",
            "auxiliary_f1": "F1@192 uses top-K selection; fixed 0.5 score F1 is diagnostic, neither is official F1",
            "score_semantics": "group-event score ranks risk of at least one label-1 row in a group; not calibrated row-label probability",
            "secondary_seeds": list(SEEDS[1:]), "bootstrap_repeats": args.bootstrap,
            "bootstrap_scope": "primary seed paired machine-stratified exact-feature-group resampling of fixed OOF",
            "max_seconds": args.max_seconds,
            "cautions": ["old 480-row holdout already exposed: not used", "historical OOF parity only after fitting",
                         "upstream scaler fit period UNKNOWN", "label-1 physical meaning UNKNOWN",
                         "future independent labels absent", "development OOF repeatedly inspected"],
            "adoption": False,
        }
        write_json(out / "experiment_plan.json", plan)
        print("Frozen plan written before the first model fit", flush=True)
        base = identity(data)
        predicted: dict[int, dict[str, np.ndarray]] = {}
        all_oof, comparisons, policy, ties, group_rows, choices, inner_rows, inner_members = [], [], [], [], [], [], [], []
        for seed in SEEDS:
            scores = {name: np.full(len(data), np.nan) for name in (*NAMES, "nested_route")}
            for outer in range(5):
                fit = data.outer_fold.ne(outer).to_numpy()
                valid = ~fit
                training = data.loc[fit].reset_index(drop=True)
                validation = data.loc[valid]
                if set(training.feature_group) & set(validation.feature_group):
                    raise ValueError("Outer group leakage")
                winner, candidate_rows, membership = choose_on_inner(training, seed, deadline)
                for row in candidate_rows:
                    inner_rows.append({"seed": seed, "outer_fold": outer, **row})
                member = identity(training)
                member["seed"], member["outer_test_fold"], member["inner_fold"] = seed, outer, membership
                inner_members.append(member)
                outer_scores = fold_scores(training, validation, seed, deadline)
                for name in NAMES:
                    scores[name][valid] = outer_scores[name]
                scores["nested_route"][valid] = outer_scores[winner]
                choices.append({"seed": seed, "outer_fold": outer, "winner": winner,
                                "train_rows": len(training), "validation_rows": len(validation),
                                "group_overlap": 0})
                print(f"seed={seed} outer={outer} selected={winner}; train={len(training)} valid={len(validation)}", flush=True)
            if any(not np.isfinite(values).all() for values in scores.values()):
                raise ValueError("Outer OOF predictions are incomplete")
            cn7 = data.machine.eq("cn7").to_numpy()
            for name in (*NAMES[1:], "nested_route"):
                if not np.array_equal(scores[name][cn7], scores["baseline"][cn7]):
                    raise ValueError("CN7 baseline preservation failed")
            predicted[seed] = scores
            for name, values in scores.items():
                row = base.copy()
                row["seed"], row["model"], row["score"] = seed, name, values
                all_oof.append(row)
                policy.append(policy_row(base, values, name, seed))
                ties.extend(tie_rows(base, values, name, seed))
                group_rows.append(rg3_group_row(base, values, name, seed))
            comparisons.extend(comparison_rows(base, scores, seed))
        parity = verify_historical_baseline(base, predicted)
        write_json(out / "historical_baseline_parity.json", parity)
        pd.concat(all_oof, ignore_index=True).to_csv(out / "oof_predictions.csv", index=False)
        pd.DataFrame(comparisons).to_csv(out / "model_comparison.csv", index=False)
        pd.DataFrame(policy).to_csv(out / "policy_summary.csv", index=False)
        pd.DataFrame(ties).to_csv(out / "topk_tie_audit.csv", index=False)
        pd.DataFrame(group_rows).to_csv(out / "rg3_group_event_summary.csv", index=False)
        pd.DataFrame(choices).to_csv(out / "selected_candidates.csv", index=False)
        pd.DataFrame(inner_rows).to_csv(out / "inner_candidate_metrics.csv", index=False)
        pd.concat(inner_members, ignore_index=True).to_csv(out / "inner_split_manifest.csv", index=False)
        print("Starting predeclared primary-seed paired group bootstrap", flush=True)
        primary = predicted[SEEDS[0]]
        uncertainty = paired_bootstrap(base, primary["baseline"], primary["nested_route"],
                                       args.bootstrap, SEEDS[0])
        pd.DataFrame(uncertainty).to_csv(out / "bootstrap_differences.csv", index=False)
        check_deadline(deadline)
        summary = pd.DataFrame(policy)
        baseline = summary.loc[summary.model.eq("baseline")].set_index("seed")
        nested = summary.loc[summary.model.eq("nested_route")].set_index("seed")
        selected = nested.loc[SEEDS[0]]
        primary_ties = pd.DataFrame(ties)
        boundary = primary_ties.loc[(primary_ties.seed.eq(SEEDS[0]))
                                    & primary_ties.model.eq("nested_route")
                                    & primary_ties.machine.isin(["cn7", "rg3"])
                                    & primary_ties.fraction.eq(0.1)]
        tie_min, tie_max = int(boundary.minimum_found.sum()), int(boundary.maximum_found.sum())
        result = {
            "research_status": "development_numeric_target_met_not_field_approved"
            if int(selected.tp) >= 17 else "development_numeric_target_not_met",
            "primary_seed": SEEDS[0], "primary_found": int(selected.tp),
            "primary_recall": float(selected.recall),
            "primary_policy_derived_accuracy": float(selected.policy_derived_accuracy),
            "primary_tie_min_found": tie_min, "primary_tie_max_found": tie_max,
            "primary_tie_robust_target_met": tie_min >= 17,
            "baseline_found_by_seed": {str(s): int(baseline.loc[s, "tp"]) for s in SEEDS},
            "nested_found_by_seed": {str(s): int(nested.loc[s, "tp"]) for s in SEEDS},
            "all_seed_numeric_target_met": bool(nested.tp.ge(17).all()),
            "all_seed_improve_on_baseline": bool(nested.tp.gt(baseline.tp).all()),
            "old_holdout_used": False, "historical_oof_used_for_selection": False,
            "label_meaning": "unconfirmed", "upstream_scaler_fit_scope": "unknown",
            "independent_future_validation": "not_available", "field_approved": False,
            "policy_accuracy_warning": "all-negative accuracy is 1880/1913=98.275pct; this metric alone is misleading",
            "bootstrap_scope": "fixed_oof_conditional_not_retraining_or_model_search_uncertainty",
        }
        write_json(out / "decision.json", result)
        write_json(out / "artifact_manifest.json", {
            str(p.relative_to(out)): digest(p) for p in sorted(out.rglob("*"))
            if p.is_file() and p.name not in {"run.log", "execution.json", "artifact_manifest.json"}})
        receipt["scope"] = "local_cpu_development_group_event_research"
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--max-seconds", type=int, default=1800)
    args = parser.parse_args()
    if args.bootstrap < 1 or args.max_seconds < 1:
        parser.error("Positive bootstrap and CPU time limit required")
    run(args)


if __name__ == "__main__":
    main()
