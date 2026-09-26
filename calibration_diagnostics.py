"""Development-only, nested grouped calibration and conditional-error review.

This never changes the published model/queue and never evaluates the old holdout.
Probabilities concern numeric label 1, whose defect meaning remains unconfirmed.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, f1_score, log_loss, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_predict

import moldguard as mg
import research_split
from research_runtime import execution_record

FROZEN = mg.ROOT / "outputs/local_runs/20260922-share-final"
CONDITION_FEATURES = ["Clamp_Close_Time", "Max_Injection_Pressure", "Max_Switch_Over_Pressure"]
BINS = np.array([0, .01, .02, .05, .1, .2, .5, 1.0])


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_development(data_dir=None, split_dir=None) -> tuple[pd.DataFrame, list[str], dict]:
    merged, features, provenance = research_split.load_validated(
        data_dir if data_dir is not None else mg.DATA_DIR, split_dir or FROZEN)
    development = merged.loc[merged.partition.eq("development")].copy().reset_index(drop=True)
    development["outer_fold"] = development.oof_validation_fold.astype(int)
    if set(development.outer_fold.unique()) != set(range(5)):
        raise ValueError("Expected exactly five frozen development folds")
    if development.groupby("feature_group").outer_fold.nunique().max() != 1:
        raise ValueError("An exact-feature group crosses outer folds")
    return development, features, provenance


def grouped_cv(data: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    """Row positions for sklearn; stratification and separation are group-level."""
    table = mg.group_table(data)
    if table.stratum.value_counts().min() < 5:
        raise ValueError("Too few groups per machine/label stratum for five inner folds")
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=mg.SEED)
    folds = []
    coverage = np.zeros(len(data), dtype=int)
    for train, valid in splitter.split(table, table.stratum):
        fit = np.flatnonzero(data.feature_group.isin(table.index[train]))
        test = np.flatnonzero(data.feature_group.isin(table.index[valid]))
        if set(data.iloc[fit].feature_group) & set(data.iloc[test].feature_group):
            raise ValueError("Group leakage in inner CV")
        if data.iloc[fit][mg.TARGET].nunique() != 2 or data.iloc[test][mg.TARGET].nunique() != 2:
            raise ValueError("Both numeric labels are required in every inner partition")
        coverage[test] += 1
        folds.append((fit, test))
    if not np.all(coverage == 1):
        raise ValueError("Each training row must have one inner OOF prediction")
    return folds


def calibrated_model(cv):
    return CalibratedClassifierCV(estimator=mg.make_model("logistic"), method="sigmoid",
                                  cv=cv, ensemble=False, n_jobs=1)


def diagnostic_threshold(y: np.ndarray, probability: np.ndarray) -> float:
    """Same F2 diagnostic criterion, fitted only on outer-training inner OOF."""
    best, best_score = 0.5, -1.0
    for threshold in np.unique(np.quantile(probability, np.linspace(0, 1, 201))):
        precision, recall, _, _ = precision_recall_fscore_support(
            y, probability >= threshold, average="binary", zero_division=0)
        f2 = 5 * precision * recall / (4 * precision + recall) if precision + recall else 0
        if f2 > best_score:
            best, best_score = float(threshold), float(f2)
    return best


def probability_metrics(y: np.ndarray, score: np.ndarray) -> dict:
    return {**mg.score_metrics(y, score), "brier": float(brier_score_loss(y, score)),
            "log_loss": float(log_loss(y, score, labels=[0, 1])),
            "mean_probability": float(np.mean(score)),
            "f1_at_0_5": float(f1_score(y, score >= .5, zero_division=0))}


def reliability_rows(y: np.ndarray, score: np.ndarray, model: str, machine: str) -> list[dict]:
    # Include p=1 in the last bin; empty bins are explicitly retained.
    assigned = np.clip(np.searchsorted(BINS, score, side="right") - 1, 0, len(BINS) - 2)
    result = []
    for index, (low, high) in enumerate(zip(BINS[:-1], BINS[1:])):
        mask = assigned == index
        result.append({"model": model, "machine": machine, "low": low, "high": high,
                       "rows": int(mask.sum()), "label1": int(y[mask].sum()),
                       "mean_probability": float(score[mask].mean()) if mask.any() else None,
                       "observed_rate": float(y[mask].mean()) if mask.any() else None})
    return result


def error_rows(predictions: pd.DataFrame) -> list[dict]:
    result = []
    definitions = [(feature, [f"band_{feature}"]) for feature in CONDITION_FEATURES]
    definitions.append(("pressure_joint", ["band_Max_Injection_Pressure", "band_Max_Switch_Over_Pressure"]))
    for kind, columns in definitions:
        for key, part in predictions.groupby(["machine", *columns], dropna=False):
            key = key if isinstance(key, tuple) else (key,)
            positive = part.label_value.eq(1)
            alert = part.raw_diagnostic_alert.eq(1)
            tp, fn = int((positive & alert).sum()), int((positive & ~alert).sum())
            fp, tn = int((~positive & alert).sum()), int((~positive & ~alert).sum())
            result.append({"machine": key[0], "condition": kind, "bands": " & ".join(key[1:]),
                           "rows": len(part), "feature_groups": part.feature_group.nunique(),
                           "positive_rows": int(positive.sum()), "tp": tp, "fn": fn, "fp": fp, "tn": tn,
                           "false_negative_rate": fn / (tp + fn) if tp + fn else None,
                           "false_positive_rate": fp / (fp + tn) if fp + tn else None,
                           "small_positive_sample": int(positive.sum()) < 10})
    return result


def paired_bootstrap(predictions: pd.DataFrame, repeats: int = 2000) -> dict:
    positions = predictions.groupby("feature_group").indices
    keys = np.array(list(positions))
    rng = np.random.default_rng(mg.SEED)
    y = predictions.label_value.to_numpy()
    losses = {name: (predictions[name].to_numpy() - y) ** 2
              for name in ["raw", "sigmoid", "prior"]}
    result = {}
    samples = {name: [] for name in ["raw", "prior"]}
    for _ in range(repeats):
        rows = np.concatenate([positions[key] for key in rng.choice(keys, len(keys), replace=True)])
        for name in samples:
            samples[name].append(float((losses["sigmoid"][rows] - losses[name][rows]).mean()))
    for name, values in samples.items():
        result[f"sigmoid_minus_{name}_brier"] = {
            "difference": float((losses["sigmoid"] - losses[name]).mean()),
            "lower_95pct": float(np.quantile(values, .025)),
            "upper_95pct": float(np.quantile(values, .975)),
        }
    return result


def run(output, data_dir=None, split_dir=None):
    with execution_record(output):
        _run(output, data_dir, split_dir or FROZEN)


def _run(output, data_dir, split_dir):
    data, features, provenance = load_development(data_dir, split_dir)
    x = mg.feature_matrix(data, features)
    y = data[mg.TARGET].to_numpy()
    predictions = data[["machine", mg.ID_COL, "feature_group", "outer_fold"]].rename(
        columns={mg.ID_COL: "source_row_id"}).copy()
    predictions["label_value"] = y
    for name in ["raw", "sigmoid", "prior", "raw_diagnostic_alert"]:
        predictions[name] = np.nan
    membership = predictions[["machine", "source_row_id", "feature_group", "outer_fold"]].copy()
    folds = []
    for outer in range(5):
        fit_mask, test_mask = data.outer_fold.ne(outer), data.outer_fold.eq(outer)
        fit = data.loc[fit_mask].reset_index(drop=True)
        inner = grouped_cv(fit)
        inner_fold = np.full(len(fit), -1)
        for k, (_, indices) in enumerate(inner):
            inner_fold[indices] = k
        membership[f"inner_fold_when_outer_{outer}"] = pd.array([None] * len(data), dtype="Int64")
        membership.loc[fit_mask, f"inner_fold_when_outer_{outer}"] = inner_fold
        raw = mg.make_model("logistic").fit(x.loc[fit_mask], y[fit_mask])
        calibrated = calibrated_model(inner).fit(x.loc[fit_mask], y[fit_mask])
        inner_raw = cross_val_predict(mg.make_model("logistic"), x.loc[fit_mask], y[fit_mask],
                                      cv=inner, method="predict_proba", n_jobs=1)[:, 1]
        threshold = diagnostic_threshold(y[fit_mask], inner_raw)
        raw_score = raw.predict_proba(x.loc[test_mask])[:, 1]
        predictions.loc[test_mask, "raw"] = raw_score
        predictions.loc[test_mask, "sigmoid"] = calibrated.predict_proba(x.loc[test_mask])[:, 1]
        predictions.loc[test_mask, "prior"] = y[fit_mask].mean()
        predictions.loc[test_mask, "raw_diagnostic_alert"] = (raw_score >= threshold).astype(int)
        boundaries = {}
        for machine in ["cn7", "rg3"]:
            machine_fit = fit[fit.machine.eq(machine)]
            validation = test_mask & data.machine.eq(machine)
            boundaries[machine] = {}
            for feature in CONDITION_FEATURES:
                median = float(machine_fit[feature].median())
                boundaries[machine][feature] = median
                predictions.loc[validation, f"band_{feature}"] = np.where(
                    data.loc[validation, feature] <= median, "le_train_median", "gt_train_median")
        folds.append({"outer_fold": outer, "train_rows": int(fit_mask.sum()),
                      "validation_rows": int(test_mask.sum()), "train_positive_rows": int(y[fit_mask].sum()),
                      "validation_positive_rows": int(y[test_mask].sum()),
                      "diagnostic_f2_threshold_inner_oof": threshold, "training_medians_by_machine": boundaries})
        print(f"outer fold {outer}: training={fit_mask.sum()}, validation={test_mask.sum()}, grouped inner folds=5", flush=True)
    if not np.isfinite(predictions[["raw", "sigmoid", "prior", "raw_diagnostic_alert"]]).all().all():
        raise RuntimeError("Incomplete nested OOF predictions")
    predictions["raw_diagnostic_alert"] = predictions.raw_diagnostic_alert.astype(int)
    metrics, reliability = {}, []
    for machine in ["all", "cn7", "rg3"]:
        part = predictions if machine == "all" else predictions[predictions.machine.eq(machine)]
        metrics[machine] = {}
        for model in ["prior", "raw", "sigmoid"]:
            metrics[machine][model] = probability_metrics(part.label_value.to_numpy(), part[model].to_numpy())
            reliability.extend(reliability_rows(part.label_value.to_numpy(), part[model].to_numpy(), model, machine))
    predictions.to_csv(output / "nested_oof_predictions.csv", index=False)
    membership.to_csv(output / "nested_split_manifest.csv", index=False)
    pd.DataFrame(reliability).to_csv(output / "reliability_bins.csv", index=False)
    errors = error_rows(predictions)
    pd.DataFrame(errors).to_csv(output / "conditional_errors.csv", index=False)
    # Research artifact only. It is not substituted into the published queue.
    final = calibrated_model(grouped_cv(data)).fit(x, y)
    joblib.dump({"model": final, "feature_columns": list(x.columns), "source_files": provenance,
                 "label_semantics": "P(numeric PassOrFail=1); defect direction unconfirmed",
                 "deployment_approved": False, "training_scope": "frozen development partition only"},
                output / "calibrated_research_model.joblib")
    report = {"status": "exploratory_development_only", "seed": mg.SEED,
              "method": "Frozen outer 5 grouped folds; sigmoid calibration with inner 5 grouped folds; ensemble=False",
              "development_rows": len(data), "development_groups": data.feature_group.nunique(),
              "old_holdout_evaluated": False, "published_model_replaced": False,
              "label_semantics_confirmed": False, "calibration_bins": BINS.tolist(),
              "metrics": metrics, "folds": folds, "paired_group_bootstrap": paired_bootstrap(predictions),
              "bootstrap_caveat": "Fixed nested OOF predictions; does not include all retraining/selection uncertainty",
              "condition_caveat": "Per-machine medians fitted on outer training only. Joint slices are descriptive associations, not causal interactions or physical thresholds.",
              "scope_caveat": "Post-selection development analysis with 33 label-1 rows. Upstream scaling/time availability remain unverified. Not independent prospective validation.",
              "source_files": provenance,
              "source_hashes": {"moldguard.py": mg.sha256(mg.ROOT / "moldguard.py"),
                                "calibration_diagnostics.py": mg.sha256(Path(__file__)),
                                "frozen_split_manifest": mg.sha256(split_dir / "split_manifest.csv")},
              "packages": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scikit-learn", "scipy"]},
              "references": ["https://scikit-learn.org/1.5/modules/calibration.html",
                             "https://scikit-learn.org/1.5/modules/generated/sklearn.calibration.CalibratedClassifierCV.html"]}
    dump(output / "calibration_review.json", report)
    dump(output / "artifact_hashes.json", {p.name: mg.sha256(p) for p in sorted(output.iterdir())
                                          if p.is_file() and p.name not in {'execution.json', 'run.log'}})
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=mg.DATA_DIR)
    parser.add_argument('--split-dir', type=Path, default=FROZEN)
    args = parser.parse_args()
    run(args.output_dir.resolve(), args.data_dir, args.split_dir)


if __name__ == "__main__":
    main()
