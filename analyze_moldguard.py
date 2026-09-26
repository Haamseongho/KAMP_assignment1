"""Independent diagnostics for the frozen MoldGuard experiment."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold

import moldguard as mg


OUTPUT = mg.ROOT / "outputs" / "moldguard"


def bootstrap_holdout(data: pd.DataFrame, scores: np.ndarray, repeats: int = 2000) -> dict:
    rng = np.random.default_rng(mg.SEED)
    positions = data.groupby("feature_group").indices
    keys = np.array(list(positions), dtype=object)
    draws = {"average_precision": [], "top10_recall": [], "top20_recall": []}
    y = data[mg.TARGET].to_numpy()
    for _ in range(repeats):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        selected = np.concatenate([positions[key] for key in sampled])
        if y[selected].sum() == 0:
            continue
        metric = mg.score_metrics(y[selected], scores[selected])
        for name in draws:
            draws[name].append(metric[name])
    return {
        name: {"lower_95pct": float(np.quantile(values, 0.025)),
               "upper_95pct": float(np.quantile(values, 0.975))}
        for name, values in draws.items()
    }


def line_quota_metrics(data: pd.DataFrame, scores: np.ndarray) -> dict:
    result = {}
    for fraction in (0.1, 0.2):
        selected = []
        for _, section in data.groupby("machine"):
            rows = np.flatnonzero(data.machine.eq(section.machine.iloc[0]).to_numpy())
            order = rows[np.argsort(-scores[rows], kind="stable")]
            selected.extend(order[: int(np.ceil(len(rows) * fraction))])
        positives = int(data.iloc[selected][mg.TARGET].sum())
        result[f"per_line_top{int(fraction * 100)}"] = {
            "inspected": len(selected), "failures_found": positives,
            "recall": positives / int(data[mg.TARGET].sum()),
            "precision": positives / len(selected),
        }
    return result


def index_order_stress(data: pd.DataFrame, features: list[str], chosen: str) -> dict:
    train_groups, test_groups = [], []
    for machine, section in data.groupby("machine"):
        ordered = section.groupby("feature_group")[mg.ID_COL].min().sort_values().index.to_numpy()
        cut = int(np.floor(len(ordered) * 0.8))
        train_groups.extend(ordered[:cut])
        test_groups.extend(ordered[cut:])
    train_mask = data.feature_group.isin(train_groups).to_numpy()
    test_mask = data.feature_group.isin(test_groups).to_numpy()
    matrix = mg.feature_matrix(data, features)
    model = mg.make_model(chosen)
    model.fit(matrix.loc[train_mask], data.loc[train_mask, mg.TARGET])
    result = {"caveat": "Index order is not a verified timestamp; this is a stress test, not a prospective validation."}
    for machine in sorted(data.machine.unique()):
        mask = test_mask & data.machine.eq(machine).to_numpy()
        scores = model.predict_proba(matrix.loc[mask])[:, 1]
        result[machine] = mg.score_metrics(data.loc[mask, mg.TARGET].to_numpy(), scores)
    return result


def anomaly_probe(data: pd.DataFrame, features: list[str], development: np.ndarray) -> dict:
    """Exploratory group-level check: are failed measurements rare anomalies?"""
    groups = data.loc[data.feature_group.isin(development)].groupby("feature_group", as_index=False).agg(
        {**{name: "first" for name in features}, "machine": "first", mg.TARGET: "max"}
    )
    result = {}
    for machine in sorted(groups.machine.unique()):
        section = groups.loc[groups.machine.eq(machine)].reset_index(drop=True)
        columns = [name for name in features if section[name].nunique() > 1]
        x = section[columns]
        y = section[mg.TARGET].to_numpy()
        predictions = np.full(len(section), np.nan)
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=mg.SEED)
        for train, validation in folds.split(x, y):
            model = IsolationForest(n_estimators=300, random_state=mg.SEED, n_jobs=4)
            model.fit(x.iloc[train[y[train] == 0]])
            predictions[validation] = -model.score_samples(x.iloc[validation])
        result[machine] = {
            "group_unit": "identical feature vector, label = any positive row",
            "positive_groups": int(y.sum()),
            "prevalence": float(y.mean()),
            "out_of_fold_metrics": mg.score_metrics(y, predictions),
        }
    return result


def main() -> None:
    global OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    OUTPUT = parser.parse_args().output_dir.resolve()
    labeled, features, _ = mg.load_data(mg.DATA_DIR, True)
    unlabeled, _, _ = mg.load_data(mg.DATA_DIR, False)
    labeled = mg.with_groups(labeled, features)
    development, test = mg.split_groups(labeled)
    test_mask = labeled.feature_group.isin(test).to_numpy()
    held = labeled.loc[test_mask].reset_index(drop=True)
    artifact = joblib.load(OUTPUT / "model.joblib")
    score = artifact["model"].predict_proba(mg.feature_matrix(held, features))[:, 1]
    group_counts = labeled.groupby("feature_group")[mg.TARGET].agg(["sum", "size"])
    ambiguous = set(group_counts.index[(group_counts["sum"] > 0) & (group_counts["sum"] < group_counts["size"])])
    threshold = artifact["diagnostic_threshold"]
    missed = held.loc[held[mg.TARGET].eq(1) & (score < threshold), ["machine", mg.ID_COL]].copy()
    false_alarm = held.loc[held[mg.TARGET].eq(0) & (score >= threshold), ["machine", mg.ID_COL]].copy()
    missed["risk_score"] = score[(held[mg.TARGET].eq(1) & (score < threshold)).to_numpy()]
    false_alarm["risk_score"] = score[(held[mg.TARGET].eq(0) & (score >= threshold)).to_numpy()]
    missed.rename(columns={mg.ID_COL: "source_row_id"}).to_csv(OUTPUT / "false_negatives.csv", index=False)
    false_alarm.rename(columns={mg.ID_COL: "source_row_id"}).to_csv(OUTPUT / "false_positives.csv", index=False)

    drift = {}
    for machine in sorted(labeled.machine.unique()):
        fit = labeled[labeled.machine.eq(machine)]
        live = unlabeled[unlabeled.machine.eq(machine)]
        statistics = {name: float(ks_2samp(fit[name], live[name]).statistic) for name in features if fit[name].nunique() > 1}
        low = pd.Series(artifact["ood_bounds"][machine]["low"])
        high = pd.Series(artifact["ood_bounds"][machine]["high"])
        outside = (live[features].lt(low) | live[features].gt(high)).sum(axis=1)
        drift[machine] = {
            "median_feature_ks": float(np.median(list(statistics.values()))),
            "largest_feature_ks": dict(sorted(statistics.items(), key=lambda item: -item[1])[:5]),
            "unlabeled_rows_outside_3_or_more_training_feature_ranges": int(outside.ge(3).sum()),
            "fraction_outside_3_or_more": float(outside.ge(3).mean()),
        }

    feature_associations = []
    if artifact["model_name"] == "logistic":
        coefficients = artifact["model"].named_steps["logisticregression"].coef_[0]
        feature_associations = [
            {"feature": name, "standardized_coefficient": float(value)}
            for name, value in sorted(zip(artifact["matrix_columns"], coefficients), key=lambda item: -abs(item[1]))[:10]
        ]

    priorities = pd.read_csv(OUTPUT / "unlabeled_priority.csv")
    diagnostic = {
        "frozen_split": {
            "development_groups": len(development), "holdout_groups": len(test),
            "cross_partition_feature_groups": len(set(development) & set(test)),
            "holdout_ambiguous_positive_rows": int(held.loc[held[mg.TARGET].eq(1), "feature_group"].isin(ambiguous).sum()),
            "holdout_positive_rows": int(held[mg.TARGET].sum()),
        },
        "holdout_95pct_group_bootstrap_interval": bootstrap_holdout(held, score),
        "inspection_with_equal_capacity_per_machine": line_quota_metrics(held, score),
        "false_negatives_at_diagnostic_threshold_by_machine": missed.machine.value_counts().to_dict(),
        "false_positives_at_diagnostic_threshold_by_machine": false_alarm.machine.value_counts().to_dict(),
        "index_order_stress_test": index_order_stress(labeled, features, artifact["model_name"]),
        "normal_only_isolation_forest_probe_not_primary_model": anomaly_probe(labeled, features, development),
        "unlabeled_distribution_shift": drift,
        "unlabeled_ood_flagged_total": int(priorities.ood_flag.sum()),
        "unlabeled_ood_action_fallback_total": int(priorities.action.eq("usual_inspection_ood").sum()),
        "unlabeled_unvalidated_machine_fallback_total": int(
            priorities.action.eq("usual_inspection_unvalidated_machine").sum()
        ),
        "largest_logistic_associations_not_causal": feature_associations,
    }
    (OUTPUT / "failure_analysis.json").write_text(json.dumps(diagnostic, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for label, mask in [
        ("All", np.ones(len(held), dtype=bool)),
        ("CN7", held.machine.eq("cn7").to_numpy()),
        ("RG3", held.machine.eq("rg3").to_numpy()),
    ]:
        truth = held.loc[mask, mg.TARGET].to_numpy()
        precision, recall, _ = precision_recall_curve(truth, score[mask])
        ap = average_precision_score(truth, score[mask])
        ax.step(recall, precision, where="post", label=f"{label}: AP={ap:.3f}")
    ax.axhline(held[mg.TARGET].mean(), color="gray", linestyle="--", linewidth=1, label="Overall prevalence")
    ax.set(xlabel="Recall", ylabel="Precision", title="Frozen holdout precision-recall (9 failures)", xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="upper right")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT / "holdout_pr.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for label, mask in [
        ("All", np.ones(len(held), dtype=bool)),
        ("CN7", held.machine.eq("cn7").to_numpy()),
        ("RG3", held.machine.eq("rg3").to_numpy()),
    ]:
        truth = held.loc[mask, mg.TARGET].to_numpy()
        order = np.argsort(-score[mask], kind="stable")
        x_axis = np.arange(1, len(truth) + 1) / len(truth)
        capture = np.cumsum(truth[order]) / truth.sum()
        ax.step(np.r_[0, x_axis], np.r_[0, capture], where="post", label=label)
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Random expectation")
    ax.set(xlabel="Fraction inspected", ylabel="Fraction of failures found", title="Frozen holdout inspection capture", xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="lower right")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT / "inspection_capture.png", dpi=180)
    plt.close(fig)

    frozen = subprocess.run([sys.executable, "-m", "pip", "freeze"], text=True, capture_output=True, check=True).stdout
    (OUTPUT / "pip-freeze.txt").write_text(frozen, encoding="utf-8")
    files = [
        mg.ROOT / "moldguard.py", mg.ROOT / "analyze_moldguard.py",
        OUTPUT / "data_audit.json", OUTPUT / "experiment.json", OUTPUT / "failure_analysis.json",
        OUTPUT / "model.joblib", OUTPUT / "holdout_diagnostics.csv", OUTPUT / "unlabeled_priority.csv",
        OUTPUT / "pip-freeze.txt", OUTPUT / "holdout_pr.png", OUTPUT / "inspection_capture.png",
    ]
    reproducibility = {
        "commands": [
            ".venv/bin/python moldguard.py audit --output-dir outputs/moldguard",
            ".venv/bin/python moldguard.py train --output-dir outputs/moldguard",
            ".venv/bin/python analyze_moldguard.py",
            ".venv/bin/python moldguard.py verify --output-dir outputs/moldguard",
            ".venv/bin/python -m pytest -q",
        ],
        "seed": mg.SEED,
        "source_file_hashes": artifact["source_hashes"],
        "artifact_hashes": {str(path.relative_to(mg.ROOT)): mg.sha256(path) for path in files},
    }
    repeat = mg.ROOT / "outputs" / "moldguard_repro"
    if repeat.exists():
        names = ["experiment.json", "holdout_diagnostics.csv", "unlabeled_priority.csv"]
        reproducibility["second_run_byte_identical"] = {
            name: mg.sha256(OUTPUT / name) == mg.sha256(repeat / name)
            for name in names
        }
    fresh = mg.ROOT / "outputs" / "moldguard_clean"
    if fresh.exists() and fresh != OUTPUT:
        names = [
            "data_audit.json", "experiment.json", "failure_analysis.json",
            "holdout_diagnostics.csv", "unlabeled_priority.csv",
            "holdout_pr.png", "inspection_capture.png",
        ]
        reproducibility["fresh_venv_byte_identical"] = {
            name: mg.sha256(OUTPUT / name) == mg.sha256(fresh / name)
            for name in names if (fresh / name).exists()
        }
    tests = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"], cwd=mg.ROOT,
        text=True, capture_output=True,
    )
    reproducibility["automated_tests"] = {
        "exit_code": tests.returncode,
        "stdout": tests.stdout.strip(),
        "stderr": tests.stderr.strip(),
    }
    (OUTPUT / "reproducibility.json").write_text(json.dumps(reproducibility, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(diagnostic, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
