"""Reproducible CPU-only analysis of the 2026 KAMP injection-molding files.

The raw CSVs remain untouched. The unnamed source index is an identifier, not a
feature or a trusted timestamp. A risk score is not a validated probability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
DATA_DIR = next((path for path in ROOT.glob("1. *AI *") if path.is_dir()), None)
SEED = 20260922
ID_COL = "Unnamed: 0"
TARGET = "PassOrFail"
UNVALIDATED_MACHINES = frozenset({"rg3"})
# The supplied CSVs have no version-specific 0/1 codebook. Keep all ranking
# decisions offline until the label mapping and prediction-time data contract
# are confirmed against the exact files used here.


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_data(data_dir: Path, labeled: bool) -> tuple[pd.DataFrame, list[str], dict]:
    kind = "labeled" if labeled else "unlabeled"
    paths = sorted(data_dir.glob(f"moldset_{kind}_*.csv"))
    if len(paths) != 2:
        raise ValueError(f"Expected two {kind} files, found {[p.name for p in paths]}")
    frames = []
    feature_cols = None
    provenance = {}
    for path in paths:
        machine = path.stem.split("_")[-1]
        frame = pd.read_csv(path)
        required = {ID_COL, TARGET} if labeled else {ID_COL}
        if not required.issubset(frame):
            raise ValueError(f"Missing required columns in {path.name}: {required - set(frame)}")
        columns = [c for c in frame if c not in {ID_COL, TARGET}]
        if feature_cols is None:
            feature_cols = columns
        elif columns != feature_cols:
            raise ValueError(f"Feature columns differ in {path.name}")
        if frame[ID_COL].isna().any() or frame[ID_COL].duplicated().any():
            raise ValueError(f"Missing or duplicate source identifiers in {path.name}")
        if labeled and not frame[TARGET].isin([0, 1]).all():
            raise ValueError(f"Labels other than 0/1 in {path.name}")
        if not all(pd.api.types.is_numeric_dtype(frame[c]) for c in columns):
            raise ValueError(f"Non-numeric feature in {path.name}")
        if not np.isfinite(frame[columns].to_numpy(dtype=float)).all():
            raise ValueError(f"Missing or non-finite feature in {path.name}")
        frame["machine"] = machine
        varying = frame[columns].loc[:, frame[columns].nunique().gt(1)]
        provenance[path.name] = {
            "rows": len(frame), "sha256": sha256(path),
            "max_abs_feature_mean": float(frame[columns].mean().abs().max()),
            "max_abs_varying_feature_std_minus_one": float((varying.std(ddof=0) - 1).abs().max()),
        }
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    if data.duplicated(["machine", ID_COL]).any():
        raise ValueError("Duplicate machine/source identifier combination")
    return data, feature_cols, provenance


def with_groups(data: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    result = data.copy()
    fingerprints = pd.util.hash_pandas_object(result[features], index=False)
    result["feature_group"] = result["machine"] + ":" + fingerprints.astype(str)
    # Hash collisions are extraordinarily unlikely, but never silently group
    # different measurements when constructing leakage-safe partitions.
    if result.groupby("feature_group")[features].nunique(dropna=False).gt(1).any().any():
        raise ValueError("Feature-hash collision detected")
    return result


def audit(data_dir: Path, out: Path) -> dict:
    labeled, features, provenance = load_data(data_dir, True)
    unlabeled, other_features, other_provenance = load_data(data_dir, False)
    if features != other_features:
        raise ValueError("Labeled and unlabeled schemas differ")
    labeled = with_groups(labeled, features)
    groups = labeled.groupby("feature_group").agg(
        machine=("machine", "first"), rows=(TARGET, "size"), failures=(TARGET, "sum")
    )
    ambiguous = (groups.failures.gt(0) & groups.failures.lt(groups.rows))
    summary = {
        "source": {**provenance, **other_provenance},
        "feature_columns": features,
        "excluded_columns": [ID_COL, TARGET],
        "labeled_rows": len(labeled),
        "unlabeled_rows": len(unlabeled),
        "positive_rows": int(labeled[TARGET].sum()),
        "positive_prevalence": float(labeled[TARGET].mean()),
        "identical_feature_groups": len(groups),
        "ambiguous_groups": int(ambiguous.sum()),
        "ambiguous_positive_rows": int(groups.loc[ambiguous, "failures"].sum()),
        "irreducible_row_errors_for_identical_features": int(
            np.minimum(groups.failures, groups.rows - groups.failures).sum()
        ),
        "constant_features": [c for c in features if labeled[c].nunique() <= 1],
        "by_machine": {},
        "normalization_warning": "Every source file is approximately centered/scaled on its own; original scaler and raw units are unavailable.",
        "time_warning": "The source index is not a validated timestamp; prospective performance cannot be established.",
    }
    for machine, section in labeled.groupby("machine"):
        local_groups = groups[groups.machine.eq(machine)]
        order = section.sort_values(ID_COL)
        early = order.iloc[: max(1, int(len(order) * 0.2))]
        late = order.iloc[int(len(order) * 0.8) :]
        summary["by_machine"][machine] = {
            "rows": len(section),
            "positive_rows": int(section[TARGET].sum()),
            "groups": len(local_groups),
            "positive_groups": int(local_groups.failures.gt(0).sum()),
            "ambiguous_groups": int(
                (local_groups.failures.gt(0) & local_groups.failures.lt(local_groups.rows)).sum()
            ),
            "first_20pct_index_positive_rows": int(early[TARGET].sum()),
            "last_20pct_index_positive_rows": int(late[TARGET].sum()),
            "unlabeled_rows": int(unlabeled.machine.eq(machine).sum()),
        }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def feature_matrix(data: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    matrix = data[[c for c in features if c != "Clamp_Open_Position"]].copy()
    matrix["machine_rg3"] = data.machine.eq("rg3").astype(int)
    return matrix


def group_table(data: pd.DataFrame) -> pd.DataFrame:
    groups = data.groupby("feature_group", sort=True).agg(
        machine=("machine", "first"), has_failure=(TARGET, "max")
    )
    groups["stratum"] = groups.machine + ":" + groups.has_failure.astype(str)
    return groups


def split_groups(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    groups = group_table(data)
    development, test = train_test_split(
        groups.index.to_numpy(), test_size=0.2, random_state=SEED,
        stratify=groups.stratum.to_numpy(),
    )
    return development, test


def make_model(name: str):
    if name.startswith("separate_"):
        return SeparateMachineModel(name.removeprefix("separate_"))
    if name == "dummy":
        return DummyClassifier(strategy="prior")
    if name == "logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000, random_state=SEED),
        )
    if name == "forest":
        return RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=4,
            class_weight="balanced_subsample", n_jobs=4, random_state=SEED,
        )
    if name == "catboost":
        return CatBoostClassifier(
            iterations=400, depth=4, learning_rate=0.03, l2_leaf_reg=5,
            loss_function="Logloss", auto_class_weights="Balanced",
            thread_count=4, random_seed=SEED, verbose=False,
            allow_writing_files=False,
        )
    raise ValueError(name)


class SeparateMachineModel:
    """Fit an independent estimator for each already standardized machine."""

    def __init__(self, base_name: str):
        self.base_name = base_name
        self.models = {}

    def fit(self, x: pd.DataFrame, y: pd.Series):
        for value in (0, 1):
            mask = x.machine_rg3.eq(value)
            model = make_model(self.base_name)
            model.fit(x.loc[mask], y.loc[mask])
            self.models[value] = model
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        scores = np.empty(len(x), dtype=float)
        for value in (0, 1):
            mask = x.machine_rg3.eq(value).to_numpy()
            if mask.any():
                scores[mask] = self.models[value].predict_proba(x.loc[mask])[:, 1]
        return np.column_stack([1 - scores, scores])


def score_metrics(y: np.ndarray, score: np.ndarray) -> dict:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    if len(y) == 0 or not np.isfinite(score).all():
        raise ValueError("Empty or invalid predictions")
    result = {
        "rows": int(len(y)), "positives": int(y.sum()),
        "prevalence": float(y.mean()),
        "average_precision": float(average_precision_score(y, score)) if y.sum() else None,
        "roc_auc": float(roc_auc_score(y, score)) if 0 < y.sum() < len(y) else None,
    }
    order = np.argsort(-score, kind="stable")
    for fraction in (0.1, 0.2):
        k = max(1, int(np.ceil(len(y) * fraction)))
        found = int(y[order[:k]].sum())
        result[f"top{int(fraction * 100)}_k"] = k
        result[f"top{int(fraction * 100)}_positives"] = found
        result[f"top{int(fraction * 100)}_recall"] = float(found / y.sum()) if y.sum() else None
        result[f"top{int(fraction * 100)}_precision"] = float(found / k)
        result[f"top{int(fraction * 100)}_lift"] = float((found / k) / y.mean()) if y.sum() else None
    return result


def make_oof(data: pd.DataFrame, x: pd.DataFrame, groups: np.ndarray, model_name: str) -> np.ndarray:
    table = group_table(data).loc[groups]
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.full(len(data), np.nan)
    for train_pos, val_pos in splitter.split(table, table.stratum):
        train_groups = table.index.to_numpy()[train_pos]
        val_groups = table.index.to_numpy()[val_pos]
        fit_mask = data.feature_group.isin(train_groups).to_numpy()
        val_mask = data.feature_group.isin(val_groups).to_numpy()
        model = make_model(model_name)
        model.fit(x.loc[fit_mask], data.loc[fit_mask, TARGET])
        oof[val_mask] = model.predict_proba(x.loc[val_mask])[:, 1]
    if not np.isfinite(oof[data.feature_group.isin(groups).to_numpy()]).all():
        raise RuntimeError("OOF predictions incomplete")
    return oof


def priority_frame(data: pd.DataFrame, scores: np.ndarray, ood_count: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame({
        "machine": data.machine.to_numpy(),
        "source_row_id": data[ID_COL].to_numpy(),
        "risk_score": np.asarray(scores, dtype=float),
        "ood_feature_count": np.asarray(ood_count, dtype=int),
    })
    if not np.isfinite(frame.risk_score).all() or not frame.risk_score.between(0, 1).all():
        raise ValueError("Risk scores must be finite in [0, 1]")
    frame = frame.sort_values(
        ["risk_score", "machine", "source_row_id"], ascending=[False, True, True], kind="stable"
    ).reset_index(drop=True)
    frame["global_rank"] = np.arange(1, len(frame) + 1)
    frame["machine_rank"] = frame.groupby("machine").cumcount() + 1
    # CN7 and RG3 are different production lines. Reserve inspection capacity
    # within each line, so a pooled score cannot starve one line entirely.
    counts = frame.groupby("machine").machine.transform("size")
    high = np.ceil(counts * 0.1).astype(int)
    review = np.ceil(counts * 0.2).astype(int)
    frame["priority"] = np.select(
        [frame.machine_rank.le(high), frame.machine_rank.le(review)],
        ["HIGH", "REVIEW"], default="STANDARD",
    )
    frame["ood_flag"] = frame.ood_feature_count.ge(3)
    frame["action"] = np.where(
        frame.machine.isin(UNVALIDATED_MACHINES), "usual_inspection_unvalidated_machine",
        np.where(
            frame.ood_flag, "usual_inspection_ood",
            "usual_inspection_label_unconfirmed",
        ),
    )
    return frame


def train(data_dir: Path, output_dir: Path) -> dict:
    data, features, provenance = load_data(data_dir, True)
    unlabeled, unlab_features, unlab_provenance = load_data(data_dir, False)
    if features != unlab_features:
        raise ValueError("Inconsistent inference schema")
    data = with_groups(data, features)
    x = feature_matrix(data, features)
    dev_groups, test_groups = split_groups(data)
    dev = data.feature_group.isin(dev_groups).to_numpy()
    test = data.feature_group.isin(test_groups).to_numpy()
    if set(data.loc[dev, "feature_group"]) & set(data.loc[test, "feature_group"]):
        raise RuntimeError("Duplicate-feature leakage across partitions")
    candidates = (
        "dummy", "logistic", "forest", "catboost",
        "separate_logistic", "separate_forest", "separate_catboost",
    )
    comparisons = {}
    oof_by_model = {}
    for name in candidates:
        oof = make_oof(data, x, dev_groups, name)
        metric = score_metrics(data.loc[dev, TARGET].to_numpy(), oof[dev])
        metric["by_machine"] = {
            machine: score_metrics(
                data.loc[dev & data.machine.eq(machine).to_numpy(), TARGET].to_numpy(),
                oof[dev & data.machine.eq(machine).to_numpy()],
            )
            for machine in sorted(data.machine.unique())
        }
        comparisons[name] = metric
        oof_by_model[name] = oof
    eligible = [name for name in candidates if name != "dummy"]
    chosen = max(
        eligible,
        key=lambda name: (comparisons[name]["average_precision"], comparisons[name]["top10_recall"]),
    )
    model = make_model(chosen)
    model.fit(x.loc[dev], data.loc[dev, TARGET])
    test_scores = model.predict_proba(x.loc[test])[:, 1]
    test_metrics = score_metrics(data.loc[test, TARGET].to_numpy(), test_scores)
    by_machine = {}
    for machine in sorted(data.machine.unique()):
        mask = test & data.machine.eq(machine).to_numpy()
        by_machine[machine] = score_metrics(data.loc[mask, TARGET].to_numpy(), model.predict_proba(x.loc[mask])[:, 1])
    # A threshold is diagnostic only. The operating policy reorders all checks.
    dev_y = data.loc[dev, TARGET].to_numpy()
    dev_score = oof_by_model[chosen][dev]
    trial = np.unique(np.quantile(dev_score, np.linspace(0, 1, 201)))
    best_threshold = 0.5
    best_f2 = -1.0
    for threshold in trial:
        precision, recall, _, _ = precision_recall_fscore_support(
            dev_y, dev_score >= threshold, average="binary", zero_division=0
        )
        f2 = 5 * precision * recall / (4 * precision + recall) if 4 * precision + recall else 0
        if f2 > best_f2:
            best_f2, best_threshold = f2, float(threshold)
    test_pred = test_scores >= best_threshold
    tn, fp, fn, tp = confusion_matrix(data.loc[test, TARGET], test_pred, labels=[0, 1]).ravel()
    test_metrics["diagnostic_f2_threshold_from_oof"] = best_threshold
    test_metrics["confusion_at_diagnostic_threshold"] = {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    test_rows = data.loc[test, ["machine", ID_COL, TARGET]].copy()
    test_rows["risk_score"] = test_scores
    test_rows["predicted_at_diagnostic_threshold"] = test_pred.astype(int)
    test_rows.rename(columns={ID_COL: "source_row_id", TARGET: "actual_fail"}).to_csv(
        output_dir / "holdout_diagnostics.csv", index=False
    )
    artifact = {
        "model": model, "model_name": chosen, "features": features,
        "matrix_columns": x.columns.tolist(), "source_hashes": {**provenance, **unlab_provenance},
        "selection_seed": SEED, "diagnostic_threshold": best_threshold,
        "ood_bounds": {
            machine: {
                "low": data.loc[dev & data.machine.eq(machine).to_numpy(), features].quantile(0.001).to_dict(),
                "high": data.loc[dev & data.machine.eq(machine).to_numpy(), features].quantile(0.999).to_dict(),
            }
            for machine in sorted(data.machine.unique())
        },
    }
    joblib.dump(artifact, output_dir / "model.joblib")
    report = {
        "method": "Machine-stratified exact-feature-group holdout (20% groups); 5-fold grouped OOF on development only. Model selected by OOF average precision.",
        "seed": SEED,
        "development_rows": int(dev.sum()), "development_positives": int(data.loc[dev, TARGET].sum()),
        "holdout_rows": int(test.sum()), "holdout_positives": int(data.loc[test, TARGET].sum()),
        "models_oof_development": comparisons,
        "selected_model": chosen,
        "holdout": test_metrics,
        "holdout_by_machine": by_machine,
        "caveats": [
            "The source index is not verified as chronological time.",
            "All four supplied files appear separately standardized; unlabeled scores have not been externally validated.",
            "Identical sensor vectors can have opposite labels; row-level distinctions are impossible from the provided features alone.",
            "Only 42 positive rows exist, so holdout uncertainty is high.",
            "Scores are model outputs, not calibrated failure probabilities.",
            "No inspection is skipped; priorities only reorder the existing inspection queue.",
            "After the prespecified primary holdout evaluation, exploratory alternatives were inspected. A new external holdout is required for any redesigned model claim.",
        ],
        "environment": {"python": platform.python_version(), "pandas": pd.__version__},
    }
    (output_dir / "experiment.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    predict(data_dir, output_dir / "model.joblib", output_dir / "unlabeled_priority.csv")
    return report


def predict(data_dir: Path, artifact_path: Path, out: Path) -> pd.DataFrame:
    artifact = joblib.load(artifact_path)
    data, features, _ = load_data(data_dir, False)
    if features != artifact["features"]:
        raise ValueError("Inference features do not match training schema")
    matrix = feature_matrix(data, features)
    if matrix.columns.tolist() != artifact["matrix_columns"]:
        raise ValueError("Inference matrix order mismatch")
    scores = artifact["model"].predict_proba(matrix)[:, 1]
    ood_count = np.zeros(len(data), dtype=int)
    for machine, bounds in artifact["ood_bounds"].items():
        mask = data.machine.eq(machine).to_numpy()
        low = pd.Series(bounds["low"])
        high = pd.Series(bounds["high"])
        ood_count[mask] = (
            data.loc[mask, features].lt(low) | data.loc[mask, features].gt(high)
        ).sum(axis=1).to_numpy()
    frame = priority_frame(data, scores, ood_count)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    check_priority(out, len(data), data)
    return frame


def check_priority(path: Path, expected_rows: int, input_data: pd.DataFrame | None = None) -> dict:
    frame = pd.read_csv(path)
    required = [
        "machine", "source_row_id", "risk_score", "ood_feature_count",
        "global_rank", "machine_rank", "priority", "ood_flag", "action",
    ]
    if frame.columns.tolist() != required:
        raise ValueError(f"Unexpected output schema: {frame.columns.tolist()}")
    if len(frame) != expected_rows or frame.duplicated(["machine", "source_row_id"]).any():
        raise ValueError("Missing or duplicate output rows")
    if not np.isfinite(frame.risk_score).all() or not frame.risk_score.between(0, 1).all():
        raise ValueError("Invalid risk score")
    if frame.global_rank.tolist() != list(range(1, len(frame) + 1)):
        raise ValueError("Global ranks not contiguous")
    if not frame.risk_score.is_monotonic_decreasing:
        raise ValueError("Scores not sorted")
    if not frame.priority.isin(["HIGH", "REVIEW", "STANDARD"]).all():
        raise ValueError("Unknown priority")
    if not frame.ood_feature_count.ge(0).all() or not frame.ood_flag.eq(frame.ood_feature_count.ge(3)).all():
        raise ValueError("OOD flags inconsistent")
    for _, section in frame.groupby("machine"):
        if section.machine_rank.tolist() != list(range(1, len(section) + 1)):
            raise ValueError("Machine ranks not contiguous")
        high = int(np.ceil(len(section) * 0.1))
        review = int(np.ceil(len(section) * 0.2))
        if section.priority.eq("HIGH").sum() != high:
            raise ValueError("HIGH tier count mismatch")
        if section.priority.eq("REVIEW").sum() != review - high:
            raise ValueError("REVIEW tier count mismatch")
        expected_priority = np.select(
            [section.machine_rank.le(high), section.machine_rank.le(review)],
            ["HIGH", "REVIEW"], default="STANDARD",
        )
        if not np.array_equal(section.priority.to_numpy(), expected_priority):
            raise ValueError("Priority does not match within-machine rank")
    expected_action = np.where(
        frame.machine.isin(UNVALIDATED_MACHINES), "usual_inspection_unvalidated_machine",
        np.where(
            frame.ood_flag, "usual_inspection_ood",
            "usual_inspection_label_unconfirmed",
        ),
    )
    if not np.array_equal(frame.action.to_numpy(), expected_action):
        raise ValueError("OOD, label, or machine fallback/priority action missing")
    if input_data is not None:
        left = set(zip(frame.machine, frame.source_row_id))
        right = set(zip(input_data.machine, input_data[ID_COL]))
        if left != right:
            raise ValueError("Output identifiers do not match input")
    return {
        "valid": True, "rows": len(frame), "high": int(frame.priority.eq("HIGH").sum()),
        "ood_flagged": int(frame.ood_flag.sum()),
        "ood_fallback": int(frame.action.eq("usual_inspection_ood").sum()),
        "unvalidated_machine_fallback": int(frame.machine.isin(UNVALIDATED_MACHINES).sum()),
        "label_unconfirmed_fallback": int(frame.action.eq("usual_inspection_label_unconfirmed").sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["audit", "train", "predict", "verify"])
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "moldguard")
    args = parser.parse_args()
    if args.data_dir is None:
        parser.error("Dataset directory not found; pass --data-dir")
    if args.command == "audit":
        result = audit(args.data_dir, args.output_dir / "data_audit.json")
    elif args.command == "train":
        result = train(args.data_dir, args.output_dir)
    elif args.command == "predict":
        result = {"rows": len(predict(args.data_dir, args.output_dir / "model.joblib", args.output_dir / "unlabeled_priority.csv"))}
    else:
        input_data, _, _ = load_data(args.data_dir, False)
        result = check_priority(args.output_dir / "unlabeled_priority.csv", len(input_data), input_data)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
