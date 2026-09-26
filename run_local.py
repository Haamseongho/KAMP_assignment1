"""Run the real-data MoldGuard checks with a new output folder and durable logs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def export_split(output: Path) -> dict:
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    import moldguard as mg

    if mg.DATA_DIR is None:
        raise FileNotFoundError("원본 데이터 폴더가 없습니다. 로컬_실행_가이드.md의 입력 준비를 확인하세요.")
    data, features, provenance = mg.load_data(mg.DATA_DIR, True)
    data = mg.with_groups(data, features)
    development, holdout = mg.split_groups(data)
    table = mg.group_table(data).loc[development]
    fold_by_group = {}
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=mg.SEED)
    for fold, (_, validation) in enumerate(splitter.split(table, table.stratum)):
        fold_by_group.update({group: fold for group in table.index.to_numpy()[validation]})
    manifest = data[["machine", mg.ID_COL, mg.TARGET, "feature_group"]].copy()
    manifest.rename(columns={mg.ID_COL: "source_row_id", mg.TARGET: "label_value"}, inplace=True)
    manifest["partition"] = np.where(data.feature_group.isin(development), "development", "holdout")
    manifest["oof_validation_fold"] = data.feature_group.map(fold_by_group).astype("Int64")
    if set(development) & set(holdout):
        raise RuntimeError("Split contains overlapping feature groups")
    if manifest.groupby("feature_group").partition.nunique().max() != 1:
        raise RuntimeError("A feature group crosses partitions")
    if manifest.loc[manifest.partition.eq("development"), "oof_validation_fold"].isna().any():
        raise RuntimeError("Missing development fold")
    manifest.to_csv(output / "split_manifest.csv", index=False)
    return {
        "seed": mg.SEED,
        "source_files": provenance,
        "development_groups": len(development),
        "holdout_groups": len(holdout),
        "cross_partition_feature_groups": 0,
        "rows_by_partition": {str(key): int(value) for key, value in manifest.partition.value_counts().items()},
        "oof_fold_rows": {str(key): int(value) for key, value in manifest.oof_validation_fold.value_counts().sort_index().items()},
        "label_semantics": "label_value=1 is the working positive class; defect semantics unconfirmed",
        "scope": "Exact-feature duplicate separation only; upstream scaling and true temporal independence unverified",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["quick", "full"])
    parser.add_argument("--run-dir", type=Path, help="New directory inside this project; existing paths are refused")
    args = parser.parse_args()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output = (args.run_dir or ROOT / "outputs" / "local_runs" / f"{stamp}-{args.mode}").resolve()
    if not output.is_relative_to(ROOT / "outputs"):
        parser.error("--run-dir must be inside the project outputs/ directory")
    output.mkdir(parents=True, exist_ok=False)
    logs = output / "logs"
    logs.mkdir()
    record = {
        "mode": args.mode, "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "cwd": str(ROOT), "output_dir": str(output), "python": sys.version,
        "python_executable": sys.executable, "platform": platform.platform(),
        "status": "running", "steps": [],
        "code_hashes": {name: digest(ROOT / name) for name in [
            "moldguard.py", "analyze_moldguard.py", "run_local.py", "requirements-model-lock.txt",
        ]},
    }

    def command(name: str, arguments: list[str]) -> None:
        argv = [sys.executable, *arguments]
        print(f"[{name}] {shlex.join(argv)}", flush=True)
        started = time.monotonic()
        env = {**os.environ, "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1"}
        log = logs / f"{len(record['steps']) + 1:02d}_{name}.log"
        with log.open("w", encoding="utf-8") as stream:
            stream.write(f"$ {shlex.join(argv)}\n")
            stream.flush()
            result = subprocess.run(argv, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        record["steps"].append({
            "name": name, "argv": argv, "exit_code": result.returncode,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "log": str(log.relative_to(output)),
        })
        dump(output / "run_manifest.json", record)
        print(f"  exit={result.returncode}; log={log}", flush=True)
        if result.returncode:
            raise RuntimeError(f"{name} failed; inspect {log}")

    try:
        if sys.version_info[:2] != (3, 12):
            raise RuntimeError("Python 3.12 is required for this pinned environment")
        pins = {}
        for line in (ROOT / "requirements-model-lock.txt").read_text().splitlines():
            if line and not line.startswith("#"):
                name, expected = line.split("==")
                actual = metadata.version(name)
                pins[name] = actual
                if actual != expected:
                    raise RuntimeError(f"Package mismatch: {name} {actual}; expected {expected}")
        record["packages"] = pins
        command("pip_check", ["-m", "pip", "check"])
        record["split"] = export_split(output)
        if args.mode == "quick":
            command("pytest", ["-m", "pytest", "-q"])
            command("verify", ["moldguard.py", "verify", "--output-dir", "outputs/moldguard"])
        else:
            relative = str(output.relative_to(ROOT))
            command("audit", ["moldguard.py", "audit", "--output-dir", relative])
            command("train", ["moldguard.py", "train", "--output-dir", relative])
            command("analysis", ["analyze_moldguard.py", "--output-dir", relative])
            tests = json.loads((output / "reproducibility.json").read_text())["automated_tests"]
            if tests["exit_code"] != 0:
                raise RuntimeError("Automated tests failed; see reproducibility.json")
            command("verify", ["moldguard.py", "verify", "--output-dir", relative])
            reference = ROOT / "outputs" / "moldguard"
            names = ["data_audit.json", "experiment.json", "failure_analysis.json",
                     "holdout_diagnostics.csv", "unlabeled_priority.csv", "holdout_pr.png", "inspection_capture.png"]
            record["reference_comparison"] = {
                name: {"reference_sha256": digest(reference / name), "run_sha256": digest(output / name),
                       "byte_identical": digest(reference / name) == digest(output / name)}
                for name in names if (reference / name).is_file()
            }
            record["comparison_policy"] = "JSON/CSV hashes must match. PNG hashes are informational because the headless Agg renderer can differ from the original Mac renderer."
            record["plot_backend"] = "Agg"
            record["plot_byte_differences"] = [
                name for name, item in record["reference_comparison"].items()
                if name.endswith(".png") and not item["byte_identical"]
            ]
            if any(not item["byte_identical"] for name, item in record["reference_comparison"].items()
                   if not name.endswith(".png")):
                raise RuntimeError("Reference JSON/CSV outputs differ; inspect reference_comparison before interpreting new results")
        record["status"] = "passed"
    except Exception as error:
        record["status"] = "failed"
        record["error"] = str(error)
        print(f"FAILED: {error}", file=sys.stderr, flush=True)
    finally:
        record["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        record["output_hashes"] = {
            str(path.relative_to(output)): digest(path)
            for path in sorted(output.rglob("*")) if path.is_file() and path.name != "run_manifest.json"
        }
        dump(output / "run_manifest.json", record)
        print(f"{record['status'].upper()}: {output / 'run_manifest.json'}", flush=True)
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
