"""Read-only official-input and NVIDIA preflight for Task 01 on another PC.

This check does not train a model or prove that a later fit uses the GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_CSV_SHA256 = {
    "moldset_labeled_cn7.csv": "f870b0c259297f20d503f2e2739966553e5db5ef1deb0cc7513eefa4bbfa2753",
    "moldset_labeled_rg3.csv": "14aab21476eea02c823ccb00f82800040280d805dbce5a3eb7c11b176a9e5aa4",
    "moldset_unlabeled_cn7.csv": "110a2d408d6cb8002caf26f50c06d06e42998e067e708788a3e37e9d1769b978",
    "moldset_unlabeled_rg3.csv": "ef9b30014ce4d6671a72f5a4c3f286802b71ff08d52f9d6a551a3f81e6af8c9c",
}
EXPECTED_SPLIT_SHA256 = "e26a82c58d0e86952b8557c2e8542f816f64db8ed49d15a4a450ef7a86b5efc7"
EXPECTED_PACKAGES = {
    "numpy": "1.26.4",
    "pandas": "2.2.2",
    "scikit-learn": "1.5.1",
    "catboost": "1.2.10",
    "lightgbm": "4.6.0",
    "xgboost": "3.0.5",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect(data_dir: Path, split_dir: Path) -> dict:
    if not data_dir.is_dir() or not split_dir.is_dir():
        raise ValueError("Official data and regenerated split directories must exist")
    sources = {}
    for name, expected in EXPECTED_CSV_SHA256.items():
        path = data_dir / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Missing or linked official CSV: {name}")
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"Official CSV hash mismatch: {name}")
        sources[name] = {"sha256": actual, "bytes": path.stat().st_size}
    split = split_dir / "split_manifest.csv"
    if not split.is_file() or split.is_symlink():
        raise ValueError("Regenerated split_manifest.csv is missing or linked")
    split_hash = sha256(split)
    if split_hash != EXPECTED_SPLIT_SHA256:
        raise ValueError("Frozen split hash mismatch; do not start GPU fitting")
    manifest = split_dir / "run_manifest.json"
    execution = split_dir / "execution.json"
    if not manifest.is_file() or not execution.is_file():
        raise ValueError("Split run_manifest.json or execution.json is missing")
    record = json.loads(manifest.read_text(encoding="utf-8"))
    run = json.loads(execution.read_text(encoding="utf-8"))
    if record.get("split_sha256") != split_hash or run.get("status") != "passed":
        raise ValueError("Regenerated split execution or manifest did not pass")
    packages = {}
    for name, expected in EXPECTED_PACKAGES.items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise ValueError(f"Required package is missing: {name}") from error
        if actual != expected:
            raise ValueError(f"Package version mismatch: {name} {actual} != {expected}")
        packages[name] = actual
    system = platform.system()
    if system not in {"Linux", "Windows"}:
        raise ValueError("NVIDIA GPU path supports Linux or Windows, not this OS")
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, check=False, timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        raise ValueError("nvidia-smi is missing or timed out") from error
    if result.returncode or not result.stdout.strip():
        raise ValueError("No usable NVIDIA GPU reported by nvidia-smi")
    return {
        "status": "passed_preflight_only",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_dir.resolve()),
        "source_files": sources,
        "split_dir": str(split_dir.resolve()),
        "split_sha256": split_hash,
        "packages": packages,
        "platform": system,
        "nvidia_smi": result.stdout.strip(),
        "gpu_fit_verified": False,
        "independent_performance_verified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = inspect(args.data_dir, args.split_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "preflight.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
