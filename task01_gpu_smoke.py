"""Bounded synthetic GPU fits; no official data or performance evaluation.

Run this before the full frozen-development comparison on a new NVIDIA PC.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
from subprocess import run as run_worker
import sys
import time

from research_runtime import execution_record
from task01_gpu_preflight import EXPECTED_PACKAGES


def write_json(path: Path, record: dict) -> None:
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def fit_synthetic(output: Path) -> None:
    packages = {name: importlib.metadata.version(name) for name in EXPECTED_PACKAGES}
    for name, expected in EXPECTED_PACKAGES.items():
        if packages[name] != expected:
            raise ValueError(f"Package version mismatch: {name} {packages[name]} != {expected}")

    import numpy as np
    import task01_compare as comparison

    runtime = comparison.check_gpu_runtime()
    rng = np.random.default_rng(20260922)
    x = rng.normal(size=(256, 24)).astype(np.float32)
    y = (x[:, 0] + x[:, 1] > 1).astype(int)
    report = {
        "status": "running",
        "scope": "synthetic_gpu_backend_smoke_only",
        "synthetic_data": True,
        "official_data_read": False,
        "independent_performance_verified": False,
        "gpu_fit_verified": False,
        "gpu_runtime": runtime,
        "packages": packages,
        "backend_checks": [],
        "rows": len(x),
        "features": x.shape[1],
        "iterations_per_model": 8,
    }
    for name in comparison.GPU_MODELS:
        print(f"synthetic GPU fit: {name}", flush=True)
        model = comparison.make_candidate(name, comparison.SEEDS[0], "gpu", y)
        model.set_params(**({"n_estimators": 8} if name.startswith("xgb_") else {"iterations": 8}))
        started = time.monotonic()
        model.fit(x, y)
        backend = comparison.check_fitted_gpu_backend(model, name)
        scores = model.predict_proba(x[:16])[:, 1]
        if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
            raise ValueError(f"Invalid GPU prediction: {name}")
        report["backend_checks"].append({
            "model": name, "configured_backend": backend,
            "elapsed_seconds": time.monotonic() - started,
            "prediction_rows": len(scores),
        })
        write_json(output / "smoke_progress.json", report)
        del model
    report.update(status="passed_synthetic_gpu_smoke_only", gpu_fit_verified=True)
    write_json(output / "smoke.json", report)
    print(json.dumps(report, indent=2), flush=True)


def run(output: Path, max_seconds: int, local_gpu_ack: bool) -> None:
    if not local_gpu_ack:
        raise ValueError("Provide --local-gpu-ack for your own NVIDIA PC")
    if max_seconds <= 0:
        raise ValueError("--max-seconds must be positive")
    with execution_record(output):
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker",
                   "--local-gpu-ack", "--output-dir", str(output.resolve())]
        environment = {**os.environ, "PYTHONUTF8": "1"}
        environment.setdefault("MPLCONFIGDIR", str(output.resolve() / "matplotlib_cache"))
        try:
            result = run_worker(command, capture_output=True, text=True, encoding="utf-8",
                                env=environment, timeout=max_seconds)
        except subprocess.TimeoutExpired as error:
            # subprocess.run kills and waits for the training process on timeout.
            for partial in (error.stdout, error.stderr):
                if partial:
                    print(partial.decode("utf-8", errors="replace") if isinstance(partial, bytes) else partial)
            raise RuntimeError(f"GPU smoke exceeded {max_seconds}s; training process stopped") from error
        print(result.stdout, end="", flush=True)
        print(result.stderr, end="", file=sys.stderr, flush=True)
        if result.returncode:
            raise RuntimeError(f"GPU smoke worker failed with exit code {result.returncode}")
        record = json.loads((output / "smoke.json").read_text(encoding="utf-8"))
        if record.get("status") != "passed_synthetic_gpu_smoke_only" or not record.get("gpu_fit_verified"):
            raise RuntimeError("GPU smoke did not produce a passed backend receipt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--local-gpu-ack", action="store_true")
    parser.add_argument("--max-seconds", type=int, default=120)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if not args.local_gpu_ack:
            parser.error("--local-gpu-ack is required")
        fit_synthetic(args.output_dir)
    else:
        run(args.output_dir, args.max_seconds, args.local_gpu_ack)


if __name__ == "__main__":
    main()
