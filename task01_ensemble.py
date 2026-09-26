"""Predeclared 50:50 development-OOF score blend; only after gated GPU results exist.

This is an exploratory ranking comparison, not fitted calibration or deployment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import task01_compare as comparison


KEYS = ["machine", "source_row_id", "feature_group", "outer_fold", "seed"]


def combine(cpu_dir: Path, gpu_dir: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    cpu_manifest = json.loads((cpu_dir / "run_manifest.json").read_text())
    gpu_manifest = json.loads((gpu_dir / "run_manifest.json").read_text())
    if cpu_manifest["device"] != "cpu" or gpu_manifest["device"] != "gpu":
        raise ValueError("Need completed CPU and GPU comparison runs")
    if gpu_manifest.get("gpu_receipt") is None:
        raise ValueError("GPU run has no verified task-04 recovery receipt")
    for key in ("frozen_manifest_sha256", "seeds", "development_rows", "development_groups"):
        if cpu_manifest[key] != gpu_manifest[key]:
            raise ValueError(f"CPU/GPU comparison basis differs: {key}")
    cpu = pd.read_csv(cpu_dir / "oof_predictions.csv")
    gpu = pd.read_csv(gpu_dir / "oof_predictions.csv")
    cpu = cpu.loc[cpu.model.eq("logistic_ref"), KEYS + ["label_value", "score"]]
    if len(cpu) != cpu_manifest["development_rows"] * len(cpu_manifest["seeds"]):
        raise ValueError("CPU reference coverage is incomplete")
    output_rows, metrics = [], []
    for candidate in comparison.GPU_MODELS:
        other = gpu.loc[gpu.model.eq(candidate), KEYS + ["label_value", "score"]]
        merged = cpu.merge(other, on=KEYS, how="outer", validate="one_to_one",
                           indicator=True, suffixes=("_cpu", "_gpu"))
        if len(merged) != len(cpu) or not merged._merge.eq("both").all():
            raise ValueError(f"Unmatched CPU/GPU OOF rows: {candidate}")
        if not merged.label_value_cpu.eq(merged.label_value_gpu).all():
            raise ValueError("Numeric labels differ")
        merged["score_50_50"] = (merged.score_cpu + merged.score_gpu) / 2
        merged["model"] = "logistic_ref_plus_" + candidate
        output_rows.append(merged.drop(columns=["_merge"]))
        for seed in comparison.SEEDS:
            seeded = merged.loc[merged.seed.eq(seed)]
            for machine in ("all", "cn7", "rg3"):
                section = seeded if machine == "all" else seeded.loc[seeded.machine.eq(machine)]
                metrics.append({"model": "logistic_ref_plus_" + candidate,
                                "seed": seed, "machine": machine,
                                **comparison.score(section.label_value_cpu.to_numpy(dtype=int),
                                                   section.score_50_50.to_numpy())})
    output.mkdir(parents=True)
    pd.concat(output_rows).to_csv(output / "blend_oof_predictions.csv", index=False)
    pd.DataFrame(metrics).to_csv(output / "blend_metrics.csv", index=False)
    comparison.write_json(output / "blend_manifest.json", {
        "status": "exploratory_development_oof_only", "ratio": "0.5 CPU logistic reference + 0.5 GPU candidate",
        "holdout_evaluated": False, "threshold_tuned": False,
        "cpu_manifest_sha256": comparison.mg.sha256(cpu_dir / "run_manifest.json"),
        "gpu_manifest_sha256": comparison.mg.sha256(gpu_dir / "run_manifest.json"),
        "caveat": "Raw weighted scores are not calibrated probabilities; blend is a prespecified ranking diagnostic only.",
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-dir", type=Path, required=True)
    parser.add_argument("--gpu-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not output.is_relative_to(comparison.ROOT / "outputs"):
        parser.error("Output must be a new directory below outputs/")
    combine(args.cpu_dir, args.gpu_dir, output)


if __name__ == "__main__":
    main()
