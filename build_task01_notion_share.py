"""Build aggregate-only Notion attachment; exclude raw and row-level records."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import moldguard as mg


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs/share/KAMP_Task01_CPU_Results_20260923_notion_v2.zip"
FILES = [
    "KAMP_과제01_CPU_GPU_비교_실행가이드_20260923.md",
    "docs/08_과제01_CPU_비교_결과_20260923.md",
    "task01_compare.py", "task01_ensemble.py", "moldguard.py",
    "calibration_diagnostics.py", "requirements-compare.txt", "requirements-model-lock.txt",
    "tests/test_task01_compare.py",
    "outputs/task01_compare_20260923_cpu_v3/comparison_summary.csv",
    "outputs/task01_compare_20260923_cpu_v3/metrics_by_seed_machine.csv",
    "outputs/task01_compare_20260923_cpu_v3/metrics_by_fold.csv",
    "outputs/task01_compare_20260923_cpu_v3/condition_slices.csv",
    "outputs/task01_compare_20260923_cpu_v3/run_manifest.json",
    "outputs/task01_compare_20260923_cpu_v3.log",
]


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    files = {name: mg.sha256(ROOT / name) for name in FILES}
    manifest = {"scope": "Task-01 code, aggregate results and execution log for Notion",
                "contains_raw_csv_or_row_level_predictions": False,
                "gpu_executed": False, "files_sha256": files}
    with zipfile.ZipFile(OUT, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for name in FILES:
            bundle.write(ROOT / name, name)
        bundle.writestr("NOTION_SHARE_MANIFEST.json",
                        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    with zipfile.ZipFile(OUT) as bundle:
        if bundle.testzip() is not None:
            raise RuntimeError("Notion ZIP CRC check failed")
    print(json.dumps({"path": str(OUT), "sha256": mg.sha256(OUT),
                      "files": len(FILES)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
