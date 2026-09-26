"""Create a versioned, raw-data-free CPU/GPU comparison handoff ZIP."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import moldguard as mg


ROOT = Path(__file__).resolve().parent
RUN = Path("outputs/task01_compare_20260923_cpu_v3")
FILES = [
    Path("KAMP_과제01_CPU_GPU_비교_실행가이드_20260923.md"),
    Path("docs/08_과제01_CPU_비교_결과_20260923.md"),
    Path("requirements-model-lock.txt"), Path("requirements-compare.txt"),
    Path("moldguard.py"), Path("calibration_diagnostics.py"),
    Path("task01_compare.py"), Path("task01_ensemble.py"),
    Path("tests/test_task01_compare.py"),
    Path("outputs/local_runs/20260922-share-final/split_manifest.csv"),
    Path("outputs/local_runs/20260922-share-final/run_manifest.json"),
    Path("outputs/task01_compare_20260923_cpu_v3.log"),
] + [path.relative_to(ROOT) for path in sorted((ROOT / RUN).iterdir()) if path.is_file()]


def build(output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Refuse to overwrite: {output}")
    if not output.parent.is_dir():
        raise FileNotFoundError(output.parent)
    hashes = {}
    for relative in FILES:
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        hashes[str(relative)] = mg.sha256(source)
    frozen = json.loads((ROOT / RUN / "run_manifest.json").read_text())
    if frozen["code_sha256"] != hashes["task01_compare.py"]:
        raise ValueError("CPU run does not match current task01_compare.py")
    if frozen["frozen_manifest_sha256"] != hashes[
        "outputs/local_runs/20260922-share-final/split_manifest.csv"]:
        raise ValueError("CPU run does not match frozen split")
    manifest = {"scope": "Task-01 CPU comparison and gated future GPU package",
                "contains_raw_source_csv": False, "gpu_run_included": False,
                "cpu_run": str(RUN), "files_sha256": hashes}
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for relative in FILES:
            bundle.write(ROOT / relative, arcname=str(relative))
        bundle.writestr("TASK01_PACKAGE_MANIFEST.json",
                        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    with zipfile.ZipFile(output) as bundle:
        if bundle.testzip() is not None:
            raise RuntimeError("Package ZIP CRC verification failed")
    return {"zip": str(output), "sha256": mg.sha256(output), "files": len(FILES)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.output.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
