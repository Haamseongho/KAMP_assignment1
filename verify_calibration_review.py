"""Verify two saved research runs without refitting or overwriting published work."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import moldguard as mg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    first, second, output = args.first.resolve(), args.second.resolve(), args.output.resolve()
    if first == second:
        parser.error("Compare two separate execution directories")
    if not output.is_relative_to(mg.ROOT / "outputs") or output.exists():
        parser.error("Use a new receipt file below outputs/")
    first_files = {p.name for p in first.iterdir() if p.is_file()}
    second_files = {p.name for p in second.iterdir() if p.is_file()}
    expected = {"artifact_hashes.json", "calibration_review.json", "nested_oof_predictions.csv",
                "nested_split_manifest.csv", "reliability_bins.csv", "conditional_errors.csv",
                "calibrated_research_model.joblib"}
    if first_files != expected or second_files != expected:
        parser.error("Each run must contain exactly the seven expected research artifacts")
    checks = {}
    pairs = {}
    for name in sorted(expected):
        a, b = mg.sha256(first / name), mg.sha256(second / name)
        pairs[name] = {"first_sha256": a, "second_sha256": b, "identical": a == b}
        checks[f"repeated_run:{name}"] = a == b
    for index, folder in enumerate([first, second]):
        manifest = json.loads((folder / "artifact_hashes.json").read_text())
        checks[f"manifest_coverage:{index}"] = set(manifest) == expected - {"artifact_hashes.json"}
        for name, digest in manifest.items():
            if name not in expected:
                parser.error("Unexpected manifest filename")
            checks[f"manifest_hash:{index}:{name}"] = mg.sha256(folder / name) == digest
    report = json.loads((first / "calibration_review.json").read_text())
    for name in ["moldguard.py", "calibration_diagnostics.py"]:
        checks[f"source:{name}"] = mg.sha256(mg.ROOT / name) == report["source_hashes"][name]
    checks["source:frozen_split"] = mg.sha256(
        mg.ROOT / "outputs/local_runs/20260922-share-final/split_manifest.csv"
    ) == report["source_hashes"]["frozen_split_manifest"]
    reference = json.loads((mg.ROOT / "outputs/moldguard/reproducibility.json").read_text())
    for name, digest in reference["artifact_hashes"].items():
        checks[f"published_unchanged:{name}"] = mg.sha256(mg.ROOT / name) == digest
    publication = json.loads((mg.ROOT / "outputs/share/notion_publication.json").read_text())
    checks["published_zip_unchanged"] = mg.sha256(
        mg.ROOT / "outputs/share/KAMP_MoldGuard_20260922_share.zip"
    ) == publication["package_sha256"]
    command = [sys.executable, "-m", "pytest", "-q"]
    tests = subprocess.run(command, cwd=mg.ROOT, text=True, capture_output=True, check=False)
    checks["automated_tests"] = tests.returncode == 0
    receipt = {
        "status": "passed" if all(checks.values()) else "failed",
        "first_run": str(first.relative_to(mg.ROOT)),
        "second_run": str(second.relative_to(mg.ROOT)),
        "artifact_pairs": pairs,
        "checks": checks,
        "automated_tests": {"command": command, "exit_code": tests.returncode,
                            "stdout": tests.stdout, "stderr": tests.stderr},
        "verifier_sha256": mg.sha256(Path(__file__)),
        "scientific_scope": "Post-selection development-only review, not a new blind test or deployment approval",
        "notion_update_performed_by_this_verifier": False,
        "competition_website_submission": False,
    }
    with output.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(tests.stdout.strip())
    print(f"{receipt['status']}: {len(pairs)} repeated artifacts; {sum(checks.values())}/{len(checks)} checks")
    print(output)
    raise SystemExit(0 if receipt["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
