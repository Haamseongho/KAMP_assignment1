"""Build an explicit, auditable sharing ZIP; exclude raw inputs and venvs."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUN = Path("outputs/local_runs/20260922-share-final")
QUICK = Path("outputs/local_runs/20260922-share-quick")
PORTABILITY = Path("outputs/local_runs/package-portability")
SHARE = Path("outputs/share")


def main() -> None:
    for run in [RUN, QUICK, PORTABILITY]:
        if json.loads((ROOT / run / "run_manifest.json").read_text())["status"] != "passed":
            raise RuntimeError("Refuse to package an unsuccessful final run")
    paths = {Path(p) for p in [
        "README.md", "로컬_실행_가이드.md", "공유_결과_요약.md",
        "moldguard.py", "analyze_moldguard.py", "run_local.py",
        "export_result_tables.py", "build_share_package.py", "kamp_prep.py",
        "requirements-model.txt", "requirements-model-lock.txt",
        "requirements.txt", "requirements-lock.txt", "config/task.template.json",
        "docs/03_사출성형_검증보고서.md", "docs/04_추가데이터_요청명세.md",
        "docs/05_라벨_의미_검증.md", "docs/06_현재_점검표.md", "docs/운영절차.md",
    ]}
    paths.update(p.relative_to(ROOT) for p in (ROOT / "tests").glob("*.py"))
    for name in ["data_audit.json", "experiment.json", "failure_analysis.json",
                 "holdout_diagnostics.csv", "unlabeled_priority.csv", "model.joblib",
                 "reproducibility.json", "pip-freeze.txt", "holdout_pr.png",
                 "inspection_capture.png"]:
        paths.add(Path("outputs/moldguard") / name)
    paths.update(p.relative_to(ROOT) for p in (ROOT / "outputs/moldguard").glob("false_*.csv"))
    for run in [RUN, QUICK, PORTABILITY]:
        paths.update(p.relative_to(ROOT) for p in (ROOT / run / "logs").glob("*.log"))
        paths.update([run / "run_manifest.json", run / "split_manifest.csv"])
    paths.add(RUN / "reproducibility.json")
    paths.add(PORTABILITY / "reproducibility.json")
    for name in ["01_model_comparison.png", "02_frozen_holdout.png",
                 "03_output_policy.png", "결과_비교표.md", "table_sources.json"]:
        paths.add(SHARE / name)
    files = []
    for rel in sorted(paths):
        path = ROOT / rel
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"Required regular file missing: {rel}")
        payload = path.read_bytes()
        files.append({"path": rel.as_posix(), "size": len(payload),
                      "sha256": hashlib.sha256(payload).hexdigest()})
    package_manifest = {"scope": "KAMP injection-molding analysis; no original inputs or venv",
                        "final_run": RUN.as_posix(), "files": files}
    output = ROOT / SHARE / "KAMP_MoldGuard_20260922_share.zip"
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for entry in files:
            archive.write(ROOT / entry["path"], entry["path"])
        archive.writestr("PACKAGE_MANIFEST.json", json.dumps(package_manifest, ensure_ascii=False, indent=2))
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        for entry in files:
            assert hashlib.sha256(archive.read(entry["path"])).hexdigest() == entry["sha256"]
    print(json.dumps({"zip": str(output), "files": len(files) + 1,
                      "bytes": output.stat().st_size,
                      "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
