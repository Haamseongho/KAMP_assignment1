"""Create a non-overwriting add-on ZIP without changing the published base ZIP."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'outputs/share/KAMP_MoldGuard_20260922_share.zip'
BASE_SHA = '1a879c41bfd6d7b832656855be174b715bb52004105987136c2e2f476be4f6d5'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert sha(BASE) == BASE_SHA
    verify = json.loads((ROOT / 'outputs/calibration-review-verification.json').read_text())
    assert verify['status'] == 'passed' and all(verify['checks'].values())
    deck_qa = json.loads((ROOT / 'reports/presentation-qa-v2.json').read_text())
    assert deck_qa['status'] == 'passed_artifact_qa_only'
    paths = [
        'README.md', '보완_공유_안내.md', '결과보고서_읽는법.md', '발표자료_읽는법.md',
        'calibration_diagnostics.py', 'verify_calibration_review.py',
        'tests/test_calibration_diagnostics.py', 'build_supplement_package.py',
        'build_review_report.py', 'verify_review_report.py', 'build_presentation.mjs',
        'export_presentation_pdf.py', 'verify_presentation.py',
        'docs/07_확률보정_및_조건부_오류분석.md',
        'outputs/calibration-review-20260922.log', 'outputs/calibration-review-20260922-repro.log',
        'outputs/calibration-review-tests.log', 'outputs/calibration-review-verification.json',
        'output/pdf/KAMP_MoldGuard_검토보고서_20260922.pdf',
        'output/pdf/KAMP_MoldGuard_발표검토본_20260922_v2.pdf',
        'output/presentation/KAMP_MoldGuard_발표검토본_20260922_v2.pptx',
        'output/pdf/report_provenance.json', 'output/pdf/report_qa.json',
        'reports/presentation-qa-v2.json',
        'tmp/presentations/kamp-v2/source_manifest.json',
    ]
    for folder in ['outputs/calibration-review-20260922', 'outputs/calibration-review-20260922-repro']:
        for name in ['artifact_hashes.json', 'calibrated_research_model.joblib', 'calibration_review.json',
                     'conditional_errors.csv', 'nested_oof_predictions.csv', 'nested_split_manifest.csv',
                     'reliability_bins.csv']:
            paths.append(f'{folder}/{name}')
    assert sha(ROOT / paths[19]) == deck_qa['pdf_sha256']
    assert sha(ROOT / paths[20]) == deck_qa['pptx_sha256']
    entries = []
    for name in sorted(paths):
        p = ROOT / name
        assert p.is_file() and not p.is_symlink(), name
        entries.append({'path': name, 'bytes': p.stat().st_size, 'sha256': sha(p)})
    manifest = {'scope': 'Research and documents supplement; not final competition submission',
                'base_zip': BASE.name, 'base_sha256': BASE_SHA, 'raw_data_included': False,
                'virtual_environment_included': False, 'files': entries}
    output = ROOT / 'outputs/share/KAMP_MoldGuard_20260922_supplement_v1.zip'
    with ZipFile(output, 'x', ZIP_DEFLATED, compresslevel=9) as archive:
        for e in entries:
            archive.write(ROOT / e['path'], e['path'])
        archive.writestr('SUPPLEMENT_MANIFEST.json', json.dumps(manifest, ensure_ascii=False, indent=2))
    with ZipFile(output) as archive:
        assert archive.testzip() is None
        for e in entries:
            assert hashlib.sha256(archive.read(e['path'])).hexdigest() == e['sha256']
    assert sha(BASE) == BASE_SHA
    receipt = {'zip': output.relative_to(ROOT).as_posix(), 'sha256': sha(output),
               'bytes': output.stat().st_size, 'entries': len(entries) + 1,
               'base_sha256_unchanged': BASE_SHA, 'manifest_verified': True,
               'notion_publication_record': 'outputs/share/supplement_publication.json'}
    (ROOT / 'outputs/share/supplement_package.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
