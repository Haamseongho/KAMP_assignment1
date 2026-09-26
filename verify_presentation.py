"""Compare native slide evidence against recorded experiments and check PDF export."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import pdfplumber
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent
NS = {'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
      'c': 'http://schemas.openxmlformats.org/drawingml/2006/chart'}


def read(name):
    return json.loads((ROOT / name).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', default='v2')
    parser.add_argument('--reviewed-pptx-sha256', required=True)
    parser.add_argument('--reviewed-pdf-sha256', required=True)
    args = parser.parse_args()
    if not args.revision.startswith('v') or not args.revision[1:].isdigit():
        parser.error('Use v plus a revision number')
    stem = f'KAMP_MoldGuard_발표검토본_20260922_{args.revision}'
    pptx, pdf = ROOT / 'output/presentation' / (stem + '.pptx'), ROOT / 'output/pdf' / (stem + '.pdf')
    assert sha(pptx) == args.reviewed_pptx_sha256
    assert sha(pdf) == args.reviewed_pdf_sha256
    manifest = read(f'tmp/presentations/kamp-{args.revision}/source_manifest.json')
    for name, expected in manifest['sources'].items():
        assert sha(ROOT / name) == expected, f'Changed source: {name}'
    assert sha(ROOT / 'build_presentation.mjs') == manifest['builder_sha256']
    exp, cal = read('outputs/moldguard/experiment.json'), read('outputs/calibration-review-20260922/calibration_review.json')
    checks, tables = [], {}
    with ZipFile(pptx) as archive:
        for number in range(1, 14):
            root = ET.fromstring(archive.read(f'ppt/slides/slide{number}.xml'))
            found = root.findall('.//a:tbl', NS)
            if number == 1:
                assert not found
            else:
                assert len(found) == 1
                tables[number] = [[''.join(t.text or '' for t in cell.findall('.//a:t', NS))
                                   for cell in row.findall('a:tc', NS)] for row in found[0].findall('a:tr', NS)]
            notes = ET.fromstring(archive.read(f'ppt/notesSlides/notesSlide{number}.xml'))
            assert '출처' in ''.join(t.text or '' for t in notes.findall('.//a:t', NS))
        expected = [[f"{m['average_precision']:.4f}", f"{m['top10_positives']}/{m['positives']}", f"{m['roc_auc']:.4f}"] for m in exp['models_oof_development'].values()]
        assert [row[1:] for row in tables[5][1:]] == expected
        checks.append('all_seven_model_comparison_rows_equal_recorded_development_metrics')
        with (ROOT / 'outputs/moldguard/holdout_diagnostics.csv').open() as handle:
            held = list(csv.DictReader(handle))
        for index, machine in enumerate([None, 'cn7', 'rg3'], 1):
            part = held if machine is None else [r for r in held if r['machine'] == machine]
            counts = [sum(int(r['actual_fail']) == a and int(r['predicted_at_diagnostic_threshold']) == b for r in part) for a,b in [(1,1),(1,0),(0,1),(0,0)]]
            assert tables[7][index][1:] == list(map(str, counts))
        checks.append('holdout_confusion_counts_recomputed_from_row_predictions')
        with (ROOT / 'outputs/calibration-review-20260922/conditional_errors.csv').open() as handle:
            joint = [r for r in csv.DictReader(handle) if r['machine'] == 'rg3' and r['condition'] == 'pressure_joint']
        for displayed, r in zip(tables[8][1:], joint, strict=True):
            expected = [f"{r['rows']} / {r['positive_rows']}", f"{r['tp']} / {r['fn']}",
                        f"{r['fp']} / {r['tn']}", f"{float(r['false_positive_rate']):.2%}"]
            assert displayed[1:] == expected
        checks.append('all_four_conditional_error_rows_equal_saved_nested_oof_analysis')
        for displayed, key in zip(tables[9][1:], ['prior','raw','sigmoid'], strict=True):
            m = cal['metrics']['all'][key]
            assert displayed[1:] == [f"{m['average_precision']:.4f}", f"{m['brier']:.6f}", f"{m['log_loss']:.6f}"]
        checks.append('calibration_metrics_equal_development_only_research')
        with (ROOT / 'outputs/moldguard/unlabeled_priority.csv').open() as handle:
            actions = Counter(r['action'] for r in csv.DictReader(handle))
        keys = ['usual_inspection_unvalidated_machine', 'usual_inspection_ood', 'usual_inspection_label_unconfirmed']
        assert [r[1] for r in tables[10][1:]] == [f'{actions[k]:,}' for k in keys]
        assert sum(actions.values()) == 71180
        checks.append('all_action_counts_equal_actual_71180_row_output')
        chart = ET.fromstring(archive.read('ppt/slides/charts/chart1.xml'))
        cached = [float(n.text) for n in chart.findall('.//c:ser/c:val/c:numRef/c:numCache/c:pt/c:v', NS)]
        actual = [exp['holdout'], exp['holdout_by_machine']['cn7'], exp['holdout_by_machine']['rg3']]
        assert cached == [round(m['average_precision'],4) for m in actual]
        checks.append('native_chart_values_equal_four_decimal_source_ap')
        assert 'ppt/embeddings/chart-data-snapshot-001.xlsx' in archive.namelist()
        slide_paragraphs = [[
            ''.join(t.text or '' for t in paragraph.findall('.//a:t', NS))
            for paragraph in ET.fromstring(archive.read(f'ppt/slides/slide{i}.xml')).findall('.//a:p', NS)
        ] for i in range(1,14)]
    reader = PdfReader(pdf)
    assert len(reader.pages) == 13
    pdf_text = [p.extract_text() or '' for p in reader.pages]
    for index, text in enumerate(pdf_text):
        assert any('\uac00' <= char <= '\ud7a3' for char in text), f'Korean text missing on page {index+1}'
        # Chart export changes object extraction order. Check each authored
        # shape/table paragraph, ignoring wrapping but not its actual content.
        normalize = lambda s: ''.join(s.split())
        for paragraph in slide_paragraphs[index]:
            assert normalize(paragraph) in normalize(text), f'Slide text mismatch on page {index+1}: {paragraph}'
    checks.append('pdf_preserves_all_authored_slide_text_including_korean')
    assert '/Users/' not in '\n'.join(pdf_text) and 'hamseongho' not in '\n'.join(pdf_text)
    with pdfplumber.open(pdf) as document:
        for index,page in enumerate(document.pages,1):
            outside = [c for c in page.chars if c['x0'] < 8 or c['x1'] > page.width-8 or c['top'] < 8 or c['bottom'] > page.height-8]
            assert not outside, f'PDF text outside margin on page {index}'
    checks.append('pdf_text_is_within_slide_margins')
    publication = read('outputs/share/notion_publication.json')
    assert sha(ROOT/'outputs/share/KAMP_MoldGuard_20260922_share.zip') == publication['package_sha256']
    receipt = {'status':'passed_artifact_qa_only', 'pptx_sha256':sha(pptx), 'pdf_sha256':sha(pdf),
               'slides':13, 'native_tables':12, 'native_charts':1, 'checks':checks,
               'visual_review':'All 13 v1 slides and font-corrected PDF pages inspected. Final v2 page 11 inspected in both renders; other 12 rendered pages were pixel-identical to inspected v1.',
               'font':'NanumGothic', 'pdf_font_correction':'Task-local FONTCONFIG_FILE, no shared runtime edits',
               'microsoft_powerpoint_application_tested':False, 'notion_published':False,
               'existing_share_zip_unchanged':True, 'field_deployment_approved':False,
               'competition_website_submission':False, 'verifier_sha256':sha(__file__)}
    output = ROOT/'reports'/f'presentation-qa-{args.revision}.json'
    output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
