"""Document QA only: does not assert scientific validity or submission readiness."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pdfplumber
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent
PDF = ROOT / "output/pdf/KAMP_MoldGuard_검토보고서_20260922.pdf"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads((ROOT / path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-reviewed-sha256", required=True,
                        help="Hash of the exact PDF whose twelve rendered pages were manually inspected")
    args = parser.parse_args()
    pdf_hash = sha(PDF)
    assert args.visual_reviewed_sha256 == pdf_hash, "Re-render and inspect the changed PDF"
    provenance = load("output/pdf/report_provenance.json")
    assert pdf_hash == provenance["pdf_sha256"]
    for name, expected in provenance["source_sha256"].items():
        assert sha(ROOT / name) == expected, f"Source changed: {name}"
    assert sha(ROOT / "build_review_report.py") == provenance["builder_sha256"]
    reader = PdfReader(PDF)
    assert len(reader.pages) == 12 and len(reader.outline) == 12
    text = [page.extract_text() for page in reader.pages]
    for value in provenance["required_text_checks"]:
        assert value in "\n".join(text), value
    experiment = load("outputs/moldguard/experiment.json")
    for m in experiment["models_oof_development"].values():
        for value in [f"{m['average_precision']:.4f}", f"{m['roc_auc']:.4f}", f"{m['top10_positives']}/{m['positives']}"]:
            assert value in text[4], value
    for m in [experiment["holdout"], *experiment["holdout_by_machine"].values()]:
        for value in [f"{m['average_precision']:.4f}", f"{m['roc_auc']:.4f}", f"{m['top10_positives']}/{m['positives']}"]:
            assert value in text[5], value
    calibration = load("outputs/calibration-review-20260922/calibration_review.json")
    for m in calibration["metrics"]["all"].values():
        for value in [f"{m['brier']:.6f}", f"{m['log_loss']:.6f}"]:
            assert value in text[8], value
    for value in ["0.0424", "0.5556", "0.0787", "0.575557"]:
        assert value in text[5]
    all_text = "\n".join(text)
    assert "/Users/" not in all_text and "hamseongho" not in all_text
    assert reader.get_fields() is None
    assert all(page.get("/Annots") is None for page in reader.pages)
    assert reader.trailer["/Root"].get("/OpenAction") is None
    embedded_korean_fonts = []
    for name, ref in reader.pages[0]["/Resources"]["/Font"].items():
        font = ref.get_object()
        if "NanumGothic" in str(font.get("/BaseFont")):
            assert font["/FontDescriptor"].get("/FontFile2") is not None
            embedded_korean_fonts.append(str(font["/BaseFont"]))
    assert embedded_korean_fonts
    page_checks = []
    with pdfplumber.open(PDF) as document:
        for index, page in enumerate(document.pages, 1):
            outside = [c for c in page.chars if c["x0"] < 20 or c["x1"] > page.width - 20
                       or c["top"] < 12 or c["bottom"] > page.height - 12]
            assert not outside, f"Page {index} has text outside safe margins"
            page_checks.append({"page": index, "characters": len(page.chars), "outside_safe_margin": 0})
    publication = load("outputs/share/notion_publication.json")
    assert sha(ROOT / "outputs/share/KAMP_MoldGuard_20260922_share.zip") == publication["package_sha256"]
    receipt = {
        "status": "passed_document_qa_only", "pdf_sha256": pdf_hash, "pages": 12,
        "source_hashes_current": True, "numeric_table_text_checks": "passed",
        "bookmarks": 12, "embedded_korean_fonts": embedded_korean_fonts,
        "page_checks": page_checks,
        "visual_review": {"status": "passed", "pages_inspected": list(range(1, 13)),
                          "method": "Poppler PNG at 110 dpi; all pages inspected via view_image after final layout change",
                          "observations": "Korean glyphs, tables, spacing, section transitions and page numbers readable; no clipping or overlap observed",
                          "renderer_warning": "Fontconfig default configuration unavailable; embedded Korean fonts rendered correctly"},
        "published_zip_unchanged": True, "scientific_completion_proven": False,
        "official_submission_ready": False, "competition_website_submission": False,
        "verifier_sha256": sha(__file__),
    }
    (PDF.parent / "report_qa.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": receipt["status"], "pages": 12, "pdf_sha256": pdf_hash}, indent=2))


if __name__ == "__main__":
    main()
