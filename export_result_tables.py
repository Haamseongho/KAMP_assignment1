"""Export shareable result tables from recorded experiment outputs (no new fitting)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import pandas as pd

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=ROOT / "outputs/moldguard")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/share")
    parser.add_argument("--font-file", type=Path, default=Path("/System/Library/Fonts/AppleSDGothicNeo.ttc"))
    args = parser.parse_args()
    if not args.font_file.is_file():
        parser.error("Use --font-file with an installed Korean font (.ttf/.ttc/.otf)")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = args.source_dir
    experiment = json.loads((source / "experiment.json").read_text())
    held = pd.read_csv(source / "holdout_diagnostics.csv")
    priorities = pd.read_csv(source / "unlabeled_priority.csv")
    font = FontProperties(fname=str(args.font_file))
    labels = {
        "dummy": "B0  Dummy", "logistic": "B1  가중 로지스틱  [선정]",
        "forest": "B2  Random Forest", "catboost": "B3  CatBoost",
        "separate_logistic": "B4  설비별 로지스틱", "separate_forest": "B5  설비별 Forest",
        "separate_catboost": "B6  설비별 CatBoost",
    }
    development = [[labels[name], f"{metric['average_precision']:.4f}",
                    f"{metric['top10_positives']}/{metric['positives']}", f"{metric['roc_auc']:.4f}"]
                   for name, metric in experiment["models_oof_development"].items()]
    holdout = []
    for label, machine in [("전체", None), ("CN7", "cn7"), ("RG3", "rg3")]:
        part = held if machine is None else held[held.machine.eq(machine)]
        y, pred = part.actual_fail.eq(1), part.predicted_at_diagnostic_threshold.eq(1)
        tp, fn, fp, tn = int((y & pred).sum()), int((y & ~pred).sum()), int((~y & pred).sum()), int((~y & ~pred).sum())
        metric = experiment["holdout"] if machine is None else experiment["holdout_by_machine"][machine]
        holdout.append([f"{label}  ({len(part)}행)", f"{metric['average_precision']:.4f} / {metric['roc_auc']:.4f}",
                        f"{metric['top10_positives']}/{metric['positives']}", f"{tp} / {fn} / {fp} / {tn}"])
    policy = []
    for action, label, meaning in [
        ("usual_inspection_unvalidated_machine", "RG3 전체", "기존 검사 유지 · 성능 미검증"),
        ("usual_inspection_ood", "CN7 분포 범위 밖", "기존 검사 유지 · OOD"),
        ("usual_inspection_label_unconfirmed", "CN7 나머지", "기존 검사 유지 · 라벨 미확정"),
    ]:
        part = priorities[priorities.action.eq(action)]
        counts = [int(part.priority.eq(tier).sum()) for tier in ["HIGH", "REVIEW", "STANDARD"]]
        policy.append([label, f"{len(part):,}", " / ".join(f"{n:,}" for n in counts), meaning])
    assert sum(int(row[1].replace(",", "")) for row in policy) == len(priorities)

    tables = [
        {"filename": "01_model_comparison.png", "title": "KAMP 7개 모델 · 개발 검증 비교",
         "subtitle": "seed=20260922 · 동일 공정값 그룹 5-fold OOF · 개발 1,913행 / 라벨 1 총 33행",
         "headers": ["모델", "개발 OOF AP", "Top 10% 발견", "ROC-AUC"], "rows": development,
         "widths": [0.42, 0.19, 0.21, 0.18], "highlight": 1,
         "notes": ["모든 모델에 같은 개발 분할 적용. AP가 가장 높은 가중 로지스틱을 선정.",
                   "발견 = 라벨 1 발견 수 / 전체 라벨 1 수. 불량 의미는 1=불량 가정에 조건부."]},
        {"filename": "02_frozen_holdout.png", "title": "KAMP 선정 모델 · 고정 홀드아웃 결과",
         "subtitle": "가중 로지스틱 · 홀드아웃 480행 / 라벨 1 총 9행 · 진단 임계값은 개발 OOF에서 결정",
         "headers": ["평가 대상", "AP / ROC-AUC", "Top 10% 발견", "TP / FN / FP / TN"], "rows": holdout,
         "widths": [0.24, 0.26, 0.20, 0.30], "highlight": None,
         "notes": ["전체 Top 10%는 통합 점수순 48행(3/9). 설비별 10%를 합치면 49행(4/9).",
                   "RG3 발견 0/5. AP 95% 그룹 부트스트랩 구간: 전체 0.013–0.629.",
                   "첫 평가 후 탐색에 사용된 홀드아웃이며 새로운 blind 평가 결과가 아님. 라벨 의미 미확정."]},
        {"filename": "03_output_policy.png", "title": "KAMP 추론 결과 · 71,180행 처리 현황",
         "subtitle": "실제 비라벨 CSV 전량 · 모델 순위는 오프라인 참고값 · 정답이 없어 예측 성능은 산출 불가",
         "headers": ["대상", "행 수", "HIGH / REVIEW / STANDARD", "현재 action 해석"], "rows": policy,
         "widths": [0.22, 0.14, 0.30, 0.34], "highlight": None,
         "notes": ["모든 행의 action은 기존 검사 유지. 이 표는 실제 Reject / Reinspect / Pass 판정이 아님.",
                   "HIGH·REVIEW·STANDARD는 각 설비 내 점수순 상위 10% / 다음 10% / 나머지."]},
    ]
    md = ["# KAMP 결과 비교표", "", "2026-09-22 · 실제 저장된 실험 결과에서 자동 추출. 라벨 1을 불량으로 보는 작업 가정에 조건부.", ""]
    for table in tables:
        height = 2.6 + 0.54 * len(table["rows"]) + 0.23 * len(table["notes"])
        fig = plt.figure(figsize=(14, height), dpi=160, facecolor="#FFFFFF")
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, height)
        ax.axis("off")
        def text(x, y, value, size=14, weight="normal", color="#15191F", ha="left"):
            ax.text(x, y, value, fontproperties=font, fontsize=size, fontweight=weight,
                    color=color, ha=ha, va="center")
        text(0.035, height - 0.40, table["title"], 23, "bold")
        text(0.035, height - 0.83, table["subtitle"], 12, color="#5C6571")
        boundaries = [0.035]
        for width in table["widths"]:
            boundaries.append(boundaries[-1] + 0.93 * width)
        top = height - 1.39
        for col, name in enumerate(table["headers"]):
            text(boundaries[col] + 0.008, top, name, 14, "bold")
        ax.plot([0.035, 0.965], [top - 0.24, top - 0.24], color="#AEB5BF", linewidth=0.8)
        for row_number, row in enumerate(table["rows"]):
            y = top - 0.55 - row_number * 0.54
            selected = row_number == table["highlight"]
            if selected:
                ax.add_patch(plt.Rectangle((0.035, y - 0.25), 0.93, 0.52, color="#EFF5FF", zorder=0))
            for col, value in enumerate(row):
                text(boundaries[col] + 0.008, y, value, 14, "bold" if selected or col > 0 else "normal")
            ax.plot([0.035, 0.965], [y - 0.27, y - 0.27], color="#E7E9ED", linewidth=0.6)
        for index, note in enumerate(table["notes"]):
            text(0.035, 0.38 + (len(table["notes"]) - 1 - index) * 0.25, note, 11, color="#5C6571")
        fig.savefig(args.output_dir / table["filename"], dpi=160, facecolor=fig.get_facecolor())
        plt.close(fig)
        md.extend([f"## {table['title']}", "", table["subtitle"], "",
                   "| " + " | ".join(table["headers"]) + " |",
                   "| " + " | ".join(["---"] * len(table["headers"])) + " |"])
        md.extend("| " + " | ".join(row) + " |" for row in table["rows"])
        md.extend(["", *table["notes"], ""])
    (args.output_dir / "결과_비교표.md").write_text("\n".join(md), encoding="utf-8")
    manifest = {
        "source_sha256": {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                          for name in ["experiment.json", "holdout_diagnostics.csv", "unlabeled_priority.csv"]},
        "tables": tables,
    }
    (args.output_dir / "table_sources.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"3 PNG tables, Markdown, and source manifest: {args.output_dir}")


if __name__ == "__main__":
    main()
