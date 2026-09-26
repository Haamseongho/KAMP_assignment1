"""Render an aggregate-only comparison image from the frozen CPU result CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "outputs/task01_compare_20260923_cpu_v3/comparison_summary.csv"
OUTPUT = ROOT / "outputs/task01_compare_20260923_cpu_v3/comparison_table.png"
ORDER = ["logistic_ref", "xgb_regularized", "catboost_regularized",
         "lgbm_regularized", "xgb_shallow", "catboost_ref", "lgbm_shallow"]
LABELS = {"logistic_ref": "Logistic 기준선", "xgb_regularized": "XGBoost regularized",
          "catboost_regularized": "CatBoost regularized", "lgbm_regularized": "LightGBM regularized",
          "xgb_shallow": "XGBoost shallow", "catboost_ref": "CatBoost reference",
          "lgbm_shallow": "LightGBM shallow"}


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source = pd.read_csv(SOURCE)
    overall = source.loc[source.machine.eq("all")].set_index("model")
    rg3 = source.loc[source.machine.eq("rg3")].set_index("model")
    if set(overall.index) != set(ORDER) or set(rg3.index) != set(ORDER):
        raise ValueError("Unexpected model coverage")
    font_path = font_manager.findfont(font_manager.FontProperties(family="AppleGothic"))
    title_font = ImageFont.truetype(font_path, 36)
    header_font = ImageFont.truetype(font_path, 22)
    row_font = ImageFont.truetype(font_path, 21)
    foot_font = ImageFont.truetype(font_path, 17)
    canvas = Image.new("RGB", (1600, 740), "white")
    draw = ImageDraw.Draw(canvas)
    ink, muted, line = "#161b25", "#526071", "#d9dfe7"
    draw.text((48, 30), "KAMP 과제①  CPU 모델 비교", fill=ink, font=title_font)
    draw.text((49, 80), "고정 개발 5-fold OOF · 3 seed 평균 · numeric PassOrFail=1", fill=muted, font=foot_font)
    columns = [(48, "모델"), (484, "전체 AP 평균 ± SD"), (850, "상위 10% 회수율"),
               (1200, "RG3 AP"), (1400, "Brier")]
    draw.line((48, 131, 1552, 131), fill="#283a53", width=2)
    for x, title in columns:
        draw.text((x, 145), title, fill=ink, font=header_font)
    draw.line((48, 184, 1552, 184), fill=line, width=2)
    for index, model in enumerate(ORDER):
        y0 = 185 + index * 63
        if index == 0:
            draw.rectangle((48, y0 + 1, 1552, y0 + 62), fill="#eef4ff")
        values = [LABELS[model],
                  f"{overall.loc[model, 'average_precision_mean']:.4f} ± {overall.loc[model, 'average_precision_std']:.4f}",
                  f"{overall.loc[model, 'top10_recall_mean']:.4f}",
                  f"{rg3.loc[model, 'average_precision_mean']:.4f}",
                  f"{overall.loc[model, 'brier_unadjusted_mean']:.4f}"]
        for (x, _), value in zip(columns, values):
            draw.text((x, y0 + 17), value, fill=ink, font=row_font)
        draw.line((48, y0 + 63, 1552, y0 + 63), fill=line, width=1)
    draw.text((48, 652), "판정  전체 AP·seed 안정성: Logistic 기준선 유지. RG3·라벨 의미·보정 확률은 미검증.",
              fill=ink, font=foot_font)
    draw.text((48, 685), "출처  comparison_summary.csv · 과거 holdout 재사용 없음 · GPU 미실행", fill=muted, font=foot_font)
    canvas.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
