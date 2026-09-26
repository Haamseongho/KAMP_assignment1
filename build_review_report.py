"""Render a source-linked Korean review PDF; never fit a model or submit anything.

Run with the bundled document Python (reportlab and pypdf). Model dependencies
stay isolated in .venv-repro. Numeric tables are read from frozen actual outputs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
from xml.sax.saxutils import escape

import reportlab
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Paragraph, Spacer, Table, TableStyle
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output/pdf/KAMP_MoldGuard_검토보고서_20260922.pdf"
W, H = A4
LEFT, RIGHT, TOP, BOTTOM = 43, 43, 80, 49
WIDTH = W - LEFT - RIGHT
NAVY, TEAL = colors.HexColor("#17334A"), colors.HexColor("#0D736D")
INK, MUTED = colors.HexColor("#172A39"), colors.HexColor("#586B77")
PALE, LINE = colors.HexColor("#EFF5F7"), colors.HexColor("#D5E0E5")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(relative):
    return json.loads((ROOT / relative).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font-file", type=Path, default=Path("/Users/hamseongho/Library/Fonts/NanumGothic.ttf"))
    parser.add_argument("--replace", action="store_true", help="Replace this builder's named review PDF only")
    args = parser.parse_args()
    if OUTPUT.exists() and not args.replace:
        parser.error("Review PDF already exists; inspect it first, then use --replace for a revision")
    pdfmetrics.registerFont(TTFont("Korean", str(args.font_file)))
    style = ParagraphStyle("body", fontName="Korean", fontSize=10.3, leading=16,
                           textColor=INK, wordWrap="CJK", spaceAfter=8)
    styles = {
        "body": style,
        "small": ParagraphStyle("small", parent=style, fontSize=8.6, leading=12.7, textColor=MUTED),
        "section": ParagraphStyle("section", parent=style, fontSize=13.1, leading=20, textColor=TEAL,
                                  spaceBefore=9, spaceAfter=9),
        "cell": ParagraphStyle("cell", parent=style, fontSize=9.1, leading=13.5, spaceAfter=0, wordWrap=None),
        "head": ParagraphStyle("head", parent=style, fontSize=9.1, leading=13.5, textColor=colors.white, spaceAfter=0),
        "title": ParagraphStyle("title", parent=style, fontSize=25, leading=36, textColor=NAVY, spaceAfter=14),
        "code": ParagraphStyle("code", parent=style, fontName="Courier", fontSize=8.5, leading=13,
                               backColor=PALE, borderPadding=8, spaceBefore=5, spaceAfter=10),
    }

    def p(value, kind="body"):
        return Paragraph(escape(str(value)).replace("\n", "<br/>"), styles[kind])

    def table(headers, rows, fractions=None):
        widths = [WIDTH * n / sum(fractions) for n in fractions] if fractions else [WIDTH / len(headers)] * len(headers)
        cells = [[p(x, "head") for x in headers]] + [[p(x, "cell") for x in row] for row in rows]
        item = Table(cells, colWidths=widths, hAlign="LEFT", repeatRows=1, spaceAfter=8)
        item.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), NAVY), ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
            ("LINEBELOW", (0, 0), (-1, -1), 0.35, LINE),
        ]))
        return item

    def s(title):
        return p(title, "section")

    def source(text):
        return p("근거: " + text, "small")

    exp = read_json("outputs/moldguard/experiment.json")
    audit = read_json("outputs/moldguard/data_audit.json")
    fail = read_json("outputs/moldguard/failure_analysis.json")
    cal = read_json("outputs/calibration-review-20260922/calibration_review.json")
    verification = read_json("outputs/calibration-review-verification.json")
    run = read_json("outputs/local_runs/20260922-share-final/run_manifest.json")
    repro = read_json("outputs/moldguard/reproducibility.json")
    if verification["status"] != "passed" or not all(verification["checks"].values()):
        raise ValueError("The calibration verification receipt is not passing")
    for name, sha in repro["artifact_hashes"].items():
        if digest(ROOT / name) != sha:
            raise ValueError(f"Frozen artifact changed: {name}")
    for name, pair in verification["artifact_pairs"].items():
        if digest(ROOT / verification["first_run"] / name) != pair["first_sha256"]:
            raise ValueError(f"Research artifact changed: {name}")

    cm = exp["holdout"]["confusion_at_diagnostic_threshold"]
    precision = cm["tp"] / (cm["tp"] + cm["fp"])
    recall = cm["tp"] / (cm["tp"] + cm["fn"])
    f1 = 2 * precision * recall / (precision + recall)
    with (ROOT / "outputs/moldguard/holdout_diagnostics.csv").open() as handle:
        held = list(csv.DictReader(handle))
    with (ROOT / "outputs/calibration-review-20260922/conditional_errors.csv").open() as handle:
        errors = list(csv.DictReader(handle))
    pages = []

    pages.append(("MOLDGUARD / REVIEW REPORT", [
        Spacer(1, 14), p("사출성형 공정데이터 기반\n품질위험 예측과 검사 우선순위", "title"),
        p("제6회 K-인공지능 제조데이터 분석 경진대회\n일반국민 / 대학(원)생 부문 · 2026-09-22 검토본", "small"),
        s("요약: 재현 가능한 분석, 현장 적용은 미승인"),
        p(f"실제 라벨 {audit['labeled_rows']:,}행과 비라벨 {audit['unlabeled_rows']:,}행을 진단하고, 동일 공정값 그룹이 학습·평가에 겹치지 않도록 고정했다. 7개 후보를 같은 개발 검증으로 비교해 가중 로지스틱을 선정했다."),
        table(["핵심 결과", "해석"], [
            [f"개발 AP {exp['models_oof_development']['logistic']['average_precision']:.4f}", "발생률 기준선과 비선형 모델을 포함한 7개 후보 중 최고 AP"],
            [f"홀드아웃 AP {exp['holdout']['average_precision']:.4f}", "양성 9행에 불과함. CN7 / RG3를 나누어 해석해야 함"],
            ["RG3 Top 10% 발견 0/5", "RG3 활용 근거 부족. 기존 검사 유지"],
            ["추가 확률보정·오류 분석", "개발 영역의 중첩 그룹 검증. 기존 모델을 교체하지 않음"],
        ], [0.38, 0.62]),
        Spacer(1, 10),
        p("본 보고서의 양성·발견·오류는 숫자 라벨 PassOrFail=1 기준이다. 1=불량이라는 의미, 사전 정규화 범위, 최종검사 전 변수 가용성은 아직 확인되지 않았다. 모델 점수는 검사 생략이나 설비 자동제어의 근거가 아니다."),
        table(["문서 상태", "확인 항목"], [
            ["6개 장 내용구조에 맞춘 검토용 편집본", "공식 HWPX 원본 서식을 채운 최종 제출본은 아님"],
            ["팀명·팀원·서명", "사용자 확인 전. 임의 이름·서명을 넣지 않음"],
            ["설문 완료 화면", "미첨부. 설문 대리 응답이나 사이트 제출을 하지 않음"],
        ], [0.48, 0.52]),
    ]))

    files = [[name, f"{info['rows']:,}", str(audit['by_machine']['cn7' if 'cn7' in name else 'rg3']['positive_rows']) if 'labeled_' in name and 'unlabeled' not in name else "정답 없음"] for name, info in audit['source'].items()]
    pages.append(("제1장. 데이터 이해 및 진단 [15점]", [
        s("1.1 분석 단위와 예측 시점"),
        p("입력은 CN7·RG3 설비의 공정 요약값과 검사 라벨이다. Cycle_Time 등 샷 전체 요약이 있으므로 목표 시점은 공정 종료 후·최종검사 전이다. 제작 전 예측이나 진행 중 조기경보를 구현했다고 주장하지 않는다. 실제 샷·제품·캐비티 연결 키가 없어 현재 검증 단위는 CSV 행과 동일 관측값 그룹이다."),
        table(["실제 입력 파일", "행 수", "라벨 1"], files, [0.65, 0.17, 0.18]),
        s("1.2 변수군과 제조공정의 연결"),
        table(["변수군", "주요 열", "분석상 역할 / 제한"], [
            ["시간", "Injection_Time, Filling_Time, Plasticizing_Time, Cycle_Time, Clamp_Close_Time", "주입·충전·가소화·사이클·형체 시간의 요약으로 해석. 정확한 계측 정의·도착 시점 확인 필요"],
            ["위치·속도·회전", "Cushion_Position, Plasticizing_Position, Clamp_Open_Position, Max_Injection_Speed, Max/Average_Screw_RPM", "위치·동작 강도 관련 공정 상태. Clamp_Open_Position은 상수로 제외"],
            ["압력", "Max_Injection_Pressure, Max_Switch_Over_Pressure, Max/Average_Back_Pressure", "주입·전환·배압 관련 변수. 원시 단위와 물리적 안전 범위 미확인"],
            ["온도", "Barrel_Temperature_1~6, Hopper_Temperature, Mold_Temperature_3~4", "가열·금형 온도 관련 상태. 표준화된 상대값이며 실측 단위는 미제공"],
        ], [0.16, 0.45, 0.39]),
        p("위 의미 연결은 열 이름에 근거한 분석상 해석이며 공식 계측 코드북을 대체하지 않는다. 모델 입력은 공정 23개 + 설비 표시 1개다. 정답 PassOrFail과 출처 인덱스 Unnamed: 0은 제외했다.", "small"),
        source("data_audit.json / 공식 과제공개 HWPX의 사출성형 문제"),
    ]))

    pages.append(("제1장. 데이터 이해 및 진단 / 계속", [
        s("1.3 데이터 품질이 성능 해석을 제한한다"),
        table(["진단", "실제 관측", "처리와 의미"], [
            ["결측·스키마", "4개 파일의 필수 열·열 순서·유한 수치·파일 내 ID 검사 통과", "결측 대체를 만들지 않음. 잘못된 입력은 실행 실패로 표시"],
            ["불균형", f"라벨 1: {audit['positive_rows']}/{audit['labeled_rows']} ({audit['positive_prevalence']:.3%})", "정확도보다 AP·Top-K 발견·FN/FP를 보고함. 1=불량 의미는 미확정"],
            ["반복 관측", f"동일 공정값 {audit['identical_feature_groups']:,}그룹", "그룹 전체를 한쪽 분할에 배정. 같은 값의 행을 임의 삭제하지 않음"],
            ["상충 라벨", f"{audit['ambiguous_groups']}그룹 / 라벨 1 중 {audit['ambiguous_positive_rows']}행", "같은 입력을 가진 0/1을 현재 센서값만으로 구별할 수 없음"],
            ["정규화", "각 파일의 변동 열 평균 ≈0, 모표준편차 ≈1", "파일별 정규화가 의심됨. 원본 변환기와 적합 범위를 확인할 수 없음"],
            ["시간·생산 키", "실제 시각·샷/제품/캐비티 관계 없음", "인덱스를 시간으로 단정하지 않음. 완전한 미래 누수 차단은 미입증"],
        ], [0.18, 0.36, 0.46]),
        s("1.4 상충 라벨을 오류라고 단정하지 않는 이유"),
        p("한 샷의 공정값이 여러 캐비티·제품에 연결되었거나 검사 단위가 다를 수 있다. 현재 데이터에서 동일 입력만 사용하는 결정적 분류기는 최소 36행의 라벨을 동시에 맞힐 수 없다. 이는 관측 데이터에 한정된 제약이며, 전체 공정의 이론적 오차 하한이라고 일반화하지 않는다."),
        s("1.5 라벨 의미와 원시 단위의 확인 조건"),
        p("외부 가이드북 사본에 서로 다른 0/1 정의가 있어 현재 CSV 버전에 대입할 수 없다. 파일명·SHA-256에 대응하는 코드북, 변환 이력, 실제 검사 결과 연결이 필요하다. 확인 전 모든 결과를 숫자 라벨 1 예측으로 제한하며, 비율만 보고 불량 방향을 확정하지 않는다."),
        source("data_audit.json / docs/04_추가데이터_요청명세.md / docs/05_라벨_의미_검증.md"),
    ]))

    pages.append(("제2장. AI 예측모델 개발 및 성능평가 [40점]", [
        s("2.1 동일 조건 비교를 위한 고정 분할"),
        table(["영역", "그룹 / 행 / 라벨 1", "사용 목적"], [
            ["개발", "957 / 1,913 / 33", "내부 5-fold 그룹 OOF로 7개 후보 비교·모델 선정·진단 임계값 결정"],
            ["홀드아웃", "240 / 480 / 9", "선정 뒤 첫 평가. 이후 탐색에 노출되어 새 모델의 미사용 평가셋은 아님"],
        ], [0.17, 0.30, 0.53]),
        p("seed=20260922. 같은 설비·24개 공정값 벡터를 그룹으로 묶고 machine × 그룹 내 라벨 1 존재 여부로 층화했다. 개발/홀드아웃 간 동일 입력 그룹 교차는 0이다. 개발 내부 OOF도 동일 그룹을 분리하지 않는다."),
        s("2.2 학습·선정 기준"),
        p("Dummy 발생률 기준선, 균형 가중 로지스틱, Random Forest, CatBoost 및 설비별 3개 변형을 비교했다. 로지스틱의 표준화는 각 학습 fold 안에서 적합한다. 모든 후보가 같은 개발 분할·입력·평가 지표를 사용한다. 최종 선정 기준은 개발 OOF AP이며 홀드아웃 수치로 후보를 다시 선택하지 않았다."),
        p("AP는 양성을 점수 상단에 배치하는 능력, Top 10/20% 발견은 제한된 검사량의 정적 발견율을 나타낸다. ROC-AUC, 진단 임계값의 F1·정밀도·재현율과 혼동행렬을 함께 기록한다. 양성 희소성을 가리는 전체 정확도는 주요 성과로 삼지 않는다."),
        s("2.3 고정됐지만 독립 미래 검증은 아니다"),
        table(["확인된 차단", "아직 확인되지 않은 항목"], [
            ["동일 관측값의 분할 교차 없음", "실제 같은 샷·제품·캐비티가 서로 다른 공정값으로 등장하는지"],
            ["라벨·원본 ID를 특징에서 제외", "센서값이 실제 예측 시점 전에 시스템에 도착하는지"],
            ["모델 내부 전처리의 fold 분리", "제공 CSV를 만들 때의 전체 파일 정규화 범위"],
            ["최초 선정 뒤 홀드아웃 평가", "후속 탐색 이후 설계된 새 모델의 독립 평가 영역"],
        ], [0.44, 0.56]),
        source("split_manifest.csv / experiment.json / moldguard.py"),
    ]))

    labels = {"dummy": "B0 발생률 기준선", "logistic": "B1 가중 로지스틱 (선정)", "forest": "B2 Random Forest", "catboost": "B3 CatBoost", "separate_logistic": "B4 설비별 로지스틱", "separate_forest": "B5 설비별 Forest", "separate_catboost": "B6 설비별 CatBoost"}
    model_rows = [[labels[name], f"{m['average_precision']:.4f}", f"{m['top10_positives']}/{m['positives']}", f"{m['roc_auc']:.4f}"] for name, m in exp['models_oof_development'].items()]
    machine_rows = []
    for name in ["logistic", "forest", "catboost", "separate_catboost"]:
        m = exp['models_oof_development'][name]['by_machine']
        machine_rows.append([labels[name], f"{m['cn7']['average_precision']:.4f}", f"{m['rg3']['average_precision']:.4f}"])
    pages.append(("제2장. 개발 검증과 모델 선정 / 계속", [
        s("2.4 개발 1,913행의 OOF 비교"),
        table(["모델", "AP", "Top 10% 발견", "ROC-AUC"], model_rows, [0.46, 0.16, 0.22, 0.16]),
        p("발견 수의 분모는 개발 라벨 1 총 33행이며 검사 대상은 상위 192행이다. 발생률 기준선의 Top-K는 동점 처리에 영향을 받으므로 유효한 순위 능력으로 해석하지 않는다.", "small"),
        s("2.5 강한 후보를 비교했지만 복잡도가 승리를 보장하지 않았다"),
        p("가중 로지스틱의 AP가 0.1154로 가장 높았다. 설비별 CatBoost는 Top 10% 발견 14/33으로 더 높았지만 선정 기준 AP는 0.0983이므로 최종 모델을 바꾸지 않았다. Forest·CatBoost를 강한 비선형 후보로 시험했다는 것과 우수한 최종 현장 모델을 확보했다는 것은 별개다."),
        table(["대표 후보", "CN7 개발 AP", "RG3 개발 AP"], machine_rows, [0.52, 0.24, 0.24]),
        p("선정 로지스틱의 RG3 AP는 RG3 양성률 0.0211에 가깝다. RG3 문제는 홀드아웃만의 우연이라고 볼 수 없다. 설비별 모델의 수치도 동일한 개발 OOF에서 계산했으며, 위 표는 최종 홀드아웃을 재평가한 비교가 아니다."),
        source("experiment.json > models_oof_development; 원본 수치에서 자동 추출"),
    ]))

    holdout_rows, confusion_rows = [], []
    for label, machine in [("전체", None), ("CN7", "cn7"), ("RG3", "rg3")]:
        metric = exp['holdout'] if machine is None else exp['holdout_by_machine'][machine]
        holdout_rows.append([label, f"{metric['rows']} / {metric['positives']}", f"{metric['average_precision']:.4f}", f"{metric['roc_auc']:.4f}", f"{metric['top10_positives']}/{metric['positives']}"])
        part = held if machine is None else [r for r in held if r['machine'] == machine]
        counts = [sum(int(r['actual_fail']) == a and int(r['predicted_at_diagnostic_threshold']) == b for r in part) for a, b in [(1, 1), (1, 0), (0, 1), (0, 0)]]
        confusion_rows.append([label, *map(str, counts)])
    interval = fail['holdout_95pct_group_bootstrap_interval']['average_precision']
    pages.append(("제2장. 선정 모델의 홀드아웃 결과 / 계속", [
        s("2.6 전체 평균 뒤에 숨은 설비별 차이"),
        table(["대상", "행 / 양성", "AP", "ROC-AUC", "Top 10% 발견"], holdout_rows, [0.14, 0.23, 0.18, 0.20, 0.25]),
        p("전체 Top 10%는 통합 점수순 48행에서 3/9를 발견한 결과다. 설비별 Top 10%를 합치면 49행에서 4/9를 발견한다. 두 정책은 검사량과 대상이 다르므로 수치를 혼용하지 않는다. CN7의 4/4는 표본 4행의 결과이지 100% 미래 검출 보증이 아니다."),
        s("2.7 진단 임계값과 오류의 비용"),
        table(["대상", "TP", "FN", "FP", "TN"], confusion_rows, [0.24, 0.19, 0.19, 0.19, 0.19]),
        p(f"진단 임계값 {exp['holdout']['diagnostic_f2_threshold_from_oof']:.6f}는 개발 OOF의 F2 기준으로 결정했다. 전체 정밀도 {precision:.4f}, 재현율 {recall:.4f}, F1 {f1:.4f}다. 118건 경보 중 113건이 숫자 라벨 0인 만큼, 경보를 확정 불량 판정으로 취급할 수 없다."),
        s("2.8 불확실성과 평가 이력"),
        p(f"양성은 9행뿐이며 그중 7행은 상충 라벨 그룹에 속한다. 동일 입력 그룹 단위 bootstrap 95% AP 구간은 [{interval['lower_95pct']:.4f}, {interval['upper_95pct']:.4f}]로 넓다. Top 10% 재현율 구간도 약 0~0.700으로 신뢰도 높은 현장 성능을 입증하지 못한다."),
        p("최초 선정 후 홀드아웃을 평가했으며 이후 추가 설계 탐색에서 이 결과를 보았다. 기존 모델은 교체하지 않았지만 이 데이터는 새 모델의 미사용 blind 평가셋이 아니다. 후속 성능 개선 주장은 새 기간의 별도 라벨로 검증해야 한다."),
        source("experiment.json / holdout_diagnostics.csv / failure_analysis.json"),
    ]))

    joint = [r for r in errors if r['machine'] == 'rg3' and r['condition'] == 'pressure_joint']
    joint_rows = []
    for r in joint:
        names = ["초과" if item == "gt_train_median" else "이하" for item in r['bands'].split(' & ')]
        joint_rows.append([" / ".join(names), f"{r['rows']} / {r['positive_rows']}", f"{r['tp']} / {r['fn']}", f"{r['fp']} / {r['tn']}", f"{float(r['false_positive_rate']):.2%}"])
    coefficient_rows = [[r['feature'], f"{r['standardized_coefficient']:+.4f}"] for r in fail['largest_logistic_associations_not_causal'][:4]]
    pages.append(("제3장. 영향요인 및 오류분석 [15점]", [
        s("3.1 모델 계수와 홀드아웃 오류"),
        table(["큰 연관 계수", "표준화 계수"], coefficient_rows, [0.7, 0.3]),
        p("계수는 예측 연관성이며 원시 단위의 효과량·인과관계가 아니다. 홀드아웃 FN 4건은 모두 RG3, FP 113건 중 85건이 RG3다. 개별 행은 false_negatives.csv / false_positives.csv에서 원본 키로 추적한다."),
        s("3.2 두 압력 조건이 함께 나타날 때의 오류"),
        p("별도 개발 중첩 검증에서 외부 학습 영역의 설비별 중앙값과 내부 OOF F2 임계값만 사용했다. 아래는 RG3의 개발 OOF 946행이며 위 홀드아웃 오류와 합산하지 않는다. 첫 조건은 Max_Injection_Pressure, 둘째는 Max_Switch_Over_Pressure다.", "small"),
        table(["중앙값 대비\n주입 / 전환 압력", "행 / 양성", "TP / FN", "FP / TN", "오경보율"], joint_rows, [0.31, 0.20, 0.16, 0.18, 0.15]),
        p("결합조건별 오경보율은 17.45~43.21%로 달랐다. 모든 구간의 양성이 10행 미만이므로 미탐률이 불안정하다. 이것은 변수 간 결합조건의 탐색적 연관성이지 통계적으로 확정된 상호작용 효과나 설비 설정 변경의 근거가 아니다."),
        s("3.3 분포 밖 사례와 순서 스트레스"),
        p(f"개발 설비별 0.1~99.9 백분위 범위 밖 변수가 3개 이상인 비라벨 행은 {fail['unlabeled_ood_flagged_total']:,}행이다. OOD는 결함 정답이 아니라 입력 분포 경고다. 인덱스 뒤 20% 스트레스에서 CN7은 양성 0건으로 평가 불가, RG3 AP는 0.0423이다. 실제 시간이 없어 전방검증으로 부르지 않는다."),
        source("failure_analysis.json / calibration-review-20260922/conditional_errors.csv"),
    ]))

    pages.append(("제4장. 현장 활용방안 [10점]", [
        s("4.1 구현된 출력과 제안하는 활용을 구분"),
        p("구현된 것은 71,180행 전체에 점수·전체/설비별 순위·우선순위·OOD·보수적 action을 붙인 정적 오프라인 분석 파일이다. 각 설비 상위 10% HIGH, 다음 10% REVIEW, 나머지 STANDARD는 참고 순위이며 실제 Pass/Reject/Reinspect 판정이 아니다."),
        table(["대상", "행 수", "현재 action"], [
            ["RG3 전체", "35,941", "usual_inspection_unvalidated_machine"],
            ["CN7 분포 범위 밖", "19,607", "usual_inspection_ood"],
            ["CN7 나머지", "15,632", "usual_inspection_label_unconfirmed"],
        ], [0.28, 0.15, 0.57]),
        p("모든 행에서 기존 검사를 유지한다. OOD 전체 27,357행에는 RG3 7,750행이 포함되지만 RG3는 성능 미검증 규칙이 먼저 적용된다. 그러므로 OOD 표시 총수와 OOD action 행 수는 다르다."),
        s("4.2 현장 검증 후 적용할 의사결정 흐름 (제안)"),
        table(["순서", "판단·조치", "필수 조건"], [
            ["1", "공정 종료 → 입력 계약·시점 확인", "예측 cutoff 이전 도착 정보만 사용"],
            ["2", "필수·긴급검사 규칙 우선", "현장 규칙과 작업자 승인, 검사 생략 금지"],
            ["3", "적용 승인 영역만 검사 순서 참고", "라벨 확정·성능 검증·분포 범위 확인"],
            ["4", "이상 입력·오류·미승인 영역은 기존 검사", "낮은 점수로 대체하지 않고 이유를 기록"],
            ["5", "실제 검사 결과 수신 후 설비별 감시", "FN·FP·발견 지연·작업자 개입 기록"],
        ], [0.09, 0.47, 0.44]),
        s("4.3 생산성과 작업환경 효과의 검증 계획"),
        p("가능한 이점은 고위험 제품의 확인 순서를 앞당기고 작업자 판단의 근거를 제공하는 것이다. 검사 가능 시각·실제 시작/종료·인력/장비 수·긴급규칙이 있어야 기존 순서 대비 발견 지연과 작업 부담을 재생 비교할 수 있다. 현재는 동적 스케줄러나 시간 절감·비용 절감 효과를 구현·입증하지 않았다."),
        source("unlabeled_priority.csv / moldguard.py verify / docs/04_추가데이터_요청명세.md"),
    ]))

    cal_rows = [[name, f"{cal['metrics']['all'][key]['average_precision']:.4f}", f"{cal['metrics']['all'][key]['brier']:.6f}", f"{cal['metrics']['all'][key]['log_loss']:.6f}"] for name, key in [("발생률 상수", "prior"), ("원점수", "raw"), ("중첩 sigmoid", "sigmoid")]]
    delta = cal['paired_group_bootstrap']['sigmoid_minus_prior_brier']
    pages.append(("제5장. 창의성 및 차별성 [10점]", [
        s("5.1 확률과 순위를 분리해 검증"),
        p("공식 문제의 확률 예측 요구에 맞춰 원점수의 신뢰성을 별도로 조사했다. 기존 개발 5개 그룹 fold 안에서 다시 5개 그룹 fold로 sigmoid 보정기를 학습했다. 외부 검증 행과 기존 홀드아웃은 보정 학습에 쓰지 않았다. 기존 모델 선정 뒤의 개발 탐색이므로 독립 성능 입증으로 보지 않는다."),
        table(["개발 OOF", "AP", "Brier (낮음)", "Log loss (낮음)"], cal_rows, [0.34, 0.18, 0.24, 0.24]),
        p("평균 점수는 원점수 29.68%에서 보정 후 1.77%로 낮아져 관측 양성률 1.73%에 가까워졌다. 하지만 평균 일치와 낮은 Brier만으로 전 구간 보정이 좋다고 결론낼 수 없다. 신뢰도 구간별 행 수·관측률을 별도 CSV로 저장했다."),
        p(f"상수 대비 보정 Brier 차이는 {delta['difference']:.6f}, 그룹 bootstrap 95% 구간 [{delta['lower_95pct']:.6f}, {delta['upper_95pct']:.6f}]로 0을 포함한다. RG3의 보정 Brier 0.021035는 상수 0.020710보다 나쁘다. 보정 모델을 최종 모델로 교체하지 않았다."),
        p("외부 fold마다 원점수와 sigmoid의 AP는 동일했지만 합친 OOF AP는 달라졌다. fold별 보정 함수의 척도가 달라졌기 때문이며, 단일 sigmoid 모델이 항상 순위를 훼손한다는 뜻은 아니다. 고위험 보정 구간은 표본 8행·2행 수준으로 확률 신뢰성 판단이 어렵다.", "small"),
        s("5.2 차별점의 실증 범위"),
        table(["접근", "관측된 기여", "넘어서는 주장 금지"], [
            ["중복 그룹 고정", "동일 입력 교차 0", "모든 종류의 누수 부재는 미입증"],
            ["설비별 실패 공개", "RG3 FN 집중·낮은 AP 확인", "전 설비 활용 가능으로 일반화 금지"],
            ["확률보정·불확실성", "과신 완화와 상수 대비 불확실성 확인", "신뢰할 수 있는 불량 확률로 확정 금지"],
            ["보수적 action", "미승인·OOD도 기존 검사 유지", "실제 검사 절감이나 안전 개선 실증 아님"],
        ], [0.25, 0.34, 0.41]),
        source("calibration_review.json / reliability_bins.csv / docs/07_확률보정_및_조건부_오류분석.md"),
    ]))

    pages.append(("제6장. 코드 구성 및 재현성 [10점]", [
        s("6.1 원본 보존과 일괄 실행"),
        p("CPU · Python 3.12.7 · macOS arm64에서 실행했다. requirements-model-lock.txt에 26개 패키지 버전을 고정했다. 원본 4개 CSV를 기존 데이터셋 폴더에 보존하고 프로젝트 루트에서 다음을 실행한다. 새 환경 설치는 루트의 로컬_실행_가이드.md를 따른다."),
        p(".venv-repro/bin/python run_local.py quick\n.venv-repro/bin/python run_local.py full", "code"),
        p("quick은 환경·테스트를 점검한다. full은 새 실행 폴더에서 audit → train → analysis → verify를 실행해 기존 outputs/moldguard/를 덮어쓰지 않는다. 행별 분할 CSV, 명령·패키지 버전·종료코드·로그·입출력 SHA-256을 남긴다."),
        table(["구성", "역할"], [
            ["moldguard.py", "입력 검증, 그룹 분할, 7개 후보 학습·선정, 추론·결과 검사"],
            ["analyze_moldguard.py", "오류·연관 계수·분포 이동·인덱스 스트레스·불확실성"],
            ["run_local.py", "환경 검증·분리된 실행 폴더·단계별 로그·참조 해시 비교"],
            ["calibration_diagnostics.py", "개발 전용 중첩 보정·신뢰도 구간·조건별 오류"],
            ["verify_calibration_review.py", "별도 연구의 반복 산출물·원본 불변성·전체 테스트 확인"],
        ], [0.43, 0.57]),
        s("6.2 재현 증거와 한계"),
        p("공유용 full 실행에서 기준 JSON/CSV 5개가 바이트 단위로 동일했다. PNG 2개는 Mac/Agg 렌더러 차이가 별도 기록되어 있으며 숫자 결과 차이와 혼동하지 않는다. 과거 공유 ZIP에는 당시 테스트 11개와 실행 로그가 들어 있다."),
        p("추가 보정 연구는 두 번 독립 실행한 7개 산출물의 SHA-256이 모두 같았다. 최신 연구 검증은 테스트 18개, 점검 항목 37/37 통과다. 기존 모델·결과·공유 ZIP도 불변이다. 실행 재현은 과학적 타당성·라벨 의미·미래 성능을 보증하지 않는다."),
        source("20260922-share-final/run_manifest.json / calibration-review-verification.json"),
    ]))

    pages.append(("부록 A. 완료 수준과 다음 검증 게이트", [
        s("A.1 평가 항목별 증거와 남은 조건"),
        table(["요구 항목", "현재 증거", "남은 조건"], [
            ["데이터 이해·진단", "규모·스키마·상충 라벨·분포 이동 진단", "라벨 방향, 생산 단위·시각·원시 단위 확인"],
            ["모델 개발·성능", "7개 후보, 고정 그룹 분할, 확률보정 연구", "강한 미래 성능·RG3 유효성은 미입증"],
            ["영향·상호작용·오류", "계수·FN/FP·두 변수 결합조건 분석", "소표본으로 상호작용 효과 확정 불가"],
            ["현장 활용", "전량 정적 순위·기존 검사 유지 action", "실제 운영 시각·자원으로 대기열 재생 검증"],
            ["차별성", "불균형·불확실성·중첩 확률보정", "상수 대비 보정 우위와 생산성 효과 미입증"],
            ["재현성", "실행 코드·버전·분할·해시·실제 로그", "새 데이터 계약과 독립 평가에서 재실행"],
        ], [0.23, 0.38, 0.39]),
        s("A.2 결과물 인계 상태"),
        p("이 PDF는 6개 장에 대한 내용 검토본이다. 공식 HWPX의 글꼴·서명란까지 완성한 제출용 파일은 아니다. 기존 공유 ZIP은 원본 CSV를 제외한 공유 패키지로, 공식 소스코드 제출 ZIP을 대신하지 않는다. 발표 PPT/PDF와 팀 정보·서명·설문 완료 화면은 별도 준비가 필요하다."),
        s("A.3 새 자료를 받으면 적용할 순서"),
        p("① 네 파일명·해시에 대응하는 라벨 코드북 확인 → ② 샷/제품 키와 변수 도착 시각·원시 스케일러 검증 → ③ 새 기간의 개발·보정·최종 평가 영역 동결 → ④ 동일 CPU 기준선과 강한 후보 비교 → ⑤ 설비별 확률·오류·대기열 효과 검증 → ⑥ 로그·산출물 재현 확인."),
        p("라벨 의미가 반대이거나 파일별로 다르면 기존 불량 해석·순위·모델을 그대로 재사용하지 않는다. 변환·분할·학습·평가를 재설계하고 이미 본 홀드아웃으로 새 성능을 주장하지 않는다."),
        s("중단조건"),
        p("라벨·예측 시점·원시 변환을 확인하지 못한 상태에서는 현장 불량 예측 완성이나 생산성 향상 실증을 선언하지 않는다. 테스트 통과나 문서 완성이 이 조건을 대신하지 않는다. 대회 사이트에는 최종 제출하지 않았다."),
    ]))

    source_names = ["outputs/moldguard/data_audit.json", "outputs/moldguard/experiment.json", "outputs/moldguard/failure_analysis.json", "outputs/moldguard/holdout_diagnostics.csv", "outputs/local_runs/20260922-share-final/run_manifest.json", "outputs/local_runs/20260922-share-final/split_manifest.csv", "outputs/calibration-review-20260922/calibration_review.json", "outputs/calibration-review-20260922/conditional_errors.csv", "outputs/calibration-review-verification.json", "moldguard.py", "analyze_moldguard.py", "run_local.py", "calibration_diagnostics.py", "verify_calibration_review.py", "requirements-model-lock.txt", "docs/03_사출성형_검증보고서.md", "docs/04_추가데이터_요청명세.md", "docs/05_라벨_의미_검증.md", "docs/07_확률보정_및_조건부_오류분석.md"]
    appendix = [s("B.1 원본 데이터 SHA-256")]
    for name, info in audit['source'].items():
        appendix += [p(name, "small"), p(info['sha256'], "code")]
    appendix += [
        s("B.2 공식 범위와 문서 근거"),
        p("제6회 K-인공지능 제조데이터 분석 경진대회 과제공개(일반국민,대학(원)생).hwpx: 사출성형 문제·6개 평가 항목·결과물 요구.\n경진대회 결과보고서 양식_일반국민,대학(원)생 부문.hwpx: 제1~6장 구조·표지 항목·설문 완료 화면 요구.", "small"),
        p("로컬 상세 근거: docs/03_사출성형_검증보고서.md, docs/04_추가데이터_요청명세.md, docs/05_라벨_의미_검증.md, docs/07_확률보정_및_조건부_오류분석.md. 외부 가이드북 사본은 현재 CSV 라벨 방향의 확정 근거로 사용하지 않았다.", "small"),
        s("B.3 확률보정 방법 참고"),
        p("scikit-learn 1.5 Probability calibration:\nhttps://scikit-learn.org/1.5/modules/calibration.html\nCalibratedClassifierCV API:\nhttps://scikit-learn.org/1.5/modules/generated/sklearn.calibration.CalibratedClassifierCV.html", "small"),
        p("보고서의 수치 근거·출력 해시·생성 환경은 같은 폴더의 report_provenance.json에 기록했다. PDF 제작은 학습 환경과 분리된 문서용 Python에서 수행했으며 모델 재학습이나 사이트 제출을 실행하지 않는다.", "small"),
    ]
    pages.append(("부록 B. 출처와 추적 가능성", appendix))

    # Measure every flowable before generating: no silent overflow or automatic
    # page split may turn a section into an unreadable/clipped layout.
    layout = []
    for number, (title, items) in enumerate(pages, 1):
        heights = [item.wrap(WIDTH, H)[1] + item.getSpaceBefore() + item.getSpaceAfter() for item in items]
        used = sum(heights)
        available = H - TOP - BOTTOM
        if used > available:
            raise ValueError(f"Page {number} overflow: {used:.1f} > {available:.1f} points")
        layout.append({"page": number, "title": title, "used_points": round(used, 2), "available_points": round(available, 2)})
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    canvas = Canvas(str(OUTPUT), pagesize=A4, invariant=1, pageCompression=1)
    canvas.setTitle("KAMP MoldGuard - 사출성형 검토보고서")
    canvas.setAuthor("MoldGuard")
    canvas.setSubject("6개 평가 항목의 연구 검토본; 라벨 및 현장 성능 미확정; 최종 제출 아님")
    for number, (title, items) in enumerate(pages, 1):
        canvas.bookmarkPage(f"page-{number}")
        canvas.addOutlineEntry(title, f"page-{number}", level=0, closed=False)
        canvas.setFillColor(TEAL)
        canvas.rect(LEFT, H - 36, 29, 3, fill=1, stroke=0)
        canvas.setFont("Korean", 10)
        canvas.setFillColor(NAVY)
        canvas.drawString(LEFT + 39, H - 38, title)
        canvas.setStrokeColor(LINE)
        canvas.line(LEFT, 38, W - RIGHT, 38)
        canvas.setFillColor(MUTED)
        canvas.setFont("Korean", 8)
        canvas.drawString(LEFT, 24, "검토용 · 숫자 라벨 1 기준 · 기존 검사 유지 · 최종 제출 아님")
        canvas.drawRightString(W - RIGHT, 24, f"{number:02d} / {len(pages):02d}")
        y = H - TOP
        for item in items:
            _, height = item.wrap(WIDTH, H)
            y -= item.getSpaceBefore() + height
            item.drawOn(canvas, LEFT, y)
            y -= item.getSpaceAfter()
        canvas.showPage()
    canvas.save()
    reader = PdfReader(OUTPUT)
    extracted = [page.extract_text() or "" for page in reader.pages]
    if len(reader.pages) != len(pages):
        raise ValueError("Unexpected PDF page count")
    for index, (title, _) in enumerate(pages):
        if title not in extracted[index]:
            raise ValueError(f"Page title missing from extracted PDF: {title}")
    all_text = "\n".join(extracted)
    required = ["0.1154", "0.2779", "0.0192", "0/5", "35,941", "19,607", "15,632", "37/37", "18개", "미확정", "후속", "설문", "최종 제출 아님"]
    for value in required:
        if value not in all_text:
            raise ValueError(f"Required numeric/status assertion missing: {value}")
    if "/Users/" in all_text or "hamseongho" in all_text:
        raise ValueError("Local identity must not leak into the review PDF")
    sources = {name: digest(ROOT / name) for name in source_names}
    sources.update({path.name: digest(path) for path in ROOT.glob("*.hwpx")})
    receipt = {
        "status": "created_text_and_geometry_checked_pending_visual_review",
        "pdf": OUTPUT.name, "pdf_sha256": digest(OUTPUT), "page_count": len(pages),
        "source_sha256": sources, "builder_sha256": digest(__file__),
        "content_as_of": "2026-09-22",
        "python": sys.version,
        "document_packages": {name: importlib.metadata.version(name) for name in ["reportlab", "pypdf", "pillow"]},
        "deterministic_pdf_metadata": "ReportLab invariant=1 uses a fixed metadata timestamp; cover date is the evidence date",
        "font_sha256": digest(args.font_file), "font_name": args.font_file.name,
        "layout": layout, "required_text_checks": required,
        "derived_holdout_metrics": {"precision": precision, "recall": recall, "f1": f1},
        "official_template_filled": False, "field_deployment_approved": False,
        "competition_website_submission": False,
        "notion_published": False,
    }
    (OUTPUT.parent / "report_provenance.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"pdf": str(OUTPUT), "pages": len(pages), "sha256": receipt['pdf_sha256'], "layout": layout}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
