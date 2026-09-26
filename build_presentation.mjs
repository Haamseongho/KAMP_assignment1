/** Build editable evidence slides from the frozen experiment artifacts. */
import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = path.dirname(fileURLToPath(import.meta.url));
const skill = '/Users/hamseongho/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.12148/skills/presentations';
const python = '/Users/hamseongho/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3';
const { importRuntimeModule } = await import(pathToFileURL(path.join(skill, 'container_tools/runtime_helpers.mjs')));
const { Presentation, PresentationFile } = await importRuntimeModule('@oai/artifact-tool');
const { finalizePresentation, applyPresentationChartFont } = await import(pathToFileURL(path.join(skill, 'container_tools/artifact_tool_utils.mjs')));

const revision = process.argv[2] ?? 'v1';
if (!/^v\d+$/.test(revision)) throw new Error('Use a revision such as v1 or v2');
const buildDir = path.join(root, 'tmp/presentations', `kamp-${revision}`);
const outputDir = path.join(root, 'output/presentation');
const finalPath = path.join(outputDir, `KAMP_MoldGuard_발표검토본_20260922_${revision}.pptx`);
await fs.mkdir(buildDir, { recursive: true });
await fs.mkdir(outputDir, { recursive: true });
try { await fs.access(finalPath); throw new Error('Final output exists; use a new revision'); }
catch (error) { if (error.code !== 'ENOENT') throw error; }
const read = async name => JSON.parse(await fs.readFile(path.join(root, name), 'utf8'));
const hash = async name => createHash('sha256').update(await fs.readFile(path.join(root, name))).digest('hex');
const sources = ['outputs/moldguard/experiment.json', 'outputs/moldguard/data_audit.json',
  'outputs/moldguard/failure_analysis.json', 'outputs/moldguard/reproducibility.json',
  'outputs/calibration-review-20260922/calibration_review.json',
  'outputs/calibration-review-verification.json', 'output/pdf/report_qa.json'];
const [exp, audit, failure, original, calibration, verified, reportQa] = await Promise.all(sources.map(read));
for (const [name, expected] of Object.entries(original.artifact_hashes)) {
  if (await hash(name) !== expected) throw new Error(`Frozen source changed: ${name}`);
}
if (verified.status !== 'passed' || reportQa.scientific_completion_proven !== false) throw new Error('Unexpected evidence scope');
const pres = Presentation.create({ slideSize: { width: 1280, height: 720 } });
const font = 'NanumGothic';
const ink = '#17334A', muted = '#506675', teal = '#0D736D', line = '#D4DFE5';
const allSlides = [], tableOwners = [], chartOwners = [];
const f4 = n => n.toFixed(4), f6 = n => n.toFixed(6);

function text(slide, value, x, y, w, h, size = 27, color = ink, bold = false) {
  const shape = slide.shapes.add({ geometry: 'textbox', name: `text-${slide.id}-${y}-${x}`,
    position: { left: x, top: y, width: w, height: h }, fill: 'none', line: { fill: 'none', width: 0 } });
  shape.text = value;
  shape.text.style = { typeface: font, fontSize: size, bold, color, autoFit: 'none',
    alignment: 'left', verticalAlignment: 'top', insets: { left: 0, top: 0, right: 0, bottom: 0 } };
  return shape;
}
function add(title, note, caveat = '숫자 라벨 1 기준. 라벨 의미와 현장 성능은 아직 미확정') {
  const slide = pres.slides.add();
  slide.background.fill = '#FFFFFF';
  allSlides.push(slide);
  text(slide, title, 64, 42, 1152, 90, 44, ink, true);
  text(slide, caveat, 64, 666, 1100, 36, 18, muted);
  text(slide, String(allSlides.length).padStart(2, '0'), 1170, 666, 46, 36, 18, muted);
  slide.speakerNotes.textFrame.setText(note);
  return slide;
}
function table(slide, values, widths, options = {}) {
  const top = options.top ?? 170, height = options.height ?? values.length * 54;
  const left = options.left ?? 64, width = options.width ?? 1152;
  const t = slide.tables.add({ rows: values.length, columns: values[0].length, left, top, width, height,
    columnWidths: widths.map(v => width * v / widths.reduce((a, b) => a + b, 0)), values });
  t.borders.assign({ fill: line, width: 0.6, style: 'solid' });
  for (let r = 0; r < values.length; r++) {
    t.rows[r].height = height / values.length;
    for (let c = 0; c < values[0].length; c++) {
      const cell = t.getCell(r, c);
      cell.fill = r === 0 ? '#17334A' : (r % 2 ? '#FFFFFF' : '#F1F6F8');
      cell.text.style = { typeface: font, fontSize: options.fontSize ?? 24, bold: r === 0,
        color: r === 0 ? '#FFFFFF' : ink, autoFit: 'none', verticalAlignment: 'middle',
        insets: { left: 12, right: 10, top: 8, bottom: 8 } };
    }
  }
  tableOwners.push(allSlides.length);
  return t;
}

// 1. A minimal, editable cover. No invented factory photo or affiliation.
{
  const s = pres.slides.add(); allSlides.push(s); s.background.fill = '#FFFFFF';
  text(s, 'MoldGuard', 64, 75, 900, 55, 30, teal, true);
  text(s, '사출성형 품질위험 예측\n검사 우선순위 결정', 64, 186, 1120, 178, 60, ink, true);
  text(s, '실제 공정데이터의 고정 그룹 검증과 실패 조건 분석', 64, 402, 1120, 58, 28, muted);
  text(s, '2026-09-22 연구 검토본\n제6회 K-인공지능 제조데이터 분석 경진대회', 64, 548, 1120, 80, 24, muted);
  s.speakerNotes.textFrame.setText('공식 과제: 사출성형 공정데이터 기반 품질불량 사전예측 및 검사 우선순위 결정. 출처: 제6회 과제공개(일반국민,대학(원)생).hwpx. 이 발표는 실제 CSV의 연구 검토본이며 팀명·서명·설문·현장 승인 미완료다. 대회 사이트에 제출하지 않았다.');
}
{
  const s = add('예측 시점은 공정 종료 후, 최종검사 전',
    '출처: 공식 과제공개 HWPX, data_audit.json의 feature_columns. Cycle_Time 등 샷 전체 요약이 있으므로 현재 구현을 제작 전 예측이나 실시간 조기경보로 해석할 수 없다. 실제 변수 도착 시각은 제공되지 않았다.');
  table(s, [['시점', '현재 입력으로 가능한 판단'],
    ['제작 전', '설계·소재·설정값만의 별도 모델과 데이터가 필요'],
    ['공정 진행 중', '시점별 센서와 도착 시각이 없어 현재 검증 불가'],
    ['공정 종료 후, 검사 전', '공정 요약값으로 위험 점수와 참고 순위 계산']], [0.30, 0.70], { height: 264 });
  text(s, '현재 구현 범위', 64, 486, 400, 42, 27, teal, true);
  text(s, '샷 종료 요약값 사용. 실제 검사 시작 전에 모든 변수가 도착하는지는 추가 확인 필요.', 64, 535, 1130, 75, 27);
}
{
  const s = add('라벨 2,393행, 비라벨 71,180행',
    '출처: outputs/moldguard/data_audit.json. 라벨 1은 CN7 17, RG3 25로 합계 42. 동일 공정값은 1,197그룹이며 상충 라벨 36그룹이 있다. 파일마다 표준화된 것으로 보이나 원시 변환 이력이 없다.');
  table(s, [['입력', '행 수', '라벨 1'], ['CN7 라벨', '1,211', '17'], ['RG3 라벨', '1,182', '25'],
    ['CN7 비라벨', '35,239', '정답 없음'], ['RG3 비라벨', '35,941', '정답 없음']], [0.50, 0.25, 0.25], { height: 285 });
  text(s, '같은 공정값인데 0과 1이 함께 있는 그룹 36개', 64, 498, 1152, 48, 29, teal, true);
  text(s, '생산 단위와 라벨 정의, 실제 시각, 원본 정규화 범위를 확인해야 함', 64, 561, 1152, 48, 26);
}
{
  const s = add('동일 공정값 그룹을 분리하지 않은 고정 검증',
    '출처: outputs/local_runs/20260922-share-final/split_manifest.csv, experiment.json. seed=20260922. machine × 그룹 내 라벨 1 존재 여부로 층화. 원본 ID와 정답을 입력에서 제외. 전처리 전 원시 정규화의 적합 범위는 미확정.');
  table(s, [['영역', '그룹', '행', '라벨 1', '역할'], ['개발', '957', '1,913', '33', '그룹 5-fold OOF로 선정'],
    ['홀드아웃', '240', '480', '9', '선정 뒤 최초 성능 평가']], [0.17, 0.13, 0.14, 0.14, 0.42], { height: 210 });
  text(s, '분할 간 동일 입력 그룹 교차 0', 64, 428, 1140, 55, 31, teal, true);
  text(s, '실제 시간·샷 키와 사전 정규화 범위가 없어 모든 누수를 차단했다고 보증할 수 없음', 64, 505, 1140, 82, 27);
  text(s, '최초 평가 뒤 탐색에 노출된 홀드아웃. 새 모델은 새로운 독립 평가가 필요.', 64, 596, 1140, 38, 22, muted);
}
{
  const names = { dummy: 'B0 발생률 기준선', logistic: 'B1 가중 로지스틱 (선정)', forest: 'B2 Random Forest', catboost: 'B3 CatBoost', separate_logistic: 'B4 설비별 로지스틱', separate_forest: 'B5 설비별 Forest', separate_catboost: 'B6 설비별 CatBoost' };
  const values = [['모델', '개발 AP', 'Top 10% 발견', 'ROC-AUC'], ...Object.entries(exp.models_oof_development).map(([key, m]) => [names[key], f4(m.average_precision), `${m.top10_positives}/${m.positives}`, f4(m.roc_auc)])];
  const s = add('7개 후보 중 개발 AP가 가장 높은 모델을 선정',
    '출처: experiment.json > models_oof_development. 모든 후보는 동일한 개발 1,913행의 고정 그룹 OOF를 사용. Top 10%는 192행. Dummy 순위는 동점 처리의 영향을 받는다. 최종 모델은 홀드아웃을 보기 전 개발 AP로 선정한 가중 로지스틱. 비선형 후보의 우위는 입증되지 않았다.');
  table(s, values, [0.45, 0.17, 0.22, 0.16], { top: 157, height: 392, fontSize: 24 });
  text(s, '설비별 CatBoost의 발견 수는 더 높지만, 선정 기준 AP는 더 낮음', 64, 574, 1152, 60, 26);
}
{
  const s = add('전체 AP는 RG3의 낮은 성능을 가린다',
    `출처: experiment.json > holdout, holdout_by_machine. AP 전체 ${exp.holdout.average_precision}, CN7 ${exp.holdout_by_machine.cn7.average_precision}, RG3 ${exp.holdout_by_machine.rg3.average_precision}. 편집 가능한 차트 워크북은 화면 표시와 동일한 소수 4자리 반올림 값을 저장한다. 완전 정밀도는 원본 JSON에 보존했다. 전체 Top 10%는 통합 순위 48행에서 3/9, 설비별 10% 합계는 49행에서 4/9. CN7 양성 표본 4행이므로 미래 100% 검출을 보장하지 않는다.`);
  const metrics = [exp.holdout, exp.holdout_by_machine.cn7, exp.holdout_by_machine.rg3];
  const chart = s.charts.add('bar', { position: { left: 64, top: 168, width: 700, height: 370 },
    categories: ['전체', 'CN7', 'RG3'], series: [{ name: '홀드아웃 AP', values: metrics.map(m => Number(f4(m.average_precision))), fill: teal, valuesFormatCode: '0.0000' }],
    barOptions: { direction: 'column', grouping: 'clustered', gapWidth: 150 }, hasLegend: false,
    xAxis: { textStyle: { typeface: font, fontSize: 25, fill: ink }, majorGridlines: null },
    yAxis: { min: 0, max: 0.8, majorUnit: 0.2, numberFormatCode: '0.0', textStyle: { typeface: font, fontSize: 20, fill: muted }, majorGridlines: { fill: line, width: 0.6 } },
    dataLabels: { showValue: true, position: 'outEnd', textStyle: { typeface: font, fontSize: 24, fill: ink } },
    chartFill: '#FFFFFF', plotAreaFill: '#FFFFFF' });
  applyPresentationChartFont(chart, { fontFamily: font }); chartOwners.push(allSlides.length);
  table(s, [['대상', '행 / 양성', 'Top 10%'], ...metrics.map((m,i) => [['전체','CN7','RG3'][i], `${m.rows} / ${m.positives}`, `${m.top10_positives}/${m.positives}`])], [0.24,0.40,0.36], { left: 810, top: 210, width: 406, height: 250, fontSize: 22 });
  text(s, '전체 AP 95% 구간 0.0129~0.6290. 양성 9행의 결과로 현장 성능을 확정할 수 없음.', 64, 566, 1152, 75, 26);
}
{
  const s = add('홀드아웃 미탐 4건은 모두 RG3에서 발생',
    '출처: holdout_diagnostics.csv, experiment.json의 confusion_at_diagnostic_threshold. threshold=0.5755568886112208를 개발 OOF의 F2로 결정. 전체 TP5 FN4 FP113 TN358. precision=5/118, recall=5/9, F1=10/127. 이 경보를 확정 불량 판정으로 취급하지 않는다.');
  table(s, [['대상', 'TP', 'FN', 'FP', 'TN'], ['전체', '5', '4', '113', '358'], ['CN7', '4', '0', '28', '212'], ['RG3', '1', '4', '85', '146']], [0.28,0.18,0.18,0.18,0.18], { height: 248 });
  text(s, '정밀도 0.0424     재현율 0.5556     F1 0.0787', 64, 467, 1152, 56, 31, teal, true);
  text(s, '진단 임계값 0.575557. 118건 경보 중 113건은 숫자 라벨 0.', 64, 545, 1152, 60, 27);
}
{
  const s = add('RG3의 압력 결합조건에 따라 오경보율이 달랐다',
    '출처: outputs/calibration-review-20260922/conditional_errors.csv. RG3 개발 OOF 946행. 첫 조건 Max_Injection_Pressure, 둘째 Max_Switch_Over_Pressure. 중앙값은 외부 학습 영역의 해당 설비, F2 임계값은 내부 OOF에서만 결정. 모든 구간 양성<10. 표는 탐색적 결합조건 연관성이며 인과효과나 확정된 통계적 상호작용이 아니다.');
  table(s, [['주입 / 전환 압력¹', '행 / 양성', 'TP / FN', 'FP / TN', '오경보율'],
    ['초과 / 초과','326 / 7','2 / 5','114 / 205','35.74%'], ['초과 / 이하','106 / 4','1 / 3','31 / 71','30.39%'],
    ['이하 / 초과','82 / 1','1 / 0','35 / 46','43.21%'], ['이하 / 이하','432 / 8','2 / 6','74 / 350','17.45%']], [0.31,0.18,0.16,0.18,0.17], { height: 295 });
  text(s, '¹ 외부 학습 영역의 설비별 중앙값 대비. 개발 OOF 결과이며 홀드아웃과 별개.', 64, 490, 1152, 64, 23, muted);
  text(s, '각 구간의 양성이 10행 미만. 설비 설정을 바꾸라는 근거로 사용할 수 없음.', 64, 569, 1152, 65, 26);
}
{
  const m = calibration.metrics.all;
  const s = add('확률보정은 과신을 줄였지만 우위는 불확실',
    '출처: calibration_review.json. 개발 전용 외부5/내부5 그룹 중첩 sigmoid ensemble=False. 기존 홀드아웃 미평가. 원점수 평균0.29683665, 보정0.01773358, 양성률0.01725039. 상수 대비 Brier 차이 95% 구간 [-0.0008958053, +0.0001665380]. 각 fold의 AP는 원점수/보정 동일하며 합산 OOF AP 차이는 fold별 척도 차이. Brier만으로 보정 품질을 확정하지 않음. https://scikit-learn.org/1.5/modules/calibration.html');
  table(s, [['개발 OOF', 'AP', 'Brier (낮음)', 'Log loss (낮음)'], ...[['발생률 상수','prior'],['원점수','raw'],['중첩 sigmoid','sigmoid']].map(([label,key]) => [label,f4(m[key].average_precision),f6(m[key].brier),f6(m[key].log_loss)])], [0.34,0.18,0.24,0.24], { height: 244 });
  text(s, '평균 점수 29.68%에서 1.77%로 감소', 64, 459, 1152, 50, 31, teal, true);
  text(s, '상수 대비 Brier 개선 구간에 0 포함. RG3도 상수보다 나쁨.\n기존 모델을 교체하지 않고 별도 연구 결과로 보존.', 64, 525, 1152, 98, 27);
}
{
  const s = add('71,180행 모두 기존 검사를 유지한다',
    '출처: unlabeled_priority.csv와 failure_analysis.json. RG3 전체35941, CN7OOD19607, CN7라벨미확정15632. OOD 표시 전체27357에는 RG3 7750이 포함. action 순서상 RG3는 미검증 설비로 먼저 처리. HIGH/REVIEW/STANDARD는 설비별상위10/다음10/나머지 정적 오프라인 순위이며 실제 Pass/Reject/Reinspect가 아니다.');
  table(s, [['대상', '행 수', '현재 조치'], ['RG3 전체', '35,941', '성능 미검증으로 기존 검사 유지'],
    ['CN7 분포 범위 밖', '19,607', 'OOD로 기존 검사 유지'], ['CN7 나머지', '15,632', '라벨 미확정으로 기존 검사 유지']], [0.32,0.18,0.50], { height: 260 });
  text(s, 'HIGH / REVIEW / STANDARD는 오프라인 참고 순위', 64, 480, 1152, 54, 31, teal, true);
  text(s, '검사 생략과 자동 설비 제어를 하지 않음. 비라벨에는 정답이 없어 실제 정확도를 계산할 수 없음.', 64, 553, 1152, 80, 27);
}
{
  const s = add('검사 순서의 효과는 실제 운영 기록으로 검증',
    '출처: docs/04_추가데이터_요청명세.md. 현재 구현은 정적 점수/순위 파일. 실시간 대기열 스케줄러나 실제 시간·비용 절감 효과를 구현·실증하지 않았다. 사용자 목표인 생산성과 작업환경 개선을 검증하려면 제품별 검사 가능 시각, 실제 시작종료, 인력장비, 현장 긴급규칙이 필요하다.');
  table(s, [['검증하려는 효과', '필요한 운영 기록'], ['고위험 제품을 더 일찍 확인', '검사 가능 시각, 검사 시작·종료, 확정 결과'],
    ['검사 자원의 과도한 집중 방지', '설비별 검사 인력·장비, 필수·긴급검사 규칙'], ['작업자의 판단 부담 감소', '오경보 대응시간, 순위 수정, 작업자 개입 이유']], [0.42,0.58], { height: 284 });
  text(s, '현장 검증 제안', 64, 500, 1152, 46, 28, teal, true);
  text(s, '같은 검사 자원에서 기존 순서와 AI 참고 순서의 발견 지연·작업 부담을 비교한다.\n절감 효과는 아직 산출하지 않았다.', 64, 548, 1152, 88, 27);
}
{
  const s = add('CPU 실행과 추적 가능한 재현 기록',
    '출처: outputs/local_runs/20260922-share-final/run_manifest.json, calibration-review-verification.json, 로컬_실행_가이드.md. Python3.12.7 CPU, 잠금26패키지. 기본 pipeline 주요JSON/CSV5개 동일. 추가 연구7개 산출물동일 및18테스트37점검통과. PNG renderer차이는 별도기록. 테스트는 실제성능이나라벨의미를보증하지않는다.');
  text(s, '.venv-repro/bin/python run_local.py quick\n.venv-repro/bin/python run_local.py full', 64, 176, 1152, 102, 30, teal, true);
  table(s, [['확인 범위', '현재 증거'], ['기본 분석 재실행', '주요 JSON/CSV 5개 해시 동일'], ['추가 확률보정 연구', '독립 반복 산출물 7개 해시 동일'],
    ['검증과 추적', '18 tests, 37/37 점검, 분할 명세와 단계별 로그']], [0.36,0.64], { top: 324, height: 248 });
  text(s, '모델과 문서 환경을 분리. 원본 데이터와 기존 공유 ZIP은 보존.', 64, 599, 1152, 43, 24, muted);
}
{
  const s = add('다음 판단에 필요한 네 가지 근거',
    '출처: docs/04_추가데이터_요청명세.md, docs/05_라벨_의미_검증.md, docs/07_확률보정_및_조건부_오류분석.md. 현장성능완성은미입증. 새기간평가영역을동결후모델보정정책선정에서제외. 기존홀드아웃재사용금지. 공식팀정보서명설문은사용자별도확인필요. 사이트제출하지않음.');
  table(s, [['필요 자료', '확인할 판단'], ['파일 해시에 대응하는 라벨 코드북', 'PassOrFail의 0/1 의미'], ['샷·제품·캐비티 키와 실제 시각', '예측 단위와 검사 전 정보 가용성'],
    ['원시 단위와 정규화 변환 이력', '전처리 범위와 신규 샷 추론'], ['새 기간의 설비별 검사 라벨', '독립 미래 성능과 RG3 활용 가능성']], [0.51,0.49], { height: 310 });
  text(s, '현재 결론', 64, 520, 1152, 48, 30, teal, true);
  text(s, '기준선과 비교 실험은 재현 가능. 현장 적용 승인과 생산성 향상 실증은 미완료.', 64, 578, 1152, 65, 27);
}

const candidatePath = path.join(buildDir, 'candidate.pptx');
await (await PresentationFile.exportPptx(pres)).save(candidatePath);
await fs.writeFile(path.join(buildDir, 'source_manifest.json'), JSON.stringify({
  evidence_date: '2026-09-22', sources: Object.fromEntries(await Promise.all(sources.map(async n => [n, await hash(n)]))),
  builder_sha256: await hash('build_presentation.mjs'), slides: allSlides.length,
  table_slides: tableOwners, chart_slides: chartOwners,
  fontPolicy: { basis: 'design', families: [font] },
  field_deployment_approved: false, competition_website_submission: false,
}, null, 2));
const result = await finalizePresentation({
  workspaceDir: root, candidatePath, finalPath, pythonExecutable: python,
  integrityValidatorPath: path.join(skill, 'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath: path.join(skill, 'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs: ['--expected-slide-size-emu', '12192000,6858000', '--validate-heading-fit',
    ...tableOwners.flatMap(n => ['--require-native-table-slide', String(n)])],
  requiredNativeTableOwnerSlides: tableOwners, requiredNativeChartOwnerSlides: chartOwners,
  materializeLiteralChartWorkbooks: true,
  fontPolicy: { basis: 'design', families: [font] }, verifyArtifactToolImport: true,
  receiptPath: path.join(buildDir, 'validation.json'),
});
console.log(JSON.stringify({ finalPath, slides: allSlides.length, result }, null, 2));
