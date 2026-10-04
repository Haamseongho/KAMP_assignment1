# KAMP 과제① — GTX 1650 기반 오탐 감소·동일 검사예산 성능 개선 실행 프롬프트

작성일: 2026-10-04 KST  
기준 저장소: https://github.com/Haamseongho/KAMP_assignment1  
확인된 `main`: `e6f7d779db02321acbd4fcb4d34dcbb409a32192`  
결과 노션: https://app.notion.com/p/1-3eec9bbb7de58192bf13e3cfeec238cd

> **사용법:** 이 문서 전체를 Codex에 전달하고 해당 프로젝트를 작업 디렉터리로 연다. 아래는 실행할 연구 지시서이지, 이 문서 작성 과정에서 새 모델을 학습하거나 성능 향상을 달성했다는 보고서가 아니다. 설계 시점에는 사용자 PC의 원격 연결이 오프라인이었다. GPU와 로컬 산출물은 실행 시작 시 다시 확인한다. 현재 기준선·공식 원본·기존 결과는 보존한다.

---

## 1. 역할과 최종 목표

당신은 희귀사건 분류, 통계적 의사결정, 제조 데이터 품질, 순위 학습, 재현 가능한 GPU 실험을 담당하는 수석 연구자 겸 구현 책임자다. 논문 이름이나 모델 종류를 나열하는 데 그치지 말고, 이 저장소의 실제 코드와 데이터를 근거로 가설·실험·반증·검증·인계까지 수행하라.

목표는 **같은 검사량에서 실제 숫자 양성 발견 수를 높이고 오탐을 줄이는 것**이다. 정확도만 높이거나, 검사량을 줄여서 오탐 감소처럼 보이게 하거나, 최고 seed만 고르는 작업은 목표가 아니다. `PassOrFail=1`의 실제 불량 의미가 확인되지 않았다면 숫자 라벨 1 연구로 진행하고 현장 불량 성능이라고 부르지 않는다.

현재의 주요 가설은 다음과 같다.

- RG3의 오탐은 단순 GPU 연산량 부족만으로 설명되지 않는다. 동일 입력의 상충 라벨, 적은 양성 그룹, 손실함수·가중치와 검사 목표의 불일치, 순위 점수의 배치 의존성, 반복적인 개발 결과 선택을 분리해 검증해야 한다.
- 현재 관측치로 불가능한 구분과 아직 학습하지 못한 구분은 다르다. 정보 한계를 확인하되, 이를 근거로 추가 개선 시도를 포기하지 않는다.
- CN7의 기존 성능을 보존하면서 RG3를 우선 개선하는 제한된 실험이, 모든 모델·특징·파라미터를 동시에 바꾸는 실험보다 원인 해석에 유리할 수 있다. 이는 검증할 연구 설계이지 성능 보장이 아니다.

**최종 결과물:** 실행 코드, 재현 명령, 전체 후보 비교, 실패 원인, 행 단위 OOF 증빙, 검사예산별 결과, 불확실성, 연구 번들, 노션에 붙여 넣을 보고서를 만든다. 목표 미달도 정당한 결과로 기록하되, 실행 가능한 개선 방법을 실제로 시험하지 않고 일반론만 작성해서 끝내지 않는다.

## 2. 작업 범위·환경·보존 원칙

기존에 확인된 로컬 경로는 다음과 같다. 존재 여부와 Git 상태를 먼저 확인한다.

```text
C:\Users\haams\Documents\ChatGPT\KAMP\KAMP_assignment1
.venv-gpu\Scripts\python.exe
data\task01_official
outputs\gpu_pc_split_20261004_03
```

과거 Mac 경로 또는 `dev_haams`를 무조건 선택하지 않는다. 이 문서의 최신 결과 기준은 위 `main` 커밋이다. 로컬 HEAD가 다르면 차이와 미커밋 변경을 기록하고 작업을 보존한다. 자동 `reset --hard`, 강제 checkout, 강제 pull, 기존 출력 덮어쓰기를 하지 않는다.

허용 범위는 해당 로컬 프로젝트의 분석·코드 보완·테스트·승인된 로컬 자원에서의 연구 실행이다. GitHub push, Notion 게시·수정, 대회 제출, 유료 GPU 생성, 외부로 원자료 전송, 드라이버 교체, 다른 프로세스 강제 종료는 이 지시서에 포함되지 않는다.

새 출력은 `outputs/precision_frontier_<실제실행시각>/`에 생성하고, 문서·실험 설정·모델 이름에 실행 ID를 넣는다. 기존 번들, 원자료, split, 보고서는 불변으로 유지한다. 기존 API와 운영 정책은 새 연구 후보로 자동 교체하지 않는다.

GPU 기준은 이전 직접 진단과 최신 실행 기록에 있는 **GTX 1650 4 GiB / RAM 16 GiB / i5-10400F / Windows 11**이다. 이전 C 드라이브 여유는 17.5 GiB였으므로 디스크도 재확인한다. 과거 측정값을 현재 여유 자원으로 간주하지 않는다.

## 3. 먼저 모든 관련 문서·코드·산출물의 증거 지도를 만들어라

`AGENTS.md` 등 프로젝트 작업 지침이 있다면 확인한다. 가상환경·Git 내부·패키지 캐시를 제외하고 저장소의 모든 Markdown 파일을 조사한다. 문서에 연결된 JSON/CSV/로그·테스트·코드를 추적한다. 로컬 `outputs`의 실제 결과가 존재하면 공개 집계표보다 먼저 대조한다.

최소 확인 대상:

```text
README.md
GPU_PC_과제01_바로실행_인계_20261004.md
모델_성능고도화_실행계획.md
모델_고도화_실험결과_20260926.md
모델_고도화_로컬테스트_가이드.md
KAMP_과제01_검증성능_개선프롬프트_20261003.md
KAMP_과제01_GPU_고도화_추가검증_아웃라인_20261003.md
KAMP_과제01_추가실험_과제04_비교_20261003.md
KAMP_과제01_04_로컬검증_개선율_20261003.md
docs/03_사출성형_검증보고서.md
docs/04_추가데이터_요청명세.md
docs/05_라벨_의미_검증.md
docs/07_확률보정_및_조건부_오류분석.md
docs/18_과제01_로컬GPU_최종결과_20261004.md
reports/gpu_pc_20261004/*
research_split.py
moldguard.py
model_upgrade.py
upgrade_models.py
upgrade_audit.py
upgrade_runtime.py
calibration_diagnostics.py
task01_compare.py
task01_group_event_experiment.py
task01_gpu_nested_search.py
task01_gpu_ensemble_probe.py
task01_gpu_calibrated_ensemble.py
task01_gpu_balance_review.py
task01_gpu_candidate_bundle.py
task01_gpu_oof_evaluate.py
verify_task01_gpu_nested.py
tests/*
requirements-compare.txt
requirements-model-lock.txt
```

특히 다음 로컬 결과를 경로 존재·manifest·해시로 확인한다. 파일명만으로 완료를 추정하지 않는다.

```text
outputs/gpu_pc_nested_*/
outputs/gpu_pc_refinement_*/
outputs/gpu_pc_ensemble_*/
outputs/gpu_pc_review_*/
outputs/gpu_pc_candidate_20261004_01/bundle/
outputs/gpu_pc_balanced_queue_20261004_01/
outputs/my_upgrade_run_01/                  # 존재할 때만 과거 비교 증거로 확인
```

각 디렉터리에서 `experiment_plan.json`, `execution.json`, `run_manifest.json`, `artifact_hashes.json`, `fit_ledger*`, `oof_predictions*`, `inner_oof*`, `model_comparison.csv`, `selected_candidates.csv`, `failure_slices.csv`, `bootstrap_differences.csv`, `rg3_positive_rank_audit.csv`, `verification.json`을 실제로 찾고 내용을 읽어라.

`document_inventory.json`과 `evidence_map.md`에 경로, SHA-256, 읽은 범위, 실험 시점, 데이터·split·모델·정책·예산·seed·판정 단위, 연결 증거, 모순 사항을 남겨라. `fully_read / partially_read / inaccessible / not_present / not_applicable`을 구분하라. 전체 파일 트리 확인을 전체 내용 검증이라고 보고하지 않는다.

증거 우선순위는 **원본·분할의 동일성 → 실행 가능 코드와 저장된 행별 예측의 재계산 → 실행 원장·검증 기록 → 요약 문서**다. 문서에 숫자가 있다는 사실과 이번 실행에서 재검증했다는 사실을 구분한다.

원자료나 OOF가 없으면 코드를 복구하거나 접근 가능한 범위의 감사·테스트를 계속하되, `blocked_missing_source`로 표시한다. 합성 데이터의 테스트 성공을 공식 데이터의 학습·성능 검증으로 대체하지 않는다.

## 4. 기준선과 지표를 먼저 재현하라

다음은 설계 시점에 공개 집계표·보고서에서 확인한 기준이다. 이번 실행에서는 동일 행·분할·정책으로 재계산하라.

| 항목 | 기준 |
|---|---|
| 개발 데이터 | 1,913행, 동일 입력 957그룹, 숫자 양성 33행 |
| CN7 | 967행, 양성 13행, 484그룹 |
| RG3 | 946행, 양성 20행, 473그룹 |
| 제외된 과거 holdout | 480행. 신규 연구의 학습·선택·평가에 사용하지 않음 |
| 주 검사정책 | 설비별 상위 7.5%, CN7 73 + RG3 71 = 144행 |
| 보조 검사정책 | 5% = 97행, 10% = 192행 |
| 기존 CPU / K144 | TP 9, FP 135, FN 24 |
| 기존 CPU / K192 | TP 12, FP 180, FN 21 |
| 현 GPU 후보 / K144 | 세 seed 각각 TP 14/14/13; 세 seed 점수 앙상블 TP 13, FP 131 |
| 현 GPU 후보 / K192 | 세 seed 각각 TP 16/17/16; 세 seed 점수 앙상블 TP 16, FP 176 |
| 현 GPU 후보 / K97 | 세 seed 점수 앙상블 TP 11, FP 86 |
| 현 조합 | CN7 local logistic C=1 + RG3 CatBoost row/event rank mean |
| K144 설비별 앙상블 TP | CN7 10/13, RG3 3/20 |
| 모델 seed | 20260922, 20260923, 20260924 |

보존 split SHA-256:

```text
e26a82c58d0e86952b8557c2e8542f816f64db8ed49d15a4a450ef7a86b5efc7
```

중요한 구분:

- `mean(TP_seed)`와 `TP(mean_or_rank_ensemble(score_seed))`는 같지 않다. 14/14/13의 산술평균 13.67을 실제 13건인 점수 앙상블의 발견 수로 보고하지 않는다.
- 초기 전체 일괄 Top-K의 TP 9와 후속 설비별 K192의 TP 12는 검사 할당 정책이 다르다. 같은 이름의 CPU 기준선으로 섞지 않는다.
- 1,986 GPU fit은 과거 공식 데이터 실행 기록상의 합계다. 이번 실행 횟수와 합쳐 과장하지 않는다. 저장된 GPU fit 원장 확인과 모든 fit 재학습은 별개다.
- `PassOrFail=1` 행 수와 독립 양성 그룹 수를 따로 보고한다. 양성 복제·여러 seed·OOF 중복 저장은 독립 표본을 늘리지 않는다.

## 5. 최적화 문제를 수학적으로 고정하라

총 행 수 N, 숫자 양성 수 P, 검사 수 K, 양성 발견 t에 대해:

```text
TP=t
FP=K-t
FN=P-t
TN=N-P-K+t
Precision@K=t/K
Recall@K=t/P
F1@K=2t/(K+P)
PolicyAccuracy@K=1-(K+P-2t)/N
FPR=FP/(N-P)
FDR=FP/K=1-Precision@K
```

같은 N·P·K에서 t가 증가하면 정밀도·재현율·F1·정확도가 함께 개선되고 FP는 감소한다. 이 관계가 저장된 표에서 어긋나면 모델 학습 전에 행 집합·양성 정의·예산·평균 방식·계산 버그를 먼저 찾아라. FPR, FDR, FP 건수를 혼용하지 않는다.

**주 목적함수:** 고정 설비별 K144에서 TP를 최대화한다. CN7은 기존 앵커를 유지하고 RG3 개선을 우선한다. TP와 Precision@K가 동등한 목적이라는 점을 이용해 불필요한 복합 가중 지표를 만들지 않는다.

**보조 평가:** K97·K192에서의 TP/FP/FN, 설비별 AP, macro AP의 정의, ROC-AUC, 확률 점수가 있는 경우 Brier·Log loss, 적합 시간·메모리, seed·동점 안정성을 함께 보고한다. 전체 AP는 비교 가능한 점수 척도인지 확인한 뒤 해석한다. AP와 사다리꼴 PR-AUC도 구분한다.

**추가 정책 연구:** 같은 TP 수준에서 K 또는 FP를 줄이는 효율 전선도 따로 탐색한다. 예산이나 임계값의 선택은 outer-training 내부에서만 수행한다. 사후 OOF 곡선에서 가장 좋은 점을 선택했다면 탐색 결과로만 표기한다.

N=1913, P=33에서 항상 0을 예측하면 정확도 약 98.275%이지만 발견은 0이다. 이를 개선 모델로 선택하지 않는다. 또한 완전 회수라도 K144의 정밀도는 33/144=22.92%, K192는 17.19%를 넘을 수 없다. 고정 예산을 유지하면서 정밀도 80~90%를 목표로 적지 않는다.

기존 방식과 완전히 동일한 역사적 평가에서 K144 TP 13→15, FP 131→129는 유의미하게 검토할 수 있는 **연구 목표 예시**다. 달성 사실이나 통계적 확신이 아니다. 주 목적은 15에서 멈추는 것이 아니라 사전 고정한 계산 예산 안에서 가장 견고한 개선을 찾는 것이다.

## 6. 라벨·중복·식별 가능성 감사를 수행하라

### 6.1 라벨과 예측 시점

파일별 `PassOrFail` 코드북, 변환 이력, 샷·제품·캐비티 연결 키, 측정·검사·변수 도착 시각을 확인한다. 현재 문서에는 라벨 의미 미확정과 사전 정규화 범위 불명확이 기록돼 있다. 외부의 비슷한 데이터셋 설명만으로 이 파일의 라벨 방향을 확정하지 않는다.

의미 미확정은 현장 판정·배포 차단 조건이지만 숫자 라벨 1에 대한 연구 중단 이유는 아니다. 라벨 방향을 바꾸면 점수가 좋아진다는 이유로 뒤집지 않는다. 코드북이 새로 확보되면 데이터 버전과 연구 목표 변경을 명시하고 재실험한다.

24개 요약 센서를 사용하는 현재 설계는 공정 종료 후·최종검사 전의 입력 가용성을 가정한다. 센서 값이 실제로 그 시점까지 도착하는지 확인되지 않았다면 조기예측 성공이라고 주장하지 않는다.

### 6.2 동일 입력의 상충 라벨

그룹 키는 원칙적으로 `machine + 원래 24개 센서값`이며 라벨·행 ID를 포함하지 않는다. 부동소수점 파싱·플랫폼 차이를 확인하되 편의를 위해 반올림해서 서로 다른 그룹을 합치지 않는다. 기존 그룹 이름을 정렬해야 한다면 구성원의 일대일 대응을 입증한다.

각 그룹 g의 `n_g=행 수`, `c_g=양성 행 수`, `q_g=c_g/n_g`를 계산하고 pure-negative / pure-positive / conflicting을 구분한다. 이는 감사·학습 fold 내부 표현이며, 평가 그룹의 c_g를 예측 입력이나 사전 조회표로 사용하지 않는다.

기존 기록의 상충 양성 29행(CN7 9, RG3 20)을 검증한다. 같은 입력에 같은 결정을 하는 규칙은 완전 회수에서 최소 29개의 반대 라벨 음성을 함께 선택해야 한다. 이 가정하에서 완전 회수 정밀도 상한은 33/62=53.23%다. 이 값은 현재 데이터의 조건부 식별 가능성 진단이지 미래 분포의 보편적 상한이나 모델이 도달할 수 있다는 보장이 아니다. 낮은 회수율에서의 정밀도 상한과도 다르다.

가능하면 그룹별 비용 n_g와 획득 양성 c_g를 사용해 작은 동적 계획법으로 **oracle 그룹 선택 전선**을 계산하라. 동일 점수의 그룹 경계에서 일부 행만 고를 때는 무작위 동점 정책의 기대값·최소·최대값을 별도로 구한다. 정답을 아는 oracle 전선을 실제 모델 성능으로 표시하지 않는다.

핵심은 두 오류를 분리하는 것이다. ① 동일 센서에서 개별 제품을 구별하지 못하는 오류, ② 위험 그룹 자체를 상위에 올리지 못하는 오류. 현재 정밀도가 낮다고 해서 두 번째 오류를 더 줄일 여지가 없다고 결론 내리지 않는다.

### 6.3 그룹 집계가 자동으로 새 정보를 만들지는 않는다

다음 binomial 손실을 진단 기준으로 사용한다.

```text
L = -sum_g [c_g*log(p_g) + (n_g-c_g)*log(1-p_g)]
```

동일한 p_g에 대해 `soft_target=c_g/n_g, sample_weight=n_g`인 교차엔트로피는 원래 행 단위 무가중 교차엔트로피와 대수적으로 같다. 손실·gradient 등가성 단위 테스트를 작성한다. 트리 histogram·샘플링·정규화 때문에 구현 결과가 달라질 수 있으므로 실제 모델까지 반드시 같다고 단정하지 않는다.

`max(y_g)`를 사용하는 기존 event 모델은 “그룹에 하나라도 1이 존재하는가”라는 다른 목표다. 기존에도 시험했으므로 새 방법처럼 재포장하지 않는다. `feature_group`은 같은 측정값 집합이지 확인된 실제 샷이 아니다. 행 단위 평가를 이벤트 단위 평가로 몰래 바꾸어 FP를 제거하지 않는다.

## 7. 이미 실패한 접근과 특징 생성 위험을 정리하라

기존 original/refinement 후보에서 가중 로지스틱, RF, SVM, XGBoost, CatBoost row/event, 양성 5배 복제, 깊이·규제 조정, 센서 비율·온도 차이, sigmoid 보정이 어느 정책에서 어떤 결과를 냈는지 표로 정리한다. 같은 실험을 반복해야 한다면 재현·버그 확인·새 가설 중 목적을 적는다.

원본 CSV의 변수는 파일별 평균 0·표준편차 1에 가까운 사전 정규화 흔적이 보고돼 있다. 따라서 표준화된 압력끼리 나눈 값을 물리적 압력비라고 부르거나, `abs(분모)`를 매우 작은 epsilon으로 대체해 거대한 비율 특징을 무분별하게 추가하지 않는다. 파일별 스케일러가 다르면 비라벨 71,180행으로의 점수 이동도 별도 문제다. 검증 가능한 원시 단위·공통 scaler가 없으면 이를 되돌렸다고 주장하지 않는다.

`Unnamed: 0`, source_row_id, 행 순서, 라벨 파일 여부를 센서 특징으로 사용하지 않는다. 실제 시간순이 입증되기 전에는 row ID로 lag·rolling feature를 만들지 않는다. 특징 선택·정규화·클리핑·결측 처리·차원 축소는 각 학습 구간에서만 적합한다.

상충 라벨 제거, 성능이 좋은 행만 선택, SMOTE/복제로 독립 양성 수 증가 주장, 비라벨을 전부 음성으로 간주, 정답 없는 큐의 정밀도·정확도 계산은 금지한다. 이상 탐지나 pseudo-labeling은 불량 탐지 개선의 자동 대체물이 아니다.

## 8. 검증 절차를 동결하라: 과거 재현과 새 절차를 구분한다

### 8.1 역사적 재현 트랙

기존 outer 5-fold와 정책·split hash·모델 seed를 그대로 사용해 표의 기준선을 재현한다. 이 트랙은 기존 결과와의 연속성 확인용이다. 과거에 반복 열람한 개발 OOF와 480행 holdout을 새 미사용 테스트라고 부르지 않는다.

### 8.2 새 절차의 nested 개발 평가 트랙

첫 본 실험 전에 `experiment_plan.json`을 확정하고 해시를 저장한다. 후보, 목적함수, 예산, tie 규칙, seed, 전처리, 조기종료 기준, 조합 비율, 계산 상한, 선택 알고리즘을 기록한다. 이전 결과를 이미 알고 설계했다는 사실도 남긴다.

각 outer-training 안에서만 inner group 3-fold를 구성한다. 같은 센서 그룹이 fit/validation에 교차하지 않게 한다. 실제 샷·lot·시간 의존성이 확인되면 그 누수 단위를 우선하되 기존 그룹도 분리하지 않는다. 실제 시각이 없는 경우 시간 검증을 했다고 부르지 않는다.

모든 후보에 같은 inner 분할을 사용하고 `split_seed`와 `model_seed`를 구분한다. 전처리, 클래스 비율, 특징 선택, 모델 선택, 앙상블 가중치, 보정, 임계값, hard-negative 선정은 모두 outer-training 내부에서 끝내야 한다. outer-validation 라벨로 early stopping도 하지 않는다.

선택은 **실제 사용할 3-seed 앙상블**의 inner OOF 결과를 기준으로 한다. 각 seed 중 최고 모델을 고르지 않는다. 모델 seed 반복·같은 outer 분할·재학습 반복을 독립 데이터셋 수로 세지 않는다.

inner에서 정한 절차로 outer-training을 적합하고 outer-validation을 예측한 뒤 선택을 되돌리지 않는다. 모든 outer 결과를 본 뒤 후보를 새로 추가하면 별도 탐색 연구로 분리한다. outer 결과가 목표에 미달한다고 seed·fold·비율을 계속 바꿔 통과할 때까지 반복하지 않는다.

추가 group split 반복은 안정성 민감도 진단일 뿐 데이터가 새로 생기는 것은 아니다. 최종 일반화 확인에는 새 기간·새 그룹의 독립 라벨 데이터가 필요하다.

## 9. 평가와 번들의 순위 계산을 같은 계약으로 맞춰라

다음 코드 사실을 우선 점검한다.

- 기존 `task01_gpu_ensemble_probe.py`는 RG3 각 모델 점수를 outer fold 내부에서 percentile rank로 만든 뒤 조합한다.
- 기존 `task01_gpu_candidate_bundle.py::rank_scores`는 실제 전달된 RG3 입력 배치 안에서 모델별 rank를 계산하고 평균한다.
- 따라서 이 점수는 고정된 불량 확률이 아니며, 원점수가 같은 한 행도 함께 들어온 배치에 따라 결합 순위가 달라질 수 있다. RG3 한 행만 입력하면 각 percentile rank는 1이 된다. 이를 확률 100%로 해석해서는 안 된다.

이 구조를 곧바로 구현 버그나 누수로 단정하지 말고, 의도한 배치 검사 정책인지 확인하라. 다음 계약 중 목적에 맞는 하나를 새 연구 전에 고정한다.

**A. 배치 검사 정책:** 전체 검사 배치를 점수 계산의 입력 단위로 명시한다. 모델 원점수는 chunk별 추론할 수 있지만 percentile rank는 전체 배치 원점수를 합친 후 계산한다. chunk별 rank를 평균·연결하지 않는다. 배치 구성 변화에 대한 의존성은 정상적인 정책 특성으로 기록한다.

**B. 고정 참조 점수 정책:** 필요할 때만 outer-training 내부에서 만든 참조 분포/CDF 또는 단순 보정·앙상블을 동결한다. 실서비스에서도 같은 참조를 사용한다. 학습 전체 적합 점수와 inner OOF 점수 분포의 차이도 점검한다. 새 OOF나 비라벨 전체 분포로 참조를 몰래 재적합하지 않는다. 참조 방식은 별도 후보이며 추가 성능 보장이 아니다.

배치 정책을 검증할 때 각 outer-validation을 가상의 평가 배치로 취급하는 결과와 과거처럼 OOF 전체를 합친 결과를 분리한다. 각 설비의 총 K를 유지해야 한다면 fold별 행 수에 비례한 사전 고정 정수 할당(최대 나머지 방식 등)을 **라벨 없이** 적용한다. 모든 fold에 독립 ceil을 적용해 총 검사량이 144보다 늘어나지 않게 한다. 두 평가 설계가 같은 TP를 내야 한다고 강제하지 않는다.

새 평가 계약에서 기준선 값이 달라지면 기준선과 후보를 함께 재채점하고 paired 차이를 보고한다. 역사적 13건과 다른 계약의 후보를 직접 비교하지 않는다.

테스트에는 입력 행 순서 변경, 특징 열 순서 변경, chunk 크기 변경 후 전체 배치 복원, 단일 RG3 행, 동일 센서 중복, score tie, 저장·복원, 지원하지 않는 설비, NaN/Inf, feature 누락을 포함한다. 최종 Top-K tie는 명시적 label-free 규칙으로 통일하고 과거 `source_row_id` 기준과 다른 정책은 별도 효과로 기록한다.

## 10. 1순위 실험: RG3 CatBoost 가중치의 영향만 분리한다

현재 최종 GPU 후보 정의에서 CatBoost row/event 경로는 `auto_class_weights='Balanced'`다. 먼저 과거 모든 관련 실험에 다른 가중치가 있었는지 조사해 중복 실험을 제외한다.

**가설 H1:** 작은·상충 라벨 데이터에서 완전 균형 가중이 유한한 트리 모델과 규제에 상호작용하면서 상위 검사 순위를 악화시킬 수 있다. 이를 검증하기 위해 다른 설정은 유지하고 가중치만 조절한다. 원인으로 이미 확정하지 않는다.

각 실제 fit의 target 표현(row 또는 event)에서 계산한 `r=n_negative/n_positive`에 대해:

```text
positive_weight = r**alpha
alpha ∈ {0, 0.25, 0.5, 1}
negative_weight = 1
```

- alpha=1은 기존 Balanced와의 재현 비교를 수행한다. 라이브러리 내부 정규화 차이도 확인한다.
- 명시적 `class_weights`와 `auto_class_weights` 또는 `scale_pos_weight`를 중복 적용하지 않는다.
- 첫 실험은 iterations=200, depth=3, learning_rate=0.03, l2_leaf_reg=10, border_count=64 등 기존 구성을 보존한다. CPU/GPU 기본값에 기대지 말고 실제 설정을 저장한다.
- row/event를 각각 평가하고, 주 후보는 두 경로에 같은 alpha를 적용한 기존 형태의 rank 평균이다. 처음부터 row alpha × event alpha × 깊이 × 규제 × 특징의 거대한 조합을 만들지 않는다.
- CN7 local logistic C=1은 고정한다. 다른 설비의 개선 때문에 CN7을 불필요하게 바꾸지 않는다.
- 비교는 같은 K144, 같은 평가 계약, 같은 입력 행, 같은 tie 규칙으로 한다.

수학적 주의: 자유로운 확률 함수에 대한 양성 가중 BCE의 최적값은 `p_w = w*q / (1-q+w*q)`다. 고정 w>0에서는 q의 단조 변환이므로 이상적 조건에서 클래스 가중치만으로 순위가 좋아지는 것은 아니다. 이번 실험은 유한 모델·정규화·샘플링의 영향을 확인하는 것이다. 가중치를 낮춰 점수 크기만 줄고 Top-K 순위가 같다면 오탐 개선으로 보고하지 않는다.

검사 집합의 overlap, 위험 그룹의 상승·하락, seed별 TP/FP, 상충·비상충 그룹 기여, RG3 AP 변화를 비교하라. 성능이 낮은 alpha도 삭제하지 않는다.

## 11. 2순위 실험: 분류 손실 대신 순위 목적을 제한적으로 비교한다

**가설 H2:** 제한된 검사 예산에서 위험 항목을 먼저 배치하는 목적에는 순위 학습이 적합할 수 있다. 하지만 양성 그룹이 적으므로 반드시 기존 분류 모델보다 좋다고 가정하지 않는다.

`requirements-compare.txt`에는 XGBoost 3.0.5가 고정되어 있다. 실제 설치 버전을 먼저 확인하고 해당 버전의 공식 API로 구현한다. 최신 문서에 있는 옵션을 검증 없이 복사하거나 동작하던 환경을 무조건 업그레이드하지 않는다.

RG3 전용 `XGBRanker` 두 개만 우선 비교한다.

```text
objective ∈ {'rank:pairwise', 'rank:ndcg'}
tree_method='hist'
device='cuda'
max_depth=2
n_estimators=250
learning_rate=0.03
reg_lambda=10
max_bin=64
lambdarank_pair_method='mean'
lambdarank_num_pair_per_sample=4
```

위 숫자는 새로운 연구의 제한된 시작 설정이지 보장된 최적값이 아니다. 해당 버전에서 지원되는지 smoke fit과 저장된 config로 확인한다. 작동하지 않는 인수를 조용히 무시하지 않는다.

**qid 설계가 핵심이다.** `feature_group`과 순위 학습의 query group은 다르다. 동일 센서값의 양성·음성 두 행만 하나의 qid로 묶으면 구별할 수 없는 쌍만 비교하게 된다. qid는 실제로 같은 검사 예산을 두고 경쟁하는 배치·lot 등을 나타내야 한다.

실제 배치 ID가 없으면 RG3의 각 학습 구간을 하나의 검사 모집단 query로 사용하는 제한된 순위 surrogate를 명시할 수 있다. 이를 실제 lot 검증으로 부르지 않는다. CV 누수 방지는 기존 센서 동일 그룹으로 수행하고, 그 뒤 학습 부분에서 qid를 구성한다. qid를 센서 특징으로 사용하지 않는다. 지도학습 결과가 잘 나오도록 인위적으로 query를 나누지 않는다.

pair 수는 독립 데이터 수가 아니다. 원자료에서 정보가 같은 상충 쌍은 순위 손실로도 개별 구분할 수 없다. pairwise 목적의 점수를 확률로 표시하지 않는다. NDCG 최적화 결과도 주 지표인 TP@K144로 검증한다.

순위 모델 단독과 기존 CatBoost pair의 0.5 고정 순위 혼합을 후보로 둘 수 있다. 혼합은 동일 배치 또는 고정 참조 계약에서 수행한다. 비율을 바꾸려면 inner에서만 선택하고 후보 수 상한에 포함한다. outer 결과를 본 뒤 여러 혼합 비율을 추가하지 않는다.

## 12. 선택적 3순위: 반드시 새로운 근거가 있을 때만 수행한다

다음은 기본 필수 실험이 아니다. 우선 실험의 원인 분석과 계산 예산에 따라 수행하거나 명확한 이유로 제외한다.

### 12.1 단순 통계 모델

설비별 절편·기울기에 축소 규제를 적용한 부분 풀링 로지스틱 또는 작은 additive 모델은 CPU로 비교할 수 있다. 기존 설비 상호작용·로지스틱 실험과 동일하다면 반복하지 않는다. 희소한 양성에 비해 특징 자유도를 크게 늘리지 않는다. 규제·특징·모형 선택은 inner에서 수행하고 독립 검증으로 오인하지 않는다.

### 12.2 2단계 재정렬과 hard-negative mining

먼저 기준 모델이 학습 구간에서 만든 **cross-fitted** 후보 중 오탐 유형이 반복되는지 확인한다. outer-validation의 FP/FN 목록으로 두 번째 모델을 학습하지 않는다.

stage-1 후보 선택과 stage-2 학습·검증까지 outer-training 안에서 교차적합해야 한다. 다른 학습 행의 OOF 특징을 생성한 모델이 현재 검증 행의 라벨을 본 적 없는지도 추적한다. 이 중첩 관계를 보장할 수 없으면 해당 실험을 제외한다.

stage-1에서 탈락한 양성은 stage-2가 복구할 수 없다. 전체 cascade의 TP/FP/FN/K를 최종 집계하고 선별된 부분집합의 정밀도만 보고하지 않는다. 검사 보류·2차 검사가 있으면 총 검사 비용과 놓친 양성을 포함한다. 두 번째 센서·이미지 같은 추가 정보가 없는 두 단계 모델은 새 정보를 만든 것이 아니므로 소표본 과적합 위험을 함께 평가한다.

## 13. 모델 개선과 검사정책 개선을 분리한다

주 실험의 CN7 73/RG3 71 배정은 유지한다. 별도 정책 연구에서만 총 K=144 안에서 설비별 배정을 조절한다. 가중 평균·전체 순위로 CN7과 RG3의 원점수를 직접 비교하지 않는다.

예산 재배정은 inner 예측으로만 설비별 비용·획득 곡선을 추정하고, 사전 고정한 최소 검사량과 불확실성을 고려해 정한다. RG3를 거의 검사하지 않아서 전체 정밀도가 좋아진 경우를 RG3 모델 개선으로 표현하지 않는다. 다음 네 경우를 분리한다.

```text
기존 모델 + 기존 정책
새 모델 + 기존 정책
기존 모델 + 새 정책
새 모델 + 새 정책
```

확률 보정은 필요한 의사결정을 지원하는 보조 실험이다. 단일 엄격 단조 보정은 같은 모델의 순위를 바꾸지 않는다. fold별 보정·모델 혼합·설비 간 점수 결합은 다르므로 별도 확인한다. 보정된 점수의 Brier 개선을 TP@K 개선으로 바꾸어 보고하지 않는다. 작은 보정 샘플에 복잡한 isotonic이나 많은 구간별 임계값을 기본 적용하지 않는다.

사업 비용이 없으면 FN/FP/검사비의 임의 비용비를 현장 최적값처럼 정하지 않는다. 필요하면 가정별 민감도 표를 제시한다. 검사 대기열은 연구용이며 기존 품질검사를 유지한다.

## 14. 실험 선택·통계·중단 규칙

### 14.1 선택 알고리즘

기본 비교는 CPU 기준선, 현 GPU 후보, H1 alpha 3개 추가 후보, H2 순위 2개와 필요한 고정 혼합 정도로 제한한다. 기본 전체 후보 파이프라인 상한은 12개다. 동일 base fit을 여러 예산으로 채점했다고 별도 독립 실험 수로 부풀리지 않는다.

inner의 주 선택 기준은 같은 검사예산에서의 TP다. 동률이면 사전 정의한 RG3 AP, 단순성, 적합 비용의 순으로 선택한다. 미래 독립 성능을 보장하는 절차라고 쓰지 않는다. 큰 변화가 없는 후보를 무수한 seed와 임계값으로 재검색하지 않는다.

CN7은 앵커 고정으로 보존한다. 주 연구 후보의 RG3 이득과 전체 이득을 별도로 보고한다. 모든 seed와 3-seed 앙상블을 공개한다. 최악 seed·동점 경계·추가 group split 민감도를 확인하되, 좋은 결과가 나오는 반복만 선택하지 않는다.

### 14.2 불확실성

고정 OOF 점수에 대해 동일 입력 그룹 전체를 보존하는 **paired group bootstrap**을 수행한다. 기준선과 후보에는 동일 재표집을 사용한다. 설비와 그룹 크기로 층화해 행 수·고정 K를 유지하거나, 변동하는 행 수에 대응한 비율 정책을 명시한다. 두 방법을 섞지 않는다. 중복 추출된 그룹의 draw ID도 구분한다.

기본 2,000회 재표집에서 ΔTP, ΔPrecision@K, ΔRecall@K, ΔFP, 설비별 AP의 구간을 산출한다. 숫자 양성이 없는 재표집과 정의되지 않는 지표의 처리 규칙을 사전에 기록한다. fold 표준편차/루트 fold 수를 독립 표본 기반 신뢰구간처럼 사용하지 않는다.

이 구간은 **고정된 OOF와 선택된 절차에 조건부인 진단**이다. 후보 탐색·전체 재학습·새 기간 불확실성을 모두 포함한 구간이라고 주장하지 않는다. 구간이 0을 포함하면 통계적으로 우월함이 입증됐다고 쓰지 않는다. 0을 제외하더라도 개발 데이터의 반복 열람 때문에 독립 검증을 대체하지 않는다.

전향적 시험 설계에는 기대 발생률, 필요한 양성 그룹 수, precision/recall 구간 폭, 그룹 의존성을 명시하고 가정별 표본 규모를 계산한다. 개발 데이터의 발생률이 미래에도 같다고 단정하지 않는다.

### 14.3 결과 등급

- `REPRODUCTION_ONLY`: 기존 결과의 재현·정정만 완료.
- `RESEARCH_CANDIDATE`: 동일 정책에서 유망한 개선을 관찰했으나 독립 검증 미완료.
- `NO_ROBUST_GAIN`: 일관된 개선 없음. 실패 후보와 원인·다음 정보 요구를 남김.
- `BLOCKED_DATA_OR_RUNTIME`: 필요한 데이터·라벨·환경에 접근 불가. 가능한 감사와 테스트는 제공.
- `PROSPECTIVE_VALIDATED`: 사전 고정 절차를 새 독립 라벨에서 평가하고 승인 조건을 충족한 경우에만 사용.

코드·누수·지표 재계산 실패 시 모델 채택 금지. 개선이 없거나 불확실하면 기존 연구 번들을 보존한다. 가설이 반증되었다는 이유로 실패 기록을 지우거나 실험 완료 사실을 숨기지 않는다.

## 15. GTX 1650 4 GiB에 맞는 실행 계획

### 15.1 사전 확인

PowerShell에서 다음과 같은 **조회**를 실행하고 결과를 저장한다. 존재하지 않는 실행 파일을 가정하지 않는다.

```powershell
Get-Location
git status --short --branch
git rev-parse HEAD
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu --format=csv
Get-PSDrive C
Test-Path .\.venv-gpu\Scripts\python.exe
.\.venv-gpu\Scripts\python.exe --version
.\.venv-gpu\Scripts\python.exe -m pip freeze
.\.venv-gpu\Scripts\python.exe task01_gpu_preflight.py --help
```

`nvidia-smi`의 CUDA Version을 설치된 Toolkit/PyTorch 버전으로 보고하지 않는다. 저장소의 기존 preflight·smoke·backend 검증을 실제 CLI에 맞게 재사용한다. 별도 시스템 Python이나 최신 CUDA 설치를 첫 해결책으로 삼지 않는다.

### 15.2 자원 정책

- GPU 동시 fit은 1개, CPU thread는 기본 2개로 시작한다. 다른 작업이 없는 것이 확인된 경우에만 합리적으로 조정한다.
- 기존 CatBoost의 `gpu_ram_part=0.45`를 출발점으로 삼는다. 메모리 여유에 따라 조절하지만 이 옵션만으로 전체 GPU 메모리가 엄격히 제한된다고 가정하지 않는다.
- XGBoost는 `hist + cuda`, 작은 깊이·bins를 우선한다. GPU fit 후 실제 backend와 모델 config를 확인한다.
- 작은 데이터는 GPU가 CPU보다 빠르다고 가정하지 않는다. 기준 적합의 wall time·peak memory를 측정하고 단순 선형 모델·통계·검증은 CPU에서 수행한다.
- GPU 이용률 100% 자체를 목표로 하지 않는다. 데이터를 복제해 부하를 만들거나 대형 Transformer/생성형 모델로 바꾸는 것을 고도화로 간주하지 않는다.
- 원본과 가상환경을 반복 복사하지 않는다. 행별 OOF는 정확한 dtype·정밀도를 보존하면서 압축 저장한다. 모든 중간 모델을 무조건 저장하지 말고 재현 설정·필요한 최종 native 모델·검증 원장을 보존한다.

기본 계산 상한 제안은 아래와 같다. 제조사 보장값이 아니라 사용자 PC 보호와 실험 남발 방지를 위한 연구 실행 예산이다. preflight의 실제 가용 자원에 맞춰 **성능 결과를 보기 전에** 더 보수적으로 조정할 수 있다.

```json
{
  "gpu_concurrency": 1,
  "cpu_threads": 2,
  "max_candidate_pipelines": 12,
  "max_new_gpu_fits": 800,
  "max_new_cpu_fits": 400,
  "max_wall_seconds": 14400,
  "max_output_gib": 2,
  "min_disk_free_gib": 5,
  "gpu_free_reserve_mib": 768,
  "bootstrap_repetitions": 2000
}
```

wall time 값은 완료 시간 예측이 아니라 실행 상한이다. 상한 도달·OOM·디스크 부족·다른 GPU 작업과 경합 시 안전하게 checkpoint하고 중단 상태를 남긴다. 실패한 GPU fit을 조용히 CPU로 실행한 뒤 GPU 결과로 기록하지 않는다. 사용자 파일을 자동 삭제하거나 다른 프로세스를 종료해 자원을 확보하지 않는다.

OOF cache는 원본·학습 행·split·seed·모델 설정·전처리·코드·라이브러리 해시가 일치할 때만 재사용한다. 전체 데이터로 적합된 변환기나 bundle을 inner/outer 예측에 재사용하지 않는다.

기본 실행 순서는 **감사/재현 → 가중치 실험 → 순위 실험 → 제한적 추가 실험 → 검증/보고**다. 외부 검증 라벨을 본 뒤 결과가 좋은 실험만 남기는 pruning을 하지 않는다. 자원 부족 시 완료된 균형 비교 단위까지만 보고하고 미실행 후보를 구분한다.

CatBoost GPU의 부동소수점 연산 비결정성을 고려해 “모든 재학습의 바이트 동일”을 요구하지 않는다. 대신 source/split/config 동일성, 저장·복원 추론 일치, 동일 입력 동일 점수, 지표·검사 집합의 변동을 구분하여 확인한다.

## 16. 새 정보를 얻는 경로도 구체적인 산출물로 남겨라

현재 센서만으로 개선이 제한되는 경우, 단순히 “데이터를 더 모으라”로 끝내지 않는다. 다음 필드를 갖춘 `additional_data_request.md`와 라벨 수집 표본 설계를 작성한다.

```text
machine_id / physical_shot_id / product_id / cavity_id / lot_id
shot_start / shot_end / sensor_available_at / inspection_at
원시 단위 센서와 scaler 학습 범위·변환 버전
PassOrFail 코드북 / 실제 검사값 / 재검사·판정 변경 이력
가능한 추가 신호: 캐비티별 센서, 제품 치수, 검사 이미지, 원재료·금형·정비 정보
```

추가 신호는 실제 존재·사용권한·예측 시점 가용성을 확인하고 나서만 사용한다. 제품의 검사 정답을 예측 이전 특징에 섞지 않는다.

추가 라벨링 예산은 대표성 있는 무작위 표본과 위험·불확실성·모델 불일치 표본을 구분해 배정한다. 선택 확률과 그룹·시각을 남긴다. 선택적으로 라벨을 받은 고위험 표본의 양성률을 전체 데이터 precision/recall로 보고하지 않는다. 역확률 보정도 선택 확률·지원 영역 등의 가정이 만족될 때만 사용한다.

새 기간의 독립 평가용 표본은 별도로 보존하고, 그 정답을 active learning이나 모델 선택에 쓰지 않는다. 비라벨 71,180행을 점수화하거나 pseudo-label을 붙여도 실제 새 정답 양성이 생긴 것이 아니다.

## 17. 반드시 생성할 산출물과 자동 테스트

기존 프로젝트 구조를 재사용하고 불필요한 새 프레임워크를 만들지 않는다. 아래 이름은 신규 산출물 요구사항이며 기존에 있다고 가정하지 않는다.

```text
outputs/precision_frontier_<run_id>/
  environment.json
  document_inventory.json
  evidence_map.md
  source_hashes.json
  data_contract.json
  collision_audit.csv
  identifiability_frontier.csv
  experiment_plan.json
  fit_ledger.csv
  selection_ledger.json
  inner_oof_predictions.csv.gz
  outer_oof_predictions.csv.gz
  baseline_reproduction.json
  policy_contract.json
  policy_parity_tests.json
  metrics_by_machine_seed_budget.csv
  ensemble_metrics.csv
  budget_pareto.csv
  paired_group_bootstrap.csv
  tie_sensitivity.csv
  failure_slices.csv
  resource_usage.csv
  experiment_failures.json
  verification.json
  decision.json
  artifact_hashes.json
  candidate_bundle/                         # 검증된 연구 후보가 있을 때만
  research_report.md
  notion_ready.md
  additional_data_request.md
  reproduction_commands.ps1
```

원장에는 후보·outer/inner fold·split/model seed·실제 fit 행/그룹·실제 양성 행/그룹·target 표현·가중치·backend·시작/종료·elapsed·실패 사유를 남긴다. full data 적합, inner 적합, outer 적합, synthetic smoke를 별도로 센다.

최소 자동 테스트:

1. 원본·split 불변, 행 ID/라벨 일치, train/validation 그룹 교차 0, holdout 제외.
2. OOF 완전 커버리지와 중복 없음. seed별 예측과 앙상블 예측의 식별자 정렬.
3. 동일 입력 동일 점수, tie 규칙·최소/최대 TP, K 합계·설비별 할당 일치.
4. TP/FP/FN/TN 합계와 직접 계산한 Precision/Recall/F1/Accuracy/FPR/FDR 일치.
5. binomial 집계와 row BCE 손실·gradient 등가성, event 목표와 행 목표 구분.
6. GPU backend 확인, 미지원 옵션·CPU fallback 탐지, 실패 fit이 성공으로 집계되지 않음.
7. 저장·복원 예측, 배치/chunk 계약, 행·열 순서, NaN/Inf·미지원 설비·훼손 번들 처리.
8. 후보 선택·가중치·보정·특징·early stopping이 outer 라벨을 보지 않았는지 원장 대조.
9. 비라벨 정답 지표 생성 금지, 기존 검사 유지, field_approved 기본 false.
10. 기존 회귀 테스트의 통과·실패·미실행 내역과 새 테스트의 적용 범위 구분.

별도 검증 함수/프로세스로 지표를 다시 계산한다. 생산 함수와 똑같은 함수를 호출한 결과만으로 독립 검증이라고 부르지 않는다. 환경 문제로 제외한 테스트를 전체 통과라고 보고하지 않는다.

## 18. 최종 보고 형식과 완료 조건

첫 화면에 다음 비교표를 넣는다.

```text
데이터/평가 계약 | 모델 | 정책 | 설비 | K | TP | FP | FN | TN |
Precision | Recall | F1 | 정책 Accuracy | AP | seed/앙상블 구분 |
불확실성 | GPU/CPU fit 수 | 실행 상태
```

본문은 아래 질문에 숫자와 파일 증거로 답한다.

- 기존 K144의 13건/131 FP와 **같은 조건**에서 얼마나 달라졌는가?
- 재현 또는 평가 계약 정정으로 기준선이 달라졌는가? 그렇다면 같은 새 계약의 paired 비교는 무엇인가?
- RG3에서 몇 건을 더 찾았고 CN7은 유지됐는가?
- 모델 개선, 예산 변경, 점수 변환, tie 변경 효과를 각각 분리했는가?
- 가장 좋았던 seed가 아니라 실제 배포 형태의 앙상블 결과는 무엇인가?
- 실패한 가설과 원인, 남은 식별 한계·정규화 문제·새 정보 요구는 무엇인가?
- 독립 데이터 없이 아직 주장할 수 없는 내용은 무엇인가?

`notion_ready.md`는 최종 결과, 같은 K 비교표, 수행·검증, 실패 실험, 한계, 다음 행동 순으로 쓴다. 실제 Notion 게시나 GitHub push는 하지 않는다. 원본 노션의 기존 숫자를 덮어쓴 것으로 보고하지 않는다.

완료는 목표 숫자의 달성만으로 판단하지 않는다. **사전 정의된 실험이 실행되고, 기준선과 같은 조건의 검증 가능한 결과가 있으며, 재현 명령·증거·실패/미실행 범위가 남아 있어야 한다.** 새 독립 성능이 없는 경우 최종 번들은 연구 후보이며 자동 출하 판정이나 검사 생략에 사용하지 않는다.

이제 위 순서대로 조사·구현·실험·검증하라. 막힌 부분은 명시하되 진행 가능한 작업까지 멈추지 말고, 실제 실행한 것과 계획만 남은 것을 구분한 결과를 제공하라.

---

## 근거 자료 — 구현 시 확인할 1차 출처

### 프로젝트 자료

- [P1] 제공된 노션 결과 페이지: https://app.notion.com/p/1-3eec9bbb7de58192bf13e3cfeec238cd
- [P2] 기준 커밋: https://github.com/Haamseongho/KAMP_assignment1/commit/e6f7d779db02321acbd4fcb4d34dcbb409a32192
- [P3] 기준 커밋의 `docs/18_과제01_로컬GPU_최종결과_20261004.md`, `GPU_PC_과제01_바로실행_인계_20261004.md`.
- [P4] `reports/gpu_pc_20261004/three_seed_mean_budgets.csv`, `verification.json`, `identifiability_bounds.json`.
- [P5] `task01_gpu_nested_search.py`, `task01_gpu_ensemble_probe.py`, `task01_gpu_candidate_bundle.py`, `research_split.py`.
- [P6] `docs/03_사출성형_검증보고서.md`, `docs/05_라벨_의미_검증.md`, `docs/07_확률보정_및_조건부_오류분석.md`.
- [P7] `모델_고도화_실험결과_20260926.md`, `KAMP_과제01_추가실험_과제04_비교_20261003.md`.
- [P8] 사용자 기존 진단 파일 `PC_Diagnostics_20261004.md`: 2026-10-04 00:33:46 KST부터의 과거 측정. 실시간 상태와 구분.

### 방법론·공식 구현 문서

- [R1] scikit-learn, Tuning the decision threshold for class prediction. 예측과 의사결정·임계값 조정의 구분. https://scikit-learn.org/stable/modules/classification_threshold.html
- [R2] Cawley & Talbot (2010), On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation, JMLR 11:2079–2107. 반복 선택의 편향. https://www.jmlr.org/papers/v11/cawley10a.html
- [R3] XGBoost 3.0.5, Learning to Rank. qid, pairwise/NDCG, 작은 데이터의 pair sampling 및 버전별 차이. https://xgboost.readthedocs.io/en/release_3.0.0/tutorials/learning_to_rank.html
- [R4] XGBoost 3.0.5, GPU Support. hist/cuda 설정과 메모리 사용. https://xgboost.readthedocs.io/en/release_3.0.0/gpu/index.html
- [R5] CatBoost, Common parameters. 클래스 가중치 설정·상호 배타 옵션. https://catboost.ai/docs/en/references/training-parameters/common
- [R6] CatBoost, FAQ. GPU 비결정성과 CPU/GPU 차이. https://catboost.ai/docs/en/concepts/faq
- [R7] CatBoost, Speeding up the training. 데이터 크기에 따른 GPU 효율. https://catboost.ai/docs/en/concepts/speed-up-training
- [R8] scikit-learn, Probability calibration. 확률 보정·순위·소표본 주의. https://scikit-learn.org/stable/modules/calibration.html

프로젝트 기준 수치는 [P1–P8]의 기록에 근거한다. 고정 K 지표 항등식, BCE 집계 등가성, 식별 가능성 계산은 해당 정의에서 도출한 수학적 검토다. H1/H2·계산 상한·추가 실험은 이 프로젝트를 위해 제안한 연구 설계이며, 위 논문이나 공식 문서가 이 데이터의 개선을 보장한다는 뜻이 아니다.
