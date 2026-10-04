# 과제 1번 추가 연구 결과 — precision_frontier_20261004_095727

| contract | pipeline | k | tp | fp | fn | tn | precision | recall | f1 | accuracy | ap | 설비 | 평가 | 구분 | 불확실성 | fit/상태 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| historical_pooled | pair_a1 | 144 | 12 | 132 | 21 | 1748 | 8.333% | 36.364% | 13.559% | 92.002% | 0.100295 | 전체 | 개발 1913행/33양성 | 3-seed 점수 앙상블 | 본문 조건부 bootstrap | 이번 GPU 737/CPU 6; 실행 완료 |
| batch_allocated | pair_a1 | 144 | 14 | 130 | 19 | 1750 | 9.722% | 42.424% | 15.819% | 92.211% | 0.100295 | 전체 | 개발 1913행/33양성 | 3-seed 점수 앙상블 | 본문 조건부 bootstrap | 이번 GPU 737/CPU 6; 실행 완료 |
| historical_pooled | pair_a0 | 144 | 11 | 133 | 22 | 1747 | 7.639% | 33.333% | 12.429% | 91.898% | 0.095824 | 전체 | 개발 1913행/33양성 | 3-seed 점수 앙상블 | 본문 조건부 bootstrap | 이번 GPU 737/CPU 6; 실행 완료 |
| batch_allocated | pair_a0 | 144 | 12 | 132 | 21 | 1748 | 8.333% | 36.364% | 13.559% | 92.002% | 0.095824 | 전체 | 개발 1913행/33양성 | 3-seed 점수 앙상블 | 본문 조건부 bootstrap | 이번 GPU 737/CPU 6; 실행 완료 |
| historical_pooled | pair_a0.25 | 144 | 12 | 132 | 21 | 1748 | 8.333% | 36.364% | 13.559% | 92.002% | 0.102959 | 전체 | 개발 1913행/33양성 | 3-seed 점수 앙상블 | 본문 조건부 bootstrap | 이번 GPU 737/CPU 6; 실행 완료 |
| batch_allocated | pair_a0.25 | 144 | 12 | 132 | 21 | 1748 | 8.333% | 36.364% | 13.559% | 92.002% | 0.102959 | 전체 | 개발 1913행/33양성 | 3-seed 점수 앙상블 | 본문 조건부 bootstrap | 이번 GPU 737/CPU 6; 실행 완료 |
| historical_pooled | pair_a0.5 | 144 | 13 | 131 | 20 | 1749 | 9.028% | 39.394% | 14.689% | 92.107% | 0.097181 | 전체 | 개발 1913행/33양성 | 3-seed 점수 앙상블 | 본문 조건부 bootstrap | 이번 GPU 737/CPU 6; 실행 완료 |
| batch_allocated | pair_a0.5 | 144 | 14 | 130 | 19 | 1750 | 9.722% | 42.424% | 15.819% | 92.211% | 0.097181 | 전체 | 개발 1913행/33양성 | 3-seed 점수 앙상블 | 본문 조건부 bootstrap | 이번 GPU 737/CPU 6; 실행 완료 |
| historical_pooled | rank_pairwise | 144 | 12 | 132 | 21 | 1748 | 8.333% | 36.364% | 13.559% | 92.002% | 0.102494 | 전체 | 개발 1913행/33양성 | 3-seed 점수 앙상블 | 본문 조건부 bootstrap | 이번 GPU 737/CPU 6; 실행 완료 |
| batch_allocated | rank_pairwise | 144 | 14 | 130 | 19 | 1750 | 9.722% | 42.424% | 15.819% | 92.211% | 0.102494 | 전체 | 개발 1913행/33양성 | 3-seed 점수 앙상블 | 본문 조건부 bootstrap | 이번 GPU 737/CPU 6; 실행 완료 |
| historical_pooled | rank_ndcg | 144 | 11 | 133 | 22 | 1747 | 7.639% | 33.333% | 12.429% | 91.898% | 0.082440 | 전체 | 개발 1913행/33양성 | 3-seed 점수 앙상블 | 본문 조건부 bootstrap | 이번 GPU 737/CPU 6; 실행 완료 |
| batch_allocated | rank_ndcg | 144 | 12 | 132 | 21 | 1748 | 8.333% | 36.364% | 13.559% | 92.002% | 0.082440 | 전체 | 개발 1913행/33양성 | 3-seed 점수 앙상블 | 본문 조건부 bootstrap | 이번 GPU 737/CPU 6; 실행 완료 |
| historical_pooled | mix_pairwise | 144 | 12 | 132 | 21 | 1748 | 8.333% | 36.364% | 13.559% | 92.002% | 0.120541 | 전체 | 개발 1913행/33양성 | 3-seed 점수 앙상블 | 본문 조건부 bootstrap | 이번 GPU 737/CPU 6; 실행 완료 |
| batch_allocated | mix_pairwise | 144 | 13 | 131 | 20 | 1749 | 9.028% | 39.394% | 14.689% | 92.107% | 0.120541 | 전체 | 개발 1913행/33양성 | 3-seed 점수 앙상블 | 본문 조건부 bootstrap | 이번 GPU 737/CPU 6; 실행 완료 |
| historical_pooled | mix_ndcg | 144 | 13 | 131 | 20 | 1749 | 9.028% | 39.394% | 14.689% | 92.107% | 0.123034 | 전체 | 개발 1913행/33양성 | 3-seed 점수 앙상블 | 본문 조건부 bootstrap | 이번 GPU 737/CPU 6; 실행 완료 |
| batch_allocated | mix_ndcg | 144 | 14 | 130 | 19 | 1750 | 9.722% | 42.424% | 15.819% | 92.211% | 0.123034 | 전체 | 개발 1913행/33양성 | 3-seed 점수 앙상블 | 본문 조건부 bootstrap | 이번 GPU 737/CPU 6; 실행 완료 |
| historical_pooled | nested_selected | 144 | 11 | 133 | 22 | 1747 | 7.639% | 33.333% | 12.429% | 91.898% | 0.105500 | 전체 | 개발 1913행/33양성 | 3-seed 점수 앙상블 | 본문 조건부 bootstrap | 이번 GPU 737/CPU 6; 실행 완료 |
| batch_allocated | nested_selected | 144 | 12 | 132 | 21 | 1748 | 8.333% | 36.364% | 13.559% | 92.002% | 0.105500 | 전체 | 개발 1913행/33양성 | 3-seed 점수 앙상블 | 본문 조건부 bootstrap | 이번 GPU 737/CPU 6; 실행 완료 |
| historical_pooled | historical_gpu | 144 | 13 | 131 | 20 | 1749 | 9.028% | 39.394% | 14.689% | 92.107% | 0.104775 | 전체 | 개발 1913행/33양성 | 3-seed 점수 앙상블 | 본문 조건부 bootstrap | 이번 GPU 737/CPU 6; 실행 완료 |
| batch_allocated | historical_gpu | 144 | 14 | 130 | 19 | 1750 | 9.722% | 42.424% | 15.819% | 92.211% | 0.104775 | 전체 | 개발 1913행/33양성 | 3-seed 점수 앙상블 | 본문 조건부 bootstrap | 이번 GPU 737/CPU 6; 실행 완료 |
| historical_pooled | historical_cpu | 144 | 9 | 135 | 24 | 1745 | 6.250% | 27.273% | 10.169% | 91.688% | 0.115419 | 전체 | 개발 1913행/33양성 | 3-seed 점수 앙상블 | 본문 조건부 bootstrap | 이번 GPU 737/CPU 6; 실행 완료 |
| batch_allocated | historical_cpu | 144 | 10 | 134 | 23 | 1746 | 6.944% | 30.303% | 11.299% | 91.793% | 0.115419 | 전체 | 개발 1913행/33양성 | 3-seed 점수 앙상블 | 본문 조건부 bootstrap | 이번 GPU 737/CPU 6; 실행 완료 |

## 최종 결과

**NO_ROBUST_GAIN**. 같은 역사적 K144 계약의 기존 GPU 앙상블 13건/131 FP에 대해, 내부 선택 절차는 11건/133 FP다. 배치 계약을 적용한 내부 선택 절차는 12건/132 FP이며 같은 계약의 새로 적합한 alpha=1 기준선 14건/130 FP와 비교해야 한다. 서로 다른 계약의 숫자를 모델 향상으로 합치지 않는다.

**구현 범위 차이를 발견했다.** 본 연구의 사전 고정 입력은 원본 24개 센서 열이다. 기존 코드는 값이 항상0인 Clamp_Open_Position을 제외하고 설비 표시 열을 붙여 다른 열 순서를 사용했다(23개 센서+표시). 설비별 적합에서는 추가 정보 차이가 없지만 트리 학습이 달라졌다. 따라서 이번 H1은 **새로 고정한 열 배치 안의 가중치 대조**이며 기존 변환기까지 완전히 보존한 순수 가중치 재현이라고 부를 수 없다. 이를 GPU 비결정성으로 설명하지 않는다. 발견 후 후보를 추가 선정하지 않고 기존 열 배치의 auto Balanced 기준선만 30 GPU fit으로 다시 적합했다. 과거 raw OOF와 최대 절대 차이0, K144 seed14/14/13·앙상블13으로 정확히 재현했다. 전체 엄격한 기존 열 배치 H1/H2 중첩 재실험은 적합 상한 내에 남은 예산이 부족해 수행하지 않았고, 이번 결과로 기존 번들을 교체하지 않는다.

검사량은 모두 CN7 73 + RG3 71 = 144행이다. 세 seed TP 평균이 아닌 실제 점수 앙상블을 평가했다. 8개 선택 가능 파이프라인과 CN7 고정 앵커를 비교했으며 row/event 8개는 원인 분석용 구성요소다. CPU·과거 GPU·중첩 선택 절차까지 전체 파이프라인 수는 사전 상한 12개 이내다.

## 같은 정책·설비·seed 비교

| contract | pipeline | machine | k | tp | fp | fn | precision | recall | ap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| historical_pooled | pair_a1 | cn7 | 73 | 10 | 63 | 3 | 0.136986 | 0.769231 | 0.325577 |
| historical_pooled | pair_a1 | rg3 | 71 | 2 | 69 | 18 | 0.028169 | 0.1 | 0.032099 |
| batch_allocated | pair_a1 | cn7 | 73 | 11 | 62 | 2 | 0.150685 | 0.846154 | 0.325577 |
| batch_allocated | pair_a1 | rg3 | 71 | 3 | 68 | 17 | 0.042254 | 0.15 | 0.032099 |
| historical_pooled | nested_selected | cn7 | 73 | 10 | 63 | 3 | 0.136986 | 0.769231 | 0.325577 |
| historical_pooled | nested_selected | rg3 | 71 | 1 | 70 | 19 | 0.014085 | 0.05 | 0.030889 |
| batch_allocated | nested_selected | cn7 | 73 | 11 | 62 | 2 | 0.150685 | 0.846154 | 0.325577 |
| batch_allocated | nested_selected | rg3 | 71 | 1 | 70 | 19 | 0.014085 | 0.05 | 0.030889 |
| historical_pooled | historical_gpu | cn7 | 73 | 10 | 63 | 3 | 0.136986 | 0.769231 | 0.325577 |
| historical_pooled | historical_gpu | rg3 | 71 | 3 | 68 | 17 | 0.042254 | 0.15 | 0.033914 |
| batch_allocated | historical_gpu | cn7 | 73 | 11 | 62 | 2 | 0.150685 | 0.846154 | 0.325577 |
| batch_allocated | historical_gpu | rg3 | 71 | 3 | 68 | 17 | 0.042254 | 0.15 | 0.033914 |

| contract | pipeline | 20260922 | 20260923 | 20260924 | ensemble |
| --- | --- | --- | --- | --- | --- |
| batch_allocated | historical_cpu | 10 | 10 | 10 | 10 |
| batch_allocated | historical_gpu | 15 | 15 | 14 | 14 |
| batch_allocated | mix_ndcg | 14 | 15 | 14 | 14 |
| batch_allocated | mix_pairwise | 13 | 13 | 13 | 13 |
| batch_allocated | nested_selected | 13 | 12 | 12 | 12 |
| batch_allocated | pair_a0 | 12 | 11 | 11 | 12 |
| batch_allocated | pair_a0.25 | 13 | 13 | 13 | 12 |
| batch_allocated | pair_a0.5 | 14 | 14 | 13 | 14 |
| batch_allocated | pair_a1 | 14 | 14 | 13 | 14 |
| batch_allocated | rank_ndcg | 12 | 12 | 12 | 12 |
| batch_allocated | rank_pairwise | 14 | 14 | 14 | 14 |
| historical_pooled | historical_cpu | 9 | 9 | 9 | 9 |
| historical_pooled | historical_gpu | 14 | 14 | 13 | 13 |
| historical_pooled | mix_ndcg | 12 | 13 | 13 | 13 |
| historical_pooled | mix_pairwise | 12 | 12 | 12 | 12 |
| historical_pooled | nested_selected | 12 | 11 | 11 | 11 |
| historical_pooled | pair_a0 | 11 | 11 | 11 | 11 |
| historical_pooled | pair_a0.25 | 12 | 12 | 12 | 12 |
| historical_pooled | pair_a0.5 | 13 | 12 | 13 | 13 |
| historical_pooled | pair_a1 | 13 | 13 | 12 | 12 |
| historical_pooled | rank_ndcg | 11 | 11 | 11 | 11 |
| historical_pooled | rank_pairwise | 12 | 12 | 12 | 12 |

CN7은 모든 새 후보에서 같은 StandardScaler + balanced logistic C=1을 유지했다. RG3 점수는 전체 배치에서 계산한 percentile rank의 평균이며 확률이 아니다. 한 RG3 행의 값 1은 100% 양성 확률을 뜻하지 않는다. 과거 pooled OOF와 새 배치별 정수 배정의 차이는 정책 효과다. 각 계약 내부에서는 원점수·순위 조합과 모델 효과를 비교했고 최종 동점 규칙은 source_row_id 오름차순으로 유지했다.

## 내부 선택과 불확실성

outer별 선택: 0: pair_a0, 1: mix_pairwise, 2: pair_a1, 3: pair_a1, 4: pair_a1, -1: rank_pairwise. -1은 모든 개발 데이터의 내부 OOF로 선택한 최종 연구 번들이다. outer 성능 순위를 보고 최종 번들을 고르지 않았다. 최종 번들의 선택된 구성은 **rank_pairwise**이며, 중첩 절차의 OOF 성능은 이 단일 최종 구성의 독립 시험 성능과 같지 않다.

기준선 alpha=1과 내부 선택 절차를 같은 그룹 draw로 2,000회 재표집했다. 설비·배치·그룹 크기 층화로 N1913/K144를 유지했다. ΔTP 95% 구간 [-4.000, 0.000], Δ정밀도 [-2.778, 0.000]%p, Δ재현율 [-12.821, 0.000]%p다. ΔFP와 설비별 AP 구간도 bootstrap_summary.json에 보존했다. 이는 고정 OOF 조건부 진단이며 탐색·재학습·미래 기간 불확실성을 포함하지 않는다. 독립 우위가 검증됐다고 주장하지 않는다.

## 수행·검증

- 원본·보존 split·기존 연구 번들 SHA를 실행 전후 대조했다. 개발 1913행/957그룹/33양성만 사용했고 과거 holdout 480행을 제외했다.
- 과거 산출물 해시 58개가 일치했다. 문서·코드·산출물 237개를 목록화하고 범위·링크를 기록했다. 부분 조사와 전체 정독을 구분하며 모든 문서를 완전히 읽었다고 주장하지 않는다.
- **이번 공식 데이터 GPU fit 737회, CPU fit 6회**. 과거 1,986회와 별도다. 사전 고정 후보·split seed·model seed·가중치·백엔드·시간·학습 그룹 해시를 fit 원장에 기록했다. 캐시 적합은 중복 횟수로 세지 않는다. synthetic smoke는 새로 수행하지 않았으며 실제 소형 첫 적합의 backend/config 검사를 이용했다.
- 본 연구 입력 배치에서 H1 row/event alpha=0/0.25/0.5/1, H2 pairwise/NDCG 및 사전 고정 50:50 혼합을 비교했다. qid는 각 RG3 학습 구간 전체를 한 query로 둔 가정이며 CV 그룹과 다르고 센서 특징에 포함하지 않았다.
- 별도 검증 프로세스 **2964개 확인**, 행별 OOF 97,563행, inner OOF 141,900행, 지표 1,368행 재계산. 테스트 48개, 실패 0, 오류 0, 생략 0. 실행한 관련 회귀 범위이며 저장소의 모든 서비스 시험을 다시 돌렸다는 뜻은 아니다.
- native 저장/복원과 outer OOF 일치, 행·열 순서, chunk 1/7/128에서 원점수 취합 후 전체 순위, singleton·중복·동점, NaN/Inf·미지원 설비·누락 특징·훼손 모델 거부를 확인했다. binomial/row BCE 손실·gradient 등가성도 검사했다.
- GPU 병렬 적합은 1개, CPU thread 2개, 시간 상한 2시간·GPU fit 상한790회. 디스크 여유 중단 후 완료 캐시를 유지하고 GC 주기를 낮춰 재개했다. 학습 중간 관측 메모리 최대 930MiB이며 실제 순간 peak를 측정했다고 주장하지 않는다. 원본 삭제·드라이버 변경·다른 프로세스 종료는 없었다.

## 실패 실험과 정보 한계

모든 alpha와 row/event, 순위 모델, 고정 혼합의 성적을 ensemble_metrics.csv에 남겼다. 결과가 낮은 후보도 삭제하지 않았다. 검사 집합 overlap, 양성 그룹 순위, 상충 그룹 기여, 동점 최소/기대/최대, 별도 group split 민감도는 각각 inspection_overlap.csv, rg3_positive_rank_audit.csv, failure_slices.csv, tie_sensitivity.csv, group_split_sensitivity.csv에 있다. 고정 OOF budget 곡선은 탐색 설명이며 새로운 최적 정책 선택이 아니다.

과거 가중 LR/RF/SVM/XGB/Cat/양성 복제와 센서 파생·규제·보정 비교는 historical_candidate_comparison.csv 및 원래 2026-09-26/10-04 보고서에서 추적한다. 같은 Cat alpha=1의 반복은 기준선 재현과 라이브러리 auto Balanced/명시 가중치의 정규화 확인 목적이다. hard-negative·새 특징·정책 재배정은 추가 정보와 적절한 교차적합 근거가 없어 이번 제한 연구에서 제외했다.

숫자 양성 33개 중 상충 그룹의 양성은 CN7 9개, RG3 20개다. 같은 센서 그룹 전체를 같은 결정으로 선택할 때 전부 회수하려면 최소 FP29, 정밀도 상한33/62=53.23%다. K144 고정 정밀도 상한은33/144=22.92%다. 이 oracle은 모델 성적이나 미래 보장이 아니다. 그룹 비용 n/이득 c의 DP oracle 전선을 별도로 보존했다. event max 라벨은 다른 target이며 정보를 새로 만든 것이 아니다.

원본의 파일별 사전 정규화·공통 scaler·24개 센서 도착시점·실제 shot/product/cavity·PassOrFail 코드북은 미확정이다. docs/04의23개 모델 센서는 상수1개 제외를 뜻하며24개 원본 그룹 센서와 모순이 아니다. RG3 순위와 CN7 확률 척도가 달라 pooled AP는 조심해서 해석한다. 전체 AP 상승을 의무적 성공으로 주장하지 않으며 설비별 AP와 두 설비 AP의 산술평균인 macro AP를 함께 저장했다. 비확률 순위에는 Brier/Log loss를 확률 평가처럼 계산하지 않으며 확률 형태의 원래 출력만 probability_diagnostics.csv에 미보정 진단으로 기록했다.

## 다음 행동과 보존

기존 연구 번들과 기존 품질검사를 유지한다. 새 번들은 검증된 추론 형식의 **연구 절차 산출물**이고 현장 채택/자동 출하/검사 생략 승인이 아니다. 새 기간·새 그룹·실제 라벨 의미를 확인한 데이터가 있어야 독립 성능을 주장할 수 있다. 구체적 필드·표본 규모·선택 확률 기록은 additional_data_request.md를 따른다.

원자료·행별 예측·native 모델은 로컬 `outputs/precision_frontier_20261004_095727`에 보존하며 공개 Git에는 코드·소형 집계·보고서만 넣는다. 재현은 run_precision_frontier.ps1에 새 출력 경로를 사용한다. 화면 잠금과 무관한 명령줄 작업이며 재현 런처는 실행 중에만 시스템 절전을 막고 종료 시 해제한다. 잠금 정책은 바꾸지 않는다.

## 1차 구현 근거

- [XGBoost 3.0.5 Learning to Rank](https://xgboost.readthedocs.io/en/release_3.0.0/tutorials/learning_to_rank.html): qid와 mean pair sampling을 설치 버전의 실제 config로 검증했다.
- CatBoost 공통 파라미터 문서 URL은 도구 접근 오류로 본문 확인에 실패했다. 설치된 라이브러리 get_all_params의 class_weights와 실제 auto Balanced 대조를 근거로 사용했다. 재학습 비결정성과 native 추론 일치는 구분한다.
