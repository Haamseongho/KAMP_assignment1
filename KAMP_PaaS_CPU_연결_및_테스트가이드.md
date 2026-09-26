# KAMP 과제① — CPU 추론 연결 준비·로컬 검증

확인일: 2026-09-26. 대상 프로젝트: `215209147869625880` / `KAMP_assign1`.

후속 변경: 사용자 추가 지시로 무료 근거 확인을 기다리지 않는 **제한된 CPU 테스트**를 진행했다.
아래는 초기 준비 당시 기록이며, 최신 원격 등록·실행 상태는 [원격 테스트 진행 기록](docs/17_KAMP_CPU_원격테스트_진행기록.md)을 따른다.

## 현재 상태

**로컬 추론 연결 후보 및 재현 검증 완료, KAMP 원격 배포·추론은 미실행**이다.
추가 비용 0원이라는 근거를 찾지 못해 업로드/빌드/학습/서비스 시작을 하지 않았다.
대시보드 설명은 재직자 부문에서 일반국민·대학(원)생 과제① 내용으로 정정했다.
첫 저장은 HTTP 401로 실패했으나 사용자 재로그인 후 “수정이 완료되었습니다”를 확인했다.
수정 문구는 `paas_cpu/PROJECT_DESCRIPTION.txt`에 있다. 참가 신청 내역은 변경하지 않았다.
GPU/유료 자원/대회 최종 제출은 수행하지 않았다. Git commit/push도 이번 작업에서는 하지 않았다.

### 부문 정정

사용자가 확인한 부문은 **일반국민·대학(원)생 과제①: 사출성형 공정데이터 기반 품질불량 사전예측 및 검사 우선순위 결정**이다.
[9/22 수정 공식 공지](https://www.kamp-ai.kr/contestNoticeDetail?CPT_NOTICE_SEQ=28)는 팀별 데이터셋 1종을 선택하도록 한다.
2종 이상 융합·추가 전후 제거실험은 [재직자 부문 공지](https://www.kamp-ai.kr/contestNoticeDetail?CPT_NOTICE_SEQ=29)의 조건이므로 이 과제에 필수로 부과하지 않는다.
외부 데이터는 선택적으로 활용할 수 있지만 공개에 문제가 없어야 하며 사유·방법·출처를 기록해야 한다.
CN7/RG3 및 라벨/비라벨 CSV 4개는 한 사출성형 데이터셋이지 독립 데이터셋 4종이 아니다.

## 이번 실제 검증 결과

기존 모델/분할/원본을 변경하지 않고 CPU 연결 후보가 기존 연구 결과를 보존하는지 확인했다.
비라벨 **71,180행** 추론: 점수 최대 절대 오차 `1.6653345369377348e-16` (허용 `1e-12`),
OOD 및 action 전 행 일치. 첫 50행은 전체 배치와 별도 배치의 점수도 비교했다.
로컬 프로세스 내 추론은 최초 약 **0.316초**, 최종 패키지 재검증은 **0.233초**였으며 데이터 로딩·네트워크·플랫폼 시작 시간은 제외했다.
따라서 이것을 KAMP 원격 속도나 운영 SLA로 제시해서는 안 된다.

개발 1,913행의 고정 그룹 5-fold OOF를 CPU에서 재계산했다. 모든 fold 그룹 교집합은 0이며
표준화와 OOD 경계는 각 훈련 fold만으로 적합했다. 기존 logistic AP를 재현했고 JSON 추론과 native 추론이 일치한다.

| 범위 | 행 / 숫자 양성 | AP | ROC-AUC | 상위 10% 양성 포착률 |
|---|---:|---:|---:|---:|
| 전체 | 1,913 / 33 | 0.115419 | 0.758422 | 27.27% |
| CN7 | 967 / 13 | 0.286265 | 0.933358 | 84.62% |
| RG3 | 946 / 20 | 0.023058 | 0.521382 | 5.00% |

이는 개발 OOF 진단이며 새로운 블라인드/외부기간 검증이 아니다. 기존에 관찰된 홀드아웃을 다시 최적화하지 않았다.
라벨 1의 의미가 미확정이므로 이 표를 실제 불량 검출률로 단정하지 않는다.
RG3는 우선순위 효과가 약해 여전히 운영 적용 불가다. 기존 강한 모델 비교는 `docs/08_과제01_CPU_비교_결과_20260923.md`를 참고한다.
이번 연결 작업에서 새 강한 모델을 선택하거나 성능 향상을 주장하지 않았다.

전체 테스트 68개 통과. 원본 없이 실행한 테스트는 56개 통과·데이터 의존 12개 건너뜀이다.
건너뜀은 데이터 검증 성공이 아니다. 시험용 산술 fixture는 단위 테스트에만 쓰며 제조 성능 산출에는 공식 CSV만 사용했다.

비라벨 추론의 기존 검사 유지 사유(우선 사유 기준, 상호 배타적):

| 사유 | 행 수 |
|---|---:|
| RG3 미검증 설비 | 35,941 |
| CN7 OOD | 19,607 |
| CN7 라벨 의미 미확정 | 15,632 |
| 합계 — 모두 기존 검사 유지 | 71,180 |

## 직접 실행하기

프로젝트 최상단에서 실행한다. 아래 `my_*` 폴더가 이미 있으면 다른 이름을 사용한다.
도구는 기존 결과를 덮어쓰지 않는다. 새 PC의 환경 구축은 `MoldGuard_서비스_실행가이드.md`를 먼저 따른다.
현재 로컬 `.venv-compare`: Python 3.12.7, numpy 1.26.4, pandas 2.2.2, scikit-learn 1.5.1.
SDK `t3qai_client`는 로컬에 없으며 로컬 추론에는 필요하지 않다.

```bash
cd /Users/hamseongho/Documents/ChatGPT/KAMP

# 환경 및 비용 근거 점검. 실행 성공이어도 remote_readiness는 별도 확인한다.
.venv-compare/bin/python -m paas_cpu.preflight \
  --output-dir outputs/my_paas_preflight

# 원격 비용 근거 검사는 현재 의도적으로 종료코드 2 (blocked).
.venv-compare/bin/python -m paas_cpu.cost_gate config/paas_cpu_cost_receipt.json

# 확정 부문은 융합 의무 없음. 미래에 외부 데이터 실험을 추가하면 별도 검토한다.
.venv-compare/bin/python -m paas_cpu.fusion_gate config/fusion_readiness.json

# 공식 데이터와 고정 분할을 사용하는 전체 코드 검사
.venv-compare/bin/python service_checks.py --layer all \
  --data-dir '1. 사출성형기 AI 데이터셋' \
  --split-dir outputs/service_dev_20260926/split \
  --output-dir outputs/my_paas_checks

# 실제 71,180행 추론 및 그룹 OOF 재현
.venv-compare/bin/python -m paas_cpu.validate_local \
  --data-dir '1. 사출성형기 AI 데이터셋' \
  --split-dir outputs/service_dev_20260926/split \
  --bundle-dir outputs/paas_cpu_20260926/bundle_final \
  --output-dir outputs/my_paas_validation
```

마지막 폴더의 `validation.json`에서 `status=passed`, `remote_executed=false`, `actions_identical=true`,
`ood_identical=true`, 오차와 설비별 지표를 확인한다. `run.log`, `execution.json`에는 명령·버전·코드/출력 해시가 남는다.
`local_inference.jsonl`은 행 단위 연구 결과이며 Git이나 공개 문서로 자동 업로드하지 않는다.
신규 다운로드 데이터는 기존 해시와 달라지면 중단한다. 임의로 해시를 갱신해 통과시키지 말고 데이터 변경을 검토한다.

### 모델 패키지 다시 생성

```bash
.venv-compare/bin/python -m paas_cpu.export_model \
  --model outputs/moldguard/model.joblib \
  --expected-sha256 daf8a177535f3ba1ef1591676224f58db62efe046294de46f888c0f44057b235 \
  --output-dir outputs/my_paas_bundle
```

검토된 기존 로컬 joblib만 역직렬화한 뒤, 배포 후보에서는 실행 코드가 없는 JSON 계수로 변환한다.
신뢰할 수 없는 joblib/pickle을 넣지 않는다. 산출 `model.json` SHA-256:
`f58243377e8fc934d9488bedf90e6d25111b6d6c722e0cd438b9c8e0e18d0341`.
고정 분할 SHA-256: `e26a82c58d0e86952b8557c2e8542f816f64db8ed49d15a4a450ef7a86b5efc7`.

## 플랫폼 연결 방식과 남은 확인

T3Q AI Platform v3.8.4의 `sklearn_LOGISTIC` 예제에서 `init_model`, `inference_dataframe`,
`T3QAI_INIT_MODEL_PATH`를 확인해 `paas_cpu/inference_service.py`에 연결 후보를 구현했다.
입력은 설비·원본 행 ID·24개 센서의 26개 이름 있는 컬럼이다. 자세한 계약은 `paas_cpu/README.md`를 따른다.
플랫폼 예제의 반환은 숫자 리스트였고 후보는 안전 상태를 포함한 레코드 배열이다. 실제 전송 규격 수용 여부는 미검증이다.

무료 근거 확인 후 다음 순서로 진행한다(아직 실행하지 않음).

1. 대상 프로젝트 ID와 로그인 상태를 확인한다. 설명 정정은 완료했지만 비용 근거는 여전히 미확인이다.
2. 운영기관이 확인한 CPU/메모리/디스크·사용량·기간·초과 시 중단 정책을 기록한다. 유료 자동전환과 GPU를 사용하지 않는다.
3. 공식 지원 Python/SDK 및 CPU 이미지 확인. 기존 템플릿의 외부 Docker 베이스와 비고정 pip 설치를 검증 없이 빌드하지 않는다.
4. `.t3qai/_runtime_config.yaml`의 실제 공식 규격, 코드 파일 업로드와 모델 파일 등록 경로를 확인한다. 추정 설정을 쓰지 않는다.
5. 코드/모델 해시를 고정하고 비공개 CPU 서비스의 `init_model`을 확인한다. 자동 무작위 Train-Test Split을 쓰지 않는다.
6. 원본의 소량 실제 행으로 입력 컬럼 유지·직렬화·인증·행 대응·오류 처리를 확인한다. 값/라벨 의미를 꾸미지 않는다.
7. 동일 전체 데이터의 원격/로컬 점수를 `1e-12` 기준으로 비교하고 누락/중복/OOD/action을 확인한다.
8. 원격 CPU/메모리/지연 로그와 실행 ID를 저장하고 즉시 중지·종료를 확인한다. 장기 서비스 대기 비용도 확인한다.

비용 설정 `config/paas_cpu_cost_receipt.json`은 현재 unconfirmed다.
`cost_gate`는 증빙 문서 해시·검토자·기간·범위를 확인하는 **사전 체크리스트**이며 실제 과금 차단 장치가 아니다.
무료라고 가정해 값을 true로 채우면 안 된다. 콘솔 자체 할당/종료 정책도 별도로 검증해야 한다.

운영기관에 확인할 질문(초안만 작성, 발송하지 않음):
“경진대회 일반국민·대학(원)생 과제① 프로젝트 215209147869625880에서 CPU 재현/추론 테스트를 하려 합니다.
코드·모델·데이터 저장, 이미지 빌드, CPU 학습/추론, 서비스 대기, 네트워크의 추가 비용이 모두 0원인지,
할당량·지원 기간·초과 시 차단 여부 및 권장 CPU 런타임/SDK 버전을 알려주세요.”
공식 기술 문의는 [일반국민 공지](https://www.kamp-ai.kr/contestNoticeDetail?CPT_NOTICE_SEQ=28)의 `dl_kamp@nhn.com`이다.

## 분석·운영 주의사항

- 같은 관측값 그룹 겹침 방지와 훈련 fold 내 전처리를 유지한다. 동일 그룹이 실제 제품/로트를 대체하지는 않는다.
- 원본 CSV가 파일별 표준화되어 있어 원시 단위 및 공급자 변환의 누수 가능성은 미해결이다.
- 라벨 의미·실제 제품 ID·예측 시점·변수의 검사 전 가용성을 확인하기 전 현장 사전예측 성능을 보증하지 않는다.
- `source_row_id`는 원본의 행 식별용이며 제품/시간 ID가 아니다. 다른 파일/수집회차와 합칠 때 별도 출처 키가 필요하다.
- 미지원 설비, 잘못된 타입/결측/Inf/중복/초과 입력은 오류로 중단한다. 오류를 0점이나 Pass로 처리하지 않는다.
- 배치마다 계산한 순위를 전체 검사 대기열 순위처럼 쓰지 않는다. 이 추론기의 순위·운영 우선순위는 null이다.
- RG3 성능 한계, OOD, 라벨 충돌을 숨기지 않는다. 기존 검사를 유지하고 자동 합격·검사 생략·설비 제어는 금지한다.
- 외부 데이터 확장 시 공개 가능 여부, 물리적 연결 근거, 검사 전 가용성, 동일 평가 행/분할의 추가 전후 비교가 필요하다.
  현재 보조 데이터는 없고 융합/제거실험을 실행하거나 성능 향상을 만들어 보고하지 않았다.
- 대회 최종 제출은 사용자 요청에 따라 하지 않는다. 블라인드 제출물의 식별 정보 제한 등 공식 제출 조건은 별도 검토 대상이다.

## 산출물 위치

`outputs/paas_cpu_20260926/` 아래에 `bundle_final/`, `validation_final/`, `checks_final/`,
`checks_no_data/`, `preflight/`, `release_preservation/` 및 로컬 ZIP을 보관한다.
앞선 `bundle/`, `validation/`, `checks/`도 실행 이력으로 보존한다. 과거 실행 영수증을 새 코드로 덮어쓰지 않는다.
원본/기존 모델/이전 분석 결과는 그대로 보존한다.

보존 감사: 기존 증빙 185개 파일의 변경 0개. `release_preservation/release_audit.json`의 테스트 수는
이전 서비스 첫 배치 기록(52개)이며, 이번 최신 68개 테스트는 `checks_final/test_report.json`을 기준으로 한다.
