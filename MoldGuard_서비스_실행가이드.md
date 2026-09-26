# MoldGuard 연구용 서비스 실행 가이드

2026-09-26 · 작업 브랜치 `dev_haams` · 첫 구현 범위 MG-001~005

서비스는 데이터 준비 상태, 기존 연구용 순위, 전체/CN7/RG3 개발 OOF 결과를 조회한다. 점수 대상은 숫자 라벨 1이며, 실제 검사 조치는 모든 항목에서 기존 검사 유지다. 새 모델 선정, 현장 승인, 점검 업무 저장은 이번 첫 배치에 포함하지 않는다.

## 1. 준비와 설치

프로젝트 최상단에서 Python 3.12를 사용한다. 이 컴퓨터에서는 기존 `.venv-compare`를 그대로 사용할 수 있다. 새 환경을 만들 경우:

```bash
python3.12 -m venv .venv-service
.venv-service/bin/python -m pip install -r requirements-compare.txt
.venv-service/bin/python service_checks.py --layer environment --output-dir outputs/service-env-new
```

아래 명령은 기존 `.venv-compare` 기준이다. 새 환경에서는 앞부분을 `.venv-service/bin/python`으로 바꾼다. 결과 폴더는 매번 새 이름을 지정한다. 실행이 실패한 폴더도 덮어쓰지 않는다.

## 2. 로컬 화면 실행

```bash
.venv-compare/bin/python -m moldguard_service.api --port 8765
```

[연구용 화면](http://127.0.0.1:8765)을 연다. 종료는 실행 터미널에서 Ctrl+C다. 서버는 `127.0.0.1`에만 바인딩하며 외부 검사/설비 시스템에 쓰지 않는다. 별도 웹 프레임워크나 DB 설치는 필요하지 않다. 다중 사용자 인증을 제공하지 않는 로컬 연구용 도구다.

기본값은 현재 프로젝트의 공식 데이터 폴더, `outputs/moldguard`, `outputs/task01_compare_20260923_cpu_v3`이다. 다른 경로는 직접 지정한다.

```bash
.venv-compare/bin/python -m moldguard_service.api \
  --data-dir '1. 사출성형기 AI 데이터셋' \
  --result-dir outputs/moldguard \
  --comparison-dir outputs/task01_compare_20260923_cpu_v3 \
  --registry-dir service_state/contracts --port 8765
```

- **데이터 준비 상태:** 8개 계약의 상태·근거·확인자·확인일, 원본 해시, 미합의 승인 기준을 확인한다.
- **연구용 검사 순위:** 원본 행 ID, 모델 점수, 설비 내 순위, 참고 분류, OOD, 보류 사유를 조회한다. 설비 필터와 페이지 이동을 제공한다.
- **모델 검증 기록:** 완료된 비교 파일에서 실제 지표를 읽는다. 화면은 첫 seed의 전체/설비별 결과를 표시하고 API는 모든 seed와 집계표를 제공한다.

순위 어댑터는 실제 4개 CSV와 과거 결과의 원본·모델·추론 파일 해시, 스키마, 행 ID를 검사한다. 모델 파일을 역직렬화하거나 브라우저 업로드 모델을 실행하지 않는다. 파일이 없거나 바뀌면 미산출/오류로 표시한다. 공개 저장소에는 행 단위 결과가 없으므로 새 clone에서는 순위가 빈 상태인 것이 정상이다. 집계 비교 자료가 있으면 모델 검증 기록은 별도로 조회할 수 있다.

## 3. 과거 outputs 없이 분할 → 비교 → 결과 검사

공식 CSV는 별도로 확보해야 한다. 명령은 실제 데이터 경로를 명시하며 과거 분할을 복사하지 않는다.

```bash
.venv-compare/bin/python research_split.py generate \
  --data-dir '1. 사출성형기 AI 데이터셋' --output-dir outputs/split-user-new
.venv-compare/bin/python research_split.py verify \
  --data-dir '1. 사출성형기 AI 데이터셋' --output-dir outputs/split-user-new
.venv-compare/bin/python task01_compare.py --device cpu \
  --data-dir '1. 사출성형기 AI 데이터셋' --split-dir outputs/split-user-new \
  --output-dir outputs/compare-user-new
.venv-compare/bin/python research_verify.py \
  --data-dir '1. 사출성형기 AI 데이터셋' --split-dir outputs/split-user-new \
  --result-dir outputs/compare-user-new --output-dir outputs/verify-user-new
```

분할은 원래 24개 특징과 설비로 그룹을 만들고 기존 seed/규칙을 사용한다. 생성 시 `--reference-split-dir outputs/local_runs/20260922-share-final`을 추가하면 기존 분할과 동일성까지 확인한다. 기준 분할을 제공하지 않은 생성 기록에는 `not_provided_new_split`이 남는다. 같은 숫자의 seed만으로 과거 분할과 같다고 보고하지 않는다.

비교는 기존 CPU 7개 후보×3 seed×5-fold를 재실행한다. 추가 가설 탐색이나 새 모델 채택 실험이 아니다. 결과 검사기는 파일 해시, OOF 1:1 행·라벨·그룹·fold 일치, 전체/설비별 지표 재계산을 확인한다. fold/조건별 표는 해시 무결성을 확인하며 의미상 동등성은 별도 전체 실행 비교로 검토한다.

보정 코드도 새 경로를 받을 수 있다.

```bash
.venv-compare/bin/python calibration_diagnostics.py \
  --data-dir '1. 사출성형기 AI 데이터셋' --split-dir outputs/split-user-new \
  --output-dir outputs/calibration-user-new
```

각 새 실행 폴더의 `execution.json`은 커밋·미커밋 변경, 코드/출력 해시, 환경, 실제 명령, 종료코드와 성공/실패를 기록한다. `run.log`는 실행 출력이다. 과거 `run_manifest.json`/지표 형식은 보존했다. 로그·시각·코드 해시는 변경될 수 있으며, 수치 CSV의 동일성과 구별한다.

## 4. 데이터 없는 검사와 실제 데이터 검사

```bash
.venv-compare/bin/python service_checks.py --layer all --no-data --output-dir outputs/checks-no-data-new
.venv-compare/bin/python service_checks.py --layer all --output-dir outputs/checks-all-new
.venv-compare/bin/python service_checks.py --layer integration \
  --data-dir '1. 사출성형기 AI 데이터셋' --split-dir outputs/split-user-new \
  --output-dir outputs/checks-integration-new
```

`--layer unit`, `--layer acceptance`도 지원한다. `test_report.json`에 passed/failed/skipped와 생략 사유를 저장한다. 데이터 없는 단위시험 객체는 TEST_ONLY 계산·정책 검사용이며 제조 성능에 사용하지 않는다. 기존 산출물을 검사하는 시험은 해당 파일이 없으면 별도 생략된다. 환경 검사 통과는 모델 성능 검증이 아니다.

새 가상환경까지 포함한 재현 확인은 아래 명령으로 수행한다. 코드와 설정만 격리 폴더에 복사하고 새 venv에 고정 패키지를 설치한다. 공식 CSV는 지정 경로에서 읽으며 과거 outputs는 복사하지 않는다. 인터넷 패키지 설치와 추가 디스크 공간이 필요하다.

```bash
.venv-compare/bin/python verify_clean_reproduction.py \
  --data-dir '1. 사출성형기 AI 데이터셋' --output-dir outputs/clean-reproduction-new
```

결과는 `execution.json`, 단계별 `.log`, `checks/test_report.json`, `reproduction_comparison.json`이다. 같은 컴퓨터에서의 새 환경 검증이며 다른 OS·하드웨어의 동등성까지 보장하지 않는다.

## 5. 계약 근거 기록

```bash
.venv-compare/bin/python service_contract.py show
```

확인자는 로컬 CLI로만 기록한다. 예를 들어 실제 근거가 확보되면 `record --item label_semantics --status confirmed --reviewer '실제 검토자' --reason '실제 확인 내용' --evidence /실제/근거파일` 인자를 사용한다. 이 예시는 실행 증빙이나 확인된 라벨 방향이 아니다.

`confirmed`/`mismatch`에는 존재하는 근거 파일이 필요하다. 레지스트리는 원본 데이터 해시, 근거 해시·경로, 검토자, 시각, 이전 버전과 변경 사유를 `service_state/contracts/`에 추가 저장한다. 근거 파일이 바뀌면 조회에서 불일치로 표시하며, 다른 데이터 버전에 과거 확인을 자동 적용하지 않는다. 이 폴더는 Git에서 제외한다. 로컬 파일을 수정할 권한이 있는 사람이 검토자라는 신뢰 경계이며 다중 사용자 전자서명 시스템은 아니다.

계약을 확인해도 현재 서비스의 현장 승인 권한은 활성화되지 않는다. `config/service_policy.json`이 검증되는 실제 연구 정책이며, 사용자가 추가한 `*.example.json`은 설계·미합의 기준으로 보존한다.

## 6. API와 아직 제공하지 않는 기능

| 경로 | 응답 |
| --- | --- |
| `GET /api/v1/readiness` | 계약·원본·모델 상태와 비활성 이유 |
| `GET /api/v1/research/rankings?machine=rg3&offset=0&limit=20` | 연구용 점수/순위와 실제 검사 유지, 모든 보류 사유 |
| `GET /api/v1/validation/runs` | 전체/CN7/RG3 및 모든 seed의 개발 OOF 지표·집계·실행 범위 |

POST/PATCH/PUT/DELETE는 405로 거절한다. 요청의 `label_confirmed=true`로 승인할 수 없다. 실제 큐 변경, 모델 업로드, 검사 생략, 설비 제어 API는 없다. 과거 결과의 `action`과 현재 정책의 `action`을 별도 필드로 제공하여 근거 확인 뒤에도 이력이 덮어써지지 않게 했다.

MG-006 실패조건 상세 화면, MG-007 사건·점검 기록, MG-008 제한된 가설 실험은 다음 배치다. MG-009 이후 실제 결과 수신·정정, 전방검증, 현장 SHADOW/승인, 반복불량/공정 중 예측은 추가 데이터와 별도 검증이 필요하다. 대회 제출은 수행하지 않는다.
