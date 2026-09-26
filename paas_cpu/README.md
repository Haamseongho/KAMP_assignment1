# KAMP CPU 추론 연결 후보 — 미배포

일반국민·대학(원)생 과제①의 기존 logistic 모델을 CPU 전용 JSON 추론기로 변환한다.
실제 KAMP 화면의 `init_model()`, `inference_dataframe(df, model_info_dict)` 콜백 이름에 맞췄다.
플랫폼 SDK 버전, 모델 등록·초기화 수명주기, 반환 JSON 수용 여부, 원격 비용은 아직 검증하지 못했다.
이 폴더/ZIP은 **바로 배포 가능한 서비스나 대회 제출물이라고 보증하지 않는다**.

## 파일

- `model_runtime.py`: 엄격한 입력 검사, 표준화·logistic 점수·설비별 OOD 계산. numpy/pandas만 필요.
- `inference_service.py`: 플랫폼 콜백. 파일 추론은 거부하며 이름 있는 DataFrame만 허용한다.
- `model.json`: 로컬 export로 생성되는 계수·표준화·OOD 정보. pickle 실행 없이 읽는다.
- `requirements-inference.txt`: Python 3.12 기준 numpy/pandas 버전 고정. 플랫폼 SDK는 별도 확인해야 한다.
- `deployment_manifest.json`: export 시 생성되는 해시/미배포 상태. 무료 확인서가 아니다.

## 초기화와 입력

운영자가 검토한 모델 해시를 환경변수 `MOLDGUARD_MODEL_SHA256`에 고정한다.
로컬에서는 `MOLDGUARD_MODEL_DIR`에 모델 디렉터리 절대경로를 지정한다.
플랫폼에서는 이를 생략하면 SDK의 `T3QAI_INIT_MODEL_PATH`를 사용하지만 실제 제공 경로는 미검증이다.
해시를 외부 요청에서 받거나 요청자가 임의로 바꾸게 하지 않는다.

입력: `machine` (`cn7` 또는 `rg3`), `source_row_id` (해당 원본 파일의 비음수 정수),
`model.json.features`에 기록한 센서 24개. 컬럼 이름을 보존해야 한다.
컬럼 누락/추가, 라벨 포함, NaN/Inf, 문자열 숫자, bool, 중복 ID, 미지원 설비는 오류로 중단한다.
현재 제공 CSV의 표준화된 표현만 대상으로 한다. 원시 센서 값을 그대로 입력하면 안 된다.

반환: 입력 순서 그대로 JSON 직렬화 가능한 레코드 배열. 숫자 라벨 1의 `research_score`이며
보정된 불량 확률이 아니다. `defect_probability`, `operational_priority`, `research_rank`는 null.
요청을 여러 배치로 나눠도 점수는 같으며, 배치 내 순위를 전체 대기열 순위로 위장하지 않는다.
모든 행은 `RESEARCH_ONLY`, `existing_queue_policy=unchanged`이다. 오류를 0점/정상품으로 바꾸지 않는다.

## 활성화 전 중단 조건

추가 비용 0원(저장·이미지 빌드·CPU·대기·네트워크까지) 근거, CPU 할당량 및 종료 정책,
공식 런타임/SDK 호환, 모델 경로, 26개 입력 컬럼 및 응답 계약, 인증/권한 확인 전 시작하지 않는다.
GPU, 자동 유료 전환, 외부 공개 API, 대회 최종 제출을 자동으로 수행하는 코드는 없다.
예제 `.t3qai/_runtime_config.yaml`의 실제 규격을 확인하지 못했으므로 추정 파일을 넣지 않았다.
플랫폼 예제의 자동 Train-Test Split으로 기존 그룹 분할을 대체하지 않는다.

전체 프로젝트의 `KAMP_PaaS_CPU_연결_및_테스트가이드.md`에 재현 명령과 검증 결과가 있다.
