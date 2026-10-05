"""Create Korean research handoff and a source-data-free public aggregate export."""
import argparse
import json
import math
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd
from research_runtime import digest
from task01_precision_audit import dump


def mdtable(frame):
    columns=list(frame.columns)
    return '| '+' | '.join(columns)+' |\n| '+' | '.join(['---']*len(columns))+' |\n'+ '\n'.join('| '+' | '.join(str(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None))


def report(out):
    out=Path(out); decision=json.loads((out/'decision.json').read_text(encoding='utf-8'))
    verification=json.loads((out/'verification.json').read_text(encoding='utf-8'))
    if verification['status']!='passed': raise ValueError('Verification required')
    table=pd.read_csv(out/'ensemble_metrics.csv')
    primary=table[table.fraction.eq(.075)&table.machine.eq('all')]
    names=['historical_cpu','historical_gpu','pair_a1','pair_a0','pair_a0.25','pair_a0.5','rank_pairwise','rank_ndcg','mix_pairwise','mix_ndcg','nested_selected']
    primary=primary[primary.pipeline.isin(names)]
    selected=primary[primary.pipeline.eq('nested_selected')].set_index('contract')
    old=primary[primary.pipeline.eq('historical_gpu')].set_index('contract')
    fresh=primary[primary.pipeline.eq('pair_a1')].set_index('contract')
    ci=json.loads((out/'bootstrap_summary.json').read_text(encoding='utf-8'))['confidence_interval']
    chosen=json.loads((out/'selection_ledger.json').read_text(encoding='utf-8'))
    ledger=pd.read_csv(out/'fit_ledger.csv');fit_counts=ledger.groupby(['stage','kind']).size().reset_index(name='fits')
    if (out/'legacy_layout_audit/fit_ledger.csv').exists():
        fit_counts=pd.concat([fit_counts,pd.DataFrame([{'stage':'legacy_layout_reproduction','kind':'cat','fits':30}])],ignore_index=True)
    fit_counts.to_csv(out/'fit_counts.csv',index=False)
    resource=pd.read_csv(out/'resource_usage.csv')
    inventory=json.loads((out/'document_inventory.json').read_text(encoding='utf-8'))
    checks=verification['count'];gpu=verification['gpu_fits'];cpu=verification['cpu_fits']
    tests=ET.parse(out/'regression_tests.xml').getroot() if (out/'regression_tests.xml').exists() else None
    suites=list(tests.iter('testsuite')) if tests is not None else []
    regression={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ['tests','failures','errors','skipped']}
    dump(out/'regression_summary.json',regression)
    # Preserve failed workflow attempts separately from successful model fits.
    failures=[]
    if (out/'training.log').exists():
        failures.append({'stage':'initial_workflow','status':'fixed_before_any_fit','reason':'original ID column contains punctuation, replaced tuple attribute access with normalized identity'})
    if (out/'runtime_resume.json').exists():
        failures.append({'stage':'resource_guard','status':'checkpointed_and_resumed','reason':'transient disk free below 5 GiB; no file deletion; process exit recovered space; GC thresholds reduced',
            'evidence':['training_attempt2.log','runtime_resume.json','training_attempt3.log']})
    failures.append({'stage':'feature_layout','status':'disclosed_and_legacy_baseline_reproduced','reason':'raw 24-column layout differs from legacy 23 sensors plus machine indicator; same within-machine information, different tree results',
        'evidence':['feature_layout_audit.json','legacy_layout_metrics.csv']})
    dump(out/'experiment_failures.json',{'fit_failures':[],'workflow_events':failures,
        'not_run':[{'experiment':'hard_negative/extra_features/partial_pooling/policy_reallocation','reason':'not mandatory; bounded H1/H2 isolation, no new information or adaptive expansion'},
                   {'experiment':'prospective validation','reason':'new independent labels unavailable'},
                   {'experiment':'strict legacy-layout full nested H1/H2 rerun','reason':'outside remaining predeclared fit budget; no model adoption; raw-layout nested results retained as explicitly limited research'}]})
    # Historical unsuccessful families: copy aggregates, not rows or model files.
    historical=[]
    for f in ['refinement_at_k192.csv','comparison_at_k192.csv','calibrated_at_k192.csv']:
        path=Path('reports/gpu_pc_20261004')/f
        if path.exists():
            h=pd.read_csv(path);h.insert(0,'evidence_file',f);historical.append(h)
    pd.concat(historical,ignore_index=True).to_csv(out/'historical_candidate_comparison.csv',index=False)
    # Approximate planning counts, not guarantees and not independent product claims.
    nrec=math.ceil(1.96**2*.45*.55/.1**2);nprecision=math.ceil(1.96**2*.10*.90/.03**2)
    prospective=pd.DataFrame([{'positive_group_prevalence':p,'required_positive_groups':nrec,
        'expected_independent_groups':math.ceil(nrec/p),'expected_rows_if_two_per_group':2*math.ceil(nrec/p)} for p in [.01,.03,.05]])
    prospective.to_csv(out/'prospective_sample_plan.csv',index=False)
    additional=f'''# 추가 데이터와 독립 검증 설계

## 공급자 요청 필드

machine_id / physical_shot_id / product_id / cavity_id / lot_id, shot_start / shot_end / sensor_available_at / inspection_at(시간대), 원시 센서 단위, scaler 적합 범위·변환 버전, PassOrFail 코드북과 실제 검사값·재검사·판정 변경 이력이 필요하다. 파일명·SHA를 source_hashes.json과 대조한다. 가능한 추가 신호는 캐비티별 센서·제품 치수·검사 이미지·원재료·금형·정비 이력이며, 사용권한과 예측시각 이전 가용성이 확인된 것만 사용한다.

## 표본 크기 가정

재현율 45%, 95% 구간 반폭 10%p의 정규근사 계획은 독립 양성 그룹 {nrec}개를 요구한다. 이는 보장된 검정력이나 정확한 Wilson 구간 설계가 아니며 실제 발생률에 따라 조정한다.

{mdtable(prospective)}

정밀도 10%, 구간 반폭 3%p에는 독립 검사 단위 약 {nprecision}개가 필요하다. 그룹당 두 행·상관계수 0.5를 가정하면 설계효과 1.5이며 약 {math.ceil(nprecision*1.5)}개 검사 행(약 {math.ceil(nprecision*1.5/2)}개 그룹)이 필요하다. 실제 샷/제품 관계를 확인한 뒤 cluster bootstrap 또는 이항 구간으로 재설계한다. 전체 검사비율 7.5%를 적용한 총 모집단 추정과 양성 그룹 확보 조건 중 큰 쪽을 확보한다.

대표성 있는 무작위 표본 70%, 위험·불확실성·모델 불일치 표본 30%를 별도 개발 라벨링 예산의 시작 가정으로 제안한다. 각 선택 확률·그룹·시각을 기록하며 고위험 선택 표본의 양성률을 전체 성능으로 보고하지 않는다. 독립 새 기간 시험 표본은 이 개발 표본과 분리하고 선택·active learning에 사용하지 않는다. 전체 회수율을 평가하려면 선택되지 않은 영역에도 정답 관측이 필요하다. 현재 71,180개 비라벨 점수는 새 정답을 생성하지 않는다.
'''
    (out/'additional_data_request.md').write_text(additional,encoding='utf-8')
    p=primary[['contract','pipeline','k','tp','fp','fn','tn','precision','recall','f1','accuracy','ap']].copy()
    for c in ['precision','recall','f1','accuracy']:p[c]=p[c].map(lambda v:f'{100*v:.3f}%')
    p['ap']=p.ap.map(lambda v:f'{v:.6f}')
    p['설비']='전체';p['평가']='개발 1913행/33양성';p['구분']='3-seed 점수 앙상블'
    p['불확실성']='본문 조건부 bootstrap';p['fit/상태']=f'이번 GPU {gpu}/CPU {cpu}; 실행 완료'
    samemachine=table[table.fraction.eq(.075)&table.pipeline.isin(['historical_gpu','pair_a1','nested_selected'])&table.machine.ne('all')]
    seeds=pd.read_csv(out/'metrics_by_machine_seed_budget.csv',dtype={'seed':str})
    seeds=seeds[seeds.fraction.eq(.075)&seeds.machine.eq('all')&seeds.pipeline.isin(names)]
    seed_summary=seeds.pivot_table(index=['contract','pipeline'],columns='seed',values='tp',aggfunc='first').reset_index()
    bas=int(old.loc['historical_pooled','tp']);new=int(selected.loc['historical_pooled','tp'])
    bnew=int(selected.loc['batch_allocated','tp']);bbase=int(fresh.loc['batch_allocated','tp'])
    report=f'''# 과제 1번 추가 연구 결과 — {out.name}

{mdtable(p)}

## 최종 결과

**{decision['status']}**. 같은 역사적 K144 계약의 기존 GPU 앙상블 {bas}건/{144-bas} FP에 대해, 내부 선택 절차는 {new}건/{144-new} FP다. 배치 계약을 적용한 내부 선택 절차는 {bnew}건/{144-bnew} FP이며 같은 계약의 새로 적합한 alpha=1 기준선 {bbase}건/{144-bbase} FP와 비교해야 한다. 서로 다른 계약의 숫자를 모델 향상으로 합치지 않는다.

**구현 범위 차이를 발견했다.** 본 연구의 사전 고정 입력은 원본 24개 센서 열이다. 기존 코드는 값이 항상0인 Clamp_Open_Position을 제외하고 설비 표시 열을 붙여 다른 열 순서를 사용했다(23개 센서+표시). 설비별 적합에서는 추가 정보 차이가 없지만 트리 학습이 달라졌다. 따라서 이번 H1은 **새로 고정한 열 배치 안의 가중치 대조**이며 기존 변환기까지 완전히 보존한 순수 가중치 재현이라고 부를 수 없다. 이를 GPU 비결정성으로 설명하지 않는다. 발견 후 후보를 추가 선정하지 않고 기존 열 배치의 auto Balanced 기준선만 30 GPU fit으로 다시 적합했다. 과거 raw OOF와 최대 절대 차이0, K144 seed14/14/13·앙상블13으로 정확히 재현했다. 전체 엄격한 기존 열 배치 H1/H2 중첩 재실험은 적합 상한 내에 남은 예산이 부족해 수행하지 않았고, 이번 결과로 기존 번들을 교체하지 않는다.

검사량은 모두 CN7 73 + RG3 71 = 144행이다. 세 seed TP 평균이 아닌 실제 점수 앙상블을 평가했다. 8개 선택 가능 파이프라인과 CN7 고정 앵커를 비교했으며 row/event 8개는 원인 분석용 구성요소다. CPU·과거 GPU·중첩 선택 절차까지 전체 파이프라인 수는 사전 상한 12개 이내다.

## 같은 정책·설비·seed 비교

{mdtable(samemachine[['contract','pipeline','machine','k','tp','fp','fn','precision','recall','ap']].round(6))}

{mdtable(seed_summary)}

CN7은 모든 새 후보에서 같은 StandardScaler + balanced logistic C=1을 유지했다. RG3 점수는 전체 배치에서 계산한 percentile rank의 평균이며 확률이 아니다. 한 RG3 행의 값 1은 100% 양성 확률을 뜻하지 않는다. 과거 pooled OOF와 새 배치별 정수 배정의 차이는 정책 효과다. 각 계약 내부에서는 원점수·순위 조합과 모델 효과를 비교했고 최종 동점 규칙은 source_row_id 오름차순으로 유지했다.

## 내부 선택과 불확실성

outer별 선택: {', '.join(str(x['outer'])+': '+x['winner'] for x in chosen)}. -1은 모든 개발 데이터의 내부 OOF로 선택한 최종 연구 번들이다. outer 성능 순위를 보고 최종 번들을 고르지 않았다. 최종 번들의 선택된 구성은 **{decision['full_bundle_candidate']}**이며, 중첩 절차의 OOF 성능은 이 단일 최종 구성의 독립 시험 성능과 같지 않다.

기준선 alpha=1과 내부 선택 절차를 같은 그룹 draw로 2,000회 재표집했다. 설비·배치·그룹 크기 층화로 N1913/K144를 유지했다. ΔTP 95% 구간 [{ci['delta_tp']['low']:.3f}, {ci['delta_tp']['high']:.3f}], Δ정밀도 [{ci['delta_precision']['low']*100:.3f}, {ci['delta_precision']['high']*100:.3f}]%p, Δ재현율 [{ci['delta_recall']['low']*100:.3f}, {ci['delta_recall']['high']*100:.3f}]%p다. ΔFP와 설비별 AP 구간도 bootstrap_summary.json에 보존했다. 이는 고정 OOF 조건부 진단이며 탐색·재학습·미래 기간 불확실성을 포함하지 않는다. 독립 우위가 검증됐다고 주장하지 않는다.

## 수행·검증

- 원본·보존 split·기존 연구 번들 SHA를 실행 전후 대조했다. 개발 1913행/957그룹/33양성만 사용했고 과거 holdout 480행을 제외했다.
- 과거 산출물 해시 58개가 일치했다. 문서·코드·산출물 {len(inventory['entries'])}개를 목록화하고 범위·링크를 기록했다. 부분 조사와 전체 정독을 구분하며 모든 문서를 완전히 읽었다고 주장하지 않는다.
- **이번 공식 데이터 GPU fit {gpu}회, CPU fit {cpu}회**. 과거 1,986회와 별도다. 사전 고정 후보·split seed·model seed·가중치·백엔드·시간·학습 그룹 해시를 fit 원장에 기록했다. 캐시 적합은 중복 횟수로 세지 않는다. synthetic smoke는 새로 수행하지 않았으며 실제 소형 첫 적합의 backend/config 검사를 이용했다.
- 본 연구 입력 배치에서 H1 row/event alpha=0/0.25/0.5/1, H2 pairwise/NDCG 및 사전 고정 50:50 혼합을 비교했다. qid는 각 RG3 학습 구간 전체를 한 query로 둔 가정이며 CV 그룹과 다르고 센서 특징에 포함하지 않았다.
- 별도 검증 프로세스 **{checks}개 확인**, 행별 OOF {verification['outer_prediction_rows']:,}행, inner OOF {verification['inner_prediction_rows']:,}행, 지표 {verification['metric_rows']:,}행 재계산. 테스트 {regression['tests']}개, 실패 {regression['failures']}, 오류 {regression['errors']}, 생략 {regression['skipped']}. 실행한 관련 회귀 범위이며 저장소의 모든 서비스 시험을 다시 돌렸다는 뜻은 아니다.
- native 저장/복원과 outer OOF 일치, 행·열 순서, chunk 1/7/128에서 원점수 취합 후 전체 순위, singleton·중복·동점, NaN/Inf·미지원 설비·누락 특징·훼손 모델 거부를 확인했다. binomial/row BCE 손실·gradient 등가성도 검사했다.
- GPU 병렬 적합은 1개, CPU thread 2개, 시간 상한 2시간·GPU fit 상한790회. 디스크 여유 중단 후 완료 캐시를 유지하고 GC 주기를 낮춰 재개했다. 학습 중간 관측 메모리 최대 {int(resource.gpu_memory_used_mib.max())}MiB이며 실제 순간 peak를 측정했다고 주장하지 않는다. 원본 삭제·드라이버 변경·다른 프로세스 종료는 없었다.

## 실패 실험과 정보 한계

모든 alpha와 row/event, 순위 모델, 고정 혼합의 성적을 ensemble_metrics.csv에 남겼다. 결과가 낮은 후보도 삭제하지 않았다. 검사 집합 overlap, 양성 그룹 순위, 상충 그룹 기여, 동점 최소/기대/최대, 별도 group split 민감도는 각각 inspection_overlap.csv, rg3_positive_rank_audit.csv, failure_slices.csv, tie_sensitivity.csv, group_split_sensitivity.csv에 있다. 고정 OOF budget 곡선은 탐색 설명이며 새로운 최적 정책 선택이 아니다.

과거 가중 LR/RF/SVM/XGB/Cat/양성 복제와 센서 파생·규제·보정 비교는 historical_candidate_comparison.csv 및 원래 2026-09-26/10-04 보고서에서 추적한다. 같은 Cat alpha=1의 반복은 기준선 재현과 라이브러리 auto Balanced/명시 가중치의 정규화 확인 목적이다. hard-negative·새 특징·정책 재배정은 추가 정보와 적절한 교차적합 근거가 없어 이번 제한 연구에서 제외했다.

숫자 양성 33개 중 상충 그룹의 양성은 CN7 9개, RG3 20개다. 같은 센서 그룹 전체를 같은 결정으로 선택할 때 전부 회수하려면 최소 FP29, 정밀도 상한33/62=53.23%다. K144 고정 정밀도 상한은33/144=22.92%다. 이 oracle은 모델 성적이나 미래 보장이 아니다. 그룹 비용 n/이득 c의 DP oracle 전선을 별도로 보존했다. event max 라벨은 다른 target이며 정보를 새로 만든 것이 아니다.

원본의 파일별 사전 정규화·공통 scaler·24개 센서 도착시점·실제 shot/product/cavity·PassOrFail 코드북은 미확정이다. docs/04의23개 모델 센서는 상수1개 제외를 뜻하며24개 원본 그룹 센서와 모순이 아니다. RG3 순위와 CN7 확률 척도가 달라 pooled AP는 조심해서 해석한다. 전체 AP 상승을 의무적 성공으로 주장하지 않으며 설비별 AP와 두 설비 AP의 산술평균인 macro AP를 함께 저장했다. 비확률 순위에는 Brier/Log loss를 확률 평가처럼 계산하지 않으며 확률 형태의 원래 출력만 probability_diagnostics.csv에 미보정 진단으로 기록했다.

## 다음 행동과 보존

기존 연구 번들과 기존 품질검사를 유지한다. 새 번들은 검증된 추론 형식의 **연구 절차 산출물**이고 현장 채택/자동 출하/검사 생략 승인이 아니다. 새 기간·새 그룹·실제 라벨 의미를 확인한 데이터가 있어야 독립 성능을 주장할 수 있다. 구체적 필드·표본 규모·선택 확률 기록은 additional_data_request.md를 따른다.

원자료·행별 예측·native 모델은 로컬 `{out.as_posix()}`에 보존하며 공개 Git에는 코드·소형 집계·보고서만 넣는다. 재현은 run_precision_frontier.ps1에 새 출력 경로를 사용한다. 화면 잠금과 무관한 명령줄 작업이며 재현 런처는 실행 중에만 시스템 절전을 막고 종료 시 해제한다. 잠금 정책은 바꾸지 않는다.

## 1차 구현 근거

- [XGBoost 3.0.5 Learning to Rank](https://xgboost.readthedocs.io/en/release_3.0.0/tutorials/learning_to_rank.html): qid와 mean pair sampling을 설치 버전의 실제 config로 검증했다.
- CatBoost 공통 파라미터 문서 URL은 도구 접근 오류로 본문 확인에 실패했다. 설치된 라이브러리 get_all_params의 class_weights와 실제 auto Balanced 대조를 근거로 사용했다. 재학습 비결정성과 native 추론 일치는 구분한다.
'''
    (out/'research_report.md').write_text(report,encoding='utf-8')
    (out/'notion_ready.md').write_text(report,encoding='utf-8')
    (out/'reproduction_commands.ps1').write_text("# Run from repository root; new directory preserves prior outputs.\n.\\run_precision_frontier.ps1\n# Resume unchanged source/config checkpoints only:\n.\\run_precision_frontier.ps1 -OutputDir '"+out.as_posix()+"'\n",encoding='utf-8')
    # Exact allow-list: no native model, source row identifiers or OOF exports.
    names=['research_report.md','additional_data_request.md','ensemble_metrics.csv','metrics_by_machine_seed_budget.csv',
        'baseline_metrics.csv','decision.json','bootstrap_summary.json','fit_counts.csv','group_split_sensitivity.csv',
        'inspection_overlap.csv','tie_sensitivity.csv','failure_slices.csv','balanced_normalization.json','regression_summary.json',
        'historical_candidate_comparison.csv','prospective_sample_plan.csv','policy_parity_tests.json','experiment_plan.json',
        'experiment_plan.sha256','experiment_failures.json','macro_ap.csv','probability_diagnostics.csv','tie_summary.csv',
        'feature_layout_audit.json','legacy_layout_metrics.csv','legacy_layout_raw_differences.csv','historical_refit_rank_differences.csv']
    public=Path('reports')/out.name;public.mkdir(exist_ok=True)
    for name in names:shutil.copy2(out/name,public/name)
    dump(public/'verification.json',{k:v for k,v in verification.items() if k!='checks'})
    dump(public/'artifact_hashes.json',{p.name:digest(p) for p in public.iterdir() if p.is_file() and p.name!='artifact_hashes.json'})
    dump(out/'artifact_hashes.json',{str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'})
    print(json.dumps({'report':str(public/'research_report.md'),'decision':decision,'regression':regression},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);report(p.parse_args().output_dir)
