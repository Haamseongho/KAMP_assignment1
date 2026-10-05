"""Read-only provenance inventory and data audit for the bounded K144 study."""
import argparse
import datetime as dt
import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

import calibration_diagnostics as cd
import moldguard as mg
from research_runtime import digest
from research_split import FEATURES
from task01_gpu_oof_evaluate import validate_frozen_contract


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False) + '\n', encoding='utf-8')


def command(args):
    p = subprocess.run(args, capture_output=True, text=True, encoding='utf-8', errors='replace')
    return {'command': args, 'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}


def main(out):
    root = Path.cwd()
    out = Path(out)
    data, _, sources = cd.load_development(root/'data/task01_official', root/'outputs/gpu_pc_split_20261004_03')
    split = root/'outputs/gpu_pc_split_20261004_03/split_manifest.csv'
    validate_frozen_contract(data, sources, digest(split))
    protected = list((root/'data/task01_official').glob('*.csv')) + [split]
    protected += list((root/'outputs/gpu_pc_candidate_20261004_01/bundle').glob('*'))
    dump(out/'source_hashes.json', {str(p.relative_to(root)): digest(p) for p in protected if p.is_file()})
    import sys
    dump(out/'environment.json', {'time': dt.datetime.now().astimezone().isoformat(), 'python': sys.version,
        'free_disk_bytes': shutil.disk_usage(root).free, 'git': command(['git','status','--short','--branch']),
        'head': command(['git','rev-parse','HEAD']), 'packages': command([sys.executable,'-m','pip','freeze']),
        'gpu': command(['nvidia-smi','--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu','--format=csv']),
        'screen_lock': 'terminal computation does not require an unlocked desktop; no mouse or security-policy changes'})
    tracked = subprocess.check_output(['git','ls-files'], text=True, encoding='utf-8').splitlines()
    # Git quotes non-ASCII paths by default: use -z for exact paths.
    tracked = subprocess.check_output(['git','ls-files','-z']).decode('utf-8').split('\0')
    paths = {root/p for p in tracked if p and (p.endswith(('.md','.py','.txt')) or p.startswith('reports/gpu_pc_'))}
    paths.update(root.glob('KAMP_*20261004.md'))
    patterns = ['gpu_pc_nested_*','gpu_pc_refinement_*','gpu_pc_ensemble_*','gpu_pc_review_*',
                'gpu_pc_candidate_20261004_01','gpu_pc_balanced_queue_20261004_01','gpu_pc_balance_*','gpu_pc_calibrated_*']
    hash_checks = []
    for pattern in patterns:
        for folder in (root/'outputs').glob(pattern):
            for p in folder.iterdir():
                if p.is_file() and p.suffix in ('.json','.csv','.log'):
                    paths.add(p)
            hp = folder/'artifact_hashes.json'
            if hp.exists():
                hashes = json.loads(hp.read_text(encoding='utf-8'))
                if isinstance(hashes, dict):
                    for name, expected in hashes.items():
                        p = folder/name
                        ok = p.is_file() and isinstance(expected,str) and digest(p) == expected
                        hash_checks.append({'path': str(p.relative_to(root)), 'matches': ok})
    entries, excerpts = [], []
    for p in sorted(paths):
        if not p.is_file():
            continue
        rel = str(p.relative_to(root)).replace('\\','/')
        raw = p.read_text(encoding='utf-8', errors='replace')
        lines = raw.splitlines()
        links = re.findall(r'\]\(([^)]+)\)', raw) if p.suffix == '.md' else []
        selected = [(i+1,l) for i,l in enumerate(lines) if re.search(r'^#{1,3} |K144|K192|TP|오탐|정밀도|누수|상충|selected|status|split_sha|1986|1,986',l)]
        entries.append({'path': rel, 'sha256': digest(p), 'bytes': p.stat().st_size,
            'status': 'partially_read', 'read_range': 'machine scanned all lines; selected evidence lines extracted; not full human/agent content verification',
            'line_count': len(lines), 'evidence_lines': selected[:30], 'linked_evidence': links,
            'experiment_time': 'see associated execution.json; filename date is not proof of execution',
            'data_split_model_policy_budget_seed_unit': 'see linked run plan/raw predictions; historical and batch contracts are separated',
            'contradictions': [],
            'schema_note': '24 raw grouping sensors; legacy model drops constant Clamp_Open_Position and adds machine_rg3 (23 sensors + indicator)' if rel.startswith('docs/04') else ''})
        if p.suffix == '.md':
            excerpts.append('## '+rel+'\n'+ '\n'.join(f'{i}: {l}' for i,l in selected[:12]))
    entries.append({'path':'outputs/my_upgrade_run_01','status':'not_present' if not (root/'outputs/my_upgrade_run_01').exists() else 'partially_read'})
    dump(out/'document_inventory.json', {'entries':entries,'old_artifact_hash_checks':hash_checks,
        'honesty_note':'Automated inventory and excerpts are not equivalent to fully reading every document.'})
    (out/'document_excerpts.md').write_text('\n\n'.join(excerpts), encoding='utf-8')
    (out/'evidence_map.md').write_text('# 증거 지도\n\n원본 CSV → 보존 split → 개발 1913행 → 저장 OOF 재계산 → 새 nested 연구.\n\n'
        '- 원본·기존 번들: source_hashes.json (완료 시 재대조)\n- 문서·코드·과거 실행: document_inventory.json, document_excerpts.md\n'
        '- 과거 GPU 조합: outputs/gpu_pc_nested_20261004_01/oof_predictions.csv에서 재구성\n'
        '- 역사적 pooled Top-K와 새 fold별 batch Top-K는 다른 계약이므로 따로 비교\n'
        '- 코드북·샷/캐비티·예측 시각·공통 scaler: 미확정. 그룹은 원본24개, 기존 모델은 상수1개 제외23개+설비표시. 새 연구의 원본24열 배치 차이는 feature_layout_audit.json 참조\n'
        '- 후보 선택: inner 3-seed 앙상블만 사용. 과거 개발 결과를 이미 알고 설계한 연구\n'
        '- 원자료·행별 OOF·모델은 로컬에만 보존; 공개 보고에는 집계만 포함\n',encoding='utf-8')
    groups = data.groupby(['machine','feature_group'])[mg.TARGET].agg(n='size',c='sum').reset_index()
    groups['q'] = groups.c/groups.n
    groups['kind'] = np.where(groups.c.eq(0),'pure_negative',np.where(groups.c.eq(groups.n),'pure_positive','conflicting'))
    groups.to_csv(out/'collision_audit.csv',index=False)
    dump(out/'data_contract.json', {'N':len(data),'P':int(data[mg.TARGET].sum()),'groups':len(groups),'features':FEATURES,
        'by_machine':groups.groupby('machine').agg(rows=('n','sum'),positives=('c','sum'),groups=('n','size')).to_dict('index'),
        'conflicting_positives':groups.loc[groups.kind.eq('conflicting')].groupby('machine').c.sum().to_dict(),
        'old_holdout_excluded':480,'label_meaning':'UNKNOWN_numeric_1_only','field_approved':False,
        'unit':'original row; feature_group is not a confirmed physical shot','source':sources})
    frontier = []
    for machine in ['cn7','rg3','all']:
        g = groups if machine == 'all' else groups[groups.machine.eq(machine)]
        dp = np.full(int(g.n.sum())+1,-100000,dtype=int); dp[0]=0
        for n,c in g[['n','c']].itertuples(index=False,name=None):
            dp[n:] = np.maximum(dp[n:],dp[:-n]+c)
        for k,t in enumerate(dp):
            if t>=0: frontier.append({'machine':machine,'k':k,'oracle_tp':int(t),'oracle_fp':int(k-t),'scope':'label-informed whole-group oracle, not achieved performance'})
    pd.DataFrame(frontier).to_csv(out/'identifiability_frontier.csv',index=False)
    print(json.dumps({'documents':len(entries),'old_hashes':len(hash_checks),'mismatches':sum(not x['matches'] for x in hash_checks),'N':len(data),'P':int(data[mg.TARGET].sum())}))


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output-dir',required=True)
    main(p.parse_args().output_dir)
