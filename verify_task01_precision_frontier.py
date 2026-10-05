"""Independent row-level verifier; does not call the production scoring/metric code."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,confusion_matrix,precision_score,recall_score,f1_score,accuracy_score

import calibration_diagnostics as cd
import moldguard as mg
from model_upgrade import identity,inner_folds
from research_runtime import digest
from task01_precision_audit import dump


def select_independent(p,fraction,batch):
    q=p.reset_index(drop=True).copy();q['position']=range(len(q));selected=[]
    for m,machine in q.groupby('machine',sort=True):
        k=int(np.ceil(len(machine)*fraction))
        groups=list(machine.groupby('batch',sort=True)) if batch else [(0,machine)]
        quotas={b:len(g)*k/len(machine) for b,g in groups};alloc={b:int(v) for b,v in quotas.items()}
        for b in sorted(alloc,key=lambda b:(-(quotas[b]-alloc[b]),b))[:k-sum(alloc.values())]:alloc[b]+=1
        for b,g in groups:
            selected.extend(g.sort_values(['score','source_row_id'],ascending=[False,True],kind='stable').head(alloc[b]).position)
    d=np.zeros(len(q),bool);d[selected]=True
    return d


def verify(out):
    out=Path(out);checks=[]
    def check(condition,name):
        checks.append({'test':name,'passed':bool(condition)})
        if not condition: raise AssertionError(name)
    hashes=json.loads((out/'source_hashes.json').read_text(encoding='utf-8'))
    check(all(digest(Path(name))==value for name,value in hashes.items()),'source_split_old_bundle_immutable')
    check(digest(out/'experiment_plan.json')==(out/'experiment_plan.sha256').read_text().strip(),'plan_immutable')
    plan=json.loads((out/'experiment_plan.json').read_text(encoding='utf-8'))
    data,_,_=cd.load_development(Path('data/task01_official'),Path('outputs/gpu_pc_split_20261004_03'))
    base=identity(data)
    outer=pd.read_csv(out/'outer_oof_predictions.csv.gz',float_precision='round_trip')
    for (p,s),rows in outer.groupby(['pipeline','seed']):
        check(len(rows)==len(base) and not rows.duplicated(['machine','source_row_id']).any(),f'coverage_{p}_{s}')
        match=base.merge(rows,on=['machine','source_row_id'],suffixes=('_base','_prediction'),validate='one_to_one')
        for c in ['label_value','feature_group','outer_fold']:
            check(match[c+'_base'].eq(match[c+'_prediction']).all(),f'identity_{c}_{p}_{s}')
        check(rows.groupby('feature_group').score.nunique().max()==1,f'identical_input_score_{p}_{s}')
    # Independent confusion matrices and metric definitions for every recorded table row.
    evaluated=pd.read_csv(out/'evaluated_predictions.csv.gz',float_precision='round_trip',dtype={'seed':str})
    reported=pd.read_csv(out/'metrics_by_machine_seed_budget.csv',dtype={'seed':str})
    for (p,s,contract,fraction),rows in reported.groupby(['pipeline','seed','contract','fraction']):
        frame=evaluated[evaluated.pipeline.eq(p)&evaluated.seed.eq(s)].reset_index(drop=True)
        d=select_independent(frame,fraction,contract=='batch_allocated')
        for r in rows.itertuples():
            mask=np.ones(len(frame),bool) if r.machine=='all' else frame.machine.eq(r.machine).to_numpy()
            y=frame.loc[mask,'label_value'];decision=d[mask];tn,fp,fn,tp=confusion_matrix(y,decision,labels=[0,1]).ravel()
            got=[tn,fp,fn,tp,int(decision.sum())];expected=[r.tn,r.fp,r.fn,r.tp,r.k]
            if got!=expected:raise AssertionError(('confusion',p,s,contract,r.machine,got,expected))
            metrics={'precision':precision_score(y,decision,zero_division=0),'recall':recall_score(y,decision,zero_division=0),
                'f1':f1_score(y,decision,zero_division=0),'accuracy':accuracy_score(y,decision),
                'fpr':fp/(tn+fp),'fdr':fp/(tp+fp),'ap':average_precision_score(y,frame.loc[mask,'score'])}
            if any(abs(v-getattr(r,k))>1e-12 for k,v in metrics.items()):raise AssertionError(('metrics',p,s,r.machine))
    check(True,f'independent_metrics_{len(reported)}_rows')
    inner=pd.read_csv(out/'inner_oof_predictions.csv.gz',float_precision='round_trip')
    selections=json.loads((out/'selection_ledger.json').read_text(encoding='utf-8'))
    assignments={}
    for outerfold in list(range(5))+[-1]:
        training=data[data.outer_fold.ne(outerfold)].reset_index(drop=True) if outerfold>=0 else data.copy()
        splits=inner_folds(training,plan['split_seed']);assignments[outerfold]=(training,splits)
        local=identity(training[training.machine.eq('rg3')])
        batches={}
        for i,(fit,valid) in enumerate(splits):
            check(not(set(training.iloc[fit].feature_group)&set(training.iloc[valid].feature_group)),f'no_group_leak_{outerfold}_{i}')
            for group in training.iloc[valid].feature_group:batches[group]=i
        local['batch']=local.feature_group.map(batches)
        arrays={}
        for (component,seed),rows in inner[inner.outer.eq(outerfold)].groupby(['component','seed']):
            check(len(rows)==len(local) and not rows.duplicated(['machine','source_row_id']).any(),f'inner_coverage_{outerfold}_{component}_{seed}')
            joined=local.merge(rows[['machine','source_row_id','label_value','batch','raw_score']],on=['machine','source_row_id'],validate='one_to_one',suffixes=('','_record'))
            check(joined.label_value.eq(joined.label_value_record).all() and joined.batch.eq(joined.batch_record).all(),f'inner_identity_{outerfold}_{component}_{seed}')
            arrays[(component,seed)]=joined.groupby('batch').raw_score.rank(method='average',pct=True).to_numpy()
        rows=[]
        for name,weights in plan['pipelines'].items():
            s=np.mean([sum(w*arrays[(c,seed)] for c,w in weights.items()) for seed in plan['model_seeds']],axis=0)
            frame=local.copy();frame['score']=s;d=select_independent(frame,.075,True)
            rows.append({'pipeline':name,'tp':int(frame.loc[d,'label_value'].sum()),'ap':average_precision_score(frame.label_value,s)})
        priority=list(plan['pipelines'])
        winner=max(rows,key=lambda r:(r['tp'],r['ap'],-len(plan['pipelines'][r['pipeline']]),-priority.index(r['pipeline'])))['pipeline']
        recorded=next(r for r in selections if r['outer']==outerfold)
        check(winner==recorded['winner'] and recorded['outer_labels_accessed'] is False,f'inner_only_selection_{outerfold}')
        for r in rows:
            expected=next(x for x in recorded['candidates'] if x['pipeline']==r['pipeline'])
            check(r['tp']==expected['tp'] and abs(r['ap']-expected['ap'])<1e-12,f'inner_objective_{outerfold}_{r["pipeline"]}')
    records=[json.loads(p.read_text(encoding='utf-8')) for p in (out/'fit_checkpoints').glob('*.json') if not p.name.endswith('.weights.json')]
    gpu=cpu=0
    for r in records:
        stage=r['stage'];fold=r['outer'];component=r['component']
        if stage=='sensitivity':
            a,_=inner_folds(data,20261005)[fold];training=data.iloc[a]
        elif stage in ['outer','balanced_check']:training=data[data.outer_fold.ne(fold)]
        elif stage=='full':training=data
        else:
            alltraining,splits=assignments[fold];training=alltraining.iloc[splits[r['inner']][0]]
        training=training[training.machine.eq('cn7' if component=='cn7_anchor' else 'rg3')]
        expected=hashlib.sha256('\n'.join(sorted(training.feature_group)).encode()).hexdigest()
        check(r['train_group_sha256']==expected and r['status']=='passed' and r['group_overlap']==0,'fit_provenance_'+str(r['provenance_hash']))
        if r['kind']!='cpu':
            gpu+=1;check(r['backend']=='GPU' or r['backend'].startswith('cuda'),'backend_'+r['provenance_hash'])
        else:cpu+=1
        if r['kind']=='cat':
            if r['target_representation']=='event': y=training.groupby('feature_group')[mg.TARGET].max()
            else:y=training[mg.TARGET]
            ratio=(len(y)-y.sum())/y.sum();weights=r['actual_config']['class_weights']
            check(abs(weights[1]-ratio**float(r['alpha']))<1e-4,'weights_'+r['provenance_hash'])
        if r['kind']=='rank':
            config=r['actual_config']['learner'];obj=config['objective']
            check(obj['name']==r['parameters']['objective'],'rank_objective_'+r['provenance_hash'])
            check('"lambdarank_pair_method": "mean"' in json.dumps(config) and '"lambdarank_num_pair_per_sample": "4"' in json.dumps(config),'rank_sampling_'+r['provenance_hash'])
    check(gpu<=790 and cpu<=100,'fit_budgets')
    legacy_count=0
    if (out/'legacy_layout_audit/fit_ledger.csv').exists():
        legacy=pd.read_csv(out/'legacy_layout_audit/fit_ledger.csv')
        legacy_count=len(legacy)
        check(legacy_count==30 and legacy.configured_backend.eq('GPU').all(),'canonical_layout_gpu_reproduction')
        check(gpu+legacy_count<=800,'total_gpu_fit_budget_including_corrective_audit')
    parity=json.loads((out/'policy_parity_tests.json').read_text(encoding='utf-8'))
    check(parity['status']=='passed' and all(t['passed'] for t in parity['tests']),'native_parity_suite')
    bootstrap=pd.read_csv(out/'paired_group_bootstrap.csv')
    check(len(bootstrap)==2000 and bootstrap.n.eq(1913).all() and bootstrap.k.eq(144).all(),'bootstrap_fixed_N_K')
    decision=json.loads((out/'decision.json').read_text(encoding='utf-8'))
    check(decision['field_approved'] is False and decision['independent_validation'] is False,'research_only_no_unlabeled_truth')
    dump(out/'verification.json',{'status':'passed','checks':checks,'count':len(checks),'gpu_fits':gpu+legacy_count,'main_study_gpu_fits':gpu,'layout_audit_gpu_fits':legacy_count,'cpu_fits':cpu,
        'outer_prediction_rows':len(outer),'inner_prediction_rows':len(inner),'metric_rows':len(reported),
        'method':'separate process; pandas sorting/rank and sklearn confusion/metrics, no production select/count import',
        'limits':'does not establish source normalization provenance, physical-label meaning or prospective generalization'})
    print(json.dumps({'status':'passed','checks':len(checks),'gpu':gpu,'cpu':cpu,'metrics':len(reported)}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);verify(p.parse_args().output_dir)
