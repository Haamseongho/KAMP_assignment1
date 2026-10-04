"""Bounded, train-only selection for the fixed K144 precision frontier.

No production model is replaced. Raw margins are gathered before batch ranking.
The frozen experiment plan and per-fit checkpoints allow audit without re-fitting.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import shutil
import subprocess
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from xgboost import XGBRanker

import calibration_diagnostics as cd
import moldguard as mg
from model_upgrade import identity, inner_folds
from research_split import FEATURES
from research_runtime import digest
from task01_precision_audit import dump
from task01_compare import check_fitted_gpu_backend

SEEDS = [20260922,20260923,20260924]
SPLIT_SEED = 20261004
ALPHAS = [1.,0.,.25,.5]
COMPONENTS = {f'cat_{rep}_a{a:g}': {'kind':'cat','representation':rep,'alpha':a}
              for a in ALPHAS for rep in ['row','event']}
COMPONENTS.update({f'rank_{obj}':{'kind':'rank','representation':'row','objective':'rank:'+obj}
                   for obj in ['pairwise','ndcg']})
PIPELINES = {f'pair_a{a:g}': {f'cat_row_a{a:g}':.5,f'cat_event_a{a:g}':.5} for a in ALPHAS}
PIPELINES.update({f'rank_{o}':{f'rank_{o}':1.} for o in ['pairwise','ndcg']})
PIPELINES.update({f'mix_{o}':{'cat_row_a1':.25,'cat_event_a1':.25,f'rank_{o}':.5} for o in ['pairwise','ndcg']})


def largest_remainder(counts, k):
    """Label-free allocation, ties resolved by ascending batch identifier."""
    keys = sorted(counts)
    exact = np.array([counts[x] for x in keys],dtype=float)*k/sum(counts.values())
    allocation = np.floor(exact).astype(int)
    order = sorted(range(len(keys)), key=lambda i: (-(exact[i]-allocation[i]),keys[i]))
    for i in order[:k-int(allocation.sum())]: allocation[i] += 1
    return dict(zip(keys,allocation.tolist()))


def choose(frame, scores, fraction=.075, batch=True):
    score = np.asarray(scores)
    if len(score)!=len(frame) or not np.isfinite(score).all(): raise ValueError('Invalid scores')
    selected = np.zeros(len(frame),bool)
    for machine in sorted(frame.machine.unique()):
        pos = np.flatnonzero(frame.machine.eq(machine).to_numpy())
        k = int(np.ceil(len(pos)*fraction))
        batches = frame.iloc[pos].batch.to_numpy() if batch else np.zeros(len(pos),int)
        counts = {int(b):int(np.sum(batches==b)) for b in np.unique(batches)}
        for b, count in largest_remainder(counts,k).items():
            p = pos[batches==b]
            ids = frame.iloc[p].source_row_id.to_numpy()
            # draw_id disambiguates repeat bootstrap draws without labels.
            draws = frame.iloc[p].draw_id.to_numpy() if 'draw_id' in frame else np.zeros(len(p),int)
            order = p[np.lexsort((draws,ids,-score[p]))]
            selected[order[:count]] = True
    return selected


def counts(y, selected, score=None):
    y=np.asarray(y,dtype=int); selected=np.asarray(selected,dtype=bool)
    n=len(y); p=int(y.sum()); k=int(selected.sum()); t=int(y[selected].sum())
    fp=k-t; fn=p-t; tn=n-p-fp
    result={'n':n,'positives':p,'k':k,'tp':t,'fp':fp,'fn':fn,'tn':tn,
        'precision':t/k if k else 0.,'recall':t/p if p else None,'f1':2*t/(k+p) if k+p else 0.,
        'accuracy':(t+tn)/n,'fpr':fp/(n-p) if n>p else None,'fdr':fp/k if k else None}
    if score is not None:
        result['ap']=float(average_precision_score(y,score)) if p else None
        result['auc']=float(roc_auc_score(y,score)) if 0<p<n else None
    return result


def raw_predict(model, frame, kind):
    x=frame[FEATURES]
    unique=x.drop_duplicates(); keys=pd.util.hash_pandas_object(x,index=False)
    uk=pd.util.hash_pandas_object(unique,index=False)
    if uk.duplicated().any(): raise ValueError('Feature hash collision')
    score=model.predict(unique) if kind=='rank' else model.predict_proba(unique)[:,1]
    result=keys.map(pd.Series(score,index=uk.to_numpy())).to_numpy(float)
    if not np.isfinite(result).all(): raise ValueError('Nonfinite predictions')
    return result


def pipeline_scores(raw, batch_ids):
    ranks={}
    for component, values in raw.items():
        r=np.empty_like(values)
        for b in np.unique(batch_ids):
            mask=batch_ids==b
            r[mask]=rankdata(values[mask],method='average')/mask.sum()
        ranks[component]=r
    result={name:sum(weight*ranks[c] for c,weight in weights.items())
            for name,weights in PIPELINES.items() if set(weights)<=set(ranks)}
    # Row/event are diagnostic components, not additional selectable pipelines.
    result.update({name:r for name,r in ranks.items() if name.startswith('cat_')})
    return result


def training_view(frame, representation):
    if representation=='row': return frame.copy()
    group=frame.groupby('feature_group',sort=True)
    fit=group.first().reset_index(); fit[mg.TARGET]=group[mg.TARGET].max().to_numpy(int)
    return fit


class Runner:
    def __init__(self,out):
        self.out=Path(out); self.start=time.monotonic(); self.ledger=[]; self.failures=[]; self.resources=[]
        self.fits=self.out/'fit_checkpoints'; self.fits.mkdir(exist_ok=True)
        self.signature=None

    def monitor(self):
        free=shutil.disk_usage(self.out).free
        if free<5*1024**3: raise RuntimeError('Disk reserve below 5 GiB')
        if time.monotonic()-self.start>7200: raise TimeoutError('Predeclared 2 hour wall limit')
        if sum(x.get('kind') in ['cat','rank'] for x in self.ledger)>=790: raise RuntimeError('GPU fit limit')
        if sum(p.stat().st_size for p in self.out.rglob('*') if p.is_file())>2*1024**3: raise RuntimeError('Output limit')
        result=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        used,available,util=[int(v.strip()) for v in result.strip().split(',')]
        self.resources.append({'time':dt.datetime.now().astimezone().isoformat(),'elapsed':time.monotonic()-self.start,
            'gpu_memory_used_mib':used,'gpu_memory_free_mib':available,'gpu_util_percent':util,'disk_free_bytes':free,
            'measurement':'between fits, not guaranteed peak'})
        if available<768: raise RuntimeError('GPU reserve below 768 MiB')

    def fit(self,train,valid,component,seed,location,save=False,auto=False):
        self.monitor()
        if set(train.feature_group)&set(valid.feature_group): raise ValueError('Group leakage')
        spec=COMPONENTS[component] if component!='cn7_anchor' else {'kind':'cpu','representation':'row'}
        fit=training_view(train,spec['representation']); y=fit[mg.TARGET].to_numpy(int)
        ratio=float((len(y)-y.sum())/y.sum()); kind=spec['kind']
        key=f"{location['stage']}_{location['outer']}_{location['inner']}_{component}_{seed}"+('_auto' if auto else '')
        cp=self.fits/(key+'.json'); scorefile=self.fits/(key+'.npz')
        provenance={'signature':self.signature,'train_ids':train[['machine',mg.ID_COL]].to_dict('list'),
                    'valid_ids':valid[['machine',mg.ID_COL]].to_dict('list'),'component':component,'seed':seed,'auto':auto}
        ph=hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest()
        if cp.exists():
            record=json.loads(cp.read_text(encoding='utf-8'))
            if record['provenance_hash']!=ph or record['status']!='passed': raise ValueError('Invalid cache')
            if digest(scorefile)!=record['score_sha256']: raise ValueError('Corrupt cached predictions')
            self.ledger.append(record)
            return np.load(scorefile)['score'],record.get('model_file')
        if kind=='cat':
            params=dict(iterations=200,depth=3,learning_rate=.03,l2_leaf_reg=10,border_count=64,
                task_type='GPU',devices='0',gpu_ram_part=.45,thread_count=2,random_seed=seed,
                verbose=False,allow_writing_files=False)
            params.update({'auto_class_weights':'Balanced'} if auto else {'class_weights':[1.,ratio**spec['alpha']]})
            model=CatBoostClassifier(**params)
        elif kind=='rank':
            params=dict(objective=spec['objective'],tree_method='hist',device='cuda',max_depth=2,
                n_estimators=250,learning_rate=.03,reg_lambda=10,max_bin=64,n_jobs=2,random_state=seed,
                lambdarank_pair_method='mean',lambdarank_num_pair_per_sample=4,validate_parameters=True)
            model=XGBRanker(**params)
        else:
            params={'C':1.,'class_weight':'balanced','max_iter':3000,'random_state':seed}
            model=make_pipeline(StandardScaler(),LogisticRegression(**params))
        started=time.monotonic()
        record={**location,'component':component,'seed':seed,'split_seed':SPLIT_SEED,'kind':kind,
            'target_representation':spec['representation'],'original_rows':len(train),'original_groups':train.feature_group.nunique(),
            'fit_rows':len(fit),'fit_groups':fit.feature_group.nunique(),'fit_positive_rows':int(y.sum()),
            'positive_groups':train.loc[train[mg.TARGET].eq(1),'feature_group'].nunique(),'negative_positive_ratio':ratio,
            'positive_weight':ratio**spec.get('alpha',0),'alpha':spec.get('alpha'),'group_overlap':0,
            'train_group_sha256':hashlib.sha256('\n'.join(sorted(train.feature_group)).encode()).hexdigest(),
            'parameters':params,'provenance_hash':ph,'started':dt.datetime.now().astimezone().isoformat()}
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                model.fit(fit[FEATURES],y,**({'qid':np.zeros(len(y),dtype=np.int32)} if kind=='rank' else {}))
            record['warnings']=[str(w.message) for w in caught]
            if any('not used' in w or 'not been used' in w for w in record['warnings']): raise ValueError('Unused GPU option')
            backend=check_fitted_gpu_backend(model,'xgb_ranker' if kind=='rank' else 'catboost_classifier') if kind!='cpu' else 'CPU'
            score=raw_predict(model,valid,kind) if len(valid) else np.array([],float)
            config=json.loads(model.get_booster().save_config()) if kind=='rank' else model.get_all_params() if kind=='cat' else params
            if kind=='rank':
                serialized=json.dumps(config)
                if 'mean' not in serialized or config['learner']['objective']['name']!=spec['objective']: raise ValueError('Rank config mismatch')
            model_file=None
            if save:
                model_file=key+('.cbm' if kind=='cat' else '.ubj' if kind=='rank' else '.weights.json')
                if kind=='cpu':
                    s,m=model[0],model[-1]
                    dump(self.fits/model_file,{'mean':s.mean_.tolist(),'scale':s.scale_.tolist(),'coef':m.coef_[0].tolist(),'intercept':float(m.intercept_[0])})
                else: model.save_model(str(self.fits/model_file))
            np.savez_compressed(scorefile,score=score)
            record.update(status='passed',backend=backend,model_file=model_file,actual_config=config,score_sha256=digest(scorefile))
        except Exception as e:
            record.update(status='failed',error=repr(e)); self.failures.append(record)
            dump(self.out/'experiment_failures.json',self.failures)
            raise
        finally:
            record.update(elapsed_seconds=time.monotonic()-started,ended=dt.datetime.now().astimezone().isoformat())
            dump(cp,record); self.ledger.append(record)
            pd.DataFrame(self.ledger).to_csv(self.out/'fit_ledger.csv',index=False)
            pd.DataFrame(self.resources).to_csv(self.out/'resource_usage.csv',index=False)
        return score,model_file


def freeze(out):
    plan={'run_id':out.name,'created':dt.datetime.now().astimezone().isoformat(),'model_seeds':SEEDS,'split_seed':SPLIT_SEED,
        'components':COMPONENTS,'pipelines':PIPELINES,'anchor':'CN7 StandardScaler + balanced LR C1',
        'objective':'inner three-seed ensemble TP at per-machine ceil(7.5%); ties RG3 AP, component count, fixed name order',
        'outer':'existing frozen 5 grouped folds','inner':'shared grouped stratified 3 fold, split_seed fixed separate from model_seed',
        'policy':'A: percentile ranks calculated once per full validation batch after raw inference; largest remainder allocation',
        'budgets':[.05,.075,.1],'tie':'descending score, ascending source_row_id; bootstrap draw_id tertiary',
        'early_stopping':False,'features':FEATURES,'prior_results_known':True,'outer_labels_for_selection':False,
        'sensitivity':{'split_seed':20261005,'folds':3,'model_seed':20260922,'pipelines':['pair_a1','pair_a0.5'],'not_used_for_selection':True},
        'bootstrap':{'repetitions':2000,'seed':20261004,'strata':['machine','batch','group_size'],
                     'unit':'whole group, unique draw_id','zero_positive':'recall/AP undefined, omit only from that CI'},
        'limits':{'gpu_fits':790,'cpu_fits':100,'seconds':7200,'output_gib':2,'disk_reserve_gib':5,'gpu_reserve_mib':768,'threads':2},
        'max_expected_gpu_fits':746,'optional_experiments':'no hard-negative, extra features or policy reallocation; limited information and compute',
        'full_bundle_selection':'same 3-seed inner procedure on all development; never outer ranking of candidates',
        'smoke':'first real fits verify backend/config; two additional auto-balanced fits compare effective class weights',
        'failure':'stop and preserve checkpoints, no silent CPU fallback or result-driven new candidates'}
    p=out/'experiment_plan.json'
    if p.exists():
        return json.loads(p.read_text(encoding='utf-8'))
    dump(p,plan); (out/'experiment_plan.sha256').write_text(digest(p)+'\n',encoding='ascii')
    dump(out/'policy_contract.json',{k:plan[k] for k in ['policy','budgets','tie','bootstrap','full_bundle_selection']})
    return plan


def baseline(data,out):
    old=Path('outputs/gpu_pc_nested_20261004_01')
    oof=pd.read_csv(old/'oof_predictions.csv',float_precision='round_trip')
    base=identity(data); base['batch']=base.outer_fold
    predictions=[]; summary=[]
    for seed in SEEDS:
        values={}
        for name in ['baseline','fixed_local_logistic_c1','fixed_cat_row_d3','fixed_cat_event_d3']:
            rows=oof[oof.seed.eq(seed)&oof.model.eq(name)]
            joined=base.merge(rows[['machine','source_row_id','label_value','score']],on=['machine','source_row_id','label_value'],validate='one_to_one',sort=False)
            if len(joined)!=len(base): raise ValueError('Baseline coverage')
            values[name]=joined.score.to_numpy()
        rg=base.machine.eq('rg3').to_numpy(); raw={f'cat_{rep}_a1':values[f'fixed_cat_{rep}_d3'][rg] for rep in ['row','event']}
        gpu=values['fixed_local_logistic_c1'].copy()
        gpu[rg]=pipeline_scores(raw,base.loc[rg,'batch'].to_numpy())['pair_a1']
        for name,score in [('historical_gpu',gpu),('historical_cpu',values['baseline'])]:
            b=base.copy(); b['pipeline']=name;b['seed']=seed;b['score']=score;predictions.append(b)
    table=pd.concat(predictions,ignore_index=True)
    for name in table.pipeline.unique():
        rows=table[table.pipeline.eq(name)]
        mean=rows.groupby(['machine','source_row_id'],sort=False).score.mean().reset_index()
        b=base.merge(mean,on=['machine','source_row_id'],validate='one_to_one',sort=False)
        b['pipeline']=name;b['seed']='ensemble';predictions.append(b)
    table=pd.concat(predictions,ignore_index=True)
    for (name,seed),p in table.groupby(['pipeline','seed'],sort=False):
        for f in [.05,.075,.1]:
            for contract in ['historical_pooled','batch_allocated']:
                d=choose(p,p.score,f,contract=='batch_allocated')
                summary.append({'pipeline':name,'seed':seed,'fraction':f,'contract':contract,**counts(p.label_value,d,p.score)})
    result=pd.DataFrame(summary)
    expected=result.query("pipeline=='historical_gpu' and seed=='ensemble' and fraction==0.075 and contract=='historical_pooled'").iloc[0]
    if expected.tp!=13 or expected.fp!=131: raise ValueError('Historical baseline mismatch')
    table.to_csv(out/'baseline_predictions.csv.gz',index=False)
    result.to_csv(out/'baseline_metrics.csv',index=False)
    dump(out/'baseline_reproduction.json',{'status':'passed','source_sha256':digest(old/'oof_predictions.csv'),
        'method':'recomputed saved raw OOF, not re-training all historical fits','metrics':summary})


def train(out):
    out=Path(out); plan=freeze(out)
    data,_,_=cd.load_development(Path('data/task01_official'),Path('outputs/gpu_pc_split_20261004_03'))
    baseline(data,out)
    runner=Runner(out)
    runner.signature=hashlib.sha256((digest(out/'experiment_plan.json')+digest(Path(__file__))+digest(out/'source_hashes.json')+digest(out/'environment.json')).encode()).hexdigest()
    selection=[]; outer_records=[]; inner_records=[]; splits=[]; bundle_records={}
    for outer in list(range(5))+[-1]:
        training=data[data.outer_fold.ne(outer)].reset_index(drop=True) if outer>=0 else data.copy().reset_index(drop=True)
        valid=data[data.outer_fold.eq(outer)].reset_index(drop=True) if outer>=0 else data.iloc[:0].copy()
        rg_train=training[training.machine.eq('rg3')].reset_index(drop=True)
        folds=inner_folds(training,SPLIT_SEED)
        assignments=np.full(len(training),-1)
        for inner,(a,b) in enumerate(folds):
            assignments[b]=inner
            splits.extend({'outer':outer,'inner':inner,'role':role,'machine':row.machine,'source_row_id':row.source_row_id,'feature_group':row.feature_group}
                for pos,role in [(a,'fit'),(b,'valid')] for row in identity(training.iloc[pos]).itertuples())
        rgmask=training.machine.eq('rg3').to_numpy(); inner_base=identity(training.loc[rgmask]);inner_base['batch']=assignments[rgmask]
        raw_by_seed={s:{} for s in SEEDS}
        for component in COMPONENTS:
            for seed in SEEDS:
                raw=np.full(len(training),np.nan)
                for inner,(a,b) in enumerate(folds):
                    tr=training.iloc[a];tr=tr[tr.machine.eq('rg3')]
                    positions=b[training.iloc[b].machine.eq('rg3').to_numpy()]
                    va=training.iloc[positions]
                    value,_=runner.fit(tr,va,component,seed,{'stage':'inner','outer':outer,'inner':inner})
                    raw[positions]=value
                raw_by_seed[seed][component]=raw[rgmask]
                rec=inner_base.copy();rec['component']=component;rec['seed']=seed;rec['outer']=outer;rec['raw_score']=raw[rgmask];inner_records.append(rec)
            print(f'outer={outer} inner component={component} fits={len(runner.ledger)}',flush=True)
        inner_scores={s:pipeline_scores(raw_by_seed[s],inner_base.batch.to_numpy()) for s in SEEDS}
        candidates=[]
        for name in PIPELINES:
            score=np.mean([inner_scores[s][name] for s in SEEDS],axis=0)
            result=counts(inner_base.label_value,choose(inner_base,score),score)
            candidates.append({'pipeline':name,**result})
        winner=max(candidates,key=lambda x:(x['tp'],x['ap'],-len(PIPELINES[x['pipeline']]),-list(PIPELINES).index(x['pipeline'])))['pipeline']
        selection.append({'outer':outer,'winner':winner,'candidates':candidates,'selection_scope':'inner ensemble only','outer_labels_accessed':False})
        dump(out/'selection_ledger.json',selection)
        required=set(PIPELINES[winner]) if outer==-1 else set(COMPONENTS)
        rg_valid=valid[valid.machine.eq('rg3')].reset_index(drop=True)
        raw_outer={s:{} for s in SEEDS}; model_files=[]
        for component in COMPONENTS:
            if component not in required: continue
            for seed in SEEDS:
                score,filename=runner.fit(rg_train,rg_valid,component,seed,{'stage':'outer' if outer>=0 else 'full','outer':outer,'inner':-1},save=component in PIPELINES[winner])
                raw_outer[seed][component]=score
                if filename:model_files.append({'component':component,'seed':seed,'file':filename,'weight':PIPELINES[winner][component]/3})
        cn_train=training[training.machine.eq('cn7')];cn_valid=valid[valid.machine.eq('cn7')]
        cn_score,cn_file=runner.fit(cn_train,cn_valid,'cn7_anchor',SEEDS[0],{'stage':'outer' if outer>=0 else 'full','outer':outer,'inner':-1},save=True)
        bundle_records[str(outer)]={'pipeline':winner,'models':model_files,'cn7_file':cn_file}
        dump(out/'bundle_sources.json',bundle_records)
        if outer>=0:
            b=identity(valid);b['batch']=outer;rg=b.machine.eq('rg3').to_numpy()
            for seed in SEEDS:
                ps=pipeline_scores(raw_outer[seed],np.zeros(len(rg_valid),int))
                ps['nested_selected']=ps[winner]
                for name,score in ps.items():
                    rec=b.copy();rec['pipeline']=name;rec['seed']=seed;rec['score']=0.
                    rec.loc[rg,'score']=score;rec.loc[~rg,'score']=cn_score;outer_records.append(rec)
                for component,score in raw_outer[seed].items():
                    rec=identity(rg_valid);rec['component']=component;rec['seed']=seed;rec['outer']=outer;rec['raw_score']=score
                    rec.to_csv(out/f'outer_raw_{outer}_{component}_{seed}.csv.gz',index=False)
        pd.concat(inner_records,ignore_index=True).to_csv(out/'inner_oof_predictions.csv.gz',index=False)
        if outer_records: pd.concat(outer_records,ignore_index=True).to_csv(out/'outer_oof_predictions.csv.gz',index=False)
        pd.DataFrame(splits).to_csv(out/'inner_split_manifest.csv.gz',index=False)
        print(f'completed outer={outer}, winner={winner}, fits={len(runner.ledger)}',flush=True)
    # Actual library normalization check, with the same real training fold.
    rg=data[data.machine.eq('rg3') & data.outer_fold.ne(0)]
    va=data[data.machine.eq('rg3') & data.outer_fold.eq(0)]
    balanced=[]
    for rep in ['row','event']:
        component=f'cat_{rep}_a1'
        score,_=runner.fit(rg,va,component,SEEDS[0],{'stage':'balanced_check','outer':0,'inner':-1},auto=True)
        params=runner.ledger[-1]['actual_config']
        explicit=pd.read_csv(out/f'outer_raw_0_{component}_{SEEDS[0]}.csv.gz').raw_score.to_numpy()
        balanced.append({'representation':rep,'actual_class_weights':params.get('class_weights'),
            'ratio':runner.ledger[-1]['negative_positive_ratio'],'max_score_difference':float(np.max(np.abs(score-explicit))),
            'note':'GPU refit nondeterminism permitted; effective class weights should match'})
    dump(out/'balanced_normalization.json',balanced)
    # Prespecified extra split diagnostic, not a candidate selection rerun.
    sensitivity=[]
    for fold,(a,b) in enumerate(inner_folds(data,20261005)):
        tr=data.iloc[a];tr=tr[tr.machine.eq('rg3')]; va=data.iloc[b];va=va[va.machine.eq('rg3')]
        raw={}
        for c in ['cat_row_a1','cat_event_a1','cat_row_a0.5','cat_event_a0.5']:
            raw[c],_=runner.fit(tr,va,c,SEEDS[0],{'stage':'sensitivity','outer':fold,'inner':-1})
        for name,score in pipeline_scores(raw,np.zeros(len(va),int)).items():
            if name.startswith('pair_'):
                rec=identity(va);rec['batch']=fold;rec['pipeline']=name;rec['seed']=SEEDS[0];rec['score']=score;sensitivity.append(rec)
    pd.concat(sensitivity,ignore_index=True).to_csv(out/'sensitivity_predictions.csv.gz',index=False)
    dump(out/'experiment_failures.json',runner.failures)
    dump(out/'training_complete.json',{'status':'passed','fits':len(runner.ledger),'elapsed_seconds':time.monotonic()-runner.start,
        'plan_sha256':digest(out/'experiment_plan.json'),'code_sha256':digest(Path(__file__))})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);args=p.parse_args()
    with threadpool_limits(limits=2): train(args.output_dir)
