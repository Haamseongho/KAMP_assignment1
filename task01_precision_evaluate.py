"""All-candidate evaluation, paired group uncertainty and native bundle checks."""
from __future__ import annotations
import argparse
import json
import shutil
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from threadpoolctl import threadpool_limits

import calibration_diagnostics as cd
from model_upgrade import identity
from research_runtime import digest
from research_split import FEATURES
from task01_precision_audit import dump
from task01_precision_frontier import SEEDS, PIPELINES, choose, counts, largest_remainder
from task01_precision_bundle import scores as bundle_scores


def ensemble(table):
    keys=['pipeline','machine','source_row_id','feature_group','outer_fold','label_value','batch']
    means=table.groupby(keys,sort=False,dropna=False).score.mean().reset_index()
    means['seed']='ensemble'
    return pd.concat([table,means],ignore_index=True)


def tie_bounds(frame,score,fraction,batch):
    out=[];score=np.asarray(score)
    for m in sorted(frame.machine.unique()):
        p=np.flatnonzero(frame.machine.eq(m)); batches=frame.iloc[p].batch.to_numpy() if batch else np.zeros(len(p),int)
        allocation=largest_remainder({int(b):int(sum(batches==b)) for b in np.unique(batches)},int(np.ceil(len(p)*fraction)))
        for b,k in allocation.items():
            ix=p[batches==b];s=score[ix];y=frame.iloc[ix].label_value.to_numpy()
            boundary=np.sort(s)[-k] if k else np.inf
            above=s>boundary;ties=s==boundary;remaining=k-int(above.sum())
            t=int(y[above].sum());c=int(y[ties].sum());n=int(ties.sum())
            out.append({'machine':m,'batch':b,'k':k,'tie_rows':n,'tie_positive':c,'boundary_slots':remaining,
                'tp_min':t+max(0,remaining-(n-c)), 'tp_expected':t+remaining*c/n if n else t,
                'tp_max':t+min(remaining,c)})
    return out


def bootstrap(table,out):
    ens=table[table.seed.astype(str).eq('ensemble')]
    baseline=ens[ens.pipeline.eq('pair_a1')].reset_index(drop=True)
    candidate=ens[ens.pipeline.eq('nested_selected')]
    b=baseline.merge(candidate[['machine','source_row_id','score']],on=['machine','source_row_id'],validate='one_to_one',suffixes=('_base','_candidate'))
    grouped=[]
    for (machine,batch,group),p in b.groupby(['machine','batch','feature_group'],sort=True):
        grouped.append({'machine':machine,'batch':batch,'group':group,'size':len(p),'positions':p.index.to_numpy()})
    g=pd.DataFrame(grouped); strata=[p for _,p in g.groupby(['machine','batch','size'])]
    rng=np.random.default_rng(20261004); rows=[]
    for rep in range(2000):
        selected=[];draws=[];draw=0
        for st in strata:
            for at in rng.integers(0,len(st),size=len(st)):
                pos=st.iloc[at].positions;selected.extend(pos);draws.extend([draw]*len(pos));draw+=1
        sample=b.iloc[selected].reset_index(drop=True);sample['draw_id']=draws
        x=counts(sample.label_value,choose(sample,sample.score_base),sample.score_base)
        y=counts(sample.label_value,choose(sample,sample.score_candidate),sample.score_candidate)
        r={'replicate':rep,'n':len(sample),'positives':int(sample.label_value.sum()),'k':x['k'],'draw_groups':draw,
           **{'delta_'+k:y[k]-x[k] if y[k] is not None and x[k] is not None else np.nan for k in ['tp','fp','precision','recall']}}
        for m in ['cn7','rg3']:
            mask=sample.machine.eq(m)
            r['delta_ap_'+m]=(average_precision_score(sample.loc[mask,'label_value'],sample.loc[mask,'score_candidate'])-
                average_precision_score(sample.loc[mask,'label_value'],sample.loc[mask,'score_base'])) if sample.loc[mask,'label_value'].sum() else np.nan
        rows.append(r)
    result=pd.DataFrame(rows);result.to_csv(out/'paired_group_bootstrap.csv',index=False)
    ci={c:{'low':float(result[c].quantile(.025)),'high':float(result[c].quantile(.975)),
           'defined_replicates':int(result[c].notna().sum())} for c in result if c.startswith('delta_')}
    dump(out/'bootstrap_summary.json',{'comparison':'nested_selected minus fresh pair_a1, batch allocation',
        'repetitions':2000,'confidence_interval':ci,'scope':'conditional on fixed OOF; no retraining/search/future uncertainty',
        'strata':['machine','batch','group_size'],'zero_positive':'undefined recall/AP omitted only from that statistic'})
    return ci


def bundles(out,data,table):
    sources=json.loads((out/'bundle_sources.json').read_text(encoding='utf-8'))
    result=[]
    for fold,source in sources.items():
        directory=out/('candidate_bundle' if fold=='-1' else f'outer_bundle_{fold}')
        directory.mkdir(exist_ok=True)
        files=[source['cn7_file']]+[r['file'] for r in source['models']]
        for name in files: shutil.copy2(out/'fit_checkpoints'/name,directory/name)
        manifest={**source,'contract':'complete_batch_rank_v1','field_approved':False,
            'meaning':'relative batch priority; singleton RG3 score 1 is not probability',
            'training_scope':'development only' if fold=='-1' else 'outer training only',
            'files_sha256':{name:digest(directory/name) for name in files},
            'plan_sha256':digest(out/'experiment_plan.json')}
        dump(directory/'bundle.json',manifest)
        frame=data if fold=='-1' else data[data.outer_fold.eq(int(fold))]
        frame=frame.reset_index(drop=True)
        ref=bundle_scores(frame,directory)
        if fold!='-1':
            saved=table[table.pipeline.eq('nested_selected')&table.seed.astype(str).eq('ensemble')&table.outer_fold.eq(int(fold))]
            aligned=identity(frame).merge(saved[['machine','source_row_id','score']],on=['machine','source_row_id'],validate='one_to_one')
            error=float(np.max(np.abs(ref-aligned.score)))
            if error>1e-12: raise ValueError(f'Native outer parity {fold}: {error}')
            result.append({'test':'saved_outer_OOF_parity','fold':fold,'max_error':error,'passed':True})
        else:
            for chunk in [1,7,128]:
                diff=float(np.max(np.abs(ref-bundle_scores(frame,directory,chunk))))
                if diff>1e-12: raise ValueError('Chunk parity')
                result.append({'test':'raw_chunk_then_global_rank','chunk':chunk,'max_error':diff,'passed':True})
            order=np.random.default_rng(41).permutation(len(frame))
            shuffled=bundle_scores(frame.iloc[order],directory)
            if not np.allclose(shuffled,ref[order],atol=1e-12,rtol=0): raise ValueError('Row order parity')
            reversed_cols=bundle_scores(frame[frame.columns[::-1]],directory)
            if not np.allclose(reversed_cols,ref,atol=1e-12,rtol=0): raise ValueError('Column order parity')
            result.extend([{'test':'row_order','passed':True},{'test':'feature_column_order','passed':True}])
            rg=frame[frame.machine.eq('rg3')].iloc[:1]
            singleton=float(bundle_scores(rg,directory)[0]);dup=bundle_scores(pd.concat([rg,rg]),directory)
            if abs(singleton-1)>1e-12 or abs(dup[0]-dup[1])>1e-12: raise ValueError('Singleton/duplicate contract')
            result.append({'test':'singleton_duplicate','passed':True,'singleton_score':singleton,'not_probability':True})
            for label in ['nan','inf','unknown_machine','missing_feature']:
                bad=frame.iloc[:2].copy()
                if label in ['nan','inf']: bad.loc[bad.index[0],FEATURES[0]]=np.nan if label=='nan' else np.inf
                elif label=='unknown_machine': bad.loc[bad.index[0],'machine']='unknown'
                else: bad=bad.drop(columns=[FEATURES[0]])
                try: bundle_scores(bad,directory)
                except ValueError: result.append({'test':label,'passed':True})
                else: raise ValueError(f'Invalid input accepted: {label}')
            with tempfile.TemporaryDirectory(dir=out) as td:
                copy=Path(td)/'bundle';shutil.copytree(directory,copy)
                with (copy/files[0]).open('ab') as stream:stream.write(b'corruption')
                try: bundle_scores(frame.iloc[:2],copy)
                except ValueError:result.append({'test':'corrupt_native_file','passed':True})
                else:raise ValueError('Corrupted bundle accepted')
    dump(out/'policy_parity_tests.json',{'status':'passed','tests':result,'tolerance':1e-12})


def evaluate(out):
    out=Path(out)
    if not (out/'training_complete.json').exists(): raise ValueError('Training incomplete')
    data,_,_=cd.load_development(Path('data/task01_official'),Path('outputs/gpu_pc_split_20261004_03'))
    raw=pd.read_csv(out/'outer_oof_predictions.csv.gz',float_precision='round_trip')
    table=ensemble(raw)
    historical=pd.read_csv(out/'baseline_predictions.csv.gz',float_precision='round_trip')
    table=pd.concat([table,historical],ignore_index=True)
    table.to_csv(out/'evaluated_predictions.csv.gz',index=False)
    rows=[];tie=[];slices=[];ranks=[]
    collision=pd.read_csv(out/'collision_audit.csv').set_index(['machine','feature_group'])
    for (pipeline,seed),part in table.groupby(['pipeline','seed'],sort=False):
        part=part.reset_index(drop=True);score=part.score.to_numpy()
        for fraction in [.05,.075,.1]:
            for contract in ['historical_pooled','batch_allocated']:
                decision=choose(part,score,fraction,contract=='batch_allocated')
                for machine in ['all','cn7','rg3']:
                    mask=np.ones(len(part),bool) if machine=='all' else part.machine.eq(machine).to_numpy()
                    row={'pipeline':pipeline,'seed':seed,'contract':contract,'fraction':fraction,'machine':machine,
                         **counts(part.loc[mask,'label_value'],decision[mask],score[mask])}
                    rows.append(row)
                if fraction==.075:
                    tie.extend({'pipeline':pipeline,'seed':seed,'contract':contract,**r} for r in tie_bounds(part,score,fraction,contract=='batch_allocated'))
                    if str(seed)=='ensemble':
                        c=part.copy();c['selected']=decision
                        c['kind']=[collision.loc[(m,g),'kind'] for m,g in zip(c.machine,c.feature_group)]
                        for (m,kind),p in c.groupby(['machine','kind']):
                            slices.append({'pipeline':pipeline,'contract':contract,'machine':m,'kind':kind,**counts(p.label_value,p.selected,p.score)})
                        if contract=='batch_allocated':
                            c['rank_in_batch']=c.groupby(['machine','batch']).score.rank(method='average',ascending=False)
                            ranks.append(c[c.machine.eq('rg3') & c.label_value.eq(1)])
    metrics=pd.DataFrame(rows);metrics.to_csv(out/'metrics_by_machine_seed_budget.csv',index=False)
    means=metrics[metrics.seed.astype(str).eq('ensemble')];means.to_csv(out/'ensemble_metrics.csv',index=False)
    macro=means[means.machine.isin(['cn7','rg3'])].groupby(['pipeline','contract','fraction']).ap.mean().reset_index(name='macro_ap')
    macro.to_csv(out/'macro_ap.csv',index=False)
    probabilities=[]
    for (name,seed,machine),p in table[table.pipeline.isin(['historical_cpu','pair_a1'])].groupby(['pipeline','seed','machine']):
        if name=='pair_a1' and machine=='rg3':continue
        probabilities.append({'component':name,'seed':seed,'machine':machine,'scope':'uncalibrated probability output, numeric row label',
            'brier':brier_score_loss(p.label_value,p.score),'log_loss':log_loss(p.label_value,p.score,labels=[0,1])})
    native=[]
    for path in out.glob('outer_raw_*_cat_row_*.csv.gz'):native.append(pd.read_csv(path,float_precision='round_trip'))
    for (name,seed),p in pd.concat(native,ignore_index=True).groupby(['component','seed']):
        probabilities.append({'component':name,'seed':seed,'machine':'rg3','scope':'weighted classifier, not calibrated',
            'brier':brier_score_loss(p.label_value,p.raw_score),'log_loss':log_loss(p.label_value,p.raw_score,labels=[0,1])})
    pd.DataFrame(probabilities).to_csv(out/'probability_diagnostics.csv',index=False)
    # Fixed predeclared budgets, descriptive only: no outer-based policy choice.
    means.to_csv(out/'budget_pareto.csv',index=False)
    pd.DataFrame(tie).to_csv(out/'tie_sensitivity.csv',index=False)
    pd.DataFrame(tie).groupby(['pipeline','seed','contract'])[['tp_min','tp_expected','tp_max']].sum().reset_index().to_csv(out/'tie_summary.csv',index=False)
    pd.DataFrame(slices).to_csv(out/'failure_slices.csv',index=False)
    pd.concat(ranks,ignore_index=True).to_csv(out/'rg3_positive_rank_audit.csv',index=False)
    overlap=[]
    e=table[table.seed.astype(str).eq('ensemble')]
    b=e[e.pipeline.eq('pair_a1')].reset_index(drop=True)
    selected=set(zip(b.loc[choose(b,b.score),'machine'],b.loc[choose(b,b.score),'source_row_id']))
    for name,p in e.groupby('pipeline'):
        p=p.reset_index(drop=True);d=choose(p,p.score); other=set(zip(p.loc[d,'machine'],p.loc[d,'source_row_id']))
        overlap.append({'pipeline':name,'intersection':len(selected&other),'union':len(selected|other),'jaccard':len(selected&other)/len(selected|other)})
    pd.DataFrame(overlap).to_csv(out/'inspection_overlap.csv',index=False)
    sensitivity=pd.read_csv(out/'sensitivity_predictions.csv.gz')
    pd.DataFrame([{'pipeline':n,**counts(p.label_value,choose(p,p.score),p.score)} for n,p in sensitivity.groupby('pipeline')]).to_csv(out/'group_split_sensitivity.csv',index=False)
    bundles(out,data,table)
    ci=bootstrap(table,out)
    fixed=means[means.contract.eq('batch_allocated')&means.fraction.eq(.075)&means.machine.eq('all')].set_index('pipeline')
    delta=int(fixed.loc['nested_selected','tp']-fixed.loc['pair_a1','tp'])
    status='RESEARCH_CANDIDATE' if delta>0 else 'NO_ROBUST_GAIN'
    dump(out/'decision.json',{'status':status,'primary_procedure':'nested_selected','comparison':'fresh pair_a1 under same batch contract',
        'delta_tp':delta,'delta_fp':-delta,'conditional_delta_tp_ci':ci['delta_tp'],
        'field_approved':False,'old_bundle_preserved':True,'independent_validation':False,
        'adopted_for_production':False,'full_bundle_candidate':json.loads((out/'bundle_sources.json').read_text(encoding='utf-8'))['-1']['pipeline']})
    print(fixed[['tp','fp','precision','recall','accuracy']].to_string())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True)
    with threadpool_limits(limits=2):evaluate(p.parse_args().output_dir)
