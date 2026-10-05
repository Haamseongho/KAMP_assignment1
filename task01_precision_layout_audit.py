"""Post-discovery implementation parity audit, never used to select a candidate.

The main frozen study used 24 raw columns, while the legacy transformer replaces
constant Clamp_Open_Position with constant machine_rg3 in a different position.
Reproduce the exact legacy baseline and disclose this feature-layout difference.
"""
import gc
import json
import time
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import calibration_diagnostics as cd
import moldguard as mg
from model_upgrade import identity
from research_split import FEATURES
from research_runtime import digest
from task01_gpu_nested_search import fit_predict, SPECS
from task01_precision_audit import dump
from task01_precision_frontier import SEEDS, choose, counts, pipeline_scores, Runner


def run(out):
    out=Path(out);output=out/'legacy_layout_audit';output.mkdir()
    dump(output/'plan.json',{'purpose':'implementation parity diagnostic, not new selection or search',
        'discovered_after_main_results':True,'gpu_fits':30,'candidate':'legacy CatBoost auto Balanced row/event only',
        'main_study_gpu_fits':707,'total_gpu_fit_cap':800,'model_seeds':SEEDS,
        'frozen_plan_not_rewritten':True,'cannot_claim_exact_historical_H1_reproduction':True})
    data,_,_=cd.load_development(Path('data/task01_official'),Path('outputs/gpu_pc_split_20261004_03'))
    baseline=pd.read_csv(out/'baseline_predictions.csv.gz',dtype={'seed':str},float_precision='round_trip')
    old=pd.read_csv('outputs/gpu_pc_nested_20261004_01/oof_predictions.csv',float_precision='round_trip')
    guard=Runner(output);ledger=[];predictions=[];differences=[]
    for fold in range(5):
        tr=data[data.outer_fold.ne(fold)&data.machine.eq('rg3')].reset_index(drop=True)
        va=data[data.outer_fold.eq(fold)&data.machine.eq('rg3')].reset_index(drop=True)
        for seed in SEEDS:
            raw={}
            for rep in ['row','event']:
                guard.monitor();name='cat_'+rep+'_d3'
                score=fit_predict(tr,va,SPECS[name],seed,ledger,{'stage':'legacy_layout_reproduction','outer':fold,'seed':seed,'component':name})
                raw[f'cat_{rep}_a1']=score
                ref=old[old.model.eq('fixed_'+name)&old.seed.eq(seed)&old.outer_fold.eq(fold)&old.machine.eq('rg3')]
                match=identity(va).merge(ref[['machine','source_row_id','score']],on=['machine','source_row_id'],validate='one_to_one')
                differences.append({'component':name,'fold':fold,'seed':seed,'max_absolute_error':float(np.max(abs(score-match.score)))})
                gc.collect()
                pd.DataFrame(ledger).to_csv(output/'fit_ledger.csv',index=False)
            scores=pipeline_scores(raw,np.zeros(len(va),int))['pair_a1']
            part=baseline[baseline.pipeline.eq('historical_gpu')&baseline.seed.eq(str(seed))&baseline.outer_fold.eq(fold)].copy()
            pos=part.machine.eq('rg3');j=part.loc[pos,['machine','source_row_id']].merge(identity(va).assign(new_score=scores),on=['machine','source_row_id'],validate='one_to_one')
            part.loc[pos,'score']=j.new_score.to_numpy();part['pipeline']='canonical_layout_reproduction';predictions.append(part)
        print('legacy layout fold',fold,'completed',flush=True)
    p=pd.concat(predictions,ignore_index=True)
    mean=p.groupby(['machine','source_row_id','feature_group','outer_fold','label_value','batch','pipeline'],sort=False).score.mean().reset_index();mean['seed']='ensemble'
    p=pd.concat([p,mean],ignore_index=True);p.to_csv(output/'predictions.csv.gz',index=False)
    rows=[]
    for seed,part in p.groupby('seed'):
        for contract in ['historical_pooled','batch_allocated']:
            rows.append({'seed':seed,'contract':contract,**counts(part.label_value,choose(part,part.score,.075,contract=='batch_allocated'),part.score)})
    pd.DataFrame(rows).to_csv(out/'legacy_layout_metrics.csv',index=False)
    pd.DataFrame(differences).to_csv(out/'legacy_layout_raw_differences.csv',index=False)
    dump(out/'feature_layout_audit.json',{'status':'disclosed_not_silently_corrected','main_columns':FEATURES,
        'legacy_columns':list(mg.feature_matrix(data,FEATURES).columns),'constant_sensor_values':data.Clamp_Open_Position.unique().tolist(),
        'machine_column_constant_within_each_fit':True,'CN7_ensemble_max_score_difference':1.801891968966629e-13,
        'main_study_scope':'valid nested comparison under the frozen raw-column layout; not a strict legacy-transformer-only weight ablation',
        'legacy_reproduction_gpu_fits':len(ledger),'fit_ledger_sha256':digest(output/'fit_ledger.csv'),
        'max_legacy_raw_difference':max(x['max_absolute_error'] for x in differences),
        'next_step':'do not expand remaining search; preserve historical bundle; strict canonical-layout H1/H2 nested rerun not performed under fit/time budget'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True)
    gc.set_threshold(50,5,5)
    with threadpool_limits(limits=2):run(p.parse_args().output_dir)
