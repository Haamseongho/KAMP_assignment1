"""Native, research-only inference under the complete-batch ranking contract."""
import json
from pathlib import Path
import numpy as np
from catboost import CatBoostClassifier
from scipy.special import expit
from scipy.stats import rankdata
from xgboost import XGBRanker
from research_runtime import digest
from research_split import FEATURES


def scores(frame, bundle, chunk_size=None):
    bundle=Path(bundle)
    manifest=json.loads((bundle/'bundle.json').read_text(encoding='utf-8'))
    if manifest.get('contract')!='complete_batch_rank_v1' or manifest.get('field_approved') is not False:
        raise ValueError('Unsupported bundle contract')
    for name,expected in manifest['files_sha256'].items():
        if Path(name).name!=name or digest(bundle/name)!=expected: raise ValueError('Corrupt bundle')
    required={manifest['cn7_file'],*(r['file'] for r in manifest['models'])}
    if not required<=set(manifest['files_sha256']): raise ValueError('Unhashed model')
    if not set(FEATURES)<=set(frame.columns): raise ValueError('Missing features')
    if not frame.machine.isin(['cn7','rg3']).all(): raise ValueError('Unsupported machine')
    x=frame[FEATURES].to_numpy(float)
    if not np.isfinite(x).all(): raise ValueError('Nonfinite features')
    if chunk_size is None: chunk_size=max(1,len(frame))
    if not isinstance(chunk_size,int) or chunk_size<=0: raise ValueError('Invalid chunk size')
    output=np.zeros(len(frame),float)
    cn=frame.machine.eq('cn7').to_numpy(); rg=~cn
    w=json.loads((bundle/manifest['cn7_file']).read_text(encoding='utf-8'))
    if cn.any(): output[cn]=expit(((x[cn]-w['mean'])/w['scale'])@np.array(w['coef'])+w['intercept'])
    if rg.any():
        rgframe=frame.loc[rg,FEATURES]
        # Gather all raw scores first. Ranking individual chunks violates this API.
        for r in manifest['models']:
            ranker=r['component'].startswith('rank_')
            model=XGBRanker() if ranker else CatBoostClassifier()
            model.load_model(str(bundle/r['file']))
            raw=[]
            for start in range(0,len(rgframe),chunk_size):
                local=rgframe.iloc[start:start+chunk_size]
                raw.append(model.predict(local) if ranker else model.predict_proba(local)[:,1])
            raw=np.concatenate(raw)
            output[rg]+=r['weight']*rankdata(raw,method='average')/len(raw)
    if not np.isfinite(output).all(): raise ValueError('Invalid output')
    return output
