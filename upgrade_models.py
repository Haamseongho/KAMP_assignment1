"""Small CPU-only candidate family and non-pickle research bundle runtime."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import moldguard as mg
from research_split import FEATURES
from research_runtime import digest, write_json

INTERACTIONS = ['Clamp_Close_Time', 'Max_Injection_Pressure', 'Max_Switch_Over_Pressure']


def candidates():
    specs = {'baseline': {'kind': 'logistic', 'c': .1, 'balanced': True}}
    for c in (.01, .1, 1.):
        for balanced in (False, True):
            if c == .1 and balanced:
                continue
            specs[f'log_c{c:g}_{"balanced" if balanced else "none"}'] = {
                'kind': 'logistic', 'c': c, 'balanced': balanced}
    specs.update({
        'log_interactions': {'kind': 'logistic', 'c': .1, 'balanced': True, 'map': 'interactions'},
        'log_group_weight': {'kind': 'logistic', 'c': .1, 'balanced': True, 'group_weight': True},
        'log_nonlinear': {'kind': 'logistic', 'c': .1, 'balanced': True, 'map': 'nonlinear'},
        'rg3_logistic': {'kind': 'logistic', 'c': .1, 'balanced': True, 'rg3_only': True},
        'lgbm_regularized': {'kind': 'lgbm_regularized'},
        'xgb_regularized': {'kind': 'xgb_regularized'},
        'xgb_shallow': {'kind': 'xgb_shallow'},
        'rg3_lgbm': {'kind': 'lgbm_regularized', 'rg3_only': True},
        'rg3_xgb': {'kind': 'xgb_regularized', 'rg3_only': True},
    })
    return specs


def check_input(frame):
    if frame.columns.duplicated().any() or not {'machine', *FEATURES}.issubset(frame):
        raise ValueError('Missing or duplicate feature columns')
    if not frame.machine.isin(['cn7', 'rg3']).all():
        raise ValueError('Unsupported machine')
    # Numeric strings are not silently coerced; schema errors must be explicit.
    if any(not pd.api.types.is_numeric_dtype(frame[c]) for c in FEATURES):
        raise ValueError('Features must be numeric')
    if not np.isfinite(frame[FEATURES].to_numpy(dtype=float)).all():
        raise ValueError('Features must be finite')


def matrix(frame, spec):
    check_input(frame)
    x = mg.feature_matrix(frame, FEATURES)
    if spec.get('map') == 'interactions':
        for col in INTERACTIONS:
            x[f'rg3_x_{col}'] = x.machine_rg3 * x[col]
    elif spec.get('map') == 'nonlinear':
        for col in INTERACTIONS:
            x[f'square_{col}'] = x[col] ** 2
        x['pressure_product'] = x.Max_Injection_Pressure * x.Max_Switch_Over_Pressure
    if not np.isfinite(x.to_numpy()).all():
        raise ValueError('Feature transformation overflow')
    return x


def fit_model(frame, spec, seed):
    fit = frame.loc[frame.machine.eq('rg3')] if spec.get('rg3_only') else frame
    y = fit[mg.TARGET].to_numpy(dtype=int)
    if set(y) != {0, 1}:
        raise ValueError('Training requires both classes')
    x = matrix(fit, spec)
    if spec['kind'] == 'logistic':
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=spec['c'], class_weight='balanced' if spec['balanced'] else None,
            max_iter=2000, random_state=mg.SEED))
        kwargs = {}
        if spec.get('group_weight'):
            w = 1 / fit.groupby('feature_group').feature_group.transform('size').to_numpy()
            w /= w.mean()
            kwargs['logisticregression__sample_weight'] = w
        model.fit(x, y, **kwargs)
        if np.any(model[-1].n_iter_ >= model[-1].max_iter):
            raise RuntimeError('Logistic did not converge')
    else:
        from task01_compare import make_candidate
        model = make_candidate(spec['kind'], seed, 'cpu', y)
        model.set_params(n_jobs=1)
        model.fit(x, y)
    return model


def predict(model, frame, spec, baseline=None):
    if spec.get('rg3_only'):
        if baseline is None:
            raise ValueError('RG3-only path needs fold-local CN7 reference')
        result = np.array(baseline, copy=True)
        mask = frame.machine.eq('rg3').to_numpy()
        if mask.any():
            result[mask] = model.predict_proba(matrix(frame.loc[mask], spec))[:, 1]
    else:
        result = model.predict_proba(matrix(frame, spec))[:, 1]
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite predictions')
    return result


def export_component(model, spec, path):
    path = Path(path)
    path.mkdir()
    meta = {'spec': spec, 'features': FEATURES}
    if spec['kind'] == 'logistic':
        scaler, estimator = model.steps[0][1], model.steps[1][1]
        write_json(path / 'weights.json', {
            'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(),
            'coef': estimator.coef_[0].tolist(), 'intercept': float(estimator.intercept_[0])})
        name = 'weights.json'
    elif spec['kind'].startswith('lgbm'):
        name = 'model.txt'
        model.booster_.save_model(str(path / name))
    else:
        name = 'model.json'
        model.save_model(path / name)
    meta.update(file=name, sha256=digest(path / name))
    write_json(path / 'component.json', meta)


class ResearchBundle:
    """Loads only locally generated JSON/native tree files, never pickle objects."""
    def __init__(self, path):
        self.path = Path(path)
        self.meta = json.loads((self.path / 'bundle.json').read_text())
        if self.meta.get('schema') != 'kamp-upgrade-research-v1' or self.meta.get('field_approved') is not False:
            raise ValueError('Unsupported research bundle')
        self.hash = digest(self.path / 'bundle.json')
        for name, expected in self.meta['files'].items():
            p = Path(name)
            if p.is_absolute() or '..' in p.parts or digest(self.path / p) != expected:
                raise ValueError('Bundle artifact hash mismatch')
        self.components = {}
        for machine in ('cn7', 'rg3'):
            folder = self.path / machine
            meta = json.loads((folder / 'component.json').read_text())
            name = meta['file']
            if Path(name).name != name or digest(folder / name) != meta['sha256'] or meta['features'] != FEATURES:
                raise ValueError('Component schema/hash mismatch')
            if meta['spec']['kind'] == 'logistic':
                model = json.loads((folder / name).read_text())
            elif meta['spec']['kind'].startswith('lgbm'):
                from lightgbm import Booster
                model = Booster(model_file=str(folder / name))
            else:
                from xgboost import XGBClassifier
                model = XGBClassifier(n_jobs=1, device='cpu')
                model.load_model(folder / name)
                model.set_params(n_jobs=1, device='cpu')
            self.components[machine] = (meta['spec'], model)

    def predict(self, frame):
        check_input(frame)
        result = np.empty(len(frame), dtype=float)
        for machine, (spec, model) in self.components.items():
            mask = frame.machine.eq(machine).to_numpy()
            if not mask.any():
                continue
            x = matrix(frame.loc[mask], spec)
            if spec['kind'] == 'logistic':
                z = ((x.to_numpy() - model['mean']) / model['scale']) @ np.array(model['coef']) + model['intercept']
                from scipy.special import expit
                score = expit(z)
            elif spec['kind'].startswith('lgbm'):
                score = model.predict(x, num_threads=1)
            else:
                score = model.predict_proba(x)[:, 1]
            result[mask] = score
        if not np.isfinite(result).all() or np.any((result < 0) | (result > 1)):
            raise ValueError('Invalid scores')
        return result
