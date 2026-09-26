"""Portable, non-pickle logistic inference. Never changes inspection policy."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

SCHEMA = 'moldguard-portable-logistic-v1'
MAX_ROWS = 100000


def load_model(path, expected_sha256):
    payload = Path(path).read_bytes()
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError('Trusted model SHA-256 is required')
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError('Model SHA-256 mismatch')
    model = json.loads(payload)
    validate_model(model)
    return model


def validate_model(model):
    if model.get('schema_version') != SCHEMA or model.get('classes') != [0, 1]:
        raise ValueError('Unsupported model or label classes')
    features = model['features']
    columns = model['matrix_columns']
    if len(features) != 24 or len(set(features)) != 24:
        raise ValueError('Expected 24 distinct original features')
    if columns != [c for c in features if c != 'Clamp_Open_Position'] + ['machine_rg3']:
        raise ValueError('Unexpected matrix layout')
    for key in ('mean', 'scale', 'coef'):
        values = np.asarray(model[key], dtype=float)
        if values.shape != (len(columns),) or not np.isfinite(values).all():
            raise ValueError('Invalid model parameter: ' + key)
    if (np.asarray(model['scale']) <= 0).any() or not np.isfinite(float(model['intercept'])):
        raise ValueError('Invalid scale/intercept')
    if model.get('mode') != 'RESEARCH_ONLY' or model.get('score_target') != 'numeric_label_1_unconfirmed':
        raise ValueError('Operational activation is not supported')
    if set(model['ood_bounds']) != {'cn7', 'rg3'}:
        raise ValueError('Missing machine bounds')
    for bounds in model['ood_bounds'].values():
        for key in ('low', 'high'):
            if set(bounds[key]) != set(features) or not np.isfinite(list(bounds[key].values())).all():
                raise ValueError('Invalid OOD bounds')
        if any(bounds['low'][c] > bounds['high'][c] for c in features):
            raise ValueError('Inverted OOD bounds')


def validate_input(frame, model):
    if not isinstance(frame, pd.DataFrame) or len(frame) > MAX_ROWS:
        raise ValueError('Expected DataFrame with at most 100000 rows')
    expected = ['machine', 'source_row_id', *model['features']]
    if frame.columns.duplicated().any() or set(frame.columns) != set(expected):
        raise ValueError('Exact named schema required; labels/approval flags/extra columns forbidden')
    if not frame.machine.isin(['cn7', 'rg3']).all():
        raise ValueError('Unknown machine')
    for column in ['source_row_id', *model['features']]:
        if not pd.api.types.is_numeric_dtype(frame[column]) or pd.api.types.is_bool_dtype(frame[column]):
            raise ValueError('Non-numeric/bool input: ' + column)
    numbers = frame[['source_row_id', *model['features']]].to_numpy(dtype=float)
    if not np.isfinite(numbers).all():
        raise ValueError('Missing or non-finite input; no score is produced')
    ids = frame.source_row_id.to_numpy(dtype=float)
    if ((ids < 0) | (ids > 2**53-1) | (ids != np.floor(ids))).any():
        raise ValueError('Source ID must be an exact nonnegative integer')
    if frame.duplicated(['machine', 'source_row_id']).any():
        raise ValueError('Duplicate source ID within machine')


def infer(frame, model):
    validate_model(model)
    validate_input(frame, model)
    if frame.empty:
        return []
    matrix = frame[[c for c in model['features'] if c != 'Clamp_Open_Position']].copy()
    matrix['machine_rg3'] = frame.machine.eq('rg3').astype(int)
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        try:
            x = (matrix.to_numpy(dtype=float) - np.asarray(model['mean'])) / np.asarray(model['scale'])
            z = x @ np.asarray(model['coef']) + model['intercept']
        except FloatingPointError as error:
            raise ValueError('Input causes numeric overflow; no score produced') from error
    if not np.isfinite(z).all():
        raise ValueError('Invalid model output')
    scores = np.empty(len(z))
    positive = z >= 0
    scores[positive] = 1 / (1 + np.exp(-z[positive]))
    exp = np.exp(z[~positive])
    scores[~positive] = exp / (1 + exp)
    counts = np.zeros(len(frame), dtype=int)
    for machine, bounds in model['ood_bounds'].items():
        mask = frame.machine.eq(machine).to_numpy()
        values = frame.loc[mask, model['features']]
        counts[mask] = (values.lt(pd.Series(bounds['low'])) | values.gt(pd.Series(bounds['high']))).sum(axis=1)
    rows = []
    for i, (machine, source_id) in enumerate(frame[['machine', 'source_row_id']].itertuples(index=False, name=None)):
        ood = bool(counts[i] >= 3)
        action = ('usual_inspection_unvalidated_machine' if machine == 'rg3' else
                  'usual_inspection_ood' if ood else 'usual_inspection_label_unconfirmed')
        reasons = ['research_only', 'label_semantics_unconfirmed', 'field_approval_unavailable',
                   'prediction_time_unconfirmed', 'physical_entity_mapping_unconfirmed',
                   'raw_units_and_supplier_transform_unconfirmed']
        if machine == 'rg3':
            reasons.append('unvalidated_machine')
        if ood:
            reasons.append('out_of_training_range')
        rows.append({'machine': machine, 'source_row_id': int(source_id),
                     'research_score': float(scores[i]), 'score_target': model['score_target'],
                     'defect_probability': None, 'operational_priority': None,
                     'research_rank': None, 'rank_scope': 'not_computed_for_request_batch',
                     'ood_feature_count': int(counts[i]), 'ood_flag': ood,
                     'action': action, 'fallback_reasons': reasons,
                     'mode': 'RESEARCH_ONLY', 'existing_queue_policy': 'unchanged',
                     'product_id': None, 'prediction_cutoff_at': None})
    return rows
