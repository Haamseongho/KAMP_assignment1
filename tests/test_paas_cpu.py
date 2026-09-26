"""TEST_ONLY arithmetic/policy fixtures; never manufacturing performance data."""
from datetime import datetime, timezone
import hashlib
import json
import numpy as np
import pandas as pd
import pytest
from paas_cpu.model_runtime import infer, load_model
from paas_cpu.cost_gate import validate, OPERATIONS
from paas_cpu.fusion_gate import assess
from paas_cpu.inference_service import init_model, inference_dataframe, inference_file

pytestmark = pytest.mark.unit


@pytest.fixture
def model():
    features = ['Clamp_Open_Position'] + ['TEST_ONLY_' + str(i) for i in range(23)]
    return {'schema_version': 'moldguard-portable-logistic-v1', 'classes': [0, 1],
            'features': features, 'matrix_columns': features[1:] + ['machine_rg3'],
            'mean': [0.] * 24, 'scale': [1.] * 24, 'coef': [1.] + [0.] * 23,
            'intercept': 0., 'mode': 'RESEARCH_ONLY', 'score_target': 'numeric_label_1_unconfirmed',
            'ood_bounds': {m: {key: {f: v for f in features} for key, v in [('low', -1.), ('high', 1.)]}
                           for m in ('cn7', 'rg3')}}


@pytest.fixture
def frame(model):
    return pd.DataFrame([{'machine': 'cn7', 'source_row_id': 1, **{f: 0. for f in model['features']}}])


def test_paas_callback_hash_pin_and_json_serialization(tmp_path, monkeypatch, model, frame):
    payload = json.dumps(model).encode()
    (tmp_path / 'model.json').write_bytes(payload)
    monkeypatch.setenv('MOLDGUARD_MODEL_DIR', str(tmp_path))
    monkeypatch.setenv('MOLDGUARD_MODEL_SHA256', hashlib.sha256(payload).hexdigest())
    result = inference_dataframe(frame, init_model())
    assert result[0]['research_score'] == .5
    assert result[0]['defect_probability'] is None and result[0]['operational_priority'] is None
    assert result[0]['research_rank'] is None and result[0]['existing_queue_policy'] == 'unchanged'
    json.dumps(result, allow_nan=False)
    with pytest.raises(ValueError):
        load_model(tmp_path / 'model.json', '0' * 64)
    monkeypatch.delenv('MOLDGUARD_MODEL_SHA256')
    with pytest.raises(ValueError):
        init_model()
    with pytest.raises(ValueError):
        inference_file([], {'model': model})


@pytest.mark.parametrize('kind', ['missing', 'extra', 'nan', 'inf', 'unknown_machine', 'duplicate', 'bool', 'str', 'bad_id', 'column_duplicate', 'overflow'])
def test_paas_bad_input_never_scores_zero(model, frame, kind):
    if kind == 'missing': frame = frame.drop(columns=['TEST_ONLY_0'])
    elif kind == 'extra': frame['PassOrFail'] = 1
    elif kind == 'nan': frame.loc[0, 'TEST_ONLY_0'] = np.nan
    elif kind == 'inf': frame.loc[0, 'TEST_ONLY_0'] = np.inf
    elif kind == 'unknown_machine': frame.loc[0, 'machine'] = 'other'
    elif kind == 'duplicate': frame = pd.concat([frame, frame])
    elif kind == 'bool': frame['TEST_ONLY_0'] = True
    elif kind == 'str': frame['TEST_ONLY_0'] = '0.0'
    elif kind == 'bad_id': frame['source_row_id'] = .5
    elif kind == 'column_duplicate': frame = pd.concat([frame, frame[['TEST_ONLY_0']]], axis=1)
    elif kind == 'overflow':
        frame['TEST_ONLY_0'] = 1e308
        model['scale'][0] = 1e-308
    with pytest.raises(ValueError):
        infer(frame, model)


def test_paas_schema_order_and_empty_and_ood(model, frame):
    assert infer(frame.iloc[:0], model) == []
    assert infer(frame, model) == infer(frame[frame.columns[::-1]], model)
    for f in model['features'][:3]: frame[f] = 2.
    r = infer(frame, model)[0]
    assert r['action'] == 'usual_inspection_ood'
    frame['machine'] = 'rg3'
    r = infer(frame, model)[0]
    assert r['action'] == 'usual_inspection_unvalidated_machine'
    assert 'out_of_training_range' in r['fallback_reasons']


def test_cost_gate_rejects_missing_paid_expired_or_changed_evidence(tmp_path):
    with pytest.raises(ValueError, match='REMOTE_BLOCKED'):
        validate({})
    p = tmp_path / 'TEST_ONLY.txt'
    p.write_text('TEST_ONLY cost-gate fixture, NOT provider confirmation')
    receipt = {'project_id': '215209147869625880', 'device': 'cpu', 'status': 'verified_zero_incremental_cost',
               'incremental_cost_cap': 0, 'gpu_allowed': False, 'paid_fallback_allowed': False,
               'reviewed_by': 'TEST_ONLY', 'provider_reference': 'TEST_ONLY',
               'verified_at': '2026-01-01T00:00:00+00:00', 'expires_at': '2026-01-02T00:00:00+00:00',
               'evidence_file': p.name, 'evidence_sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
               'operations': {k: {'no_additional_charge_confirmed': True, 'limit_and_billing_basis': 'TEST_ONLY'} for k in OPERATIONS}}
    now = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
    assert validate(receipt, tmp_path, now)['status'] == 'evidence_gate_passed'
    for changes in ({'device': 'gpu'}, {'incremental_cost_cap': 1}, {'incremental_cost_cap': False}, {'project_id': 'another'},
                    {'operations': {}}, {'expires_at': '2025-01-01T00:00:00+00:00'}, {'evidence_sha256': '0'*64}):
        with pytest.raises(ValueError):
            validate({**receipt, **changes}, tmp_path, now)


def test_fusion_cannot_count_same_dataset_twice():
    result = assess({'primary': {'dataset_id': 'same'}, 'auxiliary': {'dataset_id': 'same'}})
    assert result['status'] == 'blocked'
    assert 'second_distinct_dataset_not_verified' in result['blockers']
    assert result['claim_of_improvement_allowed'] is False


def test_general_student_does_not_require_fusion():
    plan = {'competition_track': 'general_student', 'track_evidence': 'TEST_ONLY confirmed notice'}
    assert assess(plan)['status'] == 'not_required_for_confirmed_track'
    assert assess({**plan, 'optional_fusion_requested': True})['status'] == 'blocked'
    assert assess({**plan, 'track_evidence': None})['status'] == 'blocked'
