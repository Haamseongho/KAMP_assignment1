"""TEST_ONLY policy/evidence objects; never used as manufacturing performance."""
import json
from pathlib import Path
import pytest
from moldguard_service.contracts import ContractRegistry, operational_observation
from moldguard_service.policy import decide, validate_policy, acceptance_blockers

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def policy():
    return json.loads((ROOT / 'config/service_policy.json').read_text())


@pytest.mark.parametrize('machine,ood,action', [('cn7', False, 'label_unconfirmed'),
    ('rg3', False, 'unvalidated_machine'), ('cn7', True, 'ood'), ('rg3', True, 'unvalidated_machine')])
def test_T01_T04_legacy_routes_and_all_reasons(tmp_path, policy, machine, ood, action):
    contracts = ContractRegistry(tmp_path).view({'TEST_ONLY': 'a' * 64})
    result = decide({'machine': machine, 'ood_flag': ood, 'risk_score': .8, 'priority': 'HIGH'}, contracts, policy)
    assert result['action'] == 'usual_inspection_' + action
    assert result['research_score'] == .8
    assert result['defect_probability'] is None and result['operational_priority'] is None
    assert 'label_semantics_unconfirmed' in result['fallback_reasons']
    assert 'field_approval_unavailable' in result['fallback_reasons']
    if ood:
        assert 'out_of_training_range' in result['fallback_reasons']
    assert result['existing_queue_policy'] == 'unchanged'


@pytest.mark.parametrize('value', [None, float('nan'), float('inf'), -.1, 1.1, '0.8', True])
def test_T05_T07_invalid_predictions_are_null_not_zero(tmp_path, policy, value):
    result = decide({'machine': 'cn7', 'ood_flag': False, 'risk_score': value}, ContractRegistry(tmp_path).view({}), policy)
    assert result['research_score'] is None
    assert result['prediction_status'] == 'not_produced'
    assert result['action'] == 'usual_inspection_input_error'


def test_T10_T11_T34_approvals_warnings_never_escalate(tmp_path, policy):
    c = ContractRegistry(tmp_path).view({})
    r = {'machine': 'cn7', 'ood_flag': False, 'risk_score': .8}
    for mode in ('RESEARCH_ONLY', 'SHADOW', 'ADVISORY_APPROVED'):
        for approval in ({'status': 'approved'}, {'revoked': True}, {'expires_at': '2000-01-01'}, {'scope': 'another_source'}):
            decision = decide(r, c, policy, mode=mode, approval=approval, additional_warnings=['stale_input'])
            assert decision['operational_recommendation'] is False
            assert decision['operational_priority'] is None
            assert 'stale_input' in decision['fallback_reasons']


def test_T35_T36_policy_and_missing_acceptance_block_activation(policy):
    criteria = json.loads((ROOT / 'config/acceptance_criteria.example.json').read_text())
    assert 'machine_scopes.CN7.min_positive_cases' in acceptance_blockers(criteria)
    for key in ['allow_inspection_skip', 'allow_machine_control', 'field_deployment_approved']:
        with pytest.raises(ValueError):
            validate_policy({**policy, key: True})


def test_contract_evidence_version_scope_and_tamper(tmp_path):
    registry = ContractRegistry(tmp_path / 'registry')
    sources = {'TEST_ONLY': 'a' * 64}
    with pytest.raises(ValueError, match='evidence'):
        registry.record('label_semantics', 'confirmed', sources, 'test reviewer', 'test check')
    evidence = tmp_path / 'test_evidence.txt'
    evidence.write_text('TEST_ONLY evidence for validator arithmetic, not official approval')
    first = registry.record('label_semantics', 'confirmed', sources, 'test reviewer', 'test check', evidence)
    assert registry.view(sources)['checks'][0]['status'] == 'confirmed'
    assert registry.view({'TEST_ONLY': 'b' * 64})['checks'][0]['status'] == 'unconfirmed'
    evidence.write_text('TEST_ONLY changed')
    assert registry.view(sources)['checks'][0]['status'] == 'mismatch'
    second = registry.record('label_semantics', 'unconfirmed', sources, 'test reviewer', 'retracted')
    assert second['previous_version'] == first['version']
    assert len(registry.history()) == 2
    assert first['field_approval'] is False


def test_T08_T27_operational_times_not_inferred(tmp_path):
    view = ContractRegistry(tmp_path).view({})
    with pytest.raises(ValueError, match='unconfirmed'):
        operational_observation({'source_row_id': 1}, view)
    for check in view['checks']:
        check['status'] = 'confirmed'  # TEST_ONLY boundary test, never stored
    observation = dict(product_id='TEST_ONLY', shot_id='TEST_ONLY', cavity_id='TEST_ONLY',
        prediction_cutoff_at='2026-01-01T00:00:00+00:00', shot_ended_at='2026-01-01T00:00:00+00:00',
        feature_available_at={'x':'2026-01-01T00:00:01+00:00'}, transform_hash='TEST_ONLY', raw_units={'x':'TEST_ONLY'})
    with pytest.raises(ValueError, match='unavailable'):
        operational_observation(observation, view)
