"""Fail-closed research decisions, independent of model training."""
import hashlib
import json
import math


def validate_policy(policy):
    if policy.get('schema_version') != 'moldguard-research-policy-v1' or policy.get('mode') != 'RESEARCH_ONLY':
        raise ValueError('Only the implemented RESEARCH_ONLY policy is supported')
    for key in ('field_deployment_approved', 'defect_probability_enabled', 'external_queue_writes',
                'allow_inspection_skip', 'allow_machine_control', 'allow_automatic_model_promotion'):
        if policy.get(key) is not False:
            raise ValueError('Unsupported authority: ' + key)
    if policy.get('keep_all_required_inspections') is not True or policy.get('unvalidated_machines') != ['rg3']:
        raise ValueError('Legacy inspection protections must be preserved')
    if policy.get('score_target') != 'numeric_label_1_unconfirmed':
        raise ValueError('Unconfirmed label semantics cannot be replaced by config')
    return hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()


def acceptance_blockers(criteria):
    missing = []
    def walk(value, key):
        if value is None:
            missing.append(key)
        elif isinstance(value, dict):
            for k, v in value.items():
                walk(v, key + '.' + k if key else k)
    walk(criteria, '')
    return missing


def decide(record, contracts, policy, *, model_healthy=True, mode='RESEARCH_ONLY',
           approval=None, additional_warnings=()):
    policy_hash = validate_policy(policy)
    reasons = ['research_only', 'field_approval_unavailable']
    checks = {c['item']: c['status'] for c in contracts['checks']}
    from .contracts import ITEMS
    reasons.extend(k + '_' + checks.get(k, 'unconfirmed') for k in ITEMS if checks.get(k) != 'confirmed')
    machine = record.get('machine')
    if machine != 'cn7':
        reasons.append('unvalidated_machine')
    if record.get('ood_flag') is True:
        reasons.append('out_of_training_range')
    score = record.get('risk_score')
    valid = (type(score) in (float, int) and math.isfinite(score) and 0 <= score <= 1
             and machine in ('cn7', 'rg3') and type(record.get('ood_flag')) is bool)
    if not valid:
        reasons.append('invalid_or_missing_prediction')
    if not model_healthy:
        reasons.append('model_unavailable_or_unverified')
    if mode != 'RESEARCH_ONLY':
        reasons.append('requested_mode_not_enabled')
    if approval:
        reasons.append('advisory_activation_not_implemented')
    reasons.extend(additional_warnings)
    if not valid or not model_healthy:
        action = 'usual_inspection_input_error'
    elif machine != 'cn7':
        action = 'usual_inspection_unvalidated_machine'
    elif record['ood_flag']:
        action = 'usual_inspection_ood'
    elif checks.get('label_semantics') != 'confirmed':
        action = 'usual_inspection_label_unconfirmed'
    else:
        action = 'usual_inspection_contract_or_approval_unconfirmed'
    return {'mode': 'RESEARCH_ONLY', 'research_only': True,
            'prediction_status': 'available' if valid and model_healthy else 'not_produced',
            'score_target': policy['score_target'],
            'research_score': score if valid and model_healthy else None,
            'research_priority': record.get('priority') if valid and model_healthy else None,
            'defect_probability': None, 'operational_priority': None,
            'operational_recommendation': False, 'existing_queue_policy': 'unchanged',
            'action': action, 'fallback_reasons': sorted(set(reasons)),
            'policy_version': policy['schema_version'], 'policy_hash': policy_hash,
            'data_contract_version': contracts['version']}
