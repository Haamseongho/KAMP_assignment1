"""Fail closed until project-scoped zero-incremental-cost evidence is reviewed."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

PROJECT = '215209147869625880'
OPERATIONS = ('upload_storage', 'image_build', 'cpu_training', 'cpu_inference', 'serving_idle', 'network')


def validate(receipt, base=Path('.'), now=None):
    now = now or datetime.now(timezone.utc)
    errors = []
    if receipt.get('project_id') != PROJECT or receipt.get('device') != 'cpu':
        errors.append('project_or_cpu_scope_mismatch')
    cap = receipt.get('incremental_cost_cap')
    if receipt.get('status') != 'verified_zero_incremental_cost' or type(cap) not in (int, float) or cap != 0:
        errors.append('zero_cost_unconfirmed')
    if receipt.get('gpu_allowed') is not False or receipt.get('paid_fallback_allowed') is not False:
        errors.append('paid_or_gpu_fallback_not_prohibited')
    if not receipt.get('reviewed_by') or not receipt.get('provider_reference'):
        errors.append('provider_evidence_review_missing')
    try:
        checked = datetime.fromisoformat(receipt['verified_at'])
        expires = datetime.fromisoformat(receipt['expires_at'])
        if checked.tzinfo is None or expires.tzinfo is None or not checked <= now < expires:
            errors.append('evidence_not_current')
    except (KeyError, ValueError, TypeError):
        errors.append('evidence_dates_missing')
    evidence = receipt.get('evidence_file')
    path = base / evidence if isinstance(evidence, str) and evidence else None
    if path is None or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != receipt.get('evidence_sha256'):
        errors.append('evidence_missing_or_changed')
    scopes = receipt.get('operations', {})
    for operation in OPERATIONS:
        entry = scopes.get(operation, {})
        if entry.get('no_additional_charge_confirmed') is not True or not entry.get('limit_and_billing_basis'):
            errors.append(operation + '_unconfirmed')
    if errors:
        raise ValueError('REMOTE_BLOCKED: ' + ', '.join(errors))
    return {'status': 'evidence_gate_passed', 'project_id': PROJECT, 'device': 'cpu',
            'notice': 'Not a spending enforcement system; verify console quotas and stop policy before execution'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipt', type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(validate(json.loads(args.receipt.read_text()), args.receipt.parent)))
    except ValueError as error:
        print(json.dumps({'status': 'blocked', 'reason': str(error)}))
        raise SystemExit(2)


if __name__ == '__main__':
    main()
