"""Versioned, server-side evidence registry, maintained by a local reviewer CLI."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid
from research_runtime import digest

ITEMS = ('label_semantics', 'target_unit', 'product_shot_cavity_mapping',
         'feature_availability', 'inspection_timing', 'raw_units',
         'supplier_transform', 'future_validation_data')
STATES = ('confirmed', 'unconfirmed', 'mismatch')


def scope_id(sources):
    return hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()


def source_hashes(data_dir):
    if data_dir is None:
        return {}
    return {p.name: digest(p) for p in sorted(Path(data_dir).glob('moldset_*.csv'))}


class ContractRegistry:
    def __init__(self, folder):
        self.folder = Path(folder)

    def record(self, item, status, sources, reviewer, reason, evidence=None):
        if item not in ITEMS or status not in STATES or not reviewer.strip() or not reason.strip():
            raise ValueError('Contract item, status, reviewer and reason are required')
        if not sources:
            raise ValueError('Contract must reference actual source hashes')
        if status in ('confirmed', 'mismatch') and (evidence is None or not Path(evidence).is_file()):
            raise ValueError('Confirmed/mismatch checks require an existing evidence file')
        record = {'schema_version': 'moldguard-contract-v1', 'version': str(uuid.uuid4()),
                  'item': item, 'status': status, 'source_hashes': sources, 'scope_hash': scope_id(sources),
                  'reviewer': reviewer, 'reason': reason, 'checked_at': datetime.now(timezone.utc).isoformat(),
                  'evidence_path': str(Path(evidence).resolve()) if evidence else None,
                  'evidence_sha256': digest(evidence) if evidence else None,
                  'field_approval': False}
        previous = [r for r in self.history() if r['item'] == item]
        record['previous_version'] = previous[-1]['version'] if previous else None
        self.folder.mkdir(parents=True, exist_ok=True)
        with (self.folder / (record['checked_at'].replace(':', '-') + '_' + record['version'] + '.json')).open('x') as stream:
            json.dump(record, stream, ensure_ascii=False, indent=2)
        return record

    def history(self):
        result = []
        for p in sorted(self.folder.glob('*.json')):
            value = json.loads(p.read_text())
            if value.get('schema_version') != 'moldguard-contract-v1' or value.get('item') not in ITEMS or value.get('status') not in STATES:
                raise ValueError('Invalid contract registry record: ' + p.name)
            result.append(value)
        return result

    def view(self, sources):
        history = self.history()
        entries = []
        for item in ITEMS:
            found = [r for r in history if r['item'] == item and r['scope_hash'] == scope_id(sources)]
            entry = dict(found[-1]) if found else {'item': item, 'status': 'unconfirmed',
                    'source_hashes': sources, 'scope_hash': scope_id(sources), 'version': None,
                    'reviewer': None, 'checked_at': None, 'evidence_path': None, 'evidence_sha256': None,
                    'reason': '현재 데이터 버전에 연결된 확인 근거 없음'}
            if entry['status'] in ('confirmed', 'mismatch'):
                p = Path(entry['evidence_path'] or '')
                if not entry.get('reviewer') or not entry.get('checked_at') or not p.is_file() or digest(p) != entry['evidence_sha256']:
                    entry.update(status='mismatch', reason='Evidence missing, changed, or incomplete')
            entries.append(entry)
        version = hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()
        return {'schema_version': 'moldguard-contract-view-v1', 'version': version,
                'scope_hash': scope_id(sources), 'checks': entries, 'field_approval': False}


def operational_observation(record, contracts):
    """Validate future input boundary only; never activates advisory mode."""
    if any(c['status'] != 'confirmed' for c in contracts['checks']):
        raise ValueError('Operational contracts are unconfirmed')
    required = ('product_id', 'shot_id', 'cavity_id', 'prediction_cutoff_at', 'shot_ended_at',
                'feature_available_at', 'transform_hash', 'raw_units')
    if any(not record.get(k) for k in required):
        raise ValueError('Operational keys/times/units/transform missing')
    def timestamp(v):
        t = datetime.fromisoformat(v)
        if t.tzinfo is None:
            raise ValueError('Timezone required')
        return t
    cutoff = timestamp(record['prediction_cutoff_at'])
    if timestamp(record['shot_ended_at']) > cutoff or any(timestamp(t) > cutoff for t in record['feature_available_at'].values()):
        raise ValueError('Features unavailable at prediction cutoff')
    return {'valid_structure': True, 'operational_enabled': False}
