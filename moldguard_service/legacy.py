"""Validate and read historical artifacts. Never deserialize uploaded models."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import moldguard as mg
from research_runtime import digest
from .contracts import source_hashes
from .policy import decide


class LegacyAdapter:
    def __init__(self, data_dir, result_dir, comparison_dir):
        self.data_dir = Path(data_dir) if data_dir else None
        self.result_dir = Path(result_dir)
        self.comparison_dir = Path(comparison_dir)
        self.sources = source_hashes(data_dir)
        self.frame = None
        self.errors = []
        self.model_hash = None
        self.queue_hash = None
        self.validation = None
        try:
            self._load()
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.errors.append(str(error))

    def _load(self):
        if self.data_dir is None or len(self.sources) != 4:
            raise ValueError('Official four CSVs missing; use --data-dir. Predictions are not produced.')
        p = self.result_dir / 'execution.json'
        if p.exists() and json.loads(p.read_text()).get('status') != 'passed':
            raise ValueError('Result execution is incomplete/failed')
        record = json.loads((self.result_dir / 'reproducibility.json').read_text())
        source = record['source_file_hashes']
        if self.sources != {name: info['sha256'] for name, info in source.items()}:
            raise ValueError('Result source hashes differ from provided CSVs')
        for name in ('model.joblib', 'unlabeled_priority.csv', 'data_audit.json', 'experiment.json'):
            expected = [v for k, v in record['artifact_hashes'].items() if Path(k).name == name]
            if len(expected) != 1 or digest(self.result_dir / name) != expected[0]:
                raise ValueError('Artifact hash mismatch: ' + name)
        inputs, features, _ = mg.load_data(self.data_dir, False)
        from research_split import FEATURES
        if features != FEATURES:
            raise ValueError('Feature schema mismatch')
        path = self.result_dir / 'unlabeled_priority.csv'
        self.validation = mg.check_priority(path, len(inputs), inputs)
        frame = pd.read_csv(path)
        if not frame.machine.isin(['cn7', 'rg3']).all() or frame.ood_flag.dtype != bool:
            raise ValueError('Unknown machine or invalid OOD type')
        self.model_hash = digest(self.result_dir / 'model.joblib')
        self.queue_hash = digest(path)
        self.frame = frame

    def rankings(self, contracts, policy, machine=None, offset=0, limit=30):
        if self.frame is None:
            return {'status': 'unavailable', 'rows': [], 'total': 0, 'errors': self.errors,
                    'research_score': None, 'operational_priority': None}
        if machine not in (None, 'cn7', 'rg3') or offset < 0 or not 1 <= limit <= 100:
            raise ValueError('Invalid pagination or machine')
        frame = self.frame if machine is None else self.frame[self.frame.machine.eq(machine)]
        rows = []
        for row in frame.iloc[offset:offset+limit].to_dict('records'):
            uid = hashlib.sha256(f"{self.sources['moldset_unlabeled_' + row['machine'] + '.csv']}:{row['machine']}:{row['source_row_id']}".encode()).hexdigest()
            rows.append({**row, **decide(row, contracts, policy), 'legacy_action': row['action'],
                         'observation_uid': uid, 'prediction_id': hashlib.sha256((uid + self.queue_hash).encode()).hexdigest(),
                         'model_hash': self.model_hash, 'source_sha256': self.sources['moldset_unlabeled_' + row['machine'] + '.csv'],
                         'product_id': None, 'shot_id': None, 'cavity_id': None, 'prediction_cutoff_at': None})
        return {'status': 'available', 'rows': rows, 'total': len(frame), 'offset': offset,
                'limit': limit, 'research_only': True, 'operational_queue': 'unchanged',
                'scope': 'historical_unlabeled_research_ranking_not_performance',
                'queue_sha256': self.queue_hash, 'validation': self.validation}

    def metrics(self):
        try:
            folder = self.comparison_dir
            record = json.loads((folder / 'run_manifest.json').read_text())
            if record.get('status') != 'complete_development_oof_not_independent_holdout':
                raise ValueError('Unsupported/incomplete comparison run')
            execution = folder / 'execution.json'
            if execution.exists() and json.loads(execution.read_text()).get('status') != 'passed':
                raise ValueError('Comparison execution is not complete')
            hashes = json.loads((folder / 'artifact_hashes.json').read_text())
            for name in ('run_manifest.json', 'comparison_summary.csv', 'metrics_by_seed_machine.csv'):
                if hashes.get(name) != digest(folder / name):
                    raise ValueError('Comparison hash mismatch: ' + name)
            metrics = pd.read_csv(folder / 'metrics_by_seed_machine.csv')
            if not {'model', 'machine', 'seed', 'rows', 'positives', 'average_precision'}.issubset(metrics):
                raise ValueError('Unsupported metric schema')
            if not np.isfinite(metrics.select_dtypes('number')).all().all():
                raise ValueError('Invalid numeric metrics')
            return {'status': 'available', 'run_id': folder.name, 'scope': record['source_scope'],
                    'evaluation': 'development_oof', 'rows': metrics.to_dict('records'),
                    'manifest': record, 'evidence': 'loaded_artifact_not_new_experiment',
                    'summary': pd.read_csv(folder / 'comparison_summary.csv').to_dict('records')}
        except (OSError, ValueError, KeyError, TypeError) as error:
            return {'status': 'unavailable', 'rows': [], 'errors': [str(error)]}
