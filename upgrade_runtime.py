"""Shared CPU research inference for CLI and optional local GET API."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

import moldguard as mg
from research_runtime import digest, execution_record, write_json
from upgrade_models import ResearchBundle


class UpgradeService:
    def __init__(self, run_dir, data_dir):
        self.run_dir = Path(run_dir)
        execution = json.loads((self.run_dir / 'execution.json').read_text())
        if execution['status'] != 'passed':
            raise ValueError('Upgrade research run is incomplete')
        artifacts = json.loads((self.run_dir / 'artifact_manifest.json').read_text())
        decision = json.loads((self.run_dir / 'decision.json').read_text())
        if not decision.get('bundle') or decision.get('field_approved') is not False:
            raise ValueError('No research candidate bundle')
        for name, expected in artifacts.items():
            path = Path(name)
            if path.is_absolute() or '..' in path.parts or digest(self.run_dir / path) != expected:
                raise ValueError('Upgrade artifact hash mismatch')
        self.bundle = ResearchBundle(self.run_dir / 'bundle_candidate')
        self.data, _, provenance = mg.load_data(Path(data_dir), False)
        if provenance != self.bundle.meta['inference_sources']:
            raise ValueError('Inference source mismatch')

    def readiness(self):
        return {'status': 'available', 'mode': 'RESEARCH_ONLY', 'field_approved': False,
                'run_dir': str(self.run_dir), 'bundle_sha256': self.bundle.hash,
                'routing': self.bundle.meta['routing'], 'rows': len(self.data),
                'score_semantics': self.bundle.meta['score_semantics'],
                'inspection_policy': 'existing_inspection_unchanged',
                'performance_scope': 'nested_development_selection_procedure_not_this_final_models_independent_score'}

    def predict(self, frame):
        score = self.bundle.predict(frame)
        result = frame[['machine', mg.ID_COL]].rename(columns={mg.ID_COL: 'source_row_id'}).copy()
        result['research_score'] = score
        result['ood_feature_count'] = 0
        for m in ('cn7', 'rg3'):
            mask = frame.machine.eq(m)
            bounds = self.bundle.meta['ood_bounds'][m]
            low, high = pd.Series(bounds['low']), pd.Series(bounds['high'])
            result.loc[mask, 'ood_feature_count'] = ((frame.loc[mask, low.index] < low) | (frame.loc[mask, low.index] > high)).sum(axis=1)
        result['ood_flag'] = result.ood_feature_count.ge(3)
        result['operational_priority'] = None
        result['defect_probability'] = None
        result['action'] = 'usual_inspection_research_only'
        result['bundle_sha256'] = self.bundle.hash
        return result

    def get(self, query):
        if set(query) - {'machine', 'offset', 'limit'}:
            raise ValueError('Only machine/offset/limit supported')
        machine = query.get('machine', [None])[0]
        offset, limit = int(query.get('offset', ['0'])[0]), int(query.get('limit', ['30'])[0])
        if machine not in (None, 'cn7', 'rg3') or offset < 0 or not 1 <= limit <= 100:
            raise ValueError('Invalid inference page')
        data = self.data if machine is None else self.data.loc[self.data.machine.eq(machine)]
        frame = data.iloc[offset:offset+limit]
        return {**self.readiness(), 'total': len(data), 'offset': offset, 'limit': limit,
                'rows': self.predict(frame).to_dict('records')}


def validate(run_dir, data_dir, output):
    with execution_record(output):
        service = UpgradeService(run_dir, data_dir)
        started = time.perf_counter()
        result = service.predict(service.data)
        elapsed = time.perf_counter()-started
        reference = pd.read_csv(Path(run_dir) / 'unlabeled_native_predictions.csv')
        joined = result.merge(reference, on=['machine', 'source_row_id'], validate='one_to_one', how='outer', indicator=True)
        if len(joined) != len(service.data) or not joined['_merge'].eq('both').all():
            raise ValueError('Row coverage mismatch')
        error = float(np.max(np.abs(joined.research_score-joined.native_score)))
        if error > 1e-12:
            raise ValueError('New model native/bundle parity failed')
        sample = service.data.sample(n=100, random_state=42)
        shuffled = service.predict(sample[sample.columns[::-1]])
        original = result.set_index(['machine', 'source_row_id']).research_score
        expected = original.loc[list(zip(shuffled.machine, shuffled.source_row_id))].to_numpy()
        if not np.allclose(shuffled.research_score, expected, rtol=0, atol=1e-12):
            raise ValueError('Batch/row/feature order invariance failed')
        result.to_json(output / 'predictions.jsonl', orient='records', lines=True, force_ascii=False)
        report = {**service.readiness(), 'status': 'passed', 'rows': len(result),
                  'native_max_absolute_error': error, 'inference_seconds': elapsed,
                  'ood_rows': int(result.ood_flag.sum()), 'all_existing_inspections_preserved': True,
                  'batch_row_feature_order_check_rows': 100, 'execution_location': 'local_cpu_not_kamp',
                  'performance_metrics': 'see frozen OOF model_comparison.csv; unlabeled predictions have no accuracy'}
        write_json(output / 'validation.json', report)
        print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    validate(args.run_dir, args.data_dir, args.output_dir)


if __name__ == '__main__':
    main()
