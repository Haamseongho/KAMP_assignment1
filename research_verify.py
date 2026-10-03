"""Verify completed CPU/GPU comparison outputs against official inputs and the split."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import calibration_diagnostics as cd
import moldguard as mg
import task01_compare as comparison
from research_runtime import digest, execution_record, write_json


def verify(data_dir, split_dir, result_dir):
    result_dir = Path(result_dir)
    record = json.loads((result_dir / 'run_manifest.json').read_text())
    if record.get('status') != 'complete_development_oof_not_independent_holdout':
        raise ValueError('Result is not a completed development comparison')
    execution = result_dir / 'execution.json'
    if execution.exists() and json.loads(execution.read_text()).get('status') != 'passed':
        raise ValueError('Execution is incomplete or failed')
    hashes = json.loads((result_dir / 'artifact_hashes.json').read_text())
    required = {'run_manifest.json', 'oof_predictions.csv', 'comparison_summary.csv',
                'metrics_by_seed_machine.csv', 'metrics_by_fold.csv', 'condition_slices.csv'}
    if not required.issubset(hashes):
        raise ValueError('Missing artifact integrity entries')
    for name, expected in hashes.items():
        if Path(name).name != name or digest(result_dir / name) != expected:
            raise ValueError('Artifact hash mismatch: ' + name)
    data, _, provenance = cd.load_development(data_dir, split_dir)
    if record['source_files'] != provenance or record['frozen_manifest_sha256'] != digest(Path(split_dir) / 'split_manifest.csv'):
        raise ValueError('Input/split provenance mismatch')
    predictions = pd.read_csv(result_dir / 'oof_predictions.csv')
    metrics = pd.read_csv(result_dir / 'metrics_by_seed_machine.csv')
    if predictions.duplicated(['model', 'seed', 'machine', 'source_row_id']).any():
        raise ValueError('Duplicate OOF predictions')
    if len(predictions) != len(data) * len(record['models']) * len(record['seeds']):
        raise ValueError('Missing OOF predictions')
    if set(zip(predictions.model, predictions.seed)) != {(m, s) for m in record['models'] for s in record['seeds']}:
        raise ValueError('Candidate/seed coverage mismatch')
    expected = data.rename(columns={mg.ID_COL: 'source_row_id', mg.TARGET: 'label_value'})
    keys = ['machine', 'source_row_id']
    identity = keys + ['label_value', 'feature_group', 'outer_fold']
    recomputed = []
    for (model, seed), group in predictions.groupby(['model', 'seed']):
        pd.testing.assert_frame_equal(group[identity].sort_values(keys).reset_index(drop=True),
                                      expected[identity].sort_values(keys).reset_index(drop=True), check_dtype=False)
        for machine in ('all', 'cn7', 'rg3'):
            part = group if machine == 'all' else group[group.machine.eq(machine)]
            values = comparison.score(part.label_value.to_numpy(), part.score.to_numpy())
            saved = metrics[(metrics.model == model) & (metrics.seed == seed) & (metrics.machine == machine)]
            if len(saved) != 1:
                raise ValueError('Metric coverage mismatch')
            for key, value in values.items():
                if not np.isclose(value, saved.iloc[0][key], rtol=0, atol=1e-12):
                    raise ValueError(f'Metric mismatch: {model}/{seed}/{machine}/{key}')
            recomputed.append({'model': model, 'seed': int(seed), 'machine': machine, **values})
    if len(metrics) != len(recomputed):
        raise ValueError('Extra metric rows')
    return {'status': 'passed', 'result_dir': str(result_dir), 'split_sha256': record['frozen_manifest_sha256'],
            'development_rows': len(data), 'development_groups': int(data.feature_group.nunique()),
            'oof_rows': len(predictions), 'metric_rows_recomputed': len(recomputed),
            'verified_artifact_hashes': hashes, 'absolute_metric_tolerance': 1e-12,
            'scope': 'artifact integrity + OOF identity/coverage + aggregate metrics; not prospective performance',
            'fold_and_condition_tables': 'hash integrity only; legacy full-run byte comparison recorded separately'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=mg.DATA_DIR)
    parser.add_argument('--split-dir', type=Path, default=cd.FROZEN)
    parser.add_argument('--result-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    with execution_record(args.output_dir):
        report = verify(args.data_dir, args.split_dir, args.result_dir)
        write_json(args.output_dir / 'verification.json', report)
        print(json.dumps({k: v for k, v in report.items() if k != 'verified_artifact_hashes'}, indent=2))


if __name__ == '__main__':
    main()
