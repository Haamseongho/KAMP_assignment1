"""Real-data local CPU validation; never represented as a remote PaaS run."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import moldguard as mg
import calibration_diagnostics as cd
from research_runtime import ROOT, digest, execution_record, write_json
from .model_runtime import infer, load_model
from .export_model import portable
from .inference_service import inference_dataframe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--split-dir', type=Path, required=True)
    parser.add_argument('--bundle-dir', type=Path, required=True)
    parser.add_argument('--reference-dir', type=Path, default=ROOT / 'outputs/moldguard')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    with execution_record(args.output_dir) as receipt:
        receipt['adapter_code_hashes'] = {str(p.relative_to(ROOT)): digest(p) for p in (ROOT / 'paas_cpu').glob('*.py')}
        manifest = json.loads((args.bundle_dir / 'deployment_manifest.json').read_text())
        model = load_model(args.bundle_dir / 'model.json', manifest['model_sha256'])
        data, features, sources = mg.load_data(args.data_dir, False)
        if features != model['features']:
            raise ValueError('Input schema mismatch')
        for name, source in sources.items():
            if model['training_source_hashes'][name]['sha256'] != source['sha256']:
                raise ValueError('Unexpected inference source hash')
        canonical = data.rename(columns={mg.ID_COL: 'source_row_id'})[['machine', 'source_row_id', *features]]
        started = time.perf_counter()
        rows = inference_dataframe(canonical, {'model': model})
        elapsed = time.perf_counter() - started
        pd.DataFrame(rows).to_json(args.output_dir / 'local_inference.jsonl', orient='records', lines=True, force_ascii=False)
        reference_path = args.reference_dir / 'unlabeled_priority.csv'
        provenance = json.loads((args.reference_dir / 'reproducibility.json').read_text())
        expected = provenance['artifact_hashes'].get('outputs/moldguard/unlabeled_priority.csv')
        if digest(reference_path) != expected:
            raise ValueError('Reference inference hash differs')
        result = pd.DataFrame(rows)
        compared = result.merge(pd.read_csv(reference_path), on=['machine', 'source_row_id'], validate='one_to_one', suffixes=('', '_ref'))
        if len(compared) != len(data):
            raise ValueError('Incomplete comparison coverage')
        max_error = float(np.max(np.abs(compared.research_score - compared.risk_score)))
        if max_error > 1e-12 or not compared.action.eq(compared.action_ref).all() or not compared.ood_flag.eq(compared.ood_flag_ref).all():
            raise ValueError('Portable inference differs from the existing artifact')
        batch = inference_dataframe(canonical.iloc[:50], {'model': model})
        if not np.allclose([r['research_score'] for r in batch], result.research_score.iloc[:50], rtol=0, atol=1e-12):
            raise ValueError('Batch size changed scores')
        development, features, _ = cd.load_development(args.data_dir, args.split_dir)
        x = mg.feature_matrix(development, features)
        scores = np.full(len(development), np.nan)
        native = scores.copy()
        fold_checks = []
        for fold in range(5):
            valid = development.outer_fold.eq(fold).to_numpy()
            train = ~valid
            if set(development.loc[train, 'feature_group']) & set(development.loc[valid, 'feature_group']):
                raise ValueError('Group overlap')
            estimator = mg.make_model('logistic')
            estimator.fit(x.loc[train], development.loc[train, mg.TARGET])
            bounds = {machine: {bound: development.loc[train & development.machine.eq(machine).to_numpy(), features].quantile(q).to_dict()
                       for bound, q in [('low', .001), ('high', .999)]} for machine in ('cn7', 'rg3')}
            fold_model = portable(estimator, features, x.columns.tolist(), bounds)
            frame = development.loc[valid].rename(columns={mg.ID_COL: 'source_row_id'})[['machine', 'source_row_id', *features]]
            scores[valid] = [r['research_score'] for r in infer(frame, fold_model)]
            native[valid] = estimator.predict_proba(x.loc[valid])[:, 1]
            fold_checks.append({'fold': fold, 'train_rows': int(train.sum()), 'valid_rows': int(valid.sum()), 'group_overlap': 0})
        if not np.allclose(scores, native, atol=1e-12, rtol=0):
            raise ValueError('OOF portable/native mismatch')
        metrics = {}
        for machine in ('all', 'cn7', 'rg3'):
            mask = np.ones(len(development), dtype=bool) if machine == 'all' else development.machine.eq(machine).to_numpy()
            metrics[machine] = mg.score_metrics(development.loc[mask, mg.TARGET].to_numpy(), scores[mask])
        expected_ap = {'all': .11541918898088648, 'cn7': .28626515038456846, 'rg3': .023057939107304203}
        if any(abs(metrics[k]['average_precision'] - v) > 1e-12 for k, v in expected_ap.items()):
            raise ValueError('Frozen reference AP differs')
        report = {'status': 'passed', 'execution_location': 'local_cpu_not_kamp', 'remote_executed': False,
                  'model_sha256': manifest['model_sha256'], 'split_sha256': digest(args.split_dir / 'split_manifest.csv'),
                  'rows': len(rows), 'max_score_absolute_error': max_error, 'absolute_tolerance': 1e-12,
                  'actions_identical': True, 'ood_identical': True, 'batch_invariance_checked_rows': 50,
                  'inference_seconds': elapsed, 'rows_per_second': len(rows) / elapsed,
                  'latency_scope': 'one local in-process batch, not remote transport or SLA',
                  'actions': result.action.value_counts().to_dict(), 'development_metrics': metrics,
                  'folds': fold_checks, 'development_rows': len(development),
                  'data_source_hashes': sources, 'field_validation': 'not_performed'}
        write_json(args.output_dir / 'validation.json', report)
        print(json.dumps({k: report[k] for k in ('status', 'execution_location', 'rows', 'max_score_absolute_error', 'inference_seconds', 'development_metrics')}, indent=2))


if __name__ == '__main__':
    main()
