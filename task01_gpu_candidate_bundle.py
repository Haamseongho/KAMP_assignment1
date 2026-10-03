"""Export a development research candidate and an offline unlabeled queue.

CN7: local logistic C=1. RG3: three seeds of row/event CatBoost, rank averaged.
This candidate was proposed after development OOF inspection.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.special import expit
from scipy.stats import rankdata

import moldguard as mg
import task01_compare as comparison
from research_runtime import digest, execution_record, write_json
from research_split import FEATURES
from task01_gpu_nested_search import SPECS, training_view
from task01_gpu_oof_evaluate import validate_frozen_contract


def rank_scores(frame, bundle):
    manifest = json.loads((bundle / 'bundle.json').read_text())
    for name, expected in manifest['files_sha256'].items():
        if Path(name).name != name or digest(bundle / name) != expected:
            raise ValueError('Bundle artifact hash mismatch')
    x = mg.feature_matrix(frame, FEATURES)
    if not np.isfinite(x.to_numpy()).all():
        raise ValueError('Nonfinite input')
    output = np.empty(len(frame))
    cn7 = frame.machine.eq('cn7').to_numpy()
    rg3 = frame.machine.eq('rg3').to_numpy()
    if not np.all(cn7 | rg3):
        raise ValueError('Unsupported machine')
    weights = json.loads((bundle / 'cn7_weights.json').read_text())
    if cn7.any():
        z = ((x.loc[cn7].to_numpy() - weights['mean']) / weights['scale']) @ np.array(weights['coef']) + weights['intercept']
        output[cn7] = expit(z)
    if rg3.any():
        ranks = []
        for name in manifest['rg3_models']:
            model = CatBoostClassifier()
            model.load_model(str(bundle / name))
            score = model.predict_proba(x.loc[rg3])[:, 1]
            ranks.append(rankdata(score, method='average') / len(score))
        output[rg3] = np.mean(ranks, axis=0)
    if not np.isfinite(output).all():
        raise ValueError('Invalid bundle prediction')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--split-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--local-gpu-ack', action='store_true')
    parser.add_argument('--existing-bundle-dir', type=Path, help='Reuse a verified native bundle; inference only, no new fits')
    parser.add_argument('--inspection-fraction', type=float, default=.075)
    args = parser.parse_args()
    if not 0 < args.inspection_fraction <= 1:
        parser.error('--inspection-fraction must be in (0, 1]')
    if args.existing_bundle_dir:
        with execution_record(args.output_dir):
            data, _, sources = comparison.validate_frozen_data(args.data_dir, args.split_dir)
            validate_frozen_contract(data, sources, digest(args.split_dir / 'split_manifest.csv'))
            manifest = json.loads((args.existing_bundle_dir / 'bundle.json').read_text())
            if manifest['source_files'] != sources or manifest['split_sha256'] != digest(args.split_dir / 'split_manifest.csv'):
                raise ValueError('Existing bundle provenance differs')
            unlabeled, features, unlabeled_sources = mg.load_data(args.data_dir, False)
            if features != list(FEATURES):
                raise ValueError('Unlabeled schema differs')
            scores = rank_scores(unlabeled, args.existing_bundle_dir)
            queue = unlabeled[['machine', mg.ID_COL]].rename(columns={mg.ID_COL: 'source_row_id'}).copy()
            queue['risk_score'] = scores
            queue['inspection_priority'] = queue.groupby('machine').risk_score.rank(method='first', ascending=False).astype(int)
            queue['selected_balanced_budget'] = queue.inspection_priority <= np.ceil(queue.groupby('machine').risk_score.transform('size') * args.inspection_fraction)
            queue.sort_values(['machine', 'inspection_priority']).to_csv(args.output_dir / 'unlabeled_priority.csv', index=False)
            write_json(args.output_dir / 'verification.json', {
                'status': 'passed_existing_native_bundle_hashes_and_queue_coverage', 'new_gpu_fits': 0,
                'queue_rows': len(queue), 'queue_duplicate_ids': int(queue.duplicated(['machine', 'source_row_id']).sum()),
                'queue_score_finite': bool(np.isfinite(scores).all()), 'inspection_fraction': args.inspection_fraction,
                'selected_rows': int(queue.selected_balanced_budget.sum()),
                'selected_rows_by_machine': {str(k): int(v) for k, v in queue.groupby('machine').selected_balanced_budget.sum().items()},
                'source_bundle_manifest_sha256': digest(args.existing_bundle_dir / 'bundle.json'),
                'unlabeled_sources': unlabeled_sources, 'independent_future_validation': False,
                'unlabeled_positive_count': 'unknown_without_labels', 'field_approved': False})
            print(f'Balanced queue exported; {int(queue.selected_balanced_budget.sum())}/{len(queue)} rows; no new fits')
        return
    comparison.check_gpu_authorization(None, args.local_gpu_ack)
    runtime = comparison.check_gpu_runtime()
    with execution_record(args.output_dir):
        data, _, sources = comparison.validate_frozen_data(args.data_dir, args.split_dir)
        validate_frozen_contract(data, sources, digest(args.split_dir / 'split_manifest.csv'))
        bundle = args.output_dir / 'bundle'
        bundle.mkdir()
        cn7 = data.loc[data.machine.eq('cn7')]
        model = mg.make_model('logistic')
        model[-1].set_params(C=1.)
        model.fit(mg.feature_matrix(cn7, FEATURES), cn7[mg.TARGET])
        scaler, estimator = model[0], model[-1]
        write_json(bundle / 'cn7_weights.json', {
            'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(),
            'coef': estimator.coef_[0].tolist(), 'intercept': float(estimator.intercept_[0])})
        originals, names, fits = [], [], []
        rg3 = data.loc[data.machine.eq('rg3')]
        for seed in comparison.SEEDS:
            for candidate in ('cat_row_d3', 'cat_event_d3'):
                fit = training_view(rg3, SPECS[candidate])
                model = CatBoostClassifier(iterations=200, depth=3, learning_rate=.03,
                                           l2_leaf_reg=10, auto_class_weights='Balanced',
                                           task_type='GPU', devices='0', gpu_ram_part=.45, border_count=64,
                                           random_seed=seed, thread_count=2, verbose=False,
                                           allow_writing_files=False)
                model.fit(mg.feature_matrix(fit, FEATURES), fit[mg.TARGET])
                backend = comparison.check_fitted_gpu_backend(model, 'catboost_candidate')
                name = f'rg3_{candidate}_{seed}.cbm'
                model.save_model(str(bundle / name))
                before = model.predict_proba(mg.feature_matrix(rg3, FEATURES))[:, 1]
                restored = CatBoostClassifier()
                restored.load_model(str(bundle / name))
                after = restored.predict_proba(mg.feature_matrix(rg3, FEATURES))[:, 1]
                delta = float(np.max(np.abs(before - after)))
                if delta > 1e-12:
                    raise ValueError('Native model round-trip prediction mismatch')
                originals.append(rankdata(before, method='average') / len(before))
                names.append(name)
                fits.append({'candidate': candidate, 'seed': seed, 'configured_backend': backend,
                             'fit_rows': len(fit), 'fit_positives': int(fit[mg.TARGET].sum()),
                             'native_roundtrip_max_absolute_error': delta})
        write_json(bundle / 'bundle.json', {
            'schema': 'task01-gpu-offline-research-v1', 'rg3_models': names,
            'feature_columns': FEATURES, 'cn7': 'local logistic C=1',
            'rg3': 'mean within-batch rank of six row/event CatBoost models',
            'batch_dependent_rank_score': True, 'score_is_defect_probability': False,
            'development_rows': len(data), 'development_positives': int(data[mg.TARGET].sum()),
            'old_holdout_used': False, 'independent_future_validation': False, 'field_approved': False,
            'source_files': sources, 'split_sha256': digest(args.split_dir / 'split_manifest.csv'),
            'gpu_runtime': runtime, 'fit_checks': fits,
            'files_sha256': {p.name: digest(p) for p in bundle.iterdir() if p.is_file()}})
        check = rank_scores(data, bundle)
        expected_cn7 = expit(((mg.feature_matrix(cn7, FEATURES).to_numpy() - np.array(scaler.mean_)) / np.array(scaler.scale_))
                             @ estimator.coef_[0] + estimator.intercept_[0])
        if (not np.allclose(check[data.machine.eq('cn7')], expected_cn7, rtol=0, atol=1e-12)
                or not np.allclose(check[data.machine.eq('rg3')], np.mean(originals, axis=0), rtol=0, atol=1e-12)):
            raise ValueError('Bundle combined inference round-trip mismatch')
        unlabeled, features, unlabeled_sources = mg.load_data(args.data_dir, False)
        if features != list(FEATURES):
            raise ValueError('Unlabeled schema differs')
        scores = rank_scores(unlabeled, bundle)
        queue = unlabeled[['machine', mg.ID_COL]].rename(columns={mg.ID_COL: 'source_row_id'}).copy()
        queue['risk_score'] = scores
        queue['inspection_priority'] = queue.groupby('machine').risk_score.rank(method='first', ascending=False).astype(int)
        queue['selected_top10pct'] = queue.inspection_priority <= np.ceil(queue.groupby('machine').risk_score.transform('size') * .1)
        queue['selected_balanced_budget'] = queue.inspection_priority <= np.ceil(queue.groupby('machine').risk_score.transform('size') * args.inspection_fraction)
        queue.sort_values(['machine', 'inspection_priority']).to_csv(args.output_dir / 'unlabeled_priority.csv', index=False)
        write_json(args.output_dir / 'verification.json', {
            'status': 'passed_native_roundtrip_and_queue_coverage', 'gpu_fits': len(fits),
            'original_development_positives': 33, 'new_ground_truth_positives_created': 0,
            'queue_rows': len(queue), 'queue_duplicate_ids': int(queue.duplicated(['machine', 'source_row_id']).sum()),
            'queue_score_finite': bool(np.isfinite(queue.risk_score).all()),
            'inspection_fraction': args.inspection_fraction, 'selected_rows': int(queue.selected_balanced_budget.sum()),
            'unlabeled_sources': unlabeled_sources,
            'unlabeled_true_positive_count': 'unknown_without_labels',
            'scope': 'artifact inference parity and queue completeness, not independent performance'})
        print(f'Bundle exported; GPU fits={len(fits)}, unlabeled queue rows={len(queue)}', flush=True)


if __name__ == '__main__':
    main()
