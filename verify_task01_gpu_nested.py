"""Recompute nested run metrics and audit identities, fit counts and decisions."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import moldguard as mg
import calibration_diagnostics as cd
from model_upgrade import identity
from research_runtime import digest, execution_record, write_json
from task01_gpu_nested_search import CANDIDATE_SETS, ROUTES, choose_candidates, thresholds
from task01_gpu_oof_evaluate import validate_frozen_contract


IDENTITY = ['machine', 'source_row_id', 'feature_group', 'outer_fold', 'label_value']


def independent_counts(y, chosen):
    y, chosen = np.asarray(y, dtype=int), np.asarray(chosen, dtype=bool)
    tp = int(np.sum(chosen & (y == 1)))
    fp = int(np.sum(chosen & (y == 0)))
    fn = int(np.sum(~chosen & (y == 1)))
    tn = int(np.sum(~chosen & (y == 0)))
    return {'rows': len(y), 'positives': int(y.sum()), 'k': int(chosen.sum()),
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'recall': tp / (tp + fn) if tp + fn else 0.,
            'precision': tp / (tp + fp) if tp + fp else 0.,
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
            'accuracy': (tp + tn) / len(y)}


def verify(data_dir, split_dir, result_dir):
    read = lambda name: pd.read_csv(result_dir / name, float_precision='round_trip')
    execution = json.loads((result_dir / 'execution.json').read_text())
    manifest = json.loads((result_dir / 'run_manifest.json').read_text())
    if execution.get('status') != 'passed' or manifest.get('status') != 'completed_nested_development_research':
        raise ValueError('Nested run did not finish successfully')
    hashes = json.loads((result_dir / 'artifact_hashes.json').read_text())
    for name, value in hashes.items():
        if Path(name).name != name or digest(result_dir / name) != value:
            raise ValueError(f'Artifact hash mismatch: {name}')
    plan = json.loads((result_dir / 'experiment_plan.json').read_text())
    SPECS = CANDIDATE_SETS[plan.get('candidate_set', 'original')]
    if digest(result_dir / 'experiment_plan.json') != manifest['plan_sha256'] or plan['specs'] != SPECS:
        raise ValueError('Candidate plan differs from the recorded run')
    data, _, sources = cd.load_development(data_dir, split_dir)
    validate_frozen_contract(data, sources, digest(split_dir / 'split_manifest.csv'))
    if plan['sources'] != sources or plan['split_sha256'] != digest(split_dir / 'split_manifest.csv'):
        raise ValueError('Input provenance differs')
    expected = identity(data)
    predictions = read('oof_predictions.csv')
    seeds, models = manifest['seeds'], manifest['models']
    pairs = {(model, seed) for model in models for seed in seeds}
    if (set(zip(predictions.model, predictions.seed)) != pairs
            or len(predictions) != len(data) * len(pairs)
            or predictions.duplicated(['model', 'seed', 'machine', 'source_row_id']).any()):
        raise ValueError('Incomplete or duplicated OOF predictions')
    keys = ['machine', 'source_row_id']
    for _, group in predictions.groupby(['model', 'seed']):
        pd.testing.assert_frame_equal(group[IDENTITY].sort_values(keys).reset_index(drop=True),
                                      expected[IDENTITY].sort_values(keys).reset_index(drop=True), check_dtype=False)
        if not np.isfinite(group.score).all() or not group.score.between(0, 1).all():
            raise ValueError('Invalid OOF score')
        if group.groupby('feature_group').score.nunique().gt(1).any():
            raise ValueError('Identical groups have different predictions')
    members = read('inner_split_manifest.csv')
    inner_scores = read('inner_candidate_metrics.csv')
    ledger = read('fit_ledger.csv')
    choices = read('selected_candidates.csv')
    expected_fits = set()
    train_tables = {}
    for seed in seeds:
        for outer in range(5):
            local = members.loc[members.seed.eq(seed) & members.outer_test_fold.eq(outer)]
            required = expected.loc[expected.outer_fold.ne(outer)]
            pd.testing.assert_frame_equal(local[IDENTITY].sort_values(keys).reset_index(drop=True),
                                          required[IDENTITY].sort_values(keys).reset_index(drop=True), check_dtype=False)
            if local.groupby('feature_group').inner_fold.nunique().gt(1).any():
                raise ValueError('Inner feature group leakage')
            for inner in [-1, *sorted(local.inner_fold.unique())]:
                train = local if inner == -1 else local.loc[local.inner_fold.ne(inner)]
                valid = expected.loc[expected.outer_fold.eq(outer)] if inner == -1 else local.loc[local.inner_fold.eq(inner)]
                if set(train.feature_group) & set(valid.feature_group):
                    raise ValueError('Training/validation group overlap')
                train_tables[seed, outer, inner] = train
                for name in SPECS:
                    machines = ['all'] if name == 'pooled_logistic_ref' else ['cn7', 'rg3']
                    expected_fits.update((seed, outer, inner, machine, name) for machine in machines)
    observed = list(zip(ledger.seed, ledger.outer_fold, ledger.inner_fold, ledger.machine, ledger.candidate))
    if len(observed) != len(expected_fits) or set(observed) != expected_fits:
        raise ValueError('Fit ledger coverage differs')
    for fit in ledger.itertuples(index=False):
        train = train_tables[fit.seed, fit.outer_fold, fit.inner_fold]
        if fit.machine != 'all':
            train = train.loc[train.machine.eq(fit.machine)]
        if fit.original_training_rows != len(train) or fit.original_positives != int(train.label_value.sum()):
            raise ValueError('Fit training row/class counts differ')
        spec = SPECS[fit.candidate]
        if spec.get('event'):
            labels = train.groupby('feature_group').label_value.max()
            n, positive = len(labels), int(labels.sum())
        else:
            n, positive = len(train), int(train.label_value.sum())
        factor = spec.get('oversample', 1)
        if fit.fit_rows != n + (factor - 1) * positive or fit.fit_positives != factor * positive:
            raise ValueError('Augmented fit counts differ')
        if (spec['kind'] == 'xgb' and not fit.configured_backend.startswith('cuda')
                or spec['kind'] == 'cat' and fit.configured_backend != 'GPU'
                or spec['kind'] not in {'xgb', 'cat'} and fit.configured_backend != 'CPU'):
            raise ValueError('Candidate fitted with an unexpected backend')
    decision_checks = 0
    for seed in seeds:
        for outer in range(5):
            for machine in ('cn7', 'rg3'):
                inner = inner_scores.loc[inner_scores.seed.eq(seed) & inner_scores.outer_fold.eq(outer)
                                        & inner_scores.machine.eq(machine)]
                if set(inner.candidate) != set(SPECS) or len(inner) != len(SPECS):
                    raise ValueError('Inner candidate metric coverage differs')
                parameters = inner.set_index('candidate')
                winners = choose_candidates(inner.to_dict('records'), SPECS)
                section = predictions.loc[predictions.seed.eq(seed) & predictions.outer_fold.eq(outer)
                                          & predictions.machine.eq(machine)]
                part = {name: group.sort_values('source_row_id') for name, group in section.groupby('model')}
                for route, winner in winners.items():
                    saved = choices.loc[choices.seed.eq(seed) & choices.outer_fold.eq(outer)
                                        & choices.machine.eq(machine) & choices.route.eq(route)]
                    if len(saved) != 1 or saved.iloc[0].winner != winner:
                        raise ValueError('Inner model selection differs')
                candidates = {name: part['baseline' if name == 'pooled_logistic_ref' else 'fixed_' + name].score.to_numpy()
                              for name in SPECS}
                for route in models:
                    group = part[route]
                    if route == 'nested_union':
                        margin = np.max([candidates[name] - parameters.loc[name, 'high_recall_threshold'] for name in SPECS], axis=0)
                        score, f1_decision, high = (margin + 1) / 2, margin >= 0, margin >= 0
                    else:
                        name = route.removeprefix('fixed_') if route.startswith('fixed_') else winners[route]
                        score = candidates[name]
                        f1_decision = score >= parameters.loc[name, 'f1_threshold']
                        high = score >= parameters.loc[name, 'high_recall_threshold']
                    if (not np.allclose(group.score, score, atol=1e-15, rtol=0)
                            or not np.array_equal(group.f1_decision, f1_decision)
                            or not np.array_equal(group.high_recall_decision, high)):
                        raise ValueError(f'Outer score or train-only threshold decision differs: {seed}/{outer}/{machine}/{route}')
                    decision_checks += len(group)
    summaries = read('policy_summary.csv')
    classes = read('classification_summary.csv')
    metric_checks = 0
    for saved in summaries.itertuples(index=False):
        group = predictions.loc[predictions.model.eq(saved.model) & predictions.seed.eq(saved.seed)]
        if saved.machine != 'all':
            group = group.loc[group.machine.eq(saved.machine)]
        selected = np.zeros(len(group), dtype=bool)
        for machine in group.machine.unique():
            positions = np.flatnonzero(group.machine.eq(machine))
            sub = group.iloc[positions]
            ordered = positions[np.lexsort((sub.source_row_id.to_numpy(), -sub.score.to_numpy()))]
            selected[ordered[:int(np.ceil(len(positions) * saved.fraction))]] = True
        values = independent_counts(group.label_value, selected)
        values['average_precision'] = average_precision_score(group.label_value, group.score)
        values['roc_auc'] = roc_auc_score(group.label_value, group.score)
        for name, value in values.items():
            if not np.isclose(value, getattr(saved, name), rtol=0, atol=1e-12):
                raise ValueError(f'Budget metric differs: {saved.model}/{saved.seed}/{saved.machine}/{name}')
        metric_checks += 1
    for saved in classes.itertuples(index=False):
        group = predictions.loc[predictions.model.eq(saved.model) & predictions.seed.eq(saved.seed)]
        if saved.machine != 'all':
            group = group.loc[group.machine.eq(saved.machine)]
        for name, value in independent_counts(group.label_value, group[saved.policy]).items():
            if not np.isclose(value, getattr(saved, name), rtol=0, atol=1e-12):
                raise ValueError('Threshold classification metric differs')
        metric_checks += 1
    inner_prediction_checks = 0
    if plan.get('save_inner_predictions'):
        raw = read('inner_oof_predictions.csv')
        if len(raw) != len(members) * len(SPECS) or raw.duplicated(['seed', 'outer_test_fold', 'candidate', 'machine', 'source_row_id']).any():
            raise ValueError('Inner OOF prediction coverage differs')
        for (seed, outer, candidate), group in raw.groupby(['seed', 'outer_test_fold', 'candidate']):
            local = members.loc[members.seed.eq(seed) & members.outer_test_fold.eq(outer)]
            pd.testing.assert_frame_equal(group[IDENTITY + ['inner_fold']].sort_values(keys).reset_index(drop=True),
                                          local[IDENTITY + ['inner_fold']].sort_values(keys).reset_index(drop=True), check_dtype=False)
            if not np.isfinite(group.score).all() or not group.score.between(0, 1).all() or group.groupby('feature_group').score.nunique().gt(1).any():
                raise ValueError('Invalid inner OOF scores')
            for machine, part in group.groupby('machine'):
                saved = inner_scores.loc[inner_scores.seed.eq(seed) & inner_scores.outer_fold.eq(outer)
                                        & inner_scores.machine.eq(machine) & inner_scores.candidate.eq(candidate)]
                if len(saved) != 1:
                    raise ValueError('Missing inner score summary')
                recomputed = thresholds(part.label_value.to_numpy(), part.score.to_numpy())
                recomputed['average_precision'] = average_precision_score(part.label_value, part.score)
                recomputed['roc_auc'] = roc_auc_score(part.label_value, part.score)
                chosen = np.zeros(len(part), dtype=bool)
                order = np.lexsort((part.source_row_id.to_numpy(), -part.score.to_numpy()))
                chosen[order[:int(np.ceil(len(part) * .1))]] = True
                counts = independent_counts(part.label_value, chosen)
                recomputed.update({key: counts[key] for key in ('rows', 'positives', 'k', 'recall', 'precision')})
                recomputed.update(found=counts['tp'], missed=counts['fn'], selected_negative=counts['fp'])
                for key, value in recomputed.items():
                    if not np.isclose(saved.iloc[0][key], value, atol=1e-12, rtol=0):
                        raise ValueError('Inner threshold or metric differs from raw inner OOF')
            inner_prediction_checks += len(group)
    return {'status': 'passed', 'source_and_split_hashes_verified': True,
            'inner_oof_prediction_rows_verified': inner_prediction_checks,
            'oof_rows_verified': len(predictions), 'fit_rows_verified': len(ledger),
            'gpu_fits_verified': int(ledger.configured_backend.ne('CPU').sum()),
            'metric_rows_independently_recomputed': metric_checks,
            'threshold_decision_rows_verified': decision_checks,
            'inner_group_and_outer_training_membership_verified': True,
            'scope': 'saved artifacts, OOF identity, grouped membership, fit/backend ledger, inner selection, outer threshold decisions, independently recomputed metrics',
            'limitations': ['does not replay every fit', 'inner OOF audited when saved; earlier runs only preserve inner summaries',
                            'development data previously inspected', 'no new-period independent labeled test', 'physical label meaning unknown']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--split-dir', type=Path, required=True)
    parser.add_argument('--result-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    with execution_record(args.output_dir):
        report = verify(args.data_dir, args.split_dir, args.result_dir)
        write_json(args.output_dir / 'verification.json', report)
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
