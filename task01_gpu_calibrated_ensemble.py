"""Train-only sigmoid calibration of CN7 logistic and RG3 GPU pair ranks.

Uses preserved inner OOF from the refinement run. Outer labels never enter
calibrator fitting or threshold selection. This is development research.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression

from export_task01_gpu_summary import checked_run
from research_runtime import digest, execution_record, write_json
from task01_gpu_nested_search import summarize, thresholds


def paired_rank(section, candidate_column, fold_column):
    names = ['cat_row_d3', 'cat_event_d3']
    keys = ['machine', 'source_row_id', 'feature_group', 'outer_fold', 'label_value', fold_column]
    base = section.loc[section[candidate_column].eq(names[0]), keys + ['score']].copy()
    for name in names[1:]:
        other = section.loc[section[candidate_column].eq(name), keys + ['score']]
        base = base.merge(other, on=keys, validate='one_to_one', suffixes=('_row', '_event'))
    if len(base) * 2 != len(section) or base.empty:
        raise ValueError('Incomplete paired scores')
    value = np.empty(len(base))
    for fold in base[fold_column].unique():
        mask = base[fold_column].eq(fold).to_numpy()
        value[mask] = np.mean([rankdata(base.loc[mask, col], method='average') / int(mask.sum())
                              for col in ('score_row', 'score_event')], axis=0)
    return base[keys].assign(score=value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-dir', type=Path, required=True)
    parser.add_argument('--verification-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    checked_run(args.result_dir)
    checked_run(args.verification_dir)
    verified = json.loads((args.verification_dir / 'verification.json').read_text())
    if not verified.get('inner_oof_prediction_rows_verified'):
        raise ValueError('Raw inner OOF verification is required')
    with execution_record(args.output_dir):
        write_json(args.output_dir / 'experiment_plan.json', {
            'cn7': 'local logistic C=1', 'rg3': 'row/event CatBoost depth3 within-fold rank mean',
            'calibration': 'one scalar sigmoid per machine, LogisticRegression C=10, no class weighting',
            'calibration_and_threshold_fitting': 'inner OOF of outer training only',
            'new_gpu_fits': 0, 'independent_future_validation': False,
            'source_oof_sha256': digest(args.result_dir / 'oof_predictions.csv'),
            'source_inner_oof_sha256': digest(args.result_dir / 'inner_oof_predictions.csv'),
            'caveat': 'procedure proposed after development inspection; not a fresh independent test'})
        inner = pd.read_csv(args.result_dir / 'inner_oof_predictions.csv', float_precision='round_trip')
        outer = pd.read_csv(args.result_dir / 'oof_predictions.csv', float_precision='round_trip')
        output, ledger = [], []
        replay_max_error = 0.
        for seed in sorted(outer.seed.unique()):
            for fold in range(5):
                for machine in ('cn7', 'rg3'):
                    train = inner.loc[inner.seed.eq(seed) & inner.outer_test_fold.eq(fold) & inner.machine.eq(machine)].copy()
                    valid = outer.loc[outer.seed.eq(seed) & outer.outer_fold.eq(fold) & outer.machine.eq(machine)].copy()
                    if machine == 'cn7':
                        train = train.loc[train.candidate.eq('local_logistic_c1')]
                        valid = valid.loc[valid.model.eq('fixed_local_logistic_c1')]
                    else:
                        train = paired_rank(train.loc[train.candidate.isin(['cat_row_d3', 'cat_event_d3'])], 'candidate', 'inner_fold')
                        # paired_rank expects the unprefixed candidate names.
                        valid['candidate'] = valid.model.str.removeprefix('fixed_')
                        valid['batch_fold'] = fold
                        valid = paired_rank(valid.loc[valid.candidate.isin(['cat_row_d3', 'cat_event_d3'])], 'candidate', 'batch_fold')
                    if set(train.feature_group) & set(valid.feature_group):
                        raise ValueError('Calibrator training/outer group overlap')
                    model = LogisticRegression(C=10., max_iter=3000, random_state=int(seed))
                    model.fit(train[['score']], train.label_value)
                    inner_probability = model.predict_proba(train[['score']])[:, 1]
                    threshold = thresholds(train.label_value.to_numpy(), inner_probability)
                    score = model.predict_proba(valid[['score']])[:, 1]
                    replay = expit(valid.score.to_numpy() * float(model.coef_[0, 0]) + float(model.intercept_[0]))
                    delta = float(np.max(np.abs(score - replay)))
                    if delta > 1e-12:
                        raise ValueError('Sigmoid coefficient replay mismatch')
                    replay_max_error = max(replay_max_error, delta)
                    row = valid[['machine', 'source_row_id', 'feature_group', 'outer_fold', 'label_value']].copy()
                    row['model'], row['seed'], row['score'] = 'inner_calibrated_gpu_pair', int(seed), score
                    row['f1_decision'] = score >= threshold['f1_threshold']
                    row['high_recall_decision'] = score >= threshold['high_recall_threshold']
                    output.append(row)
                    ledger.append({'seed': int(seed), 'outer_fold': fold, 'machine': machine,
                                   'training_rows': len(train), 'training_positives': int(train.label_value.sum()),
                                   'coefficient': float(model.coef_[0, 0]), 'intercept': float(model.intercept_[0]),
                                   'group_overlap': 0, **threshold})
        predictions = pd.concat(output, ignore_index=True)
        if predictions.duplicated(['seed', 'machine', 'source_row_id']).any() or len(predictions) != 1913 * 3:
            raise ValueError('Calibrated OOF identity coverage differs')
        budgets, classification, ties = summarize(predictions)
        predictions.to_csv(args.output_dir / 'oof_predictions.csv', index=False)
        budgets.to_csv(args.output_dir / 'policy_summary.csv', index=False)
        classification.to_csv(args.output_dir / 'classification_summary.csv', index=False)
        ties.to_csv(args.output_dir / 'tie_audit.csv', index=False)
        pd.DataFrame(ledger).to_csv(args.output_dir / 'calibration_ledger.csv', index=False)
        write_json(args.output_dir / 'verification.json', {
            'status': 'passed_source_inner_oof_verification_and_sigmoid_replay', 'calibrator_fits': len(ledger),
            'oof_rows': len(predictions), 'max_coefficient_replay_error': replay_max_error,
            'training_outer_group_overlap': 0, 'independent_future_validation': False})
        write_json(args.output_dir / 'artifact_hashes.json', {
            p.name: digest(p) for p in args.output_dir.iterdir() if p.is_file() and p.name not in {'execution.json', 'run.log', 'artifact_hashes.json'}})
        print(budgets.loc[budgets.machine.eq('all') & budgets.fraction.eq(.1),
                          ['seed', 'k', 'tp', 'recall', 'precision', 'f1', 'accuracy', 'average_precision', 'roc_auc']].to_string(index=False))


if __name__ == '__main__':
    main()
