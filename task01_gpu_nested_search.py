"""GPU candidate expansion with train-only nested selection and threshold fitting.

The historical holdout is excluded. Outputs concern numeric label 1 only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.special import expit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

import moldguard as mg
import task01_compare as comparison
from model_upgrade import inner_folds, identity
from research_runtime import execution_record, digest, write_json
from research_split import FEATURES
from task01_gpu_oof_evaluate import validate_frozen_contract
from upgrade_audit import metrics, select, tie_rows


SPECS = {
    'pooled_logistic_ref': {'kind': 'pooled_logistic'},
    **{f'local_logistic_c{c:g}': {'kind': 'logistic', 'c': c} for c in (.01, .1, 1)},
    'event_logistic': {'kind': 'logistic', 'c': .1, 'event': True},
    'event_forest': {'kind': 'forest', 'event': True},
    **{f'rbf_c{c:g}': {'kind': 'rbf', 'c': c} for c in (1, 10)},
    'xgb_row_d1': {'kind': 'xgb', 'depth': 1, 'weight': 'balanced'},
    'xgb_row_d2': {'kind': 'xgb', 'depth': 2, 'weight': 'sqrt'},
    'xgb_row_d3': {'kind': 'xgb', 'depth': 3, 'weight': 'balanced'},
    'xgb_event_d1': {'kind': 'xgb', 'depth': 1, 'weight': 'balanced', 'event': True},
    'xgb_event_d2': {'kind': 'xgb', 'depth': 2, 'weight': 'balanced', 'event': True},
    'xgb_event_d3': {'kind': 'xgb', 'depth': 3, 'weight': 'sqrt', 'event': True},
    'xgb_oversample5': {'kind': 'xgb', 'depth': 2, 'weight': 'none', 'oversample': 5},
    'cat_row_d3': {'kind': 'cat', 'depth': 3},
    'cat_event_d3': {'kind': 'cat', 'depth': 3, 'event': True},
}
CANDIDATE_SETS = {
    'original': SPECS,
    'refinement': {
        'pooled_logistic_ref': {'kind': 'pooled_logistic'},
        'local_logistic_c1': {'kind': 'logistic', 'c': 1},
        'cat_row_d3': {'kind': 'cat', 'depth': 3},
        'cat_event_d3': {'kind': 'cat', 'depth': 3, 'event': True},
        'cat_row_d2_reg30': {'kind': 'cat', 'depth': 2, 'l2': 30, 'iterations': 400},
        'cat_event_d2_reg30': {'kind': 'cat', 'depth': 2, 'l2': 30, 'iterations': 400, 'event': True},
        'cat_row_engineered': {'kind': 'cat', 'depth': 3, 'l2': 20, 'engineered': True},
        'cat_event_engineered': {'kind': 'cat', 'depth': 3, 'l2': 20, 'engineered': True, 'event': True},
        'xgb_row_engineered': {'kind': 'xgb', 'depth': 2, 'weight': 'sqrt', 'engineered': True},
    },
}
ROUTES = ('baseline', 'nested_recall', 'nested_guarded', 'nested_f1', 'nested_high_recall', 'nested_union')
FRACTIONS = (.05, .1, .15, .2, .3, .5, 1.)


def training_view(frame, spec):
    """Augmentation is confined to fitting; evaluation keeps every original label."""
    if spec.get('event'):
        grouped = frame.groupby('feature_group', sort=True)
        result = grouped.first().reset_index()
        result[mg.TARGET] = grouped[mg.TARGET].max().to_numpy(dtype=int)
    else:
        result = frame.copy()
    if spec.get('oversample'):
        positive = result.loc[result[mg.TARGET].eq(1)]
        result = pd.concat([result, *[positive] * (spec['oversample'] - 1)], ignore_index=True)
    return result


def candidate_matrix(frame, spec):
    """Deterministic sensor interactions; no labels, IDs or fitted statistics."""
    x = mg.feature_matrix(frame, FEATURES).copy()
    if spec.get('engineered'):
        ratio_pairs = {
            'filling_injection_ratio': ('Filling_Time', 'Injection_Time'),
            'plasticizing_cycle_ratio': ('Plasticizing_Time', 'Cycle_Time'),
            'pressure_switch_ratio': ('Max_Injection_Pressure', 'Max_Switch_Over_Pressure'),
            'rpm_average_max_ratio': ('Average_Screw_RPM', 'Max_Screw_RPM'),
            'backpressure_average_max_ratio': ('Average_Back_Pressure', 'Max_Back_Pressure'),
        }
        for name, (a, b) in ratio_pairs.items():
            x[name] = x[a] / np.maximum(np.abs(x[b]), 1e-8)
        temperatures = x[[f'Barrel_Temperature_{i}' for i in range(1, 7)]]
        x['barrel_mean'] = temperatures.mean(axis=1)
        x['barrel_spread'] = temperatures.max(axis=1) - temperatures.min(axis=1)
        x['mold_temperature_delta'] = x.Mold_Temperature_3 - x.Mold_Temperature_4
        x['position_delta'] = x.Plasticizing_Position - x.Cushion_Position
    if not np.isfinite(x.to_numpy()).all():
        raise ValueError('Nonfinite candidate features')
    return x


def unique_predictions(model, frame, kind, spec=None):
    """Identical measured inputs always receive exactly the same score."""
    x = candidate_matrix(frame, spec or {})
    unique = x.drop_duplicates()
    # Hashes are used for lookup only, never as model input.
    keys = pd.util.hash_pandas_object(x, index=False)
    unique_keys = keys.loc[unique.index]
    if unique_keys.duplicated().any():
        raise ValueError('Prediction feature hash collision')
    score = expit(model.decision_function(unique)) if kind == 'rbf' else model.predict_proba(unique)[:, 1]
    result = keys.map(pd.Series(score, index=unique_keys.to_numpy())).to_numpy(dtype=float)
    if not np.isfinite(result).all() or np.any((result < 0) | (result > 1)):
        raise ValueError('Invalid candidate predictions')
    return result


def fit_predict(training, validation, spec, seed, ledger, location):
    if set(training.feature_group) & set(validation.feature_group):
        raise ValueError('Fit/validation feature group leakage')
    fit = training_view(training, spec)
    x = candidate_matrix(fit, spec)
    y = fit[mg.TARGET].to_numpy(dtype=int)
    if set(y) != {0, 1}:
        raise ValueError('Training partition needs both classes')
    kind = spec['kind']
    ratio = (len(y) - int(y.sum())) / int(y.sum())
    if kind == 'pooled_logistic':
        model = mg.make_model('logistic')
    elif kind == 'logistic':
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=spec['c'], class_weight='balanced', max_iter=3000, random_state=seed))
    elif kind == 'forest':
        model = RandomForestClassifier(n_estimators=200, max_depth=3, min_samples_leaf=8,
                                       max_features='sqrt', class_weight='balanced_subsample',
                                       n_jobs=1, random_state=seed)
    elif kind == 'rbf':
        model = make_pipeline(StandardScaler(), SVC(C=spec['c'], kernel='rbf', gamma='scale',
                                                  class_weight='balanced', probability=False))
    elif kind == 'xgb':
        weight = ratio if spec['weight'] == 'balanced' else np.sqrt(ratio) if spec['weight'] == 'sqrt' else 1.
        model = XGBClassifier(n_estimators=250, max_depth=spec['depth'], learning_rate=.03,
                              min_child_weight=3, reg_lambda=10, max_bin=64, subsample=.85,
                              colsample_bytree=.85, scale_pos_weight=weight, tree_method='hist',
                              device='cuda', n_jobs=2, random_state=seed, verbosity=0)
    elif kind == 'cat':
        model = CatBoostClassifier(iterations=spec.get('iterations', 200), depth=spec['depth'], learning_rate=.03,
                                   l2_leaf_reg=spec.get('l2', 10), auto_class_weights='Balanced',
                                   task_type='GPU', devices='0', gpu_ram_part=.45, border_count=64,
                                   random_seed=seed, thread_count=2, verbose=False,
                                   allow_writing_files=False)
    else:
        raise ValueError(kind)
    started = time.monotonic()
    model.fit(x, y)
    backend = comparison.check_fitted_gpu_backend(model, 'xgb_candidate' if kind == 'xgb' else 'catboost_candidate') if kind in {'xgb', 'cat'} else 'CPU'
    result = unique_predictions(model, validation, kind, spec)
    ledger.append({**location, 'kind': kind, 'configured_backend': backend,
                   'original_training_rows': len(training), 'original_positives': int(training[mg.TARGET].sum()),
                   'fit_rows': len(fit), 'fit_positives': int(y.sum()),
                   'independent_positive_groups': int(training.loc[training[mg.TARGET].eq(1), 'feature_group'].nunique()),
                   'fit_validation_group_overlap': 0, 'elapsed_seconds': time.monotonic() - started})
    return result


def thresholds(y, scores, required_recall=1.):
    precision, recall, threshold = precision_recall_curve(y, scores)
    precision, recall = precision[:-1], recall[:-1]
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(precision), where=precision + recall > 0)
    best = max(range(len(threshold)), key=lambda i: (f1[i], recall[i], precision[i], threshold[i]))
    eligible = np.flatnonzero(recall >= required_recall - 1e-12)
    high = max(eligible, key=lambda i: (precision[i], threshold[i]))
    return {'f1_threshold': float(threshold[best]), 'inner_f1': float(f1[best]),
            'f1_recall': float(recall[best]), 'high_recall_threshold': float(threshold[high]),
            'high_recall_precision': float(precision[high]), 'required_inner_recall': required_recall}


def choose_candidates(rows, specs=None):
    priority = {name: i for i, name in enumerate(SPECS if specs is None else specs)}
    reference = next(r for r in rows if r['candidate'] == 'pooled_logistic_ref')
    key = lambda r: (r['found'], r['average_precision'], -priority[r['candidate']])
    eligible = [r for r in rows if r['average_precision'] >= reference['average_precision'] - 1e-12
                and r['roc_auc'] >= reference['roc_auc'] - 1e-12]
    return {
        'baseline': 'pooled_logistic_ref',
        'nested_recall': max(rows, key=key)['candidate'],
        'nested_guarded': max(eligible, key=key)['candidate'],
        'nested_f1': max(rows, key=lambda r: (r['inner_f1'], r['f1_recall'], -priority[r['candidate']]))['candidate'],
        'nested_high_recall': max(rows, key=lambda r: (r['high_recall_precision'], -priority[r['candidate']]))['candidate'],
    }


def score_partition(training, validation, seed, ledger, location):
    scores = {name: np.full(len(validation), np.nan) for name in SPECS}
    scores['pooled_logistic_ref'] = fit_predict(training, validation, SPECS['pooled_logistic_ref'], seed,
                                               ledger, {**location, 'machine': 'all', 'candidate': 'pooled_logistic_ref'})
    for machine in ('cn7', 'rg3'):
        train = training.loc[training.machine.eq(machine)].reset_index(drop=True)
        mask = validation.machine.eq(machine).to_numpy()
        valid = validation.loc[mask].reset_index(drop=True)
        for name, spec in SPECS.items():
            if name != 'pooled_logistic_ref':
                scores[name][mask] = fit_predict(train, valid, spec, seed, ledger,
                                                 {**location, 'machine': machine, 'candidate': name})
    return scores


def classification_row(base, decision):
    y = base.label_value.to_numpy(dtype=int)
    tn, fp, fn, tp = confusion_matrix(y, decision, labels=[0, 1]).ravel()
    return {'rows': len(y), 'positives': int(y.sum()), 'k': int(np.sum(decision)),
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
            'recall': float(recall_score(y, decision, zero_division=0)),
            'precision': float(precision_score(y, decision, zero_division=0)),
            'f1': float(f1_score(y, decision, zero_division=0)), 'accuracy': float((tp + tn) / len(y))}


def summarize(predictions):
    budgets, classifications, ties = [], [], []
    for (model, seed), part in predictions.groupby(['model', 'seed'], sort=False):
        scores = part.score.to_numpy()
        for machine in ('all', 'cn7', 'rg3'):
            mask = np.ones(len(part), dtype=bool) if machine == 'all' else part.machine.eq(machine).to_numpy()
            local = part.loc[mask]
            for fraction in FRACTIONS:
                row = metrics(local, scores[mask], fraction, 'P_MACHINE')
                chosen = select(local, scores[mask], fraction, 'P_MACHINE')
                budgets.append({'model': model, 'seed': int(seed), 'machine': machine,
                                'fraction': fraction, **row, **classification_row(local, chosen)})
            for policy in ('f1_decision', 'high_recall_decision'):
                classifications.append({'model': model, 'seed': int(seed), 'machine': machine,
                                        'policy': policy, **classification_row(local, local[policy].to_numpy(dtype=bool))})
        ties.extend(tie_rows(part, scores, model, int(seed)))
    return pd.DataFrame(budgets), pd.DataFrame(classifications), pd.DataFrame(ties)


def run(args):
    comparison.check_gpu_authorization(None, args.local_gpu_ack)
    runtime = comparison.check_gpu_runtime()
    with execution_record(args.output_dir), threadpool_limits(limits=2):
        data, _, sources = comparison.validate_frozen_data(args.data_dir, args.split_dir)
        validate_frozen_contract(data, sources, digest(args.split_dir / 'split_manifest.csv'))
        base = identity(data)
        seeds = [int(s) for s in args.seeds.split(',')]
        plan = {'status': 'frozen_before_fitting', 'specs': SPECS, 'candidate_set': args.candidate_set,
                'save_inner_predictions': args.save_inner_predictions, 'seeds': seeds,
                'outer_folds': [0, 1, 2, 3, 4], 'inner_folds': 'train-only grouped three folds',
                'routes': ROUTES, 'fractions': FRACTIONS, 'required_inner_recall': 1.,
                'selection': 'machine-specific inner OOF only; no outer labels used to select models or thresholds',
                'sources': sources, 'split_sha256': digest(args.split_dir / 'split_manifest.csv'),
                'gpu_runtime': runtime, 'original_development_positives': 33,
                'augmentation': 'training-only class weights, group events, or five copies of existing positives; no new ground truth',
                'old_holdout_used': False, 'independent_future_validation': False,
                'scope': 'development research; candidate design informed by previously inspected development results',
                'max_seconds': args.max_seconds}
        write_json(args.output_dir / 'experiment_plan.json', plan)
        audit = data.groupby(['machine', 'feature_group'])[mg.TARGET].agg(['size', 'sum'])
        write_json(args.output_dir / 'identifiability.json', {
            machine: {'positives': int(g['sum'].sum()),
                      'conflicting_groups': int(((g['sum'] > 0) & (g['sum'] < g['size'])).sum()),
                      'unavoidable_false_positives_at_full_recall': int((g.loc[g['sum'].gt(0), 'size'] - g.loc[g['sum'].gt(0), 'sum']).sum())}
            for machine, g in audit.groupby(level=0)})
        all_predictions, choices, inner_metrics, membership, ledger, inner_predictions = [], [], [], [], [], []
        for seed in seeds:
            scores = {name: np.full(len(data), np.nan) for name in (*ROUTES, *[f'fixed_{n}' for n in SPECS if n != 'pooled_logistic_ref'])}
            decisions = {route: {key: np.zeros(len(data), dtype=bool) for key in ('f1_decision', 'high_recall_decision')} for route in scores}
            for outer in range(5):
                fit = data.outer_fold.ne(outer).to_numpy()
                valid = ~fit
                training = data.loc[fit].reset_index(drop=True)
                validation = data.loc[valid].reset_index(drop=True)
                inner_scores = {name: np.full(len(training), np.nan) for name in SPECS}
                for inner, (a, b) in enumerate(inner_folds(training, seed)):
                    values = score_partition(training.iloc[a].reset_index(drop=True), training.iloc[b].reset_index(drop=True),
                                             seed, ledger, {'seed': seed, 'outer_fold': outer, 'inner_fold': inner})
                    for name, value in values.items():
                        inner_scores[name][b] = value
                    row = identity(training.iloc[b])
                    row['seed'], row['outer_test_fold'], row['inner_fold'] = seed, outer, inner
                    membership.append(row)
                    if args.save_inner_predictions:
                        for name, value in values.items():
                            saved = row.copy()
                            saved['candidate'], saved['score'] = name, value
                            inner_predictions.append(saved)
                if any(not np.isfinite(value).all() for value in inner_scores.values()):
                    raise ValueError('Incomplete inner OOF coverage')
                outer_scores = score_partition(training, validation, seed, ledger,
                                               {'seed': seed, 'outer_fold': outer, 'inner_fold': -1})
                train_base = identity(training)
                for machine in ('cn7', 'rg3'):
                    m_train = training.machine.eq(machine).to_numpy()
                    m_valid = validation.machine.eq(machine).to_numpy()
                    positions = np.flatnonzero(valid)[m_valid]
                    rows = []
                    for name in SPECS:
                        row = {'candidate': name, **metrics(train_base.loc[m_train], inner_scores[name][m_train]),
                               **thresholds(training.loc[m_train, mg.TARGET].to_numpy(), inner_scores[name][m_train])}
                        rows.append(row)
                        inner_metrics.append({'seed': seed, 'outer_fold': outer, 'machine': machine, **row})
                    winners = choose_candidates(rows)
                    parameters = {r['candidate']: r for r in rows}
                    for route, name in winners.items():
                        scores[route][positions] = outer_scores[name][m_valid]
                        for policy, threshold_key in (('f1_decision', 'f1_threshold'), ('high_recall_decision', 'high_recall_threshold')):
                            decisions[route][policy][positions] = outer_scores[name][m_valid] >= parameters[name][threshold_key]
                        choices.append({'seed': seed, 'outer_fold': outer, 'machine': machine, 'route': route,
                                        'winner': name, **parameters[name], 'fit_validation_group_overlap': 0})
                    margins = []
                    for name in SPECS:
                        margins.append(outer_scores[name][m_valid] - parameters[name]['high_recall_threshold'])
                        if name != 'pooled_logistic_ref':
                            route = 'fixed_' + name
                            scores[route][positions] = outer_scores[name][m_valid]
                            decisions[route]['f1_decision'][positions] = outer_scores[name][m_valid] >= parameters[name]['f1_threshold']
                            decisions[route]['high_recall_decision'][positions] = outer_scores[name][m_valid] >= parameters[name]['high_recall_threshold']
                    margin = np.max(margins, axis=0)
                    scores['nested_union'][positions] = (margin + 1.) / 2.
                    decisions['nested_union']['f1_decision'][positions] = margin >= 0
                    decisions['nested_union']['high_recall_decision'][positions] = margin >= 0
                pd.DataFrame(ledger).to_csv(args.output_dir / 'fit_ledger.csv', index=False)
                print(f'seed={seed} outer={outer} complete; fits={len(ledger)}', flush=True)
            if any(not np.isfinite(value).all() for value in scores.values()):
                raise ValueError('Incomplete outer OOF coverage')
            for route, value in scores.items():
                row = base.copy()
                row['model'], row['seed'], row['score'] = route, seed, value
                for key, decision in decisions[route].items():
                    row[key] = decision
                all_predictions.append(row)
            pd.concat(all_predictions, ignore_index=True).to_csv(args.output_dir / 'oof_predictions_checkpoint.csv', index=False)
        predictions = pd.concat(all_predictions, ignore_index=True)
        if inner_predictions:
            pd.concat(inner_predictions, ignore_index=True).to_csv(args.output_dir / 'inner_oof_predictions.csv', index=False)
        budgets, classification, ties = summarize(predictions)
        for name, table in {'oof_predictions': predictions, 'policy_summary': budgets,
                            'classification_summary': classification, 'tie_audit': ties,
                            'selected_candidates': pd.DataFrame(choices), 'inner_candidate_metrics': pd.DataFrame(inner_metrics),
                            'inner_split_manifest': pd.concat(membership, ignore_index=True)}.items():
            table.to_csv(args.output_dir / f'{name}.csv', index=False)
        write_json(args.output_dir / 'run_manifest.json', {
            'status': 'completed_nested_development_research', 'plan_sha256': digest(args.output_dir / 'experiment_plan.json'),
            'development_rows': len(data), 'development_positives': int(data[mg.TARGET].sum()),
            'seeds': seeds, 'models': list(scores), 'fit_count': len(ledger),
            'gpu_fit_count': sum(r['configured_backend'] != 'CPU' for r in ledger),
            'cpu_fit_count': sum(r['configured_backend'] == 'CPU' for r in ledger),
            'outer_group_overlap': 0, 'old_holdout_used': False, 'independent_future_validation': False})
        write_json(args.output_dir / 'artifact_hashes.json', {
            p.name: digest(p) for p in args.output_dir.iterdir() if p.is_file() and p.name not in {'run.log', 'execution.json', 'artifact_hashes.json'}})
        print(budgets.loc[budgets.machine.eq('all') & budgets.fraction.eq(.1) & budgets.model.isin(ROUTES),
                          ['model', 'seed', 'tp', 'recall', 'precision', 'f1', 'average_precision', 'roc_auc']].to_string(index=False), flush=True)


def main():
    global SPECS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--split-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--local-gpu-ack', action='store_true')
    parser.add_argument('--seeds', default='20260922,20260923,20260924')
    parser.add_argument('--max-seconds', type=int, default=3600)
    parser.add_argument('--candidate-set', choices=CANDIDATE_SETS, default='original')
    parser.add_argument('--save-inner-predictions', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    SPECS = CANDIDATE_SETS[args.candidate_set]
    if not args.local_gpu_ack or args.max_seconds <= 0:
        parser.error('Explicit local GPU use and a positive hard time limit are required')
    if args.worker:
        run(args)
        return
    if args.output_dir.exists():
        parser.error('Output folder must not exist')
    started = time.monotonic()
    watchdog = {'max_seconds': args.max_seconds, 'status': 'running'}
    try:
        result = subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), *sys.argv[1:], '--worker'],
                                timeout=args.max_seconds, check=False)
        watchdog.update(status='passed' if result.returncode == 0 else 'failed', exit_code=result.returncode)
    except subprocess.TimeoutExpired as error:
        watchdog.update(status='timed_out_process_stopped', exit_code=124)
        receipt_path = args.output_dir / 'execution.json'
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            receipt.update(status='failed', exit_code=124, error='Hard time limit; training process stopped')
            write_json(receipt_path, receipt)
        raise RuntimeError('Nested GPU search timed out; process stopped') from error
    finally:
        watchdog['elapsed_seconds'] = time.monotonic() - started
        if args.output_dir.is_dir():
            write_json(args.output_dir / 'watchdog.json', watchdog)
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
