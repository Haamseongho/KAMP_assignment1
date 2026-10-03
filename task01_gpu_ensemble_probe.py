"""Explicit exploratory combinations of completed OOF models, never a new test."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import f1_score

import calibration_diagnostics as cd
from model_upgrade import identity
from research_runtime import digest, execution_record, write_json
from task01_gpu_nested_search import SPECS
from task01_gpu_oof_evaluate import validate_frozen_contract
from upgrade_audit import metrics, select, tie_rows


FAMILIES = {
    'xgb_rows': ['fixed_xgb_row_d1', 'fixed_xgb_row_d2', 'fixed_xgb_row_d3'],
    'xgb_events': ['fixed_xgb_event_d1', 'fixed_xgb_event_d2', 'fixed_xgb_event_d3'],
    'cat_pair': ['fixed_cat_row_d3', 'fixed_cat_event_d3'],
    'event_trees': ['fixed_event_forest', 'fixed_xgb_event_d1', 'fixed_xgb_event_d2', 'fixed_xgb_event_d3', 'fixed_cat_event_d3'],
    'all_trees': ['fixed_event_forest', *['fixed_' + name for name, spec in SPECS.items() if spec['kind'] in {'xgb', 'cat'}]],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--split-dir', type=Path, required=True)
    parser.add_argument('--result-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    with execution_record(args.output_dir):
        source = args.result_dir
        if json.loads((source / 'execution.json').read_text())['status'] != 'passed':
            raise ValueError('Incomplete source run')
        hashes = json.loads((source / 'artifact_hashes.json').read_text())
        for name, expected_hash in hashes.items():
            if Path(name).name != name or digest(source / name) != expected_hash:
                raise ValueError('Source artifact hash mismatch')
        data, _, provenance = cd.load_development(args.data_dir, args.split_dir)
        validate_frozen_contract(data, provenance, digest(args.split_dir / 'split_manifest.csv'))
        base = identity(data)
        specs = json.loads((source / 'experiment_plan.json').read_text())['specs']
        available = {'fixed_' + name for name in specs}
        families = {name: members for name, members in FAMILIES.items() if set(members) <= available}
        families['all_trees'] = ['fixed_' + name for name, spec in specs.items() if spec['kind'] in {'xgb', 'cat', 'forest'}]
        if 'cat_row_engineered' in specs:
            families.update(cat_shallow_pair=['fixed_cat_row_d2_reg30', 'fixed_cat_event_d2_reg30'],
                            cat_engineered_pair=['fixed_cat_row_engineered', 'fixed_cat_event_engineered'],
                            all_cat=['fixed_' + name for name, spec in specs.items() if spec['kind'] == 'cat'])
        plan = {'status': 'exploratory_plan_before_combination_scoring',
                'selection_caveat': 'new combinations proposed after reading development outcomes; outer OOF is not independent validation',
                'cn7_heads': ['baseline', 'fixed_local_logistic_c1'], 'rg3_families': families,
                'aggregation': ['raw_mean', 'within_outer_fold_rank_mean'],
                'fractions': [.1, .15, .2, .3, .5, 1.],
                'source_oof_sha256': digest(source / 'oof_predictions.csv'),
                'split_sha256': digest(args.split_dir / 'split_manifest.csv'),
                'new_fits': 0, 'adoption': False, 'independent_validation': False}
        write_json(args.output_dir / 'experiment_plan.json', plan)
        original = pd.read_csv(source / 'oof_predictions.csv', float_precision='round_trip')
        summaries, ties, predictions = [], [], []
        for seed in sorted(original.seed.unique()):
            values = {}
            for model, part in original.loc[original.seed.eq(seed)].groupby('model'):
                merged = base.merge(part[['machine', 'source_row_id', 'label_value', 'score']],
                                    on=['machine', 'source_row_id', 'label_value'], validate='one_to_one', sort=False)
                if len(merged) != len(base) or not np.isfinite(merged.score).all():
                    raise ValueError('OOF identity or score coverage differs')
                values[model] = merged.score.to_numpy()
            rg3 = base.machine.eq('rg3').to_numpy()
            candidates = {name: values[name][rg3] for name in values if name.startswith('fixed_')}
            for family, names in families.items():
                candidates[family + '_mean'] = np.mean([values[name][rg3] for name in names], axis=0)
                ranks = []
                for name in names:
                    rank = np.empty(int(rg3.sum()))
                    folds = base.loc[rg3, 'outer_fold'].to_numpy()
                    for fold in range(5):
                        positions = np.flatnonzero(folds == fold)
                        rank[positions] = rankdata(values[name][rg3][positions], method='average') / len(positions)
                    ranks.append(rank)
                candidates[family + '_rank_mean'] = np.mean(ranks, axis=0)
            for head in plan['cn7_heads']:
                for tail, score in candidates.items():
                    model = f'{head}__rg3_{tail}'
                    combined = values[head].copy()
                    combined[rg3] = score
                    row = base.copy()
                    row['model'], row['seed'], row['score'] = model, int(seed), combined
                    predictions.append(row)
                    for machine in ('all', 'cn7', 'rg3'):
                        mask = np.ones(len(base), dtype=bool) if machine == 'all' else base.machine.eq(machine).to_numpy()
                        local, s = base.loc[mask], combined[mask]
                        for fraction in plan['fractions']:
                            summary = metrics(local, s, fraction, 'P_MACHINE')
                            chosen = select(local, s, fraction, 'P_MACHINE')
                            independent = np.zeros(len(local), dtype=bool)
                            for m in local.machine.unique():
                                pos = np.flatnonzero(local.machine.eq(m))
                                ids = local.iloc[pos].source_row_id.to_numpy()
                                order = pos[np.lexsort((ids, -s[pos]))]
                                independent[order[:int(np.ceil(len(pos) * fraction))]] = True
                            if not np.array_equal(chosen, independent):
                                raise ValueError('Independent policy selection differs')
                            y = local.label_value.to_numpy()
                            tp = int(y[chosen].sum())
                            fp, fn = int(chosen.sum()) - tp, int(y.sum()) - tp
                            tn = len(y) - int(y.sum()) - fp
                            summaries.append({'model': model, 'seed': int(seed), 'machine': machine,
                                              'fraction': fraction, **summary, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
                                              'f1': float(f1_score(y, chosen, zero_division=0)),
                                              'policy_derived_accuracy': (tp + tn) / len(y)})
                    ties.extend(tie_rows(base, combined, model, int(seed)))
        table = pd.DataFrame(summaries)
        table.to_csv(args.output_dir / 'policy_summary.csv', index=False)
        pd.DataFrame(ties).to_csv(args.output_dir / 'tie_audit.csv', index=False)
        pd.concat(predictions, ignore_index=True).to_csv(args.output_dir / 'oof_predictions.csv', index=False)
        score = table.loc[table.machine.eq('all') & table.fraction.eq(.1)]
        aggregate = score.groupby('model').tp.agg(['min', 'mean', 'max']).sort_values(['min', 'mean'], ascending=False)
        aggregate.to_csv(args.output_dir / 'seed_summary.csv')
        write_json(args.output_dir / 'verification.json', {
            'status': 'passed_source_hashes_and_two_independent_topk_implementations',
            'metric_rows': len(table), 'new_fits': 0, 'independent_future_validation': False})
        write_json(args.output_dir / 'artifact_hashes.json', {
            p.name: digest(p) for p in args.output_dir.iterdir() if p.is_file() and p.name not in {'execution.json', 'run.log', 'artifact_hashes.json'}})
        print(aggregate.head(12).to_string())


if __name__ == '__main__':
    main()
