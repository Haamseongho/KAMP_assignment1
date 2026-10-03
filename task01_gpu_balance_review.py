"""Compare inspection budgets using verified OOF; prefer fewer false positives.

The policy choice is exploratory on exposed development data, not a deployment
threshold. Original and repeated GPU pair runs are assessed together.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from export_task01_gpu_summary import checked_run
from research_runtime import digest, execution_record, write_json
from task01_gpu_result_review import CANDIDATE
from upgrade_audit import metrics, select
from verify_task01_gpu_nested import independent_counts


FRACTIONS = (.02, .03, .05, .075, .10, .15)


def tie_bounds(part, fraction):
    lower = upper = 0
    for _, group in part.groupby('machine'):
        k = int(np.ceil(len(group) * fraction))
        boundary = np.sort(group.score.to_numpy())[-k]
        above, tied = group.loc[group.score.gt(boundary)], group.loc[group.score.eq(boundary)]
        places = k - len(above)
        positive, negative = int(tied.label_value.sum()), int((tied.label_value == 0).sum())
        lower += int(above.label_value.sum()) + max(0, places - negative)
        upper += int(above.label_value.sum()) + min(places, positive)
    return lower, upper


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outputs-root', type=Path, default=Path('outputs'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    sources = {
        'cpu_reference': ('gpu_pc_nested_20261004_01', 'baseline'),
        'gpu_pair_original': ('gpu_pc_ensemble_20261004_01', CANDIDATE),
        'gpu_pair_repeat': ('gpu_pc_refinement_ensemble_20261004_01', CANDIDATE),
        'inner_calibrated_pair': ('gpu_pc_calibrated_20261004_01', 'inner_calibrated_gpu_pair'),
    }
    for folder, _ in sources.values():
        checked_run(args.outputs_root / folder)
    with execution_record(args.output_dir):
        write_json(args.output_dir / 'experiment_plan.json', {
            'fractions': FRACTIONS, 'sources': sources,
            'policy': 'P_MACHINE; fixed share for each machine',
            'selection_rule': 'smallest budget retaining at least reference 12 positives in every seed of both GPU pair runs, with strictly fewer false positives and higher precision/F1 than CPU at K192',
            'selection_scope': 'post-hoc development inspection; no fresh test or field approval',
            'new_fits': 0})
        records, predictions, mean_records = [], [], []
        for name, (folder, model) in sources.items():
            table = pd.read_csv(args.outputs_root / folder / 'oof_predictions.csv', float_precision='round_trip')
            table = table.loc[table.model.eq(model)].copy()
            if len(table) != 1913 * 3 or table.duplicated(['seed', 'machine', 'source_row_id']).any():
                raise ValueError('Balance source OOF coverage differs')
            for seed, part in table.groupby('seed'):
                for fraction in FRACTIONS:
                    for machine in ('all', 'cn7', 'rg3'):
                        local = part if machine == 'all' else part.loc[part.machine.eq(machine)]
                        selected = select(local, local.score.to_numpy(), fraction, 'P_MACHINE')
                        counts = independent_counts(local.label_value, selected)
                        lower, upper = tie_bounds(local, fraction)
                        records.append({'model': name, 'seed': int(seed), 'machine': machine, 'fraction': fraction,
                                        **metrics(local, local.score.to_numpy(), fraction, 'P_MACHINE'), **counts,
                                        'tie_min_tp': lower, 'tie_max_tp': upper})
            keys = ['machine', 'source_row_id', 'feature_group', 'outer_fold', 'label_value']
            mean = table.groupby(keys, sort=False).score.mean().reset_index()
            mean['model'] = name
            predictions.append(mean)
            for fraction in FRACTIONS:
                selected = select(mean, mean.score.to_numpy(), fraction, 'P_MACHINE')
                lower, upper = tie_bounds(mean, fraction)
                mean_records.append({'model': name, 'fraction': fraction,
                                     **independent_counts(mean.label_value, selected),
                                     'tie_min_tp': lower, 'tie_max_tp': upper})
        result = pd.DataFrame(records)
        means = pd.DataFrame(mean_records)
        eligible = []
        for fraction in FRACTIONS:
            part = result.loc[result.model.isin(['gpu_pair_original', 'gpu_pair_repeat'])
                              & result.machine.eq('all') & result.fraction.eq(fraction)]
            mean = means.loc[means.model.isin(['gpu_pair_original', 'gpu_pair_repeat']) & means.fraction.eq(fraction)]
            if (len(part) == 6 and part.tp.ge(12).all() and part.fp.lt(180).all()
                    and part.precision.gt(12 / 192).all() and part.f1.gt(24 / 225).all()
                    and mean.tp.ge(12).all()):
                eligible.append(fraction)
        recommendation = min(eligible) if eligible else None
        result.to_csv(args.output_dir / 'budget_comparison.csv', index=False)
        means.to_csv(args.output_dir / 'three_seed_mean_budgets.csv', index=False)
        if recommendation is not None:
            result.loc[result.fraction.eq(recommendation)].to_csv(args.output_dir / 'recommended_policy.csv', index=False)
        selected = result.loc[result.machine.eq('all') & result.model.isin(['gpu_pair_original', 'gpu_pair_repeat'])
                              & result.fraction.eq(recommendation)]
        decision = {'status': 'exploratory_balanced_budget_found' if recommendation is not None else 'no_budget_meets_balance_rule',
                    'recommended_fraction': recommendation, 'policy': 'P_MACHINE',
                    'basis': 'retain reference K192 positive recovery while reducing false positives; both GPU runs and their three-seed means',
                    'lowest_observed_tp': int(selected.tp.min()) if len(selected) else None,
                    'lowest_tp_under_boundary_tie_permutations': int(selected.tie_min_tp.min()) if len(selected) else None,
                    'independent_future_validation': False, 'field_approved': False,
                    'all_33_recovered': False, 'new_ground_truth_positives_created': 0}
        write_json(args.output_dir / 'decision.json', decision)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
        for name, color in [('cpu_reference', '#637387'), ('gpu_pair_original', '#007f87'), ('gpu_pair_repeat', '#bd7142')]:
            points = result.loc[result.model.eq(name) & result.machine.eq('all')].groupby('fraction').agg(
                fp=('fp', 'mean'), tp=('tp', 'mean'), precision=('precision', 'mean'), recall=('recall', 'mean')).reset_index()
            axes[0].plot(points.fp, points.tp, 'o-', label=name, color=color)
            axes[1].plot(points.recall, points.precision, 'o-', label=name, color=color)
        axes[0].set(xlabel='False positives (mean across three seeds)', ylabel='True positives recovered')
        axes[1].set(xlabel='Recall', ylabel='Precision')
        for ax in axes:
            ax.grid(alpha=.2)
            ax.legend(fontsize=7)
        fig.suptitle('Development OOF: balance false alarms and positive recovery\nPolicy budget chosen after development inspection', fontsize=10)
        fig.savefig(args.output_dir / 'balance_tradeoff.png', dpi=180)
        plt.close(fig)
        write_json(args.output_dir / 'verification.json', {
            'status': 'passed_source_hashes_and_independent_confusion_counts', 'metric_rows': len(result),
            'oof_mean_rows': sum(len(p) for p in predictions), 'boundary_tie_bounds_computed': True,
            'source_oof_sha256': {name: digest(args.outputs_root / folder / 'oof_predictions.csv') for name, (folder, _) in sources.items()},
            'scope': 'fixed verified OOF and preset budget arithmetic, no independent performance validation'})
        write_json(args.output_dir / 'artifact_hashes.json', {
            p.name: digest(p) for p in args.output_dir.iterdir() if p.is_file() and p.name not in {'execution.json', 'run.log', 'artifact_hashes.json'}})
        print(json.dumps(decision, indent=2))
        print(selected[['model', 'seed', 'k', 'tp', 'fp', 'recall', 'precision', 'f1', 'accuracy', 'tie_min_tp', 'tie_max_tp']].to_string(index=False))


if __name__ == '__main__':
    main()
