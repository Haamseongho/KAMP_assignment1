"""Aggregate confirmed local experiments, uncertainty and inspection tradeoffs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from research_runtime import digest, execution_record, write_json
from upgrade_audit import paired_bootstrap


CANDIDATE = 'fixed_local_logistic_c1__rg3_cat_pair_rank_mean'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nested-dir', type=Path, required=True)
    parser.add_argument('--ensemble-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=1000)
    args = parser.parse_args()
    with execution_record(args.output_dir):
        for source in (args.nested_dir, args.ensemble_dir):
            if json.loads((source / 'execution.json').read_text())['status'] != 'passed':
                raise ValueError('An input run did not finish')
            for name, expected in json.loads((source / 'artifact_hashes.json').read_text()).items():
                if Path(name).name != name or digest(source / name) != expected:
                    raise ValueError('Source artifact mismatch')
        nested = pd.read_csv(args.nested_dir / 'policy_summary.csv')
        ensemble = pd.read_csv(args.ensemble_dir / 'policy_summary.csv')
        baseline = nested.loc[nested.model.eq('baseline')].copy()
        candidate = ensemble.loc[ensemble.model.eq(CANDIDATE)].copy()
        comparison = pd.concat([baseline, candidate], ignore_index=True)
        comparison.loc[comparison.fraction.eq(.1)].to_csv(args.output_dir / 'comparison_at_k192.csv', index=False)
        high = pd.read_csv(args.nested_dir / 'classification_summary.csv')
        high.loc[high.model.eq('baseline') & high.policy.eq('high_recall_decision')].to_csv(
            args.output_dir / 'full_recall_tradeoff.csv', index=False)
        old = pd.read_csv(args.nested_dir / 'oof_predictions.csv', float_precision='round_trip')
        new = pd.read_csv(args.ensemble_dir / 'oof_predictions.csv', float_precision='round_trip')
        base = old.loc[old.model.eq('baseline') & old.seed.eq(20260922)].reset_index(drop=True)
        new = new.loc[new.model.eq(CANDIDATE)]
        aligned = base.merge(new.loc[new.seed.eq(20260922), ['machine', 'source_row_id', 'score']],
                             on=['machine', 'source_row_id'], validate='one_to_one', suffixes=('', '_candidate'), sort=False)
        uncertainty = paired_bootstrap(base, base.score.to_numpy(), aligned.score_candidate.to_numpy(),
                                       args.bootstrap, 20260922)
        pd.DataFrame(uncertainty).to_csv(args.output_dir / 'conditional_group_bootstrap.csv', index=False)
        # The three-seed mean is a separate exploratory ensemble, not the best seed.
        from upgrade_audit import metrics, select
        from sklearn.metrics import f1_score
        keys = ['machine', 'source_row_id', 'label_value', 'feature_group', 'outer_fold']
        mean = new.groupby(keys, sort=False).score.mean().reset_index()
        rows = []
        for machine in ('all', 'cn7', 'rg3'):
            local = mean if machine == 'all' else mean.loc[mean.machine.eq(machine)]
            row = metrics(local, local.score.to_numpy())
            row['f1'] = float(f1_score(local.label_value, select(local, local.score.to_numpy()), zero_division=0))
            rows.append({'machine': machine, **row})
        pd.DataFrame(rows).to_csv(args.output_dir / 'three_seed_mean_candidate.csv', index=False)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
        for name, source, color in [('CPU reference', baseline, '#546375'), ('Exploratory GPU combination', candidate, '#007f87')]:
            points = source.loc[source.machine.eq('all')].groupby(['fraction', 'k'])
            stats = points.agg(tp_mean=('tp', 'mean'), tp_min=('tp', 'min'), tp_max=('tp', 'max'),
                               recall=('recall', 'mean'), precision=('precision', 'mean')).reset_index().sort_values('k')
            axes[0].plot(stats.k, stats.tp_mean, 'o-', color=color, label=name)
            axes[0].fill_between(stats.k, stats.tp_min, stats.tp_max, color=color, alpha=.13)
            axes[1].plot(stats.recall, stats.precision, 'o-', color=color, label=name)
        axes[0].axhline(33, color='#c05249', linestyle='--', linewidth=1, label='33 numeric positives')
        axes[0].axvline(192, color='#888888', linestyle=':', linewidth=1)
        axes[0].set(xlabel='Inspection budget (rows)', ylabel='Positives recovered (of 33)', ylim=(0, 35))
        axes[1].set(xlabel='Recall', ylabel='Precision', xlim=(0, 1.02), ylim=(0, .13))
        for ax in axes:
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        fig.suptitle('Development OOF: inspection cost and positive recovery\nShading: three-seed range; no new-period independent test', fontsize=11)
        fig.savefig(args.output_dir / 'inspection_tradeoff.png', dpi=180)
        plt.close(fig)
        at_k = candidate.loc[candidate.machine.eq('all') & candidate.fraction.eq(.1)]
        bounds = {'numeric_positives': 33, 'same_input_opposite_label_pairs': 29,
                  'unavoidable_false_positives_at_100pct_recall_for_feature_only_rule': 29,
                  'precision_upper_bound_at_full_recall_even_with_oracle_group_detection': 33 / 62,
                  'f1_upper_bound_at_full_recall_even_with_oracle_group_detection': 66 / 95,
                  'precision_upper_bound_at_k192': 33 / 192,
                  'f1_upper_bound_at_k192': 66 / 225,
                  'interpretation': 'identifiability bounds, not an achieved model score'}
        write_json(args.output_dir / 'identifiability_bounds.json', bounds)
        write_json(args.output_dir / 'decision.json', {
            'status': 'measured_development_improvement_with_remaining_requirements',
            'candidate': CANDIDATE, 'found_by_seed': {str(int(r.seed)): int(r.tp) for r in at_k.itertuples()},
            'same_k192_all_seed_improvement_over_reference': bool(at_k.tp.gt(12).all()),
            'all_seeds_at_least_17': bool(at_k.tp.ge(17).all()), 'all_seeds_33_at_k192': bool(at_k.tp.eq(33).all()),
            'three_seed_mean_ensemble_found_at_k192': int(rows[0]['found']),
            'candidate_selection': 'exploratory after inspecting development OOF; not nested procedure adoption',
            'conditional_bootstrap_caution': 'fixed OOF and post-hoc candidate; does not account for search or future generalization',
            'independent_new_period_validation': False, 'field_approved': False,
            'ground_truth_positives_increased': False,
            'remaining': ['stable 17/33 at K192', '33/33 with acceptable precision and F1',
                          'new independent labels', 'shot/product/cavity metadata', 'label semantics']})
        write_json(args.output_dir / 'artifact_hashes.json', {
            p.name: digest(p) for p in args.output_dir.iterdir() if p.is_file() and p.name not in {'run.log', 'execution.json', 'artifact_hashes.json'}})
        print(at_k[['seed', 'k', 'tp', 'recall', 'precision', 'f1']].to_string(index=False))


if __name__ == '__main__':
    main()
