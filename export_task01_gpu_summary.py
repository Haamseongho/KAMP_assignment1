"""Publish only reviewed aggregate tables, a plot, hashes and verification counts.

Raw CSVs, row-level predictions, split membership and fitted models stay local.
Run only after both nested searches and their separate verifiers have passed.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil

import pandas as pd

from research_runtime import digest, write_json


AGGREGATE_COLUMNS = {
    'model', 'seed', 'machine', 'fraction', 'policy', 'rows', 'groups', 'positives',
    'prevalence', 'k', 'found', 'missed', 'selected_negative', 'recall', 'precision',
    'lift', 'average_precision', 'roc_auc', 'tp', 'fp', 'fn', 'tn', 'f1', 'accuracy',
    'policy_derived_accuracy', 'metric', 'difference', 'lower95', 'upper95',
    'valid_repeats', 'invalid_repeats', 'requested_repeats', 'scope',
    'tie_min_tp', 'tie_max_tp',
}


def checked_run(folder):
    receipt = json.loads((folder / 'execution.json').read_text())
    if receipt['status'] != 'passed':
        raise ValueError(f'Incomplete run: {folder.name}')
    hashes_file = folder / 'artifact_hashes.json'
    hashes = json.loads(hashes_file.read_text()) if hashes_file.exists() else receipt['output_hashes']
    for name, expected in hashes.items():
        if Path(name).name != name or digest(folder / name) != expected:
            raise ValueError(f'Artifact mismatch: {folder.name}/{name}')
    return receipt


def export_table(source, destination, top10=False):
    table = pd.read_csv(source, float_precision='round_trip')
    if not set(table) <= AGGREGATE_COLUMNS:
        raise ValueError(f'Non-aggregate schema blocked: {source.name}')
    if top10:
        table = table.loc[table.fraction.eq(.1)]
    table.to_csv(destination, index=False, lineterminator='\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outputs-root', type=Path, default=Path('outputs'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    root, output = args.outputs_root, args.output_dir
    folders = {name: root / folder for name, folder in {
        'gpu': 'gpu_pc_gpu_20261004_01',
        'gpu_verify': 'gpu_pc_gpu_verify_20261004_02',
        'nested': 'gpu_pc_nested_20261004_01',
        'nested_verify': 'gpu_pc_nested_verify_20261004_02',
        'ensemble': 'gpu_pc_ensemble_20261004_01',
        'review': 'gpu_pc_review_20261004_01',
        'bundle': 'gpu_pc_candidate_20261004_01',
        'refinement': 'gpu_pc_refinement_20261004_01',
        'refinement_verify': 'gpu_pc_refinement_verify_20261004_01',
        'refinement_ensemble': 'gpu_pc_refinement_ensemble_20261004_01',
        'calibrated': 'gpu_pc_calibrated_20261004_01',
        'balance': 'gpu_pc_balance_20261004_01',
        'balanced_queue': 'gpu_pc_balanced_queue_20261004_01',
    }.items()}
    for folder in folders.values():
        checked_run(folder)
    output.mkdir(parents=True, exist_ok=False)
    for name in ('comparison_at_k192', 'full_recall_tradeoff', 'conditional_group_bootstrap', 'three_seed_mean_candidate'):
        export_table(folders['review'] / f'{name}.csv', output / f'{name}.csv')
    export_table(folders['refinement'] / 'policy_summary.csv', output / 'refinement_at_k192.csv', top10=True)
    export_table(folders['refinement_ensemble'] / 'policy_summary.csv', output / 'refinement_combinations_at_k192.csv', top10=True)
    export_table(folders['calibrated'] / 'policy_summary.csv', output / 'calibrated_at_k192.csv', top10=True)
    for name in ('budget_comparison', 'three_seed_mean_budgets', 'recommended_policy'):
        export_table(folders['balance'] / f'{name}.csv', output / f'{name}.csv')
    write_json(output / 'balanced_decision.json', json.loads((folders['balance'] / 'decision.json').read_text()))
    shutil.copyfile(folders['balance'] / 'balance_tradeoff.png', output / 'balance_tradeoff.png')
    original = pd.read_csv(folders['ensemble'] / 'policy_summary.csv')
    candidate = json.loads((folders['review'] / 'decision.json').read_text())['candidate']
    original = original.loc[original.model.eq(candidate)]
    if not set(original) <= AGGREGATE_COLUMNS:
        raise ValueError('Inspection tradeoff schema blocked')
    original.to_csv(output / 'inspection_tradeoff.csv', index=False, lineterminator='\n')
    for name in ('decision', 'identifiability_bounds'):
        document = json.loads((folders['review'] / f'{name}.json').read_text())
        if name == 'decision':
            document['final_research_policy'] = json.loads((folders['balance'] / 'decision.json').read_text())
            document['final_scope'] = 'user-prioritized false-positive/precision balance; K144 policy, exploratory development evidence'
        write_json(output / f'{name}.json', document)
    shutil.copyfile(folders['review'] / 'inspection_tradeoff.png', output / 'inspection_tradeoff.png')
    verifications = {}
    for name in ('nested_verify', 'refinement_verify'):
        verifications[name] = json.loads((folders[name] / 'verification.json').read_text())
    bundle = json.loads((folders['bundle'] / 'verification.json').read_text())
    verifications['bundle'] = {key: bundle[key] for key in (
        'status', 'gpu_fits', 'queue_rows', 'queue_duplicate_ids', 'queue_score_finite',
        'original_development_positives', 'new_ground_truth_positives_created', 'scope')}
    verifications['gpu_comparison'] = {key: json.loads((folders['gpu_verify'] / 'verification.json').read_text())[key]
                                       for key in ('status', 'development_rows', 'development_groups', 'oof_rows', 'metric_rows_recomputed', 'scope')}
    verifications['calibrated'] = json.loads((folders['calibrated'] / 'verification.json').read_text())
    verifications['balance'] = json.loads((folders['balance'] / 'verification.json').read_text())
    queue = json.loads((folders['balanced_queue'] / 'verification.json').read_text())
    verifications['balanced_queue'] = {key: value for key, value in queue.items() if key != 'unlabeled_sources'}
    verifications['source_run_manifests_sha256'] = {
        name: digest(folder / 'run_manifest.json') for name, folder in folders.items() if (folder / 'run_manifest.json').exists()}
    verifications['source_plans_sha256'] = {
        name: digest(folder / 'experiment_plan.json') for name, folder in folders.items() if (folder / 'experiment_plan.json').exists()}
    verifications['official_data_gpu_fits_total'] = 60 + verifications['nested_verify']['gpu_fits_verified'] + verifications['refinement_verify']['gpu_fits_verified'] + bundle['gpu_fits']
    verifications['independent_new_period_performance_validation'] = False
    write_json(output / 'verification.json', verifications)
    write_json(output / 'artifact_hashes.json', {p.name: digest(p) for p in output.iterdir() if p.is_file()})
    print(f'Aggregate-only report exported: {output}; GPU fits={verifications["official_data_gpu_fits_total"]}')


if __name__ == '__main__':
    main()
