"""Recompute research metrics, audit nested selection, compare independent runs."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import calibration_diagnostics as cd
from model_upgrade import identity
from research_runtime import digest, execution_record, write_json
from upgrade_audit import comparison_rows, metrics


def verify_one(run_dir, data_dir, split_dir):
    run_dir = Path(run_dir)
    if json.loads((run_dir / 'execution.json').read_text())['status'] != 'passed':
        raise ValueError('Incomplete run')
    hashes = json.loads((run_dir / 'artifact_manifest.json').read_text())
    for name, value in hashes.items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or digest(run_dir / path) != value:
            raise ValueError('Artifact integrity failure: ' + name)
    plan = json.loads((run_dir / 'experiment_plan.json').read_text())
    data, _, sources = cd.load_development(data_dir, split_dir)
    if sources != plan['sources'] or digest(Path(split_dir) / 'split_manifest.csv') != plan['split_sha256']:
        raise ValueError('Input/split mismatch')
    base = identity(data)
    columns = base.columns.tolist()
    keys = ['machine', 'source_row_id']
    oof = pd.read_csv(run_dir / 'oof_predictions.csv')
    expected_models = {*plan['specs'], 'nested_route', 'prior'}
    if set(oof.model) != expected_models or set(oof.seed) != set(plan['seeds']):
        raise ValueError('Candidate/seed coverage mismatch')
    recomputed = []
    for seed in plan['seeds']:
        scores = {}
        for model in expected_models:
            part = oof.loc[oof.seed.eq(seed) & oof.model.eq(model)]
            if part.duplicated(keys).any():
                raise ValueError('Duplicate OOF key')
            pd.testing.assert_frame_equal(part[columns].reset_index(drop=True), base, check_dtype=False)
            scores[model] = part.score.to_numpy()
        cn = base.machine.eq('cn7').to_numpy()
        if not np.array_equal(scores['baseline'][cn], scores['nested_route'][cn]):
            raise ValueError('CN7 changed')
        recomputed.extend(comparison_rows(base, scores, seed))
    saved = pd.read_csv(run_dir / 'model_comparison.csv')
    sort = ['seed', 'model', 'fold', 'machine', 'policy', 'fraction']
    pd.testing.assert_frame_equal(saved.sort_values(sort).reset_index(drop=True),
                                  pd.DataFrame(recomputed).sort_values(sort).reset_index(drop=True),
                                  check_dtype=False, atol=1e-12, rtol=0)
    membership = pd.read_csv(run_dir / 'inner_split_manifest.csv')
    inner = pd.read_csv(run_dir / 'inner_oof_predictions.csv')
    selections = pd.read_csv(run_dir / 'selected_candidates.csv')
    inner_metrics = pd.read_csv(run_dir / 'inner_candidate_metrics.csv')
    audited_selections = 0
    for seed in plan['seeds']:
        for outer in range(5):
            rows = membership.loc[membership.seed.eq(seed) & membership.outer_test_fold.eq(outer)]
            expected = base.loc[base.outer_fold.ne(outer)].reset_index(drop=True)
            pd.testing.assert_frame_equal(rows[columns].reset_index(drop=True), expected, check_dtype=False)
            if rows.groupby('feature_group').inner_fold.nunique().max() != 1:
                raise ValueError('Inner group split')
            outer_groups = set(base.loc[base.outer_fold.eq(outer), 'feature_group'])
            if set(rows.feature_group) & outer_groups:
                raise ValueError('Outer validation entered inner selection')
            computed = []
            for name in plan['specs']:
                p = inner.loc[inner.seed.eq(seed) & inner.outer_test_fold.eq(outer) & inner.candidate.eq(name)]
                pd.testing.assert_frame_equal(p[columns+['inner_fold']].reset_index(drop=True),
                                              rows[columns+['inner_fold']].reset_index(drop=True), check_dtype=False)
                rg = p.machine.eq('rg3')
                result = metrics(p.loc[rg], p.loc[rg, 'score'].to_numpy())
                saved_metric = inner_metrics.loc[inner_metrics.seed.eq(seed) & inner_metrics.outer_fold.eq(outer)
                                                 & inner_metrics.candidate.eq(name)]
                if len(saved_metric) != 1:
                    raise ValueError('Inner metric coverage')
                for key, value in result.items():
                    if value is not None and not np.isclose(saved_metric.iloc[0][key], value, rtol=0, atol=1e-12):
                        raise ValueError('Inner metric mismatch')
                computed.append((result['found'], result['average_precision'], -list(plan['specs']).index(name), name))
            winner = max(computed)[-1]
            chosen = selections.loc[selections.seed.eq(seed) & selections.outer_fold.eq(outer)]
            if len(chosen) != 1 or chosen.iloc[0].winner != winner:
                raise ValueError('Inner selection mismatch')
            routed = oof.loc[oof.seed.eq(seed) & oof.outer_fold.eq(outer) & oof.model.eq('nested_route')]
            expected_route = pd.concat([
                oof.loc[oof.seed.eq(seed) & oof.outer_fold.eq(outer) & oof.model.eq('baseline') & oof.machine.eq('cn7')],
                oof.loc[oof.seed.eq(seed) & oof.outer_fold.eq(outer) & oof.model.eq(winner) & oof.machine.eq('rg3')]])
            np.testing.assert_allclose(routed.sort_values(keys).score, expected_route.sort_values(keys).score, rtol=0, atol=1e-12)
            audited_selections += 1
    protected = json.loads((run_dir / 'protected_artifacts_before.json').read_text())
    for path, expected in protected.items():
        from research_runtime import ROOT
        if digest(ROOT / path) != expected:
            raise ValueError('Original artifact changed: ' + path)
    decision = json.loads((run_dir / 'decision.json').read_text())
    primary = oof.loc[oof.seed.eq(plan['seeds'][0])]
    for model, key in [('baseline', 'baseline_machine_policy'), ('nested_route', 'nested_route_machine_policy')]:
        part = primary.loc[primary.model.eq(model)]
        for name, value in metrics(part, part.score.to_numpy()).items():
            if not np.isclose(value, decision[key][name], rtol=0, atol=1e-12):
                raise ValueError('Decision metric mismatch')
    final_selection_audited = False
    if decision['bundle']:
        if decision['nested_route_machine_policy']['found'] <= decision['baseline_machine_policy']['found']:
            raise ValueError('Research export gate not met')
        final_membership = pd.read_csv(run_dir / 'final_inner_split.csv')
        pd.testing.assert_frame_equal(final_membership[columns], base, check_dtype=False)
        if final_membership.groupby('feature_group').inner_fold.nunique().max() != 1:
            raise ValueError('Final inner split overlap')
        final_oof = pd.read_csv(run_dir / 'final_inner_oof.csv')
        candidates = []
        for index, name in enumerate(plan['specs']):
            part = final_oof.loc[final_oof.candidate.eq(name)]
            pd.testing.assert_frame_equal(part[columns+['inner_fold']].reset_index(drop=True),
                                          final_membership[columns+['inner_fold']], check_dtype=False)
            rg = part.machine.eq('rg3')
            m = metrics(part.loc[rg], part.loc[rg, 'score'].to_numpy())
            candidates.append((m['found'], m['average_precision'], -index, name))
        winner = max(candidates)[-1]
        bundle = json.loads((run_dir / 'bundle_candidate/bundle.json').read_text())
        if winner != decision['bundle']['winner'] or winner != bundle['routing']['rg3']:
            raise ValueError('Final bundle selection mismatch')
        if bundle['field_approved'] is not False or decision['field_approved'] is not False:
            raise ValueError('Research was improperly approved')
        final_selection_audited = True
    return {'status': 'passed', 'artifact_files': len(hashes), 'metric_rows_recomputed': len(recomputed),
            'inner_selections_recomputed': audited_selections, 'original_artifacts_unchanged': len(protected),
            'full_development_bundle_selection_recomputed': final_selection_audited,
            'oof_rows': len(oof), 'inner_oof_rows': len(inner), 'hashes': hashes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first', type=Path, required=True)
    parser.add_argument('--second', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--split-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.first.resolve() == args.second.resolve():
        parser.error('Two distinct run directories required')
    with execution_record(args.output_dir):
        first = verify_one(args.first, args.data_dir, args.split_dir)
        second = verify_one(args.second, args.data_dir, args.split_dir)
        if first['hashes'] != second['hashes']:
            mismatch = sorted(k for k in first['hashes'].keys() | second['hashes'].keys()
                              if first['hashes'].get(k) != second['hashes'].get(k))
            raise ValueError('Independent execution artifacts differ: ' + ', '.join(mismatch))
        report = {'status': 'passed', 'first': str(args.first), 'second': str(args.second),
                  'scope': 'two_separate_local_CPU_processes_same_environment_not_future_validation',
                  'first_checks': first, 'second_checks': second, 'all_scientific_artifacts_byte_identical': True}
        write_json(args.output_dir / 'reproduction_report.json', report)
        print(json.dumps({k: v for k, v in report.items() if k not in ('first_checks', 'second_checks')}, indent=2))


if __name__ == '__main__':
    main()
