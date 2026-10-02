"""Bounded, nested-group CPU research. Never replaces published artifacts."""
import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

import calibration_diagnostics as cd
import moldguard as mg
from research_runtime import ROOT, digest, execution_record, write_json
from research_split import FEATURES
from upgrade_models import candidates, fit_model, predict, export_component, ResearchBundle
from upgrade_audit import metrics, select, ranking, comparison_rows, tie_rows, paired_bootstrap

SEEDS = (20260922, 20260923, 20260924)
CONDITIONS = ('Clamp_Close_Time', 'Max_Injection_Pressure', 'Max_Switch_Over_Pressure')


def inner_folds(frame, seed):
    groups = mg.group_table(frame)
    # Fixed adaptive rule uses only the current outer-training stratum counts.
    count = min(3, int(groups.stratum.value_counts().min()))
    if count < 2:
        raise ValueError('Too few independent groups for inner selection')
    splitter = StratifiedKFold(count, shuffle=True, random_state=seed)
    result = []
    for fit, valid in splitter.split(groups, groups.stratum):
        a = np.flatnonzero(frame.feature_group.isin(groups.index[fit]))
        b = np.flatnonzero(frame.feature_group.isin(groups.index[valid]))
        if set(frame.feature_group.iloc[a]) & set(frame.feature_group.iloc[b]):
            raise ValueError('Inner group leakage')
        for m in ('cn7', 'rg3'):
            for positions in (a, b):
                part = frame.iloc[positions]
                if part.loc[part.machine.eq(m), mg.TARGET].nunique() != 2:
                    raise ValueError('Insufficient machine/class coverage in inner fold')
        result.append((a, b))
    return result


def identity(frame):
    return frame[['machine', mg.ID_COL, 'feature_group', 'outer_fold', mg.TARGET]].rename(
        columns={mg.ID_COL: 'source_row_id', mg.TARGET: 'label_value'}).reset_index(drop=True)


def selection(frame, specs, seed, deadline):
    folds = inner_folds(frame, seed)
    scores = {name: np.full(len(frame), np.nan) for name in specs}
    assignments = np.full(len(frame), -1)
    for inner, (fit, valid) in enumerate(folds):
        assignments[valid] = inner
        training, validation = frame.iloc[fit], frame.iloc[valid]
        for name, spec in specs.items():
            if time.monotonic() > deadline:
                raise TimeoutError('Predeclared CPU experiment time limit reached')
            model = fit_model(training, spec, seed)
            scores[name][valid] = predict(model, validation, spec, scores['baseline'][valid])
    base = identity(frame)
    mask = frame.machine.eq('rg3').to_numpy()
    rows = []
    for name, s in scores.items():
        if not np.isfinite(s).all():
            raise ValueError('Incomplete inner predictions')
        rows.append({'candidate': name, **metrics(base.loc[mask], s[mask])})
    # All candidates contribute RG3 only; CN7 is fixed to the outer-fold baseline.
    # Prefer simpler baseline on an exact objective/AP tie, then frozen name order.
    priority = {name: i for i, name in enumerate(specs)}
    winner = max(rows, key=lambda r: (r['found'], r['average_precision'], -priority[r['candidate']]))['candidate']
    for row in rows:
        row['selected'] = row['candidate'] == winner
    return winner, rows, scores, assignments


def contract_report(data, provenance, output):
    groups = data.groupby(['machine', 'feature_group'])[mg.TARGET].agg(['size', 'sum', 'nunique'])
    groups['conflicting'] = groups['nunique'].gt(1)
    groups.reset_index().to_csv(output / 'development_group_audit.csv', index=False)
    rows = []
    for machine in ('cn7', 'rg3'):
        g = groups.loc[machine]
        rows.append({'machine': machine, 'development_rows': int(g['size'].sum()),
                     'positive_rows': int(g['sum'].sum()), 'groups': len(g),
                     'conflicting_groups': int(g.conflicting.sum()),
                     'positive_rows_in_conflicting_groups': int(g.loc[g.conflicting, 'sum'].sum()),
                     'group_size_counts': {str(k): int(v) for k, v in g['size'].value_counts().items()}})
    write_json(output / 'data_contract_review.json', {
        'scope': 'development_only_no_old_holdout_performance', 'by_machine': rows, 'sources': provenance,
        'unconfirmed': ['numeric_label_meaning', 'shot_product_cavity_keys', 'timestamps_and_cutoff',
                        'supplier_scaler_fit_scope', 'raw_units', 'unused_future_labels'],
        'confirmed': ['frozen_source_hashes', 'exact_feature_group_separation', '24_feature_schema'],
        'prohibited': ['label_flipping', 'conflict_deletion', 'row_id_as_feature', 'inspection_skip']})
    (output / 'data_contract_review.md').write_text(
        '# 데이터 계약 확인\n\n숫자 라벨 1 의미, 샷/캐비티, 예측 시각, 원시 단위, 공급자 정규화는 미확정이다.\n'
        '현재 파일·분할 해시와 동일 입력 그룹 분리만 검증했다. 개발 영역 외 홀드아웃 성능은 사용하지 않는다.\n\n'
        + '\n'.join(f'- {r["machine"]}: 개발 {r["development_rows"]}행, 양성 {r["positive_rows"]}행, '
                    f'상충 그룹 {r["conflicting_groups"]}개, 그 안의 양성 {r["positive_rows_in_conflicting_groups"]}행.' for r in rows)
        + '\n\n상충 라벨은 원형 유지. 추가 정보 없이는 불량 의미·현장 성능을 확정하지 않는다.\n')
    return data.groupby('feature_group')[mg.TARGET].transform('nunique').gt(1).to_numpy()


def training_bands(data, features):
    audit = identity(data)
    audit['ood_count'] = 0
    boundaries = []
    for fold in range(5):
        for machine in ('cn7', 'rg3'):
            fit = data.loc[data.outer_fold.ne(fold) & data.machine.eq(machine)]
            mask = data.outer_fold.eq(fold) & data.machine.eq(machine)
            lo, hi = fit[features].quantile(.001), fit[features].quantile(.999)
            audit.loc[mask, 'ood_count'] = ((data.loc[mask, features] < lo) | (data.loc[mask, features] > hi)).sum(axis=1)
            for col in CONDITIONS:
                median = float(fit[col].median())
                audit.loc[mask, 'band_' + col] = np.where(data.loc[mask, col] > median, 'gt_train_median', 'le_train_median')
                boundaries.append({'outer_fold': fold, 'machine': machine, 'feature': col, 'median': median,
                                   'fit_rows': len(fit), 'source': 'outer_training_only'})
    audit['ood_flag'] = audit.ood_count.ge(3)
    return audit, boundaries


def diagnostics(base, scores, seed, output_rows, failure_rows, rank_rows, allocation_rows):
    for name, s in scores.items():
        output_rows.extend(tie_rows(base, s, name, seed))
        for policy in ('P_GLOBAL', 'P_MACHINE'):
            chosen = select(base, s, .1, policy)
            for fold in [-1, *range(5)]:
                for machine in ('cn7', 'rg3'):
                    mask = base.machine.eq(machine).to_numpy()
                    if fold != -1:
                        mask &= base.outer_fold.eq(fold).to_numpy()
                    allocation_rows.append({'model': name, 'seed': seed, 'policy': policy, 'fold': fold,
                        'machine': machine, 'rows': int(mask.sum()), 'selected': int((mask & chosen).sum()),
                        'found': int(base.loc[mask & chosen, 'label_value'].sum()),
                        'missed': int(base.loc[mask & ~chosen, 'label_value'].sum()),
                        'score_q10': float(np.quantile(s[mask], .1)), 'score_q50': float(np.quantile(s[mask], .5)),
                        'score_q90': float(np.quantile(s[mask], .9)),
                        'selection_scope': 'pooled_oof_selection_then_slice_not_fold_reallocation'})
        chosen = select(base, s)
        for col in ['conflicting_label_group', 'ood_flag', *['band_' + c for c in CONDITIONS]]:
            for (machine, value), part in base.groupby(['machine', col]):
                pos = part.index.to_numpy()
                y = part.label_value.to_numpy()
                selected = chosen[pos]
                failure_rows.append({'model': name, 'seed': seed, 'machine': machine, 'condition': col,
                    'value': str(value), 'rows': len(pos), 'groups': part.feature_group.nunique(),
                    'positives': int(y.sum()), 'selected': int(selected.sum()),
                    'found': int(y[selected].sum()), 'missed': int(y[~selected].sum()),
                    'selected_negative': int((1-y[selected]).sum()), 'policy': 'P_MACHINE',
                    'interpretation': 'topk_selection_not_threshold_confusion_matrix'})
        machine_pos = np.flatnonzero(base.machine.eq('rg3'))
        local_order = ranking(base.iloc[machine_pos], s[machine_pos])
        ranks = np.empty(len(machine_pos), dtype=int)
        ranks[local_order] = np.arange(1, len(machine_pos)+1)
        selected20 = select(base, s, .2)
        threshold = s[machine_pos[local_order[int(np.ceil(len(machine_pos)*.1))-1]]]
        for local, pos in enumerate(machine_pos):
            if base.label_value.iloc[pos] != 1:
                continue
            rank_rows.append({**base.iloc[pos].to_dict(), 'model': name, 'seed': seed,
                              'score': float(s[pos]), 'rank_in_machine': int(ranks[local]),
                              'selected_top10': bool(chosen[pos]), 'selected_top20': bool(selected20[pos]),
                              'boundary_tie_flag': bool(s[pos] == threshold)})


def export_bundle(output, data, specs, seed, deadline, plan, data_dir):
    winner, rows, final_scores, assignments = selection(data, specs, seed, deadline)
    pd.DataFrame(rows).to_csv(output / 'final_inner_selection.csv', index=False)
    member = identity(data)
    member['inner_fold'] = assignments
    member.to_csv(output / 'final_inner_split.csv', index=False)
    inner_records = []
    for name, scores in final_scores.items():
        part = member.copy()
        part['candidate'], part['score'] = name, scores
        inner_records.append(part)
    pd.concat(inner_records, ignore_index=True).to_csv(output / 'final_inner_oof.csv', index=False)
    reference = fit_model(data, specs['baseline'], seed)
    candidate = reference if winner == 'baseline' else fit_model(data, specs[winner], seed)
    bundle = output / 'bundle_candidate'
    bundle.mkdir()
    export_component(reference, specs['baseline'], bundle / 'cn7')
    export_component(candidate, specs[winner], bundle / 'rg3')
    files = {str(p.relative_to(bundle)): digest(p) for p in sorted(bundle.rglob('*')) if p.is_file()}
    unlabeled, _, inference_sources = mg.load_data(data_dir, False)
    write_json(bundle / 'bundle.json', {'schema': 'kamp-upgrade-research-v1', 'field_approved': False,
        'score_semantics': 'numeric_label_1_research_score_not_defect_probability',
        'features': FEATURES, 'routing': {'cn7': 'baseline', 'rg3': winner}, 'policy': 'P_MACHINE',
        'files': files, 'split_sha256': plan['split_sha256'], 'sources': plan['sources'],
        'inference_sources': inference_sources,
        'seed': seed, 'fit_scope': '1913_development_rows_no_old_holdout',
        'selection_scope': 'full_development_inner_group_cv_not_outer_oof_winner',
        'runtime': {'python': '3.12', 'cpu_threads': 1}, 'label_mapping': 'unconfirmed',
        'ood_bounds': {m: {bound: data.loc[data.machine.eq(m), FEATURES].quantile(q).to_dict()
                          for bound, q in [('low', .001), ('high', .999)]} for m in ('cn7', 'rg3')}})
    native = predict(reference, data, specs['baseline'])
    candidate_scores = predict(candidate, data, specs[winner], native)
    native[data.machine.eq('rg3')] = candidate_scores[data.machine.eq('rg3')]
    actual = ResearchBundle(bundle).predict(data)
    error = float(np.max(np.abs(native-actual)))
    if error > 1e-12:
        raise ValueError('New bundle does not match NEW native model')
    ref = identity(data)
    ref['native_score'] = native
    ref.to_csv(output / 'bundle_reference_predictions.csv', index=False)
    native_unlabeled = predict(reference, unlabeled, specs['baseline'])
    candidate_unlabeled = predict(candidate, unlabeled, specs[winner], native_unlabeled)
    mask = unlabeled.machine.eq('rg3')
    native_unlabeled[mask] = candidate_unlabeled[mask]
    exported = ResearchBundle(bundle).predict(unlabeled)
    unseen_error = float(np.max(np.abs(native_unlabeled-exported)))
    if unseen_error > 1e-12:
        raise ValueError('Unlabeled native/bundle mismatch')
    ref = unlabeled[['machine', mg.ID_COL]].rename(columns={mg.ID_COL: 'source_row_id'})
    ref['native_score'] = native_unlabeled
    ref.to_csv(output / 'unlabeled_native_predictions.csv', index=False)
    return {'winner': winner, 'native_bundle_max_error': error,
            'unlabeled_native_bundle_max_error': unseen_error, 'unlabeled_rows': len(unlabeled),
            'scope': 'native_vs_bundle_numeric_parity_not_performance'}


def run(args):
    deadline = time.monotonic()+args.max_seconds
    with execution_record(args.output_dir) as receipt, threadpool_limits(limits=1):
        out = args.output_dir
        specs = candidates()
        data, features, sources = cd.load_development(args.data_dir, args.split_dir)
        artifact_source = ROOT / 'outputs/service_dev_20260926/original_artifacts.json'
        old = json.loads(artifact_source.read_text()) if artifact_source.exists() else {}
        # A missing historical inventory is explicit, never a false preservation pass.
        for name, expected in old.items():
            if digest(ROOT / name) != expected:
                raise ValueError('Existing historical artifact changed before run: ' + name)
        protected = {str(p.relative_to(ROOT)): digest(p) for folder in (
            ROOT / 'outputs/moldguard', ROOT / 'outputs/paas_cpu_20260926/bundle_final')
                     for p in folder.rglob('*') if p.is_file()}
        plan = {'schema': 'kamp-upgrade-plan-v1', 'status': 'frozen_before_fitting',
            'sources': sources, 'split_sha256': digest(args.split_dir / 'split_manifest.csv'),
            'addendum_sha256': digest(args.addendum), 'seeds': list(SEEDS), 'specs': specs,
            'primary': 'P_MACHINE numeric-positive capture at 10pct budget',
            'inner_selection': 'RG3 found@10pct then RG3 AP then frozen candidate order',
            'cn7': 'exact outer-fold baseline predictions preserved in nested_route',
            'tie_rule': 'score descending, machine ascending, numeric source_row_id ascending',
            'outer_folds': 5, 'inner_folds': 'min(3, smallest outer-training group-stratum count), minimum 2',
            'max_seconds': args.max_seconds, 'cpu_threads': 1, 'bootstrap_repeats': args.bootstrap,
            'operational_acceptance': {'status': 'not_agreed', 'min_gain': None, 'max_overall_ap_loss': None},
            'research_export_rule': 'primary-seed nested_route finds more at same machine budget; CN7 identical; NOT deployment approval',
            'old_holdout': 'not_used_for_training_selection_or_performance',
            'e6_blend': 'deferred: test routing first; no mixture search without evidence',
            'no_external_upload_no_submission_no_gpu': True}
        write_json(out / 'experiment_plan.json', plan)
        shutil.copyfile(args.split_dir / 'split_manifest.csv', out / 'split_manifest.csv')
        shutil.copyfile(args.addendum, out / 'source_addendum.md')
        write_json(out / 'protected_artifacts_before.json', {**old, **protected})
        conflict = contract_report(data, sources, out)
        base, boundaries = training_bands(data, features)
        base['conflicting_label_group'] = conflict
        pd.DataFrame(boundaries).to_csv(out / 'condition_boundaries.csv', index=False)
        comparisons, oof, choices, memberships, inners, inner_predictions = [], [], [], [], [], []
        ties, failures, ranks, allocations, uncertainty = [], [], [], [], []
        all_scores = {}
        for seed in SEEDS:
            scores = {name: np.full(len(data), np.nan) for name in [*specs, 'nested_route']}
            scores['prior'] = np.full(len(data), np.nan)
            for outer in range(5):
                train = data.outer_fold.ne(outer).to_numpy()
                valid = ~train
                training, validation = data.loc[train].reset_index(drop=True), data.loc[valid]
                if set(training.feature_group) & set(validation.feature_group):
                    raise ValueError('Outer group overlap')
                winner, rows, inner_scores, assignments = selection(training, specs, seed, deadline)
                member = identity(training)
                member['inner_fold'], member['outer_test_fold'], member['seed'] = assignments, outer, seed
                memberships.append(member)
                for name, s in inner_scores.items():
                    inner_frame = member.copy()
                    inner_frame['candidate'], inner_frame['score'] = name, s
                    inner_predictions.append(inner_frame)
                choices.append({'seed': seed, 'outer_fold': outer, 'winner': winner,
                                'train_rows': len(training), 'valid_rows': len(validation), 'group_overlap': 0})
                inners.extend({**r, 'seed': seed, 'outer_fold': outer} for r in rows)
                for name, spec in specs.items():
                    if time.monotonic() > deadline:
                        raise TimeoutError('Predeclared CPU experiment time limit reached')
                    model = fit_model(training, spec, seed)
                    scores[name][valid] = predict(model, validation, spec, scores['baseline'][valid])
                scores['nested_route'][valid] = scores['baseline'][valid]
                rg_valid = valid & data.machine.eq('rg3').to_numpy()
                scores['nested_route'][rg_valid] = scores[winner][rg_valid]
                scores['prior'][valid] = training[mg.TARGET].mean()
                print(f'seed={seed} outer={outer} RG3 selected={winner}; train={len(training)} valid={len(validation)}', flush=True)
            if any(not np.isfinite(s).all() for s in scores.values()):
                raise ValueError('OOF missing predictions')
            cn = data.machine.eq('cn7').to_numpy()
            if not np.array_equal(scores['baseline'][cn], scores['nested_route'][cn]):
                raise ValueError('CN7 reference preservation failed')
            # Verify against row-keyed historical predictions, not just a headline AP.
            historical = pd.read_csv(ROOT / 'outputs/task01_compare_20260923_cpu_v3/oof_predictions.csv')
            historical = historical.loc[historical.model.eq('logistic_ref') & historical.seed.eq(seed)]
            merged = base.merge(historical[['machine', 'source_row_id', 'label_value', 'score']],
                                on=['machine', 'source_row_id', 'label_value'], validate='one_to_one', how='left')
            if merged.score.isna().any() or not np.allclose(merged.score, scores['baseline'], rtol=0, atol=1e-12):
                raise ValueError('Historical baseline OOF parity failed')
            comparisons.extend(comparison_rows(base, scores, seed))
            diagnostics(base, scores, seed, ties, failures, ranks, allocations)
            for name, s in scores.items():
                frame = base.copy()
                frame['model'], frame['seed'], frame['score'] = name, seed, s
                oof.append(frame)
            all_scores[seed] = scores
            # Save completed seeds before bootstrap; failed runs remain explicitly failed.
            pd.concat(oof, ignore_index=True).to_csv(out / 'oof_predictions.csv', index=False)
            pd.DataFrame(comparisons).to_csv(out / 'model_comparison.csv', index=False)
        pd.concat(memberships, ignore_index=True).to_csv(out / 'inner_split_manifest.csv', index=False)
        pd.concat(inner_predictions, ignore_index=True).to_csv(out / 'inner_oof_predictions.csv', index=False)
        pd.DataFrame(choices).to_csv(out / 'selected_candidates.csv', index=False)
        pd.DataFrame(inners).to_csv(out / 'inner_candidate_metrics.csv', index=False)
        pd.DataFrame(ties).to_csv(out / 'topk_tie_audit.csv', index=False)
        pd.DataFrame(failures).to_csv(out / 'failure_slices.csv', index=False)
        pd.DataFrame(ranks).to_csv(out / 'rg3_positive_rank_audit.csv', index=False)
        pd.DataFrame(allocations).to_csv(out / 'global_allocation_audit.csv', index=False)
        # Primary seed predeclared; other seeds are stability diagnostics, not best-seed selection.
        primary = all_scores[SEEDS[0]]
        print('Starting paired group bootstrap for predeclared primary seed', flush=True)
        uncertainty = paired_bootstrap(base, primary['baseline'], primary['nested_route'], args.bootstrap, SEEDS[0])
        pd.DataFrame(uncertainty).to_csv(out / 'bootstrap_differences.csv', index=False)
        if time.monotonic() > deadline:
            raise TimeoutError('Time limit reached after bootstrap; no candidate exported')
        random_rows = []
        rng = np.random.default_rng(SEEDS[0])
        # Uniform row selection is the null for this row-budget research policy.
        # It does not assume that rows are independent manufacturing outcomes.
        for machine in ('cn7', 'rg3'):
            part = base.loc[base.machine.eq(machine)]
            n, positive = len(part), int(part.label_value.sum())
            k = int(np.ceil(n*.1))
            draws = rng.hypergeometric(positive, n-positive, k, size=10000)
            mask = base.machine.eq(machine).to_numpy()
            observed = metrics(part, primary['nested_route'][mask])['found']
            random_rows.append({'machine': machine, 'rows': n, 'positives': positive, 'k': k,
                'random_expected_found': k*positive/n, 'random_lower95': float(np.quantile(draws, .025)),
                'random_upper95': float(np.quantile(draws, .975)), 'nested_found': observed,
                'random_ge_observed_fraction': float((draws >= observed).mean()),
                'repeats': 10000, 'interpretation': 'fixed_cohort_uniform_row_selection_reference_not_generalization_test'})
        pd.DataFrame(random_rows).to_csv(out / 'random_policy_reference.csv', index=False)
        b, c = metrics(base, primary['baseline']), metrics(base, primary['nested_route'])
        export = None
        if c['found'] > b['found']:
            export = export_bundle(out, data, specs, SEEDS[0], deadline, plan, args.data_dir)
        decision = {'research_status': 'promising_development_candidate' if export else 'no_primary_capture_gain_keep_baseline',
                    'baseline_machine_policy': b, 'nested_route_machine_policy': c, 'bundle': export,
                    'field_approved': False, 'label_meaning': 'unconfirmed',
                    'independent_future_validation': 'not_available', 'operational_acceptance': 'not_agreed',
                    'candidate_tables': 'fixed_candidates_exploratory_not_posthoc_selection',
                    'primary_seed': SEEDS[0], 'external_execution': 'not_performed'}
        write_json(out / 'decision.json', decision)
        (out / 'decision.md').write_text('# 연구 후보 판정\n\n'
            f'- 판정: `{decision["research_status"]}`\n'
            f'- 같은 설비별 10% 예산: 기준선 {b["found"]}/{b["positives"]}, 중첩 선택 {c["found"]}/{c["positives"]}.\n'
            '- CN7은 fold별 기준선 예측을 유지했다. 전체 AP와 seed별 결과는 비교표에 함께 보존했다.\n'
            '- 새 번들이 있더라도 개발 연구용이며 실제 불량 확률·현장 합격을 뜻하지 않는다.\n'
            '- 새 기간 라벨 및 업무상 승인 기준 미확보. 검사 생략·대회 제출·외부 업로드 없음.\n')
        preserved = {**old, **protected}
        for name, expected in preserved.items():
            if digest(ROOT / name) != expected:
                raise ValueError('Protected artifact changed: ' + name)
        write_json(out / 'preservation_report.json', {'status': 'passed', 'files': len(preserved),
                    'historical_inventory_available': bool(old), 'source_hashes': sources})
        write_json(out / 'artifact_manifest.json', {str(p.relative_to(out)): digest(p)
            for p in sorted(out.rglob('*')) if p.is_file() and p.name not in ('run.log', 'execution.json', 'artifact_manifest.json')})
        receipt['scope'] = 'local_cpu_development_research_not_remote_or_independent_validation'
        print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--split-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--addendum', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--max-seconds', type=int, default=1800)
    args = parser.parse_args()
    if args.bootstrap < 1 or args.max_seconds < 1:
        parser.error('Positive bootstrap/time limits required')
    run(args)


if __name__ == '__main__':
    main()
