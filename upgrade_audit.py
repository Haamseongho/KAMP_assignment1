"""Explicit inspection policies, tie diagnostics and paired group uncertainty."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def ranking(frame, scores):
    if len(frame) != len(scores) or not np.isfinite(scores).all():
        raise ValueError('Invalid score coverage')
    table = frame[['machine', 'source_row_id']].copy().reset_index(drop=True)
    table['score'] = scores
    # Original positional index is only the last fallback for bootstrap copies.
    return table.sort_values(['score', 'machine', 'source_row_id'],
                             ascending=[False, True, True], kind='stable').index.to_numpy()


def select(frame, scores, fraction=.1, policy='P_MACHINE'):
    if policy not in ('P_MACHINE', 'P_GLOBAL') or not 0 < fraction <= 1:
        raise ValueError('Unsupported policy/fraction')
    selected = np.zeros(len(frame), dtype=bool)
    groups = [np.arange(len(frame))] if policy == 'P_GLOBAL' else [
        np.flatnonzero(frame.machine.eq(m)) for m in sorted(frame.machine.unique())]
    for positions in groups:
        if len(positions):
            order = ranking(frame.iloc[positions], np.asarray(scores)[positions])
            selected[positions[order[:int(np.ceil(len(positions) * fraction))]]] = True
    return selected


def metrics(frame, scores, fraction=.1, policy='P_MACHINE'):
    y = frame.label_value.to_numpy(dtype=int)
    chosen = select(frame, scores, fraction, policy)
    positives, k = int(y.sum()), int(chosen.sum())
    found = int(y[chosen].sum())
    return {'rows': len(frame), 'groups': int(frame.feature_group.nunique()),
            'positives': positives, 'prevalence': float(y.mean()), 'k': k, 'found': found,
            'missed': positives-found, 'selected_negative': k-found,
            'recall': found / positives if positives else None,
            'precision': found / k if k else None,
            'lift': (found / k) / y.mean() if positives and k else None,
            'average_precision': float(average_precision_score(y, scores)) if positives else None,
            'roc_auc': float(roc_auc_score(y, scores)) if len(set(y)) == 2 else None}


def comparison_rows(base, predictions, seed):
    result = []
    for model, scores in predictions.items():
        for fold in [-1, *sorted(base.outer_fold.unique())]:
            for machine in ('all', 'cn7', 'rg3'):
                mask = np.ones(len(base), dtype=bool)
                if fold != -1:
                    mask &= base.outer_fold.eq(fold).to_numpy()
                if machine != 'all':
                    mask &= base.machine.eq(machine).to_numpy()
                part = base.loc[mask]
                for policy in ('P_GLOBAL', 'P_MACHINE'):
                    for fraction in (.1, .2):
                        result.append({'model': model, 'seed': seed, 'fold': int(fold),
                                       'machine': machine, 'policy': policy, 'fraction': fraction,
                                       'scope': 'pooled_oof' if fold == -1 else 'within_fold',
                                       **metrics(part, scores[mask], fraction, policy)})
    return result


def tie_rows(base, scores, model, seed):
    result = []
    for machine in ('all', 'cn7', 'rg3'):
        pos = np.arange(len(base)) if machine == 'all' else np.flatnonzero(base.machine.eq(machine))
        frame, s = base.iloc[pos], scores[pos]
        y = frame.label_value.to_numpy()
        for fraction in (.1, .2):
            k = int(np.ceil(len(pos) * fraction))
            boundary = s[ranking(frame, s)[k-1]]
            above, tied = s > boundary, s == boundary
            a, b, t, q = int(above.sum()), int(y[above].sum()), int(tied.sum()), int(y[tied].sum())
            r = k-a
            result.append({'model': model, 'seed': seed, 'machine': machine, 'fraction': fraction,
                           'k': k, 'boundary_score': float(boundary), 'above_rows': a,
                           'above_positives': b, 'tied_rows': t, 'tied_positives': q, 'remaining': r,
                           'minimum_found': b+max(0, r-(t-q)), 'expected_found': b+r*q/t,
                           'maximum_found': b+min(r, q), 'unique_scores': len(np.unique(s)),
                           'interpretation': 'exact_tie_sensitivity_not_confidence_interval'})
    return result


def paired_bootstrap(base, baseline, candidate, repeats, seed):
    """Machine-stratified cluster bootstrap; redraw K and rankings per replicate."""
    rng = np.random.default_rng(seed)
    machine_groups = {m: list(base.loc[base.machine.eq(m)].groupby('feature_group').indices.values())
                      for m in ('cn7', 'rg3')}
    machine_positions = {m: np.flatnonzero(base.machine.eq(m)) for m in machine_groups}
    values = {(m, p, metric): [] for m in ('all', 'cn7', 'rg3')
              for p in ('P_GLOBAL', 'P_MACHINE') for metric in ('recall', 'average_precision')}
    invalid = {key: 0 for key in values}
    for _ in range(repeats):
        drawn = []
        for m, groups in machine_groups.items():
            local = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
            drawn.append(machine_positions[m][local])
        positions = np.concatenate(drawn)
        for machine in ('all', 'cn7', 'rg3'):
            pos = positions if machine == 'all' else positions[base.machine.iloc[positions].eq(machine)]
            frame = base.iloc[pos]
            y = frame.label_value.to_numpy()
            for policy in ('P_GLOBAL', 'P_MACHINE'):
                if y.sum() == 0:
                    for metric in ('recall', 'average_precision'):
                        invalid[machine, policy, metric] += 1
                    continue
                for metric in ('recall', 'average_precision'):
                    if metric == 'recall':
                        delta = (y[select(frame, candidate[pos], .1, policy)].sum()
                                 - y[select(frame, baseline[pos], .1, policy)].sum()) / y.sum()
                    else:
                        delta = average_precision_score(y, candidate[pos])-average_precision_score(y, baseline[pos])
                    values[machine, policy, metric].append(float(delta))
    rows = []
    for (machine, policy, metric), sample in values.items():
        mask = np.ones(len(base), dtype=bool) if machine == 'all' else base.machine.eq(machine).to_numpy()
        b = metrics(base.loc[mask], baseline[mask], .1, policy)
        c = metrics(base.loc[mask], candidate[mask], .1, policy)
        rows.append({'machine': machine, 'policy': policy, 'metric': metric, 'seed': seed,
                     'difference': c[metric]-b[metric], 'lower95': np.quantile(sample, .025) if sample else None,
                     'upper95': np.quantile(sample, .975) if sample else None, 'valid_repeats': len(sample),
                     'invalid_repeats': invalid[machine, policy, metric], 'requested_repeats': repeats,
                     'scope': 'conditional_fixed_oof_not_retraining_or_model_search_uncertainty'})
    return rows
