import numpy as np
import pandas as pd
import pytest

from export_task01_gpu_summary import export_table
from task01_gpu_calibrated_ensemble import paired_rank
from task01_gpu_balance_review import tie_bounds


def paired_scores():
    records = []
    for candidate in ('cat_row_d3', 'cat_event_d3'):
        for row_id, score in enumerate((.2, .2, .7)):
            records.append(dict(machine='rg3', source_row_id=row_id, feature_group=f'g{row_id // 2}',
                                outer_fold=0, inner_fold=1, label_value=row_id % 2,
                                candidate=candidate, score=score))
    return pd.DataFrame(records)


def test_rank_combination_preserves_ties_and_ignores_labels():
    original = paired_scores()
    first = paired_rank(original, 'candidate', 'inner_fold')
    original['label_value'] = 1 - original.label_value
    second = paired_rank(original, 'candidate', 'inner_fold')
    np.testing.assert_array_equal(first.score, second.score)
    assert first.score.iloc[0] == first.score.iloc[1]


def test_pair_combination_rejects_missing_prediction():
    with pytest.raises(ValueError, match='Incomplete paired scores'):
        paired_rank(paired_scores().iloc[:-1], 'candidate', 'inner_fold')


def test_public_export_rejects_row_level_identity(tmp_path):
    private = tmp_path / 'private.csv'
    public = tmp_path / 'public.csv'
    pd.DataFrame({'source_row_id': [1], 'score': [.5], 'label_value': [1]}).to_csv(private, index=False)
    with pytest.raises(ValueError, match='Non-aggregate schema blocked'):
        export_table(private, public)
    assert not public.exists()


def test_boundary_ties_report_both_possible_positive_counts():
    frame = paired_scores().loc[lambda x: x.candidate.eq('cat_row_d3')]
    assert tie_bounds(frame, .5) == (0, 1)
