"""Reference portability may rename a group, but cannot change its members."""

import pandas as pd
import pytest

import moldguard as mg
from research_split import align_group_fingerprints


def example():
    data = pd.DataFrame({'machine': ['cn7'] * 4, mg.ID_COL: [0, 1, 2, 3],
                         mg.TARGET: [0, 1, 0, 0],
                         'feature_group': ['local_a', 'local_a', 'local_b', 'local_c']})
    reference = data.rename(columns={mg.ID_COL: 'source_row_id', mg.TARGET: 'label_value'}).copy()
    reference['feature_group'] = ['ref_a', 'ref_a', 'ref_b', 'ref_c']
    return data, reference


def test_reference_group_names_preserve_members_labels_and_row_order():
    data, reference = example()
    result = align_group_fingerprints(data, reference.iloc[::-1])
    assert result.feature_group.tolist() == ['ref_a', 'ref_a', 'ref_b', 'ref_c']
    pd.testing.assert_frame_equal(result.drop(columns='feature_group'), data.drop(columns='feature_group'))


@pytest.mark.parametrize('groups', [
    ['ref_a', 'ref_other', 'ref_b', 'ref_c'],
    ['ref_a', 'ref_a', 'ref_a', 'ref_c'],
])
def test_reference_cannot_split_or_merge_groups(groups):
    data, reference = example()
    reference['feature_group'] = groups
    with pytest.raises(ValueError, match='merges or splits'):
        align_group_fingerprints(data, reference)


def test_reference_cannot_change_labels():
    data, reference = example()
    reference.loc[1, 'label_value'] = 0
    with pytest.raises(ValueError, match='labels differ'):
        align_group_fingerprints(data, reference)
