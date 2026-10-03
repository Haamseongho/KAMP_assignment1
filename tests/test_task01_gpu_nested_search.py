"""Selection uses training scores and augmented rows never enter evaluation."""
import numpy as np
import pandas as pd

import moldguard as mg
import task01_gpu_nested_search as search
from research_split import FEATURES


def test_full_recall_threshold_is_learned_from_training_positives():
    y = np.array([1, 0, 1, 0])
    scores = np.array([.9, .8, .7, .1])
    result = search.thresholds(y, scores)
    assert result['high_recall_threshold'] == .7
    assert np.all((scores >= result['high_recall_threshold'])[y == 1])
    assert result['inner_f1'] == .8


def test_oversampling_keeps_original_labels_and_does_not_mutate_source():
    source = pd.DataFrame({'feature_group': ['a', 'a', 'b'], mg.TARGET: [0, 1, 0]})
    result = search.training_view(source, {'oversample': 5})
    assert source[mg.TARGET].tolist() == [0, 1, 0]
    assert len(result) == 7
    assert result[mg.TARGET].sum() == 5
    assert result.loc[result[mg.TARGET].eq(1), 'feature_group'].nunique() == 1


def test_identical_inputs_share_one_exact_prediction():
    source = pd.DataFrame({name: [1., 1., 2.] for name in FEATURES})
    source['machine'] = 'rg3'
    calls = []

    class Model:
        def predict_proba(self, x):
            calls.append(len(x))
            assert mg.ID_COL not in x and mg.TARGET not in x and 'feature_group' not in x
            return np.array([[.2, .8], [.3, .7]])

    scores = search.unique_predictions(Model(), source, 'xgb')
    assert calls == [2]
    assert scores.tolist() == [.8, .8, .7]


def test_guarded_route_rejects_more_captures_with_lower_ap():
    def row(name, found, ap):
        return {'candidate': name, 'found': found, 'average_precision': ap, 'roc_auc': .8,
                'inner_f1': .2, 'f1_recall': .5, 'high_recall_precision': .1}
    baseline = row('pooled_logistic_ref', 3, .2)
    candidate = row('xgb_row_d1', 5, .1)
    choice = search.choose_candidates([baseline, candidate])
    assert choice['nested_recall'] == 'xgb_row_d1'
    assert choice['nested_guarded'] == 'pooled_logistic_ref'


def test_engineered_features_do_not_use_labels_and_handle_zero_denominators():
    frame = pd.DataFrame({name: [0., 1.] for name in FEATURES})
    frame['machine'] = 'rg3'
    frame[mg.TARGET] = [0, 1]
    frame[mg.ID_COL] = [7, 8]
    first = search.candidate_matrix(frame, {'engineered': True})
    frame[mg.TARGET] = [1, 0]
    frame[mg.ID_COL] = [99, 100]
    second = search.candidate_matrix(frame, {'engineered': True})
    pd.testing.assert_frame_equal(first, second)
    assert first.shape[1] == len(FEATURES) + 9
    assert np.isfinite(first.to_numpy()).all()
