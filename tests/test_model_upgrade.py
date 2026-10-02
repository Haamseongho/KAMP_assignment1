"""Calculation fixtures are TEST_ONLY; performance uses official data only."""
import json
import numpy as np
import pandas as pd
import pytest

import calibration_diagnostics as cd
import moldguard as mg
from model_upgrade import inner_folds, identity
from research_runtime import digest, write_json
from research_split import FEATURES
from upgrade_audit import select, metrics, tie_rows, paired_bootstrap
from upgrade_models import candidates, fit_model, predict, export_component, ResearchBundle, matrix


def test_policy_budget_and_exact_tie_bounds():
    # TEST_ONLY ranking arithmetic, not fabricated manufacturing measurements.
    frame = pd.DataFrame({'machine': ['cn7']*3 + ['rg3']*3, 'source_row_id': [0, 1, 2]*2,
                          'label_value': [1, 0, 0, 0, 1, 0],
                          'feature_group': [f'TEST_ONLY:{i}' for i in range(6)]})
    scores = np.array([.9, .8, .7, .6, .6, .6])
    assert select(frame, scores, .1, 'P_GLOBAL').sum() == 1
    assert select(frame, scores, .1, 'P_MACHINE').sum() == 2
    tied = [r for r in tie_rows(frame, scores, 'test', 0) if r['machine'] == 'rg3' and r['fraction'] == .1][0]
    assert tied['minimum_found'] == 0 and tied['maximum_found'] == 1
    assert tied['expected_found'] == pytest.approx(1/3)
    shuffled = frame.sample(frac=1, random_state=42)
    chosen = select(shuffled, scores[shuffled.index], .1)
    assert set(zip(shuffled.loc[chosen, 'machine'], shuffled.loc[chosen, 'source_row_id'])) == {('cn7', 0), ('rg3', 0)}


def test_no_positive_is_unavailable_not_zero_performance():
    frame = pd.DataFrame({'machine': ['rg3']*2, 'source_row_id': [0, 1],
                          'label_value': [0, 0], 'feature_group': ['TEST_ONLY:0', 'TEST_ONLY:1']})
    row = metrics(frame, np.array([.1, .2]))
    assert row['recall'] is None and row['average_precision'] is None and row['roc_auc'] is None


def test_paired_bootstrap_identical_scores_zero_difference():
    frame = pd.DataFrame({'machine': ['cn7']*4 + ['rg3']*4, 'source_row_id': list(range(4))*2,
                          'label_value': [0, 1, 0, 0]*2,
                          'feature_group': [f'TEST_ONLY:{i//2}' for i in range(8)]})
    scores = np.array([.3, .3, .1, .1]*2)
    rows = paired_bootstrap(frame, scores, scores, repeats=20, seed=2)
    assert all(r['difference'] == 0 for r in rows)
    assert all(r['lower95'] == 0 and r['upper95'] == 0 for r in rows)
    assert all(r['valid_repeats'] + r['invalid_repeats'] == 20 for r in rows)
    assert any(r['invalid_repeats'] > 0 for r in rows)


@pytest.fixture
def development():
    return cd.load_development(mg.DATA_DIR, cd.FROZEN)[0]


@pytest.mark.integration
def test_nested_groups_and_frozen_reference(development):
    assert len(development) == 1913
    for outer in range(5):
        train = development[development.outer_fold.ne(outer)].reset_index(drop=True)
        outer_groups = set(development.loc[development.outer_fold.eq(outer), 'feature_group'])
        covered = np.zeros(len(train), dtype=int)
        for fit, valid in inner_folds(train, mg.SEED):
            assert not (set(train.feature_group.iloc[fit]) & set(train.feature_group.iloc[valid]))
            assert not (set(train.feature_group) & outer_groups)
            covered[valid] += 1
        assert (covered == 1).all()


@pytest.mark.integration
@pytest.mark.parametrize('name', ['baseline', 'log_interactions', 'log_nonlinear', 'rg3_lgbm', 'rg3_xgb'])
def test_native_bundle_parity_and_schema(development, tmp_path, name):
    # Only official development measurements used for fit and parity checks.
    training = development[development.outer_fold.ne(0)]
    validation = development[development.outer_fold.eq(0)]
    specs = candidates()
    reference = fit_model(training, specs['baseline'], mg.SEED)
    model = fit_model(training, specs[name], mg.SEED)
    export_component(reference, specs['baseline'], tmp_path / 'cn7')
    export_component(model, specs[name], tmp_path / 'rg3')
    write_json(tmp_path / 'bundle.json', {'schema': 'kamp-upgrade-research-v1', 'field_approved': False,
        'files': {str(p.relative_to(tmp_path)): digest(p) for p in tmp_path.rglob('*') if p.is_file()}})
    runtime = ResearchBundle(tmp_path)
    baseline = predict(reference, validation, specs['baseline'])
    expected = predict(model, validation, specs[name], baseline)
    expected[validation.machine.eq('cn7')] = baseline[validation.machine.eq('cn7')]
    actual = runtime.predict(validation)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    shuffled = validation.sample(frac=1, random_state=42)
    np.testing.assert_allclose(runtime.predict(shuffled), pd.Series(actual, index=validation.index).loc[shuffled.index], rtol=0, atol=1e-12)
    np.testing.assert_allclose(runtime.predict(validation[validation.columns[::-1]]), actual, rtol=0, atol=1e-12)
    assert not set(matrix(validation, specs[name]).columns) & {mg.ID_COL, mg.TARGET, 'outer_fold', 'feature_group'}
    with pytest.raises(ValueError, match='Missing'):
        runtime.predict(validation.drop(columns=[FEATURES[0]]))
    for invalid in [float('nan'), float('inf')]:
        bad = validation.copy()
        bad.iloc[0, bad.columns.get_loc(FEATURES[0])] = invalid
        with pytest.raises(ValueError, match='finite'):
            runtime.predict(bad)
    bad = validation.copy()
    bad.iloc[0, bad.columns.get_loc('machine')] = 'unknown'
    with pytest.raises(ValueError, match='Unsupported machine'):
        runtime.predict(bad)
    component = tmp_path / 'cn7' / 'weights.json'
    component.write_text(component.read_text() + ' ')
    with pytest.raises(ValueError, match='hash'):
        ResearchBundle(tmp_path)
