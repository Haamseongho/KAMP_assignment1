import json
import shutil
import pandas as pd
import pytest
import moldguard as mg
import calibration_diagnostics as cd
import research_split as split
from research_runtime import digest
from moldguard_service.legacy import LegacyAdapter
from moldguard_service.contracts import ContractRegistry


@pytest.mark.unit
def test_T12_T13_split_membership_rejects_mutation():
    # TEST_ONLY membership fixture: no invented manufacturing features or performance.
    data = pd.DataFrame([{'machine': m, mg.ID_COL: g*2+i, mg.TARGET: g % 2,
                          'feature_group': f'{m}:TEST_ONLY:{g}'}
                         for m in ('cn7', 'rg3') for g in range(40) for i in range(2)])
    valid = split.build_table(data)
    split.validate_membership(data, valid)
    for column, value in [('label_value', 9), ('partition', 'unknown'), ('feature_group', 'other'),
                          ('source_row_id', -1), ('oof_validation_fold', 99)]:
        bad = valid.copy()
        bad.loc[0, column] = value
        with pytest.raises(ValueError):
            split.validate_membership(data, bad)
    bad = valid.copy()
    bad.loc[0, 'partition'] = 'holdout' if bad.loc[1, 'partition'] == 'development' else 'development'
    with pytest.raises(ValueError, match='crosses'):
        split.validate_membership(data, bad)


@pytest.mark.integration
def test_T14_generated_split_exactly_matches_reference(tmp_path):
    result = split.generate(mg.DATA_DIR, tmp_path, cd.FROZEN)
    assert result['reference_comparison']['status'] == 'identical'
    assert digest(tmp_path / 'split_manifest.csv') == digest(cd.FROZEN / 'split_manifest.csv')
    data, features, _ = cd.load_development(mg.DATA_DIR, tmp_path)
    assert len(data) == 1913 and len(features) == 24
    assert not list(tmp_path.glob('*.joblib'))


@pytest.mark.integration
def test_T13_source_and_split_hash_mutations_rejected(tmp_path):
    split.generate(mg.DATA_DIR, tmp_path)
    record = json.loads((tmp_path / 'run_manifest.json').read_text())
    record['split']['source_files']['moldset_labeled_cn7.csv']['sha256'] = '0' * 64
    (tmp_path / 'run_manifest.json').write_text(json.dumps(record))
    with pytest.raises(ValueError, match='Input hash'):
        split.load_validated(mg.DATA_DIR, tmp_path)


@pytest.mark.integration
def test_T16_actual_adapter_preserves_71180_rows_and_actions(tmp_path):
    if not (mg.ROOT / 'outputs/moldguard/reproducibility.json').is_file():
        pytest.skip('Historical ranking artifacts not installed; actual 71,180-row check not executed')
    adapter = LegacyAdapter(mg.DATA_DIR, mg.ROOT / 'outputs/moldguard', mg.ROOT / 'outputs/task01_compare_20260923_cpu_v3')
    assert adapter.errors == []
    assert len(adapter.frame) == 71180
    assert adapter.frame.action.value_counts().to_dict() == {
        'usual_inspection_unvalidated_machine': 35941, 'usual_inspection_ood': 19607,
        'usual_inspection_label_unconfirmed': 15632}
    policy = json.loads((mg.ROOT / 'config/service_policy.json').read_text())
    page = adapter.rankings(ContractRegistry(tmp_path).view(adapter.sources), policy, 'rg3')
    assert all(r['action'] == r['legacy_action'] and r['operational_priority'] is None for r in page['rows'])
    assert all(r['product_id'] is None and r['prediction_cutoff_at'] is None for r in page['rows'])


@pytest.mark.integration
def test_T06_model_hash_failure_produces_no_scores(tmp_path):
    base = mg.ROOT / 'outputs/moldguard'
    if not (base / 'reproducibility.json').is_file():
        pytest.skip('Historical model provenance not installed; integration corruption check not executed')
    shutil.copyfile(base / 'reproducibility.json', tmp_path / 'reproducibility.json')
    (tmp_path / 'model.joblib').write_bytes(b'TEST_ONLY invalid model, never deserialized')
    adapter = LegacyAdapter(mg.DATA_DIR, tmp_path, tmp_path)
    assert adapter.frame is None and adapter.errors
    assert adapter.rankings({}, {})['research_score'] is None
