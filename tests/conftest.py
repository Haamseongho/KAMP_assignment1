"""Explicitly separate real-data checks from data-free code checks."""
from pathlib import Path
import pytest

LEGACY_INTEGRATION = {
    'test_identical_measurements_never_cross_frozen_split',
    'test_audit_finds_ambiguous_duplicates',
    'test_frozen_development_has_no_holdout_or_fold_overlap',
    'test_outer_development_membership_is_unchanged',
    'test_inner_folds_never_access_outer_validation_groups',
    'test_published_numeric_artifacts_and_model_are_untouched',
    'test_saved_review_contains_only_frozen_development_rows',
    'test_error_slices_account_for_every_development_row',
}


def pytest_addoption(parser):
    parser.addoption('--no-data', action='store_true', help='Explicitly skip official-data integration checks')
    parser.addoption('--source-data-dir', default=None)
    parser.addoption('--frozen-split-dir', default=None)


def pytest_sessionstart(session):
    import moldguard as mg
    import calibration_diagnostics as cd
    if session.config.getoption('--source-data-dir'):
        mg.DATA_DIR = Path(session.config.getoption('--source-data-dir'))
    if session.config.getoption('--frozen-split-dir'):
        cd.FROZEN = Path(session.config.getoption('--frozen-split-dir'))


def pytest_collection_modifyitems(config, items):
    import moldguard as mg
    import calibration_diagnostics as cd
    absent = (config.getoption('--no-data') or mg.DATA_DIR is None or not mg.DATA_DIR.is_dir()
              or not (cd.FROZEN / 'split_manifest.csv').is_file())
    for item in items:
        if item.name in LEGACY_INTEGRATION:
            item.add_marker(pytest.mark.integration)
        elif not any(item.get_closest_marker(m) for m in ('unit', 'integration', 'acceptance')):
            item.add_marker(pytest.mark.unit)
        if item.get_closest_marker('integration') and absent:
            item.add_marker(pytest.mark.skip(reason='Official CSVs/reviewed split unavailable, or --no-data selected; not a validation pass'))
