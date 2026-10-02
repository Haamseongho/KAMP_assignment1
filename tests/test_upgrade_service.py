"""Real candidate/API acceptance. No synthetic performance data."""
import json
from http.server import ThreadingHTTPServer
from threading import Thread
from urllib.request import urlopen

import numpy as np
import pytest

import moldguard as mg
from moldguard_service.api import Application, handler_for
from upgrade_runtime import UpgradeService


@pytest.mark.unit
def test_upgrade_api_not_silently_enabled(tmp_path):
    app = Application(data_dir=tmp_path)
    assert app.get('/api/v1/upgrade/readiness', {})['status'] == 'unavailable'


@pytest.fixture
def service():
    path = mg.ROOT / 'outputs/model_upgrade_20260926_final'
    if not (path / 'execution.json').exists() or json.loads((path / 'execution.json').read_text())['status'] != 'passed':
        pytest.skip('Completed real research candidate not available; not an inference validation pass')
    return UpgradeService(path, mg.DATA_DIR)


@pytest.mark.integration
def test_new_candidate_full_batch_and_cli_same_native_predictions(service):
    import pandas as pd
    actual = service.predict(service.data)
    expected = pd.read_csv(service.run_dir / 'unlabeled_native_predictions.csv')
    joined = actual.merge(expected, on=['machine', 'source_row_id'], validate='one_to_one')
    assert len(joined) == 71180
    np.testing.assert_allclose(joined.research_score, joined.native_score, rtol=0, atol=1e-12)
    assert actual.operational_priority.isna().all()
    assert actual.defect_probability.isna().all()
    assert actual.action.eq('usual_inspection_research_only').all()


@pytest.mark.integration
def test_actual_http_api_cli_bundle_identity(service):
    app = Application(data_dir=mg.DATA_DIR, upgrade_run_dir=service.run_dir)
    http = ThreadingHTTPServer(('127.0.0.1', 0), handler_for(app))
    thread = Thread(target=http.serve_forever, daemon=True)
    thread.start()
    try:
        url = f'http://127.0.0.1:{http.server_port}'
        ready = json.load(urlopen(url + '/api/v1/upgrade/readiness'))
        assert ready['bundle_sha256'] == service.bundle.hash
        for machine in ('cn7', 'rg3'):
            page = json.load(urlopen(url + f'/api/v1/upgrade/predictions?machine={machine}&limit=50'))
            expected = service.get({'machine': [machine], 'limit': ['50']})
            assert page['rows'] == expected['rows']
            assert all(r['defect_probability'] is None and r['operational_priority'] is None for r in page['rows'])
        from urllib.error import HTTPError
        with pytest.raises(HTTPError) as error:
            urlopen(url + '/api/v1/upgrade/predictions?machine=unsupported')
        assert error.value.code == 422
    finally:
        http.shutdown()
        http.server_close()
        thread.join()
