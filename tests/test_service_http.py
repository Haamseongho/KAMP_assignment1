import json
from http.server import ThreadingHTTPServer
from threading import Thread
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import pytest
from moldguard_service.api import Application, handler_for

pytestmark = pytest.mark.acceptance


@pytest.fixture
def server(tmp_path):
    app = Application(tmp_path / 'absent', tmp_path / 'results', tmp_path / 'compare', tmp_path / 'registry')
    http = ThreadingHTTPServer(('127.0.0.1', 0), handler_for(app))
    thread = Thread(target=http.serve_forever, daemon=True)
    thread.start()
    yield f'http://127.0.0.1:{http.server_port}'
    http.shutdown()
    http.server_close()
    thread.join()


def test_empty_readiness_and_rankings_are_not_fabricated(server):
    state = json.load(urlopen(server + '/api/v1/readiness'))
    assert state['research_available'] is False and state['data_sources'] == {}
    assert all(c['status'] == 'unconfirmed' for c in state['contracts']['checks'])
    assert state['unset_acceptance_criteria']
    result = json.load(urlopen(server + '/api/v1/research/rankings'))
    assert result['rows'] == [] and result['research_score'] is None
    assert json.load(urlopen(server + '/api/v1/validation/runs'))['status'] == 'unavailable'
    assert '연구 모드'.encode() in urlopen(server + '/').read()


@pytest.mark.parametrize('path', ['/api/v1/approvals', '/api/v1/research/score', '/api/v1/operational/observations',
                                 '/api/v1/inspection-skip', '/api/v1/control', '/api/v1/models/upload'])
def test_T09_T10_T36_T38_writes_and_spoofed_approvals_rejected(server, path):
    request = Request(server + path, data=json.dumps({'label_confirmed': True, 'deployment_approved': True}).encode(),
                      headers={'Content-Type':'application/json'})
    with pytest.raises(HTTPError) as error:
        urlopen(request)
    assert error.value.code == 405
    assert json.load(error.value)['operational_priority'] is None


def test_client_query_approval_not_trusted(server):
    with pytest.raises(HTTPError) as error:
        urlopen(server + '/api/v1/research/rankings?label_confirmed=true')
    assert error.value.code == 422


def test_arbitrary_files_not_served(server):
    with pytest.raises(HTTPError) as error:
        urlopen(server + '/../../moldguard.py')
    assert error.value.code == 404
