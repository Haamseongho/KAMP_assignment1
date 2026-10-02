"""Loopback-only read-only UI/API. All mutation endpoints are unavailable."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import moldguard as mg
from research_runtime import digest
from .contracts import ContractRegistry, source_hashes
from .legacy import LegacyAdapter
from .policy import acceptance_blockers, validate_policy

ROOT = Path(__file__).resolve().parents[1]


class Application:
    def __init__(self, data_dir=None, result_dir=None, comparison_dir=None, registry_dir=None,
                 policy_path=None, acceptance_path=None, upgrade_run_dir=None):
        self.data_dir = data_dir
        self.result_dir = Path(result_dir or ROOT / 'outputs/moldguard')
        self.comparison_dir = Path(comparison_dir or ROOT / 'outputs/task01_compare_20260923_cpu_v3')
        self.registry = ContractRegistry(registry_dir or ROOT / 'service_state/contracts')
        self.policy_path = Path(policy_path or ROOT / 'config/service_policy.json')
        self.acceptance_path = Path(acceptance_path or ROOT / 'config/acceptance_criteria.example.json')
        self.upgrade_run_dir = upgrade_run_dir
        self.upgrade_service = None

    def get(self, path, query):
        if path in ('/api/v1/upgrade/readiness', '/api/v1/upgrade/predictions'):
            if self.upgrade_run_dir is None:
                return {'status': 'unavailable', 'reason': 'No --upgrade-run-dir supplied', 'mode': 'RESEARCH_ONLY'}
            if self.upgrade_service is None:
                from upgrade_runtime import UpgradeService
                self.upgrade_service = UpgradeService(self.upgrade_run_dir, self.data_dir)
            if path.endswith('/readiness'):
                return self.upgrade_service.readiness()
            return self.upgrade_service.get(query)
        policy = json.loads(self.policy_path.read_text())
        policy_hash = validate_policy(policy)
        adapter = LegacyAdapter(self.data_dir, self.result_dir, self.comparison_dir)
        contracts = self.registry.view(adapter.sources)
        if path == '/api/v1/readiness':
            criteria = json.loads(self.acceptance_path.read_text())
            return {'mode': 'RESEARCH_ONLY', 'field_deployment_approved': False,
                    'research_available': adapter.frame is not None, 'errors': adapter.errors,
                    'data_sources': adapter.sources, 'contracts': contracts, 'policy': policy,
                    'policy_hash': policy_hash, 'acceptance_hash': digest(self.acceptance_path),
                    'unset_acceptance_criteria': acceptance_blockers(criteria),
                    'model_hash': adapter.model_hash, 'result_dir': str(self.result_dir),
                    'comparison_dir': str(self.comparison_dir), 'validation': adapter.validation,
                    'disabled_features': ['operational_queue', 'defect_probability', 'approvals',
                                          'in_cycle_prediction', 'time_savings', 'machine_control', 'inspection_skip']}
        if path == '/api/v1/research/rankings':
            if set(query) - {'machine', 'offset', 'limit'}:
                raise ValueError('Only machine/offset/limit are accepted; client approvals are not trusted')
            return adapter.rankings(contracts, policy, query.get('machine', [None])[0],
                                    int(query.get('offset', ['0'])[0]), int(query.get('limit', ['30'])[0]))
        if path == '/api/v1/validation/runs':
            return adapter.metrics()
        raise LookupError('Endpoint is not implemented')


def handler_for(app):
    class Handler(BaseHTTPRequestHandler):
        def send(self, status, payload, content_type='application/json; charset=utf-8'):
            body = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.headers.get('Host', '').split(':')[0] not in ('localhost', '127.0.0.1'):
                return self.send(403, {'error': 'Loopback host required'})
            parsed = urlparse(self.path)
            assets = {'/': ('index.html', 'text/html; charset=utf-8'), '/app.js': ('app.js', 'text/javascript; charset=utf-8')}
            if parsed.path in assets:
                name, mime = assets[parsed.path]
                return self.send(200, (Path(__file__).parent / 'static' / name).read_bytes(), mime)
            try:
                self.send(200, app.get(parsed.path, parse_qs(parsed.query)))
            except LookupError as error:
                self.send(404, {'error': str(error), 'mode': 'RESEARCH_ONLY'})
            except (OSError, ValueError, TypeError, KeyError) as error:
                self.send(422, {'error': str(error), 'research_score': None,
                                'operational_priority': None, 'existing_inspection': 'unchanged'})

        def do_POST(self):
            self.send(405, {'error': 'Read-only local service; approvals, uploads and operational writes are disabled',
                            'mode': 'RESEARCH_ONLY', 'operational_priority': None})

        do_PATCH = do_POST
        do_PUT = do_POST
        do_DELETE = do_POST
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=mg.DATA_DIR)
    parser.add_argument('--result-dir', type=Path)
    parser.add_argument('--comparison-dir', type=Path)
    parser.add_argument('--registry-dir', type=Path)
    parser.add_argument('--upgrade-run-dir', type=Path, help='Optional verified CPU research candidate; does not replace legacy results')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    app = Application(args.data_dir, args.result_dir, args.comparison_dir, args.registry_dir,
                      upgrade_run_dir=args.upgrade_run_dir)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler_for(app))
    print(f'MoldGuard RESEARCH_ONLY http://127.0.0.1:{server.server_port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
