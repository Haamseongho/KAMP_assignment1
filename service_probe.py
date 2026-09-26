"""Measure and verify the running loopback research API; no field latency claim."""
import argparse
import json
from pathlib import Path
import statistics
import time
from urllib.parse import urlparse
from urllib.request import urlopen
from research_runtime import execution_record, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8765')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if urlparse(args.url).hostname not in ('127.0.0.1', 'localhost'):
        parser.error('Use the loopback service only')
    with execution_record(args.output_dir):
        report = {'concurrency': 1, 'samples_per_endpoint': 5, 'scope': 'local historical artifact API, not model inference or production latency', 'endpoints': {}}
        for endpoint in ('/api/v1/readiness', '/api/v1/research/rankings?limit=20', '/api/v1/validation/runs'):
            times = []
            for _ in range(5):
                started = time.perf_counter()
                with urlopen(args.url + endpoint, timeout=60) as response:
                    payload = json.load(response)
                    assert response.status == 200
                times.append(time.perf_counter() - started)
            if endpoint.endswith('readiness'):
                assert payload['mode'] == 'RESEARCH_ONLY' and payload['field_deployment_approved'] is False
            elif 'rankings' in endpoint:
                assert payload['total'] == 71180
                assert all(r['operational_priority'] is None and r['defect_probability'] is None for r in payload['rows'])
            else:
                assert len(payload['rows']) == 63 and payload['evaluation'] == 'development_oof'
            values = sorted(times)
            report['endpoints'][endpoint] = {'status': 'passed', 'seconds': times, 'p50_seconds': statistics.median(times),
                                            'p95_seconds_linear': values[3] + .8 * (values[4] - values[3])}
        write_json(args.output_dir / 'probe.json', report)
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
