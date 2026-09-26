"""Read-only environment/cost check: never installs or starts remote resources."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
from research_runtime import execution_record, write_json, digest
from .cost_gate import validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path, default=Path('config/paas_cpu_cost_receipt.json'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    with execution_record(args.output_dir):
        packages = {}
        for name in ('numpy', 'pandas', 'scikit-learn', 't3qai_client'):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = None
        inference_ready = sys.version_info[:2] == (3, 12) and packages['numpy'] == '1.26.4' and packages['pandas'] == '2.2.2'
        try:
            cost = validate(json.loads(args.receipt.read_text()), args.receipt.parent)
        except ValueError as error:
            cost = {'status': 'blocked', 'reason': str(error)}
        report = {'local_inference_dependencies_match': inference_ready,
                  'packages': packages, 'python': sys.version, 'device': 'cpu',
                  'cost_gate': cost, 'cost_receipt_sha256': digest(args.receipt),
                  'remote_readiness': 'unverified', 'remote_actions_performed': [],
                  'note': 'Local dependency match does not verify platform SDK, runtime lifecycle, endpoint transport or free allocation.'}
        write_json(args.output_dir / 'preflight.json', report)
        print(json.dumps(report, indent=2))
        if not inference_ready:
            raise RuntimeError('Use the documented pinned local environment')


if __name__ == '__main__':
    main()
