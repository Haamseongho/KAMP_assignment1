"""Fresh isolated code copy + venv; no old outputs copied; official inputs read in place."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from research_runtime import ROOT, digest, execution_record, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    data_dir = args.data_dir.resolve()
    with execution_record(output) as receipt:
        checkout = output / 'checkout'
        checkout.mkdir()
        paths = [*ROOT.glob('*.py'), *ROOT.glob('requirements*.txt'), ROOT / 'pytest.ini']
        for folder in ('tests', 'config', 'moldguard_service'):
            paths.extend(p for p in (ROOT / folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
        receipt['copied_code_hashes'] = {}
        for path in paths:
            target = checkout / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            receipt['copied_code_hashes'][str(path.relative_to(ROOT))] = digest(path)
        assert not (checkout / 'outputs').exists()
        receipt['historical_outputs_copied'] = False
        steps = []
        def command(name, argv):
            print(name, flush=True)
            with (output / (name + '.log')).open('w') as log:
                result = subprocess.run(argv, cwd=checkout, stdout=log, stderr=subprocess.STDOUT,
                    env={**os.environ, 'PYTHONUNBUFFERED': '1', 'MPLBACKEND':'Agg'})
            steps.append({'name': name, 'argv': argv, 'exit_code': result.returncode})
            receipt['steps'] = steps
            if result.returncode:
                raise RuntimeError(f'{name} failed; inspect {output / (name + ".log")}')
        command('01_venv', [sys.executable, '-m', 'venv', str(checkout / '.venv')])
        python = str(checkout / '.venv/bin/python')
        command('02_install', [python, '-m', 'pip', 'install', '-r', 'requirements-compare.txt'])
        command('03_environment', [python, 'service_checks.py', '--layer', 'environment', '--output-dir', str(output / 'environment')])
        command('04_split', [python, 'research_split.py', 'generate', '--data-dir', str(data_dir), '--output-dir', str(output / 'split')])
        command('05_verify_split', [python, 'research_split.py', 'verify', '--data-dir', str(data_dir), '--output-dir', str(output / 'split')])
        command('06_compare', [python, 'task01_compare.py', '--device', 'cpu', '--data-dir', str(data_dir), '--split-dir', str(output / 'split'), '--output-dir', str(output / 'cpu')])
        command('06b_verify_results', [python, 'research_verify.py', '--data-dir', str(data_dir), '--split-dir', str(output / 'split'), '--result-dir', str(output / 'cpu'), '--output-dir', str(output / 'result_verification')])
        command('07_tests', [python, 'service_checks.py', '--layer', 'all', '--data-dir', str(data_dir), '--split-dir', str(output / 'split'), '--output-dir', str(output / 'checks')])
        pairs = {'split/split_manifest.csv': ROOT / 'outputs/local_runs/20260922-share-final/split_manifest.csv'}
        for name in ('comparison_summary.csv', 'metrics_by_seed_machine.csv', 'metrics_by_fold.csv', 'condition_slices.csv', 'oof_predictions.csv'):
            pairs['cpu/' + name] = ROOT / 'outputs/task01_compare_20260923_cpu_v3' / name
        report = {}
        for relative, reference in pairs.items():
            report[relative] = {'reference_available': reference.is_file(), 'sha256': digest(output / relative),
                                'reference_sha256': digest(reference) if reference.is_file() else None,
                                'byte_identical': digest(output / relative) == digest(reference) if reference.is_file() else None}
        write_json(output / 'reproduction_comparison.json', report)
        if any(item['byte_identical'] is False for item in report.values()):
            raise RuntimeError('Numerical output differs from reference; investigate before claiming equivalence')


if __name__ == '__main__':
    main()
