"""Run explicit environment/unit/integration/acceptance layers with durable receipts."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
from research_runtime import execution_record, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layer', choices=['environment', 'unit', 'integration', 'acceptance', 'all'], required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--split-dir', type=Path)
    parser.add_argument('--no-data', action='store_true')
    args = parser.parse_args()
    project = Path(__file__).resolve().parent
    with execution_record(args.output_dir) as receipt:
        if args.layer == 'environment':
            if sys.version_info[:2] != (3, 12):
                raise RuntimeError('Python 3.12 required')
            command = [sys.executable, '-m', 'pip', 'check']
        else:
            command = [sys.executable, '-m', 'pytest', '-c', str(project / 'pytest.ini'),
                       '--rootdir', str(project), '--confcutdir', str(project), str(project / 'tests'),
                       '-q', '-rs', '--junitxml', str(args.output_dir.resolve() / 'junit.xml')]
            if args.layer != 'all':
                command.extend(['-m', args.layer])
            if args.no_data:
                command.append('--no-data')
            if args.data_dir:
                command.extend(['--source-data-dir', str(args.data_dir)])
            if args.split_dir:
                command.extend(['--frozen-split-dir', str(args.split_dir)])
        receipt['executed_command'] = command
        p = subprocess.run(command, cwd=project, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(p.stdout)
        result = {'layer': args.layer, 'command': command, 'exit_code': p.returncode,
                  'scope': 'local_code_or_official_data_checks_not_field_validation'}
        xml = args.output_dir / 'junit.xml'
        if xml.exists():
            cases = ET.parse(xml).findall('.//testcase')
            result['tests'] = [{'name': c.attrib['name'], 'status': 'skipped' if c.find('skipped') is not None else
                               'failed' if c.find('failure') is not None or c.find('error') is not None else 'passed',
                               'reason': c.find('skipped').attrib.get('message') if c.find('skipped') is not None else None}
                              for c in cases]
            result['counts'] = {s: sum(t['status'] == s for t in result['tests']) for s in ('passed', 'failed', 'skipped')}
        write_json(args.output_dir / 'test_report.json', result)
        if p.returncode:
            raise RuntimeError('Checks failed; inspect test_report.json and run.log')


if __name__ == '__main__':
    main()
