"""Audit preservation and numerical equivalence after service development."""
import argparse
import json
from pathlib import Path
from research_runtime import ROOT, digest, execution_record, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    base = args.evidence_dir
    with execution_record(args.output_dir):
        before = json.loads((base / 'original_artifacts.json').read_text())
        changed = [name for name, old in before.items() if not (ROOT / name).is_file() or digest(ROOT / name) != old]
        comparisons = {}
        for kind, reference, names in [
            ('cpu', ROOT / 'outputs/task01_compare_20260923_cpu_v3',
             ['oof_predictions.csv', 'comparison_summary.csv', 'condition_slices.csv', 'metrics_by_fold.csv', 'metrics_by_seed_machine.csv']),
            ('calibration', ROOT / 'outputs/calibration-review-20260922',
             ['nested_oof_predictions.csv', 'nested_split_manifest.csv', 'reliability_bins.csv', 'conditional_errors.csv'])]:
            comparisons[kind] = {name: {'new_sha256': digest(base / kind / name), 'reference_sha256': digest(reference / name),
                                       'byte_identical': digest(base / kind / name) == digest(reference / name)} for name in names}
        report = {'original_file_count': len(before), 'changed_original_files': changed,
                  'numeric_comparisons': comparisons,
                  'checks': {folder: json.loads((base / folder / 'test_report.json').read_text())['counts']
                             for folder in ('checks_all', 'checks_no_data', 'checks_final')},
                  'clean_reproduction': json.loads((base / 'clean_v2/execution.json').read_text())['status'],
                  'clean_numeric_comparison': json.loads((base / 'clean_v2/reproduction_comparison.json').read_text()),
                  'oof_verification': json.loads((base / 'cpu_verification/verification.json').read_text()),
                  'field_validation': 'not_performed', 'new_model_selected': False}
        write_json(args.output_dir / 'release_audit.json', report)
        if changed or any(not i['byte_identical'] for group in comparisons.values() for i in group.values()):
            raise RuntimeError('Original artifact or numeric equivalence failure')
        if report['clean_reproduction'] != 'passed':
            raise RuntimeError('Clean reproduction incomplete')
        if any(counts['failed'] for counts in report['checks'].values()):
            raise RuntimeError('Recorded test failure')
        if report['oof_verification']['status'] != 'passed':
            raise RuntimeError('OOF verification incomplete')
        print(json.dumps({k: report[k] for k in ('original_file_count', 'changed_original_files', 'checks', 'clean_reproduction')}, indent=2))


if __name__ == '__main__':
    main()
