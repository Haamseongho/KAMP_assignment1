"""Read-only workspace investigation, written to a new evidence file."""
import argparse
import json
from pathlib import Path
import platform
import sys
import unicodedata
import moldguard as mg
from research_runtime import digest, git_info


def inventory():
    sources = json.loads((mg.ROOT / 'source_manifest.json').read_text())
    current = {unicodedata.normalize('NFC', p.name): p for p in [*mg.ROOT.glob('*.md'), *mg.ROOT.glob('docs/*.md')]}
    mapping = []
    for item in sources['files']:
        p = current.get(unicodedata.normalize('NFC', item['source_filename']))
        mapping.append({'source_filename': item['source_filename'], 'provided_package_path': item['included_path'],
                        'actual_path': str(p.relative_to(mg.ROOT)) if p else None,
                        'sha256': digest(p) if p else None,
                        'matches_provided_manifest': bool(p and digest(p) == item['sha256'])})
    files = [p for p in mg.ROOT.glob('1. */*.csv')]
    artifacts = [mg.ROOT / p for p in ['outputs/moldguard/model.joblib', 'outputs/moldguard/experiment.json',
        'outputs/moldguard/unlabeled_priority.csv', 'outputs/moldguard/failure_analysis.json',
        'outputs/local_runs/20260922-share-final/split_manifest.csv',
        'outputs/task01_compare_20260923_cpu_v3/oof_predictions.csv']]
    return {'git': git_info(), 'python': sys.version, 'executable': sys.executable,
            'platform': platform.platform(), 'data_dir': str(mg.DATA_DIR),
            'official_csvs': {str(p.relative_to(mg.ROOT)): digest(p) for p in files},
            'historical_artifacts': {str(p.relative_to(mg.ROOT)): {'exists': p.is_file(), 'sha256': digest(p) if p.is_file() else None} for p in artifacts},
            'supplied_source_mapping': mapping,
            'scope': 'workspace inspection, not new model or field validation'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with args.output.open('x') as stream:
        json.dump(inventory(), stream, ensure_ascii=False, indent=2)
    print(args.output)
