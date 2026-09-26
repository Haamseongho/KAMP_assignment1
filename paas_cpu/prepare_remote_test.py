"""Package 100 real unlabeled rows, not synthetic performance data."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
import moldguard as mg
from research_runtime import ROOT, execution_record, digest, write_json
from .model_runtime import load_model, infer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--bundle-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    with execution_record(args.output_dir):
        package = args.output_dir / 'algorithm'
        package.mkdir()
        for name in ('model.json', 'model_runtime.py', 'inference_service.py'):
            shutil.copyfile(args.bundle_dir / name, package / name)
        shutil.copyfile(ROOT / 'paas_cpu/train.py', package / 'train.py')
        model = load_model(package / 'model.json', 'f58243377e8fc934d9488bedf90e6d25111b6d6c722e0cd438b9c8e0e18d0341')
        data, features, sources = mg.load_data(args.data_dir, False)
        if features != model['features'] or any(model['training_source_hashes'].get(name, {}).get('sha256') != source['sha256'] for name, source in sources.items()):
            raise ValueError('Unexpected official input sources/schema')
        frame = data.groupby('machine', sort=True).head(50).rename(columns={mg.ID_COL: 'source_row_id'})[['machine', 'source_row_id', *features]]
        write_json(package / 'smoke_input.json', frame.to_dict(orient='records'))
        write_json(package / 'smoke_expected.json', infer(frame, model))
        write_json(package / 'smoke_manifest.json', {'rows': len(frame), 'source': '100 actual official unlabeled rows; 50 each machine',
                   'full_source_hashes': sources, 'input_hashes': {name: digest(package / name) for name in ('smoke_input.json', 'smoke_expected.json')}})
        command = [sys.executable, '-c', 'from train import run; run(".", "../local_model", "../local_result")']
        subprocess.run(command, cwd=package, check=True)
        with zipfile.ZipFile(args.output_dir / 'MoldGuard_CPU_smoke_algorithm.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
            for file in sorted(package.iterdir()):
                if file.is_file():
                    archive.write(file, file.name)
        write_json(args.output_dir / 'upload_manifest.json', {file.name: digest(file) for file in package.iterdir() if file.is_file()})
        print('Prepared and locally verified real-data smoke package, not yet run on KAMP.')


if __name__ == '__main__':
    main()
