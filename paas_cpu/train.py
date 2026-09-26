"""T3Q one-shot import + real-data inference smoke test. Does NOT retrain."""
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import time
import numpy as np
import pandas as pd
from model_runtime import infer, load_model

MODEL_SHA256 = 'f58243377e8fc934d9488bedf90e6d25111b6d6c722e0cd438b9c8e0e18d0341'


def run(module_dir, model_dir, output_dir):
    module_dir, model_dir, output_dir = map(Path, (module_dir, model_dir, output_dir))
    model = load_model(module_dir / 'model.json', MODEL_SHA256)
    manifest = json.loads((module_dir / 'smoke_manifest.json').read_text())
    for name, expected_hash in manifest['input_hashes'].items():
        if hashlib.sha256((module_dir / name).read_bytes()).hexdigest() != expected_hash:
            raise ValueError('Smoke test input hash mismatch: ' + name)
    frame = pd.DataFrame(json.loads((module_dir / 'smoke_input.json').read_text()))
    expected = json.loads((module_dir / 'smoke_expected.json').read_text())
    started = time.perf_counter()
    actual = infer(frame, model)
    elapsed = time.perf_counter() - started
    if len(actual) != len(expected) or not actual:
        raise ValueError('Result count mismatch')
    error = max(abs(a['research_score'] - b['research_score']) for a, b in zip(actual, expected))
    if error > 1e-12:
        raise ValueError('CPU inference numeric mismatch')
    for a, b in zip(actual, expected):
        if {k: v for k, v in a.items() if k != 'research_score'} != {k: v for k, v in b.items() if k != 'research_score'}:
            raise ValueError('ID/order/OOD/safety policy mismatch')
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(module_dir / 'model.json', model_dir / 'model.json')
    report = {'status': 'passed', 'test': 'frozen_model_real_data_inference', 'retrained': False,
              'rows': len(actual), 'max_score_absolute_error': error, 'safety_outputs_identical': True,
              'seconds': elapsed, 'model_sha256': MODEL_SHA256,
              'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
              'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'mode': 'RESEARCH_ONLY', 'field_validation': False,
              'note': 'Transport/model import smoke test, not an independent accuracy evaluation.'}
    (output_dir / 'remote_smoke_report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    (output_dir / 'remote_smoke_predictions.json').write_text(json.dumps(actual, allow_nan=False))
    print('MOLDGUARD_SMOKE_RESULT ' + json.dumps(report, allow_nan=False), flush=True)
    return report


def main():
    import t3qai_client as tc
    from t3qai_client import T3QAI_TRAIN_MODEL_PATH, T3QAI_TRAIN_OUTPUT_PATH
    tc.train_start()
    try:
        run(Path(__file__).resolve().parent, T3QAI_TRAIN_MODEL_PATH, T3QAI_TRAIN_OUTPUT_PATH)
    except Exception as error:
        tc.train_finish(error, str(error))
        raise
    tc.train_finish(None, 'MoldGuard CPU inference smoke test passed; no retraining')


if __name__ == '__main__':
    main()
