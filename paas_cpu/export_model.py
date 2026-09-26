"""Export only the trusted existing logistic pipeline to non-executable JSON."""
import argparse
import json
from pathlib import Path
import shutil
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from research_runtime import ROOT, digest, execution_record, write_json
from .model_runtime import SCHEMA, validate_model


def portable(model, features, matrix_columns, ood_bounds):
    if not isinstance(model, Pipeline) or len(model.steps) != 2:
        raise ValueError('Only reviewed StandardScaler + LogisticRegression is supported')
    scaler, estimator = [step for _, step in model.steps]
    if not isinstance(scaler, StandardScaler) or not isinstance(estimator, LogisticRegression):
        raise ValueError('Unsupported pipeline')
    if not scaler.with_mean or not scaler.with_std or estimator.coef_.shape != (1, len(matrix_columns)):
        raise ValueError('Unsupported scaler/classifier layout')
    result = {'schema_version': SCHEMA, 'classes': estimator.classes_.tolist(),
              'features': features, 'matrix_columns': matrix_columns,
              'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(),
              'coef': estimator.coef_[0].tolist(), 'intercept': float(estimator.intercept_[0]),
              'ood_bounds': ood_bounds, 'mode': 'RESEARCH_ONLY',
              'score_target': 'numeric_label_1_unconfirmed',
              'input_representation': 'provided_standardized_features_raw_units_unverified'}
    validate_model(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--expected-sha256', required=True, help='Pin from a previously trusted receipt, not an uploaded model')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if digest(args.model) != args.expected_sha256:
        raise ValueError('Refusing to deserialize untrusted or changed local model')
    with execution_record(args.output_dir) as receipt:
        artifact = joblib.load(args.model)  # Only the explicitly pinned local artifact.
        if artifact['model_name'] != 'logistic':
            raise ValueError('Expected the existing logistic baseline')
        model = portable(artifact['model'], artifact['features'], artifact['matrix_columns'], artifact['ood_bounds'])
        model['source_model_sha256'] = args.expected_sha256
        model['training_source_hashes'] = artifact['source_hashes']
        write_json(args.output_dir / 'model.json', model)
        for name in ('model_runtime.py', 'inference_service.py', 'requirements-inference.txt', 'README.md'):
            shutil.copyfile(ROOT / 'paas_cpu' / name, args.output_dir / name)
        receipt['adapter_code_hashes'] = {str(p.relative_to(ROOT)): digest(p) for p in (ROOT / 'paas_cpu').glob('*.py')}
        write_json(args.output_dir / 'deployment_manifest.json', {
            'status': 'prepared_not_deployed', 'device': 'cpu', 'mode': 'RESEARCH_ONLY',
            'model_sha256': digest(args.output_dir / 'model.json'),
            'source_model_sha256': args.expected_sha256, 'raw_data_included': False,
            'remote_cost_verified': False, 'remote_lifecycle_verified': False,
            'code_hashes': receipt['adapter_code_hashes']})
        print(json.dumps({'status': 'prepared_not_deployed', 'model_sha256': digest(args.output_dir / 'model.json')}))


if __name__ == '__main__':
    main()
