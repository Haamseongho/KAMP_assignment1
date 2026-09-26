"""T3Q callback shape observed in the project's sklearn_LOGISTIC sample.

Platform transport/lifecycle is unverified until a zero-cost CPU run is allowed.
Set MOLDGUARD_MODEL_SHA256 from the reviewed export receipt, not a client request.
No training, installs, external requests, or raw-row logging happen here.
"""
import os
from pathlib import Path

try:
    from .model_runtime import infer, load_model
except ImportError:  # Files uploaded at the algorithm's top level.
    from model_runtime import infer, load_model


def init_model():
    root = os.environ.get('MOLDGUARD_MODEL_DIR')
    if root is None:
        from t3qai_client import T3QAI_INIT_MODEL_PATH
        root = T3QAI_INIT_MODEL_PATH
    model = load_model(Path(root) / 'model.json', os.environ.get('MOLDGUARD_MODEL_SHA256'))
    return {'model': model}


def inference_dataframe(df, model_info_dict):
    return infer(df, model_info_dict['model'])


def inference_file(files, model_info_dict):
    raise ValueError('File inference disabled; use the validated named DataFrame schema')
