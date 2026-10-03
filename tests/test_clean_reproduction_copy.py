"""TEST_ONLY: the clean checkout must collect its tests without the source tree."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

from verify_clean_reproduction import source_paths


ROOT = Path(__file__).resolve().parents[1]


def test_isolated_copy_collects_full_suite(tmp_path):
    checkout = tmp_path / 'checkout'
    checkout.mkdir()
    paths = source_paths(ROOT)
    assert ROOT / 'paas_cpu' / '__init__.py' in paths
    assert ROOT / 'paas_cpu' / 'model_runtime.py' in paths
    for path in paths:
        target = checkout / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    environment = os.environ.copy()
    environment.pop('PYTHONPATH', None)
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', '--collect-only', '-q'],
        cwd=checkout,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    assert 'test_paas_cpu.py' in result.stdout
