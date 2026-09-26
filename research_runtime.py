"""Durable local run receipts; no model or data uploads."""
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def git_info():
    def read(*args):
        p = subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True)
        return p.stdout.strip() if p.returncode == 0 else None
    changes = read('status', '--porcelain')
    return {'commit': read('rev-parse', 'HEAD'), 'branch': read('branch', '--show-current'),
            'dirty': bool(changes), 'changes': changes}


class Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, text):
        self.console.write(text)
        return self.log.write(text)

    def flush(self):
        self.console.flush()
        self.log.flush()


@contextmanager
def execution_record(output):
    """Exclusively create a run. Failed partial runs never receive passed status."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    receipt = {'status': 'running', 'started_at': datetime.now(timezone.utc).isoformat(),
               'argv': [sys.executable, *sys.argv], 'git': git_info(),
               'python': sys.version, 'platform': platform.platform(),
               'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
               'code_hashes': {str(p.relative_to(ROOT)): digest(p)
                               for p in [*ROOT.glob('*.py'), *ROOT.glob('moldguard_service/*.py'), *ROOT.glob('paas_cpu/*.py')]}}
    write_json(output / 'execution.json', receipt)
    try:
        with (output / 'run.log').open('w') as log:
            with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
                try:
                    yield receipt
                except BaseException:
                    traceback.print_exc()
                    raise
        receipt.update(status='passed', exit_code=0)
    except BaseException as error:
        receipt.update(status='failed', exit_code=1, error=str(error))
        raise
    finally:
        receipt['finished_at'] = datetime.now(timezone.utc).isoformat()
        receipt['output_hashes'] = {p.name: digest(p) for p in output.iterdir()
                                    if p.is_file() and p.name != 'execution.json'}
        write_json(output / 'execution.json', receipt)
