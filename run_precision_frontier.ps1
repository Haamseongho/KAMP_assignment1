param([string]$OutputDir = ('outputs/precision_frontier_' + (Get-Date -Format 'yyyyMMdd_HHmmss')))
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$env:MPLCONFIGDIR = Join-Path $PSScriptRoot 'outputs\matplotlib_cache'
Set-Location $PSScriptRoot
$python = Join-Path $PSScriptRoot '.venv-gpu\Scripts\python.exe'
if (!(Test-Path -LiteralPath $python)) { throw 'Existing GPU virtual environment is required.' }
if (!(Test-Path -LiteralPath $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
    & $python task01_precision_audit.py --output-dir $OutputDir
    if ($LASTEXITCODE -ne 0) { throw 'Audit failed.' }
}
# GC releases native estimator cycles promptly on the 16 GiB Windows host.
# Execution-state protection is temporary, lasts only for this Python process,
# and allows the normal screen lock while preventing idle system sleep.
$env:KAMP_PRECISION_OUTPUT = $OutputDir
@'
import ctypes, gc, os, runpy, sys
gc.set_threshold(50, 5, 5)
ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
try:
    sys.argv=['task01_precision_frontier.py','--output-dir',os.environ['KAMP_PRECISION_OUTPUT']]
    runpy.run_path('task01_precision_frontier.py',run_name='__main__')
finally:
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
'@ | & $python -u -
if ($LASTEXITCODE -ne 0) { throw 'Training stopped; preserved checkpoints may be resumed with the same output directory.' }
& $python task01_precision_evaluate.py --output-dir $OutputDir
if ($LASTEXITCODE -ne 0) { throw 'Evaluation failed.' }
if (!(Test-Path -LiteralPath (Join-Path $OutputDir 'feature_layout_audit.json'))) {
    & $python task01_precision_layout_audit.py --output-dir $OutputDir
    if ($LASTEXITCODE -ne 0) { throw 'Legacy layout parity audit failed.' }
}
& $python verify_task01_precision_frontier.py --output-dir $OutputDir
if ($LASTEXITCODE -ne 0) { throw 'Independent verification failed.' }
$testArgs = @('-m','pytest','tests/test_precision_frontier.py','tests/test_research_split_reference.py',
    'tests/test_task01_compare.py','tests/test_task01_gpu_smoke.py','tests/test_task01_gpu_nested_search.py',
    'tests/test_task01_gpu_calibrated_ensemble.py','tests/test_task01_gpu_oof_evaluate.py',
    'tests/test_task01_gpu_preflight.py','tests/test_task01_group_event_experiment.py',
    '--source-data-dir','data/task01_official','--frozen-split-dir','outputs/gpu_pc_split_20261004_03',
    ('--junitxml=' + (Join-Path $OutputDir 'regression_tests.xml')),'-q')
& $python @testArgs
if ($LASTEXITCODE -ne 0) { throw 'Regression tests failed.' }
& $python report_task01_precision_frontier.py --output-dir $OutputDir
if ($LASTEXITCODE -ne 0) { throw 'Report failed.' }
