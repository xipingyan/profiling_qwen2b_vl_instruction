# Verify this script pass for local laptop.

# python -m venv python-env
. .\python-env\Scripts\Activate.ps1

# Windows install all dependencies first
# .\python-env\Scripts\python.exe -m pip install -r .\openvino.genai\tests\python_tests\requirements.txt

# Windows nightly don't contain python package, so build ov from source.
# . ..\openvino\build\install\setupvars.ps1
. .\openvino_toolkit_windows_2026.0.0.dev20260115_x86_64\setupvars.ps1

# pip install --pre -U openvino openvino-tokenizers==2025.4.0.0.dev20250929 --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly

$workspaceRoot = $PSScriptRoot
$env:SCRIPT_RUN_GENAI_PYTEST_DIR = Join-Path $workspaceRoot "openvino.genai"
Set-Location $env:SCRIPT_RUN_GENAI_PYTEST_DIR

$env:GENAI_ROOT_DIR = Join-Path $env:SCRIPT_RUN_GENAI_PYTEST_DIR "install\python"
$env:PYTHONPATH = "$env:GENAI_ROOT_DIR;$env:PYTHONPATH"

$genaiRuntimeBin = Resolve-Path (Join-Path $env:GENAI_ROOT_DIR "..\runtime\bin\intel64\Release")
$env:PATH = "$genaiRuntimeBin;$env:PATH"

# Python 3.8+ doesn't reliably search PATH for dependent DLLs of extension modules.
# openvino uses OPENVINO_LIB_PATHS to register DLL search directories via add_dll_directory.
$env:OPENVINO_LIB_PATHS = "$genaiRuntimeBin;$env:OPENVINO_LIB_PATHS"

$env:HF_ENDPOINT = "https://hf-mirror.com"

Set-Location "tests\python_tests"
$env:OV_CACHE = "./ov_cache"
# python -m pytest ./ -s -m precommit -k test_vlm_continuous_batching_generate_vs_add_request
# python -m pytest ./ -s -m precommit -k test_vlm_pipeline_match_optimum_preresized

# python -m pytest ./ -s -m precommit -k test_vlm_pipeline_chat_with_video

python -m pytest . -k "test_stop_strings_facebook_opt"

Set-Location $workspaceRoot