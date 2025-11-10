
# python -m venv python-env
. .\python-env\Scripts\Activate.ps1
# pip install -r .\openvino.genai\tests\python_tests\requirements.txt

# Windows nightly don't contain python package, so build ov from source.
. ..\openvino\build\install\setupvars.ps1

# pip install --pre -U openvino openvino-tokenizers==2025.4.0.0.dev20250929 --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly

$env:SCRIPT_RUN_GENAI_PYTEST_DIR = "C:\Users\openvino-ci-88\xiping\profiling_qwen2b_vl_instruction"
cd $env:SCRIPT_RUN_GENAI_PYTEST_DIR

$env:HF_ENDPOINT = "https://hf-mirror.com"

cd openvino.genai/tests/python_tests
$env:OV_CACHE = "./ov_cache"
# python -m pytest ./ -s -m precommit -k test_vlm_continuous_batching_generate_vs_add_request
# python -m pytest ./ -s -m precommit -k test_vlm_pipeline_match_optimum_preresized

python -m pytest ./ -s -m precommit -k test_vlm_pipeline_chat_with_video

cd ../../../