
# python -m venv python-env
. ..\python-env\Scripts\Activate.ps1
# pip install pillow requests numpy

# OpenVINO
. ..\openvino_toolkit_windows_2026.0.0.dev20251117_x86_64\setupvars.ps1

$env:CustomVIT_PATH = "C:\ov_task\profiling_qwen2b_vl_instruction\custom_vit"

# GenAI
$env:GENAI_ROOT_DIR = "..\openvino.genai\install\python\"
$env:PYTHONPATH = "$env:GENAI_ROOT_DIR;$env:PYTHONPATH"
$env:PATH = "$env:GENAI_ROOT_DIR\..\runtime\bin\intel64\Debug;$env:PATH;$env:CustomVIT_PATH"

python .\test_pipeline_genai.py 