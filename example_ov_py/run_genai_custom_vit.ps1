
# python -m venv python-env
. ..\python-env\Scripts\Activate.ps1
# . ..\py_ov\Scripts\Activate.ps1
# py

# # OpenVINO
# . ..\openvino_toolkit_windows_2026.0.0.dev20251117_x86_64\setupvars.ps1
. ..\openvino_toolkit_windows_2025.4.0.dev20251105_x86_64\setupvars.ps1

# GenAI
$env:GENAI_ROOT_DIR = "D:\\xiping\\profiling_qwen2b_vl_instruction\\openvino.genai\\install\\python"

$env:GENAI_ROOT_DIR = "C:\ov_task\\profiling_qwen2b_vl_instruction\\openvino.genai\\install\\python"
$env:PYTHONPATH = "$env:GENAI_ROOT_DIR;$env:PYTHONPATH"
$env:PATH = "$env:GENAI_ROOT_DIR\\..\\runtime\\bin\\intel64\\Debug;$env:PATH;$env:CustomVIT_PATH"

$env:CUSTOM_VIT_PATH = "C:\\ov_task\\profiling_qwen2b_vl_instruction\\custom_vit"
$env:CUSTOM_VIT_IMG_PATH = "C:\\ov_task\\profiling_qwen2b_vl_instruction\\custom_vit\\home.jpg"
$env:ENABLE_CUSTOM_VIT = "0"

python .\test_pipeline_genai.py 