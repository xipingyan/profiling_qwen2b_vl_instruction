# Note: 
# Powershell, no tip for lost dll.
# We need to copy all dll manually.
$env:SCRIPT_DIR_EXAMPLE_OV_CPP_RUN = "C:\\ov_task\\profiling_qwen2b_vl_instruction\\example_ov_cpp"

echo "workpath = $env:SCRIPT_DIR_EXAMPLE_OV_CPP_RUN"
cd $env:SCRIPT_DIR_EXAMPLE_OV_CPP_RUN

# . ..\openvino_toolkit_windows_2026.0.0.dev20251117_x86_64\setupvars.ps1
. ..\openvino_toolkit_windows_2025.4.0.dev20251105_x86_64\setupvars.ps1

cd $env:SCRIPT_DIR_EXAMPLE_OV_CPP_RUN

$env:model_id = "../models/ov/Qwen2.5-VL-3B-Instruct/INT4/"

$env:type="img"
$env:prompt = "prompt.txt"
# echo "Please describe the image." > $env:prompt

$env:prompt2 = "prompt2.txt"
# echo "How many chars in this image?" > $env:prompt2

$env:device='GPU'
$env:GENAI_ROOT_DIR = "$env:SCRIPT_DIR_EXAMPLE_OV_CPP_RUN\\.."
$env:OPENCV_BIN = "C:\\Users\\xipingya\\Downloads\\opencv\\build\\bin"
$env:PATH = "$env:GENAI_ROOT_DIR\\openvino.genai\\install\\runtime\\bin\\intel64\\Debug;$env:OPENCV_BIN;$env:PATH"

echo "== Start: qwen2vl_app_cpp"

$env:CUSTOM_VIT_PATH = "C:\\ov_task\\profiling_qwen2b_vl_instruction\\custom_vit"
$env:CUSTOM_VIT_IMG_PATH = "C:\\ov_task\\profiling_qwen2b_vl_instruction\\custom_vit\\home.jpg"
$env:ENABLE_CUSTOM_VIT = "0"

# .\\build\\Debug\\qwen2vl_app_cpp.exe $env:model_id $env:type $env:CUSTOM_VIT_IMG_PATH $env:device $env:prompt $env:prompt2
.\\build\\Release\\qwen2vl_app_cpp.exe $env:model_id $env:type $env:CUSTOM_VIT_IMG_PATH $env:device $env:prompt $env:prompt2