$env:SCRIPT_DIR_EXAMPLE_OV_CPP_RUN = "C:\\ov_task\\profiling_qwen2b_vl_instruction\\example_ov_cpp"

echo "workpath = $env:SCRIPT_DIR_EXAMPLE_OV_CPP_RUN"
cd $env:SCRIPT_DIR_EXAMPLE_OV_CPP_RUN

. ..\openvino_toolkit_windows_2026.0.0.dev20251117_x86_64\setupvars.ps1

cd $env:SCRIPT_DIR_EXAMPLE_OV_CPP_RUN

$env:model_id = "../models/ov/Qwen2.5-VL-3B-Instruct/INT4/"

$env:type="img"
$env:video_img_path = "../test_video/home.jpg"
$env:prompt = "prompt.txt"
echo "Please describe the image." > $prompt

$env:device='GPU'
$env:GENAI_ROOT_DIR = "$env:SCRIPT_DIR_EXAMPLE_OV_CPP_RUN\\.."
$env:OPENCV_BIN = "C:\\Users\\xipingya\\Downloads\\opencv\\build\\bin"
$env:PATH = "$env:GENAI_ROOT_DIR\\openvino.genai\\install\\runtime\\bin\\intel64\\Debug;$env:OPENCV_BIN;$env:PATH"

echo "== Start: qwen2vl_app_cpp"
.\\build\\Debug\\qwen2vl_app_cpp.exe $env:model_id $env:type $env:video_img_path $env:device $env:prompt