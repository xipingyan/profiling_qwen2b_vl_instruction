SCRIPT_DIR_EXAMPLE_OV_CPP_RUN="$(dirname "$(readlink -f "$BASH_SOURCE")")"
cd ${SCRIPT_DIR_EXAMPLE_OV_CPP_RUN}

source ../python-env/bin/activate
source ../source_ov.sh

cd ${SCRIPT_DIR_EXAMPLE_OV_CPP_RUN}

model_id=../models/ov/Qwen2.5-VL-3B-Instruct/INT4/

type="img"
video_img_path=../test_video/home.jpg
prompt="prompt.txt"
echo "Please describe the image." > $prompt

device='GPU'
export CUSTOM_VIT_PATH="../custom_vit"
export CUSTOM_VIT_IMG_PATH="../test_video/home.jpg"
export ENABLE_CUSTOM_VIT=1
./build/qwen2vl_app_cpp $model_id $type $video_img_path $device $prompt