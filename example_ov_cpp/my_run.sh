SCRIPT_DIR="$(dirname "$(readlink -f "$BASH_SOURCE")")"
cd ${SCRIPT_DIR}

source ../python-env/bin/activate
source ../source_ov.sh

cd ${SCRIPT_DIR}

model_id=../models/ov/Qwen2.5-VL-3B-Instruct/INT4/
model_id=../openvino.genai/tests/python_tests/ov_cache/20251015/optimum-intel-1.25.2_transformers-4.53.3/test_models/katuni4ka_tiny-random-qwen2.5-vl/

# type="img"
# video_img_path=../openvino.genai/tests/python_tests/.pytest_cache/d/images/handwritten.png

type="video"
video_img_path=../test_video/rsz_video

device='GPU'
device='CPU'

# gdb --args 
# onetrace --chrome-call-logging --chrome-device-timeline 
# heaptrack 
# LD_LIBRAY_PATH=/mnt/xiping/my_tools/pti-gpu/tools/unitrace/build/:$LD_LIBRAY_PATH

# python /mnt/xiping/my_tools/pti-gpu/tools/unitrace/scripts/roofline/roofline.py  --memory mem.csv --device GPU --output xx.log  
# gdb --args 
# DISABLE_VLSDPA=1
# OV_VERBOSE=6
# ONEDNN_VERBOSE=2 OV_GPU_DUMP_GRAPHS_PATH="./dump_graphs/" 
./build/qwen2vl_app_cpp $model_id $type $video_img_path $device

# unitrace -g VectorEngine138 -q --chrome-kernel-logging -o xx.log ./build/qwen2vl_app_cpp $model_id $type $video_img_path