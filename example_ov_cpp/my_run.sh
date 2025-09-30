source ../python-env/bin/activate
source ../openvino_toolkit_ubuntu24_2025.4.0.dev20250929_x86_64/setupvars.sh

model_id=../models/ov/Qwen2.5-VL-3B-Instruct/INT4/

# type="img"
# video_img_path=../cat_1.jpg

type="video"
video_img_path=../test_video/rsz_video

device='GPU'

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