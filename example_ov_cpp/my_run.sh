source ../python-env/bin/activate
# source ../openvino_toolkit_ubuntu22_2025.3.0.dev20250725_x86_64/setupvars.sh 
# source ../openvino_toolkit_ubuntu22_2025.2.0.19140.c01cd93e24d_x86_64/setupvars.sh
source /mnt/xiping/gpu_profiling/openvino/build/install/setupvars.sh

model_id=../models/ov/Qwen2.5-VL-3B-Instruct/INT4/

type="video"
type="img"
# video_img_path=../cat_1.jpg
video_img_path=/mnt/xiping/gpu_profiling/profiling_qwen2b_vl_instruction/test_video/rsz_video

# gdb --args 
# onetrace --chrome-call-logging --chrome-device-timeline 
# heaptrack 
# LD_LIBRAY_PATH=/mnt/xiping/my_tools/pti-gpu/tools/unitrace/build/:$LD_LIBRAY_PATH

# python /mnt/xiping/my_tools/pti-gpu/tools/unitrace/scripts/roofline/roofline.py  --memory mem.csv --device GPU --output xx.log  
# gdb --args 
# DISABLE_VLSDPA=1
# OV_VERBOSE=6
# ONEDNN_VERBOSE=2 OV_GPU_DUMP_GRAPHS_PATH="./dump_graphs/" 
./build/qwen2vl_app_cpp $model_id $type $video_img_path

# unitrace -g VectorEngine138 -q --chrome-kernel-logging -o xx.log ./build/qwen2vl_app_cpp $model_id $type $video_img_path