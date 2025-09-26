
workpath=/mnt/xiping/gpu_profiling/profiling_qwen2b_vl_instruction
cd ${workpath}

source ./python-env/bin/activate
# source ./openvino_toolkit_ubuntu22_2025.3.0.dev20250725_x86_64/setupvars.sh
source /mnt/xiping/gpu_profiling/openvino/build/install/setupvars.sh

# Download codes:
# ================================
# git clone https://github.com/xipingyan/openvino.genai.git
# cd openvino.genai/
# git submodule update --init

export INSTALL_DIR=/mnt/xiping/gpu_profiling/openvino/build/install/
export OpenVINO_DIR=${INSTALL_DIR}/runtime
export PYTHONPATH=${INSTALL_DIR}/python:./build/:$PYTHONPATH
export LD_LIBRARY_PATH=${INSTALL_DIR}/runtime/lib/intel64:$LD_LIBRARY_PATH

GENAI_ROOT_DIR=/mnt/xiping/gpu_profiling/profiling_qwen2b_vl_instruction/openvino.genai/install/python/
export PYTHONPATH=${GENAI_ROOT_DIR}:$PYTHONPATH
export LD_LIBRARY_PATH=${GENAI_ROOT_DIR}/../runtime/lib/intel64/:$LD_LIBRARY_PATH

export HF_ENDPOINT=https://hf-mirror.com

cd openvino.genai/tests/python_tests
# export OV_CACHE=
python -m pytest ./ -s -m precommit -k test_vlm_pipeline_video_input