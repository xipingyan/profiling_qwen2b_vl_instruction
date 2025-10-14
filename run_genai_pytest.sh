
SCRIPT_RUN_GENAI_PYTEST_DIR="$(dirname "$(readlink -f "$BASH_SOURCE")")"
cd ${SCRIPT_RUN_GENAI_PYTEST_DIR}

source ./python-env/bin/activate
# openvino_tokenizer: 
# pip install --pre -U openvino openvino-tokenizers==2025.4.0.0.dev20250929 --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
source ./openvino_toolkit_ubuntu24_2025.4.0.dev20250929_x86_64/setupvars.sh

# source /mnt/xiping/gpu_profiling/openvino/build/install/setupvars.sh
# export INSTALL_DIR=/mnt/xiping/gpu_profiling/openvino/build/install/
# export OpenVINO_DIR=${INSTALL_DIR}/runtime
# export PYTHONPATH=${INSTALL_DIR}/python:./build/:$PYTHONPATH
# export LD_LIBRARY_PATH=${INSTALL_DIR}/runtime/lib/intel64:$LD_LIBRARY_PATH

# Download codes:
# ================================
# git clone https://github.com/xipingyan/openvino.genai.git
# cd openvino.genai/
# git submodule update --init

GENAI_ROOT_DIR=${SCRIPT_RUN_GENAI_PYTEST_DIR}/openvino.genai/install/python/
export PYTHONPATH=${GENAI_ROOT_DIR}:$PYTHONPATH
export LD_LIBRARY_PATH=${GENAI_ROOT_DIR}../runtime/lib/intel64/:$LD_LIBRARY_PATH

export HF_ENDPOINT=https://hf-mirror.com
# export HF_ENDPOINT=https://huggingface.co/

cd openvino.genai/tests/python_tests
export OV_CACHE=./ov_cache
# python -m pytest ./ -s -m precommit -k test_vlm_pipeline_video_input
# python -m pytest ./ -s -m precommit -k test_vlm_continuous_batching_generate_vs_add_request
# python -m pytest ./ -s -m precommit -k test_vlm_pipeline_match_optimum_preresized

 gdb --args python -m pytest ./ -s -m precommit -k test_vlm_pipeline_chat_with_video
