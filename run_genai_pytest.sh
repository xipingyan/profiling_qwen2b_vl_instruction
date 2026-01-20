
SCRIPT_RUN_GENAI_PYTEST_DIR="$(dirname "$(readlink -f "$BASH_SOURCE")")"
cd ${SCRIPT_RUN_GENAI_PYTEST_DIR}

source ./python-env/bin/activate
# pip install --pre -U openvino openvino-tokenizers==2025.4.0.0.dev20250929 --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
# pip install -r openvino.genai/tests/python_tests/requirements.txt
source ./source_ov.sh
cd ${SCRIPT_RUN_GENAI_PYTEST_DIR}

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
export LD_LIBRARY_PATH=${GENAI_ROOT_DIR}/../runtime/lib/intel64/:$LD_LIBRARY_PATH

# Ensure the Python openvino_tokenizers package uses the GenAI-built extension
# (must be ABI-compatible with the OpenVINO runtime used by setupvars.sh).
export OV_TOKENIZER_PREBUILD_EXTENSION_PATH=${GENAI_ROOT_DIR}/../runtime/lib/intel64/libopenvino_tokenizers.so

export HF_ENDPOINT=https://hf-mirror.com

cd openvino.genai/tests/python_tests
export OV_CACHE=./ov_cache
# python -m pytest ./ -s -m precommit -k test_vlm_continuous_batching_generate_vs_add_request
# python -m pytest ./ -s -k test_stop_strings_facebook_opt
#  gdb --args python -m pytest ./ -s -m precommit -k test_vlm_pipeline_chat_with_video

# python -m pytest ./ -m precommit -k test_add_extension
# python -m pytest ./test_vlm_pipeline.py -m precommit -k test_vlm_pipeline_add_extension

python -m pytest ./samples/test_prompt_lookup_decoding_vlm.py -s -k test_prompt_lookup_decoding_vlm
# python -m pytest ./samples/test_prompt_lookup_decoding_lm.py -s -k test_prompt_lookup_decoding_lm

