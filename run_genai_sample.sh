
SCRIPT_RUN_GENAI_SAMPLE_DIR="$(dirname "$(readlink -f "$BASH_SOURCE")")"
cd ${SCRIPT_RUN_GENAI_SAMPLE_DIR}

source ./python-env/bin/activate
# OV
source ./source_ov.sh
cd ${SCRIPT_RUN_GENAI_SAMPLE_DIR}

# GenAI
GENAI_ROOT_DIR=${SCRIPT_RUN_GENAI_SAMPLE_DIR}/openvino.genai/install/python/
export PYTHONPATH=${GENAI_ROOT_DIR}:$PYTHONPATH
export LD_LIBRARY_PATH=${GENAI_ROOT_DIR}../runtime/lib/intel64/:$LD_LIBRARY_PATH

export HF_ENDPOINT=https://hf-mirror.com

MODEL_ID=${SCRIPT_RUN_GENAI_SAMPLE_DIR}/models/ov/Qwen2.5-VL-3B-Instruct/INT4/
IMG_FN=${SCRIPT_RUN_GENAI_SAMPLE_DIR}/./test_video/home.jpg
IMG_FN=${SCRIPT_RUN_GENAI_SAMPLE_DIR}/test_video/IMG_20250723_145708_008_IN.jpg
IMG_FN=${SCRIPT_RUN_GENAI_SAMPLE_DIR}/test_video/cat_120_100.png
PROMPT="Please describe this image"
PROMPT="请描述图片"

# python sample
# ==============================================
# cd openvino.genai/samples/python/visual_language_chat/
# export OV_CACHE=./ov_cache

# python prompt_lookup_decoding_vlm.py $MODEL_ID $IMG_FN "$PROMPT"
# # python prompt_lookup_decoding_vlm.py $MODEL_ID $IMG_FN "$PROMPT" --disable_lookup

# cpp sample
# ==============================================
cd openvino.genai/samples/cpp/visual_language_chat
mkdir -p build
cd build

export OV_CACHE=./ov_cache

export CMAKE_PREFIX_PATH=${GENAI_ROOT_DIR}../runtime/cmake
echo $CMAKE_PREFIX_PATH
cmake ..
make -j20

# ./prompt_lookup_decoding_vlm $MODEL_ID $IMG_FN "$PROMPT"