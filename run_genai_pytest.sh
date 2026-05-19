
SCRIPT_RUN_GENAI_PYTEST_DIR="$(dirname "$(readlink -f "$BASH_SOURCE")")"
cd ${SCRIPT_RUN_GENAI_PYTEST_DIR}

source ./python-env/bin/activate
# pip install --pre -U openvino openvino-tokenizers==2026.2.0.0.dev20250929 --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
# uv pip install --prerelease=allow -r openvino.genai/tests/python_tests/requirements.txt
# uv pip install pytest

# Build OV also need this VENV.
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

# Avoid external download instability for cat fixture image in test_vlm_pipeline.py
# (pytest cache path used by pytestconfig.cache.mkdir("images")).
mkdir -p .pytest_cache/d/images
if [ ! -f .pytest_cache/d/images/cat.jpg ]; then
python - <<'PY'
from PIL import Image, ImageDraw

img = Image.new("RGB", (512, 512), color=(245, 245, 245))
draw = ImageDraw.Draw(img)
draw.ellipse((140, 120, 380, 360), fill=(180, 170, 160), outline=(90, 90, 90), width=4)
draw.ellipse((190, 180, 235, 225), fill=(30, 30, 30))
draw.ellipse((285, 180, 330, 225), fill=(30, 30, 30))
draw.polygon([(256, 235), (236, 270), (276, 270)], fill=(200, 110, 110))
draw.line((220, 295, 292, 295), fill=(70, 70, 70), width=3)
img.save(".pytest_cache/d/images/cat.jpg", format="JPEG", quality=95)
print("Generated local .pytest_cache/d/images/cat.jpg")
PY
fi

# python -m pytest ./ -s -m precommit -k test_vlm_continuous_batching_generate_vs_add_request
# python -m pytest ./ -s -k test_stop_strings_facebook_opt
#  gdb --args python -m pytest ./ -s -m precommit -k test_vlm_pipeline_chat_with_video

# python -m pytest ./ -m precommit -k test_add_extension
# python -m pytest ./test_vlm_pipeline.py -m precommit -k test_vlm_pipeline_add_extension

# python -m pytest ./samples/test_prompt_lookup_decoding_vlm.py -s -k test_prompt_lookup_decoding_vlm
# python -m pytest ./samples/test_prompt_lookup_decoding_lm.py -s -k test_prompt_lookup_decoding_lm

python -m pytest -s './test_vlm_pipeline.py::test_vision_pos_embeds_modes_equivalence[optimum-intel-internal-testing/tiny-random-qwen3-vl/PA]'
