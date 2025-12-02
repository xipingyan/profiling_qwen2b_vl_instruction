
SCRIPT_RUN_GENAI_PYTEST_DIR="$(dirname "$(readlink -f "$BASH_SOURCE")")"
cd ${SCRIPT_RUN_GENAI_PYTEST_DIR}

source ./python-env/bin/activate
source ./source_ov.sh
cd ${SCRIPT_RUN_GENAI_PYTEST_DIR}

GENAI_ROOT_DIR=${SCRIPT_RUN_GENAI_PYTEST_DIR}/openvino.genai/install/python/
export PYTHONPATH=${GENAI_ROOT_DIR}:$PYTHONPATH
export LD_LIBRARY_PATH=${GENAI_ROOT_DIR}../runtime/lib/intel64/:$LD_LIBRARY_PATH


export OV_CACHE=./ov_cache
app=./openvino.genai/samples/python/visual_language_chat/benchmark_vlm.py

model_path=./models/eagle3/qwen2.5-vl-7b-ov-int4
draft_model=./models/eagle3/qwen2.5-vl-7b-eagle3-ov-int4
img_path=./test_video/IMG_20250723_145708_008_IN.jpg

# gdb --args 
python $app -n 3 -d GPU -mt 100 -m $model_path -dm $draft_model -p "简单描述一下这张图"  -i $img_path