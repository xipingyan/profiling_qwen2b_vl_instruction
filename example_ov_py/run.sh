SCRIPT_DIR="$(dirname "$(readlink -f "$BASH_SOURCE")")"
cd ${SCRIPT_DIR}

# PYTHON API test: openvino/genai need to share same python ENV.
source ../python-env/bin/activate
source ../openvino_toolkit_ubuntu24_2025.4.0.dev20250929_x86_64/setupvars.sh

GENAI_ROOT_DIR=../openvino.genai/install/python/
export PYTHONPATH=${GENAI_ROOT_DIR}:$PYTHONPATH
export LD_LIBRARY_PATH=${GENAI_ROOT_DIR}/../runtime/lib/intel64/:$LD_LIBRARY_PATH

# source ov_venv/bin/activate

# gdb --args 
python ./test_pipeline_genai.py
# python ./test_pipeline_optimum.py
# python test_llm_genai.py
# python test_vlm_continuous_batching_generate_vs_add_request.py
