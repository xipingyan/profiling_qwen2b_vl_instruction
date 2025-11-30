SCRIPT_DIR_OV_PY="$(dirname "$(readlink -f "$BASH_SOURCE")")"
cd ${SCRIPT_DIR_OV_PY}

# PYTHON API test: openvino/genai need to share same python ENV.
source ../python-env/bin/activate
source ../source_ov.sh
cd ${SCRIPT_DIR_OV_PY}

GENAI_ROOT_DIR=../openvino.genai/install/python/
export PYTHONPATH=${GENAI_ROOT_DIR}:$PYTHONPATH
export LD_LIBRARY_PATH=${GENAI_ROOT_DIR}/../runtime/lib/intel64/:$LD_LIBRARY_PATH

# gdb --args 
# python ./test_pipeline_genai.py
python ./test_pipeline_genai_eagle3.py

# python ./test_pipeline_optimum.py
# python test_llm_genai.py
# python test_vlm_continuous_batching_generate_vs_add_request.py
