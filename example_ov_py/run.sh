
# PYTHON API test: openvino/genai need to share same python ENV.
source ../python-env/bin/activate
source ../../openvino/build/install/setupvars.sh
GENAI_ROOT_DIR=/mnt/xiping/gpu_profiling/profiling_qwen2b_vl_instruction/openvino.genai/install/python/
export PYTHONPATH=${GENAI_ROOT_DIR}:$PYTHONPATH
export LD_LIBRARY_PATH=${GENAI_ROOT_DIR}/../runtime/lib/intel64/:$LD_LIBRARY_PATH

# source ov_venv/bin/activate

# gdb --args 
python ./test_pipeline_genai.py
# python ./test_pipeline_optimum.py
# python test_llm_genai.py
