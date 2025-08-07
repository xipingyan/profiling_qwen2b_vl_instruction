source ../python-env/bin/activate
# source ~/openvino/build/install/setupvars.sh

GENAI_ROOT_DIR=/mnt/xiping/gpu_profiling/profiling_qwen2b_vl_instruction/openvino.genai/install/python/
OV_ROOT_DIR=/mnt/xiping/gpu_profiling/profiling_qwen2b_vl_instruction/openvino_toolkit_ubuntu22_2025.3.0.dev20250725_x86_64
export PYTHONPATH=${GENAI_ROOT_DIR}:${GENAI_ROOT_DIR}openvino_genai:${OV_ROOT_DIR}/python/:$PYTHONPATH
export LD_LIBRARY_PATH=${GENAI_ROOT_DIR}/../runtime/lib/intel64/:${OV_ROOT_DIR}/runtime/lib/intel64/:$LD_LIBRARY_PATH

# gdb --args 
python ./test_pipeline_genai.py
#  python ./test_pipeline_optimum.py
