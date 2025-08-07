
workpath=/mnt/xiping/gpu_profiling/profiling_qwen2b_vl_instruction
cd ${workpath}

source ./python-env/bin/activate
source ./openvino_toolkit_ubuntu22_2025.3.0.dev20250725_x86_64/setupvars.sh
# source ./openvino_toolkit_ubuntu22_2025.2.0.19140.c01cd93e24d_x86_64/setupvars.sh

# Download codes:
# ================================
# git clone https://github.com/xipingyan/openvino.genai.git
# cd openvino.genai/
# git submodule update --init

cd openvino.genai
# cmake -DCMAKE_BUILD_TYPE=Release -S ./ -B ./build/
# cmake --build ./build/ --config Release -j 200
# cmake --install ./build/ --config Release --prefix ./install

# Debug
# cmake -DCMAKE_BUILD_TYPE=Debug -S ./ -B ./build/
cmake --build ./build/ --config Debug -j
cmake --install ./build/ --config Debug --prefix ./install
