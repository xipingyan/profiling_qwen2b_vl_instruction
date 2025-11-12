
SCRIPT_DIR_BUILD_GENAI="$(dirname "$(readlink -f "$BASH_SOURCE")")"
cd ${SCRIPT_DIR_BUILD_GENAI}

source ./python-env/bin/activate
source ./source_ov.sh

echo $SCRIPT_DIR_BUILD_GENAI

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

cd openvino.genai
# cmake -DCMAKE_BUILD_TYPE=Release -S ./ -B ./build/
# cmake --build ./build/ --config Release -j 200
# cmake --install ./build/ --config Release --prefix ./install

# Debug
# cmake -DCMAKE_BUILD_TYPE=Debug -S ./ -B ./build/
cmake --build ./build/ --config Debug -j 20
cmake --install ./build/ --config Debug --prefix ./install
