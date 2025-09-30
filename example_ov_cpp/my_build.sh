SCRIPT_DIR="$(dirname "$(readlink -f "$BASH_SOURCE")")"
cd ${SCRIPT_DIR}

source ../python-env/bin/activate
source ../openvino_toolkit_ubuntu24_2025.4.0.dev20250929_x86_64/setupvars.sh
# source /mnt/xiping/gpu_profiling/openvino/build/install/setupvars.sh

mkdir -p build
cd build

# Based on myown build openvino.genai
export CMAKE_PREFIX_PATH="../../openvino.genai/install/runtime/cmake/"

cmake -DCMAKE_BUILD_TYPE=Debug ..
# cmake -DCMAKE_BUILD_TYPE=Release ..
make -j32
