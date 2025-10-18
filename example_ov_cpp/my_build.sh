SCRIPT_DIR="$(dirname "$(readlink -f "$BASH_SOURCE")")"
cd ${SCRIPT_DIR}

source ../python-env/bin/activate
source ../source_ov.sh

cd ${SCRIPT_DIR}

mkdir -p build
cd build

# Based on myown build openvino.genai
export CMAKE_PREFIX_PATH="../../openvino.genai/install/runtime/cmake/"

cmake -DCMAKE_BUILD_TYPE=Debug ..
# cmake -DCMAKE_BUILD_TYPE=Release ..
make -j32
