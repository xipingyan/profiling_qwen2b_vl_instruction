SCRIPT_DIR_EXAMPLE_OV_CPP="$(dirname "$(readlink -f "$BASH_SOURCE")")"
cd ${SCRIPT_DIR_EXAMPLE_OV_CPP}

source ../python-env/bin/activate
source ../source_ov.sh

cd ${SCRIPT_DIR_EXAMPLE_OV_CPP}

mkdir -p build
cd build

# Based on myown build openvino.genai
export CMAKE_PREFIX_PATH="../../openvino.genai/install/runtime/cmake/"

cmake -DCMAKE_BUILD_TYPE=Debug ..
# cmake -DCMAKE_BUILD_TYPE=Release ..
make -j32
