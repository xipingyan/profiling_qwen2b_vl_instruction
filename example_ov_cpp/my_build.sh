source ../python-env/bin/activate
source ../../../openvino/build/install/setupvars.sh

mkdir -p build
cd build

# Based on myown build openvino.genai
export CMAKE_PREFIX_PATH="../../openvino.genai/install/runtime/cmake/"

cmake ..
make -j32
