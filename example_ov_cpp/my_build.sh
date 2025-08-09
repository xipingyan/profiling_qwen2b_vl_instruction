source ../python-env/bin/activate
# source ../../../openvino/build/install/setupvars.sh
source ../openvino_toolkit_ubuntu22_2025.3.0.dev20250725_x86_64/setupvars.sh 
# source ../openvino_toolkit_ubuntu22_2025.2.0.19140.c01cd93e24d_x86_64/setupvars.sh

mkdir -p build
cd build

# Based on myown build openvino.genai
export CMAKE_PREFIX_PATH="../../openvino.genai/install/runtime/cmake/"
export CMAKE_PREFIX_PATH="../../openvino.genai/install/runtime/cmake/"

# cmake -DCMAKE_BUILD_TYPE=Debug ..
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j32
