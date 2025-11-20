
# python -m venv python-env
. ..\python-env\Scripts\Activate.ps1
# pip install -r .\openvino.genai\tests\python_tests\requirements.txt

# Windows nightly don't contain python package, so build ov from source.
# . ..\..\openvino\build\install\setupvars.ps1
# . ..\openvino_toolkit_windows_2026.0.0.dev20251117_x86_64\setupvars.ps1
. ..\openvino_toolkit_windows_2025.4.0.dev20251105_x86_64\setupvars.ps1


# mkdir -p build
cd build

# Based on myown build openvino.genai
$env:CMAKE_PREFIX_PATH = "../../openvino.genai/install/runtime/cmake/;C:\Users\xipingya\Downloads\opencv\build"

cmake -DCMAKE_BUILD_TYPE=Debug ..
# cmake -DCMAKE_BUILD_TYPE=Release ..
# make -j32

cd ..