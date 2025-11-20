
$env:http_proxy = "http://proxy-dmz.intel.com:912"
$env:https_proxy = "http://proxy-dmz.intel.com:912"

# python -m venv python-env
. .\python-env\Scripts\Activate.ps1
# pip install -r .\openvino.genai\tests\python_tests\requirements.txt

# Windows nightly don't contain python package, so build ov from source.
# . ..\openvino\build\install\setupvars.ps1
. openvino_toolkit_windows_2025.4.0.dev20251105_x86_64\setupvars.ps1
# . openvino_toolkit_windows_2026.0.0.dev20251117_x86_64\setupvars.ps1


cd openvino.genai
# cmake -DCMAKE_BUILD_TYPE=Release -DOpenVINO_DIR=C:\Users\openvino-ci-88\xiping\openvino\build -DENABLE_PYTHON_PACKAGING=ON -S ./ -B ./build/
# cmake --build ./build/ --config Release -j 96
# cmake --install ./build/ --config Release --prefix ./install

# Debug
cmake -G "Visual Studio 17 2022" -DCMAKE_BUILD_TYPE=Debug -S ./ -B ./build/
cmake --build ./build/ --config Debug -j 30
cmake --install ./build/ --config Debug --prefix ./install

cd ..