
# python -m venv python-env
. .\python-env\Scripts\Activate.ps1
# pip install -r .\openvino.genai\tests\python_tests\requirements.txt

# Windows nightly don't contain python package, so build ov from source.
. ..\openvino\build\install\setupvars.ps1

cd openvino.genai
python -m pip wheel . -w dist/ --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/pre-release --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
cd ..