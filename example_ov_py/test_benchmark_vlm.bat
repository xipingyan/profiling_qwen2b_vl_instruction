@REM set http_proxy=
@REM set http_proxy=
@REM pip install --pre -U openvino --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly


call .\python-env\Scripts\activate.ba
set PATH="C:\Users\test\Desktop\xiping\openvino.genai\tests\python_tests";%PATH%
python ..\\..\\samples\\python\\visual_language_chat\\benchmark_vlm.py -m C:\\Users\\test\\Desktop\\xiping\\Qwen2.5-VL-3B-Instruct\\INT4 -d GPU -i C:\\Users\\test\\Desktop\\xiping\\home.jpg -mt 64  -p "Please describe the image."
