# profiling_qwen2b_vl_instruction
Profiling and try to optimize qwen2_vl_2b_instruction model based on OpenVINO.

# Download model and convert OV

1. Download model 'Qwen/Qwen2-VL-2B-Instruct' (refer: https://hf-mirror.com/)

```
    # pip install -U huggingface_hub
    export HF_ENDPOINT=https://hf-mirror.com
    model_id='Qwen/Qwen2-VL-2B-Instruct'
    huggingface-cli download --token <your token> --resume-download $model_id --local-dir $model_id
```

2. Convert to OV.

Refer: https://docs.openvino.ai/2024/notebooks/qwen2-vl-with-output.html

```
    python3 -m venv pyenv
    source pyenv/bin/activate
    pip install openvino-genai
    pip install requests pillow

    cvt_ov.sh
```

3. Run example python.

```
    source pyenv/bin/activate

    pip freeze | grep genai
    openvino-genai==2025.1.0.0

    python example_ov_py/test_pipeline.py 
```

4. Run example cpp.

Download openvino_genai and decompress to:(for example) <br>
download_ov/openvino_genai_ubuntu24_2025.1.0.0_x86_64/  <br>

```
    deactivate
    source download_ov/openvino_genai_ubuntu24_2025.1.0.0_x86_64/setupvars.sh

    cd example_ov_cpp
    mkdir build && cd build
    ./qwen2vl_app_cpp
    ./qwen2vl_app_cpp <model_dir> <img fn>
```

4.1 Update OpenVINO.genai, local build and test.

Build genai, please [refer](https://github.com/xipingyan/openvino.genai/blob/xp/add_get_logits_score/src/docs/BUILD.md#build-openvino-openvino-tokenizers-and-openvino-genai-from-source).

```
    git clone https://github.com/xipingyan/openvino.genai.git --branch xp/add_get_logits_score
    git submodule update --init

    cd openvino.genai
    source ../download_ov/openvino_genai_ubuntu24_2025.1.0.0_x86_64/setupvars.sh

    cmake -DCMAKE_BUILD_TYPE=Release -S ./ -B ./build/
    cmake --build ./build/ --config Release -j
    cmake --install ./build/ --config Release --prefix ./install_release
```

Copy libs to genai libs. and test app.

```
    cp openvino.genai/install_release/runtime/lib/intel64/* ./download_ov/openvino_genai_ubuntu24_2025.1.0.0_x86_64/runtime/lib/intel64/

    source download_ov/openvino_genai_ubuntu24_2025.1.0.0_x86_64/setupvars.sh
    example_ov_cpp/build
    rm -rf *
    cmake ..
    make -j20
    ./qwen2vl_app_cpp 
```

onetrace profiling

```
    onetrace --chrome-call-logging --chrome-device-timeline ./qwen2vl_app_cpp
```

4.2 Use latest OpenVINO.genai(with patch) + latest OV, local build and test.

    python3 -m venv env_cpp
    source env_cpp/bin/activate

    <!-- build ov -->
    cd openvino && mkdir build && cd build
    cmake -DENABLE_DEBUG_CAPS=ON -DENABLE_AUTO=OFF -DENABLE_AUTO_BATCH=OFF -DENABLE_INTEL_NPU=OFF -DCMAKE_INSTALL_PREFIX=install ..
    make -j32 && make install

    <!-- build genai -->
    git clone https://github.com/xipingyan/openvino.genai.git --branch xp/return_logits_via_score
    cd openvino.genai
    cmake -DCMAKE_BUILD_TYPE=Release -S ./ -B ./build/
    cmake --build ./build/ --config Release -j

    <!-- Run sample -->
    ./build/samples/cpp/visual_language_chat/visual_language_chat ../ov_model_i8/ ../cat_1.jpg GPU

    <!-- onetrace profiling -->
    onetrace --chrome-call-logging --chrome-device-timeline ./build/samples/cpp/visual_language_chat/visual_language_chat ../ov_model_i8/ ../cat_1.jpg GPU