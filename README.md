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

```
    Refer: https://docs.openvino.ai/2024/notebooks/qwen2-vl-with-output.html
    cvt_ov.sh
```

3. Run example python.

```
    python3 -m venv pyenv
    source pyenv/bin/activate
    pip install openvino-genai
    pip install requests pillow

    pip freeze | grep genai
    openvino-genai==2025.1.0.0
```

4. Run example cpp.

```
    Download openvino_genai and decompress to:(for example)
    download_ov/openvino_genai_ubuntu24_2025.1.0.0_x86_64/

    deactivate
    source download_ov/openvino_genai_ubuntu24_2025.1.0.0_x86_64/setupvars.sh
```