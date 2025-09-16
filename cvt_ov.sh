source python-env/bin/activate
# dependcy
# pip install openvino-tokenizers openvino nncf optimum[intel]
# pip install -U huggingface_hub

mkdir -p models
cd models

# model_id='Qwen/Qwen2.5-VL-7B-Instruct'
# model_id='Qwen/Qwen2.5-VL-3B-Instruct'

# Refer: https://hf-mirror.com/
# export HF_ENDPOINT=https://hf-mirror.com
# huggingface-cli download --token [your token] --resume-download $model_id --local-dir $model_id

# compress to int4
# optimum-cli export openvino --model $model_id --task image-text-to-text ov/Qwen2.5-VL-3B-Instruct/INT4 --weight-format int4 --trust-remote-code
# optimum-cli export openvino --model $model_id --task video-text-to-text ov/Qwen2-VL-2B-Instruct_video/INT4 --weight-format int4 --trust-remote-code

model_id='Qwen/Qwen2.5-VL-3B-Instruct'
model_id='Qwen/Qwen2.5-VL-7B-Instruct'
optimum-cli export openvino --model $model_id --task image-text-to-text Qwen2.5-VL-7B-Instruct/INT4 --weight-format int4 --trust-remote-code