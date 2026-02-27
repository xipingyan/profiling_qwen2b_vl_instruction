source python-env/bin/activate
# dependcy
# pip install openvino-tokenizers openvino nncf optimum[intel]
# pip install -U huggingface_hub

mkdir -p models
cd models

# model_id='Qwen/Qwen2.5-VL-7B-Instruct'
# model_id='Qwen/Qwen2.5-VL-3B-Instruct'
model_id='Qwen/Qwen3.5-35B-A3B-Base'
# model_id='katuni4ka/tiny-random-qwen2.5-vl/'
# model_id='katuni4ka/tiny-random-minicpmv-2_6'
# model_id='OpenVINO/Qwen2-0.5B-int8-ov'


# Refer: https://hf-mirror.com/
export HF_ENDPOINT=https://hf-mirror.com
# huggingface-cli download --token [your token] --resume-download $model_id --local-dir $model_id

# compress to int4
# optimum-cli export openvino --model $model_id --task image-text-to-text ov/Qwen2.5-VL-3B-Instruct/INT4 --weight-format int4 --trust-remote-code
# optimum-cli export openvino --model $model_id --task video-text-to-text ov/Qwen2-VL-2B-Instruct_video/INT4 --weight-format int4 --trust-remote-code

model_id='Qwen/Qwen2.5-VL-3B-Instruct'
model_id='Qwen/Qwen2.5-VL-7B-Instruct'
model_id='katuni4ka/tiny-random-qwen2.5-vl/'
model_id='katuni4ka/tiny-random-qwen2vl'
model_id='katuni4ka/tiny-random-minicpmv-2_6'
optimum-cli export openvino --model $model_id --task image-text-to-text $model_id/INT4 --weight-format int4 --trust-remote-code