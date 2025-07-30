source python-env/bin/activate
# dependcy
# pip install openvino-tokenizers openvino nncf

# model_id='./Qwen/Qwen2-VL-2B-Instruct'
# # compress to int4
# # optimum-cli export openvino --model $model_id --task image-text-to-text Qwen2-VL-2B-Instruct/INT4 --weight-format int4 --trust-remote-code
# # optimum-cli export openvino --model $model_id --task video-text-to-text Qwen2-VL-2B-Instruct_video/INT4 --weight-format int4 --trust-remote-code

model_id='Qwen/Qwen2.5-VL-3B-Instruct'
optimum-cli export openvino --model $model_id --task image-text-to-text Qwen2.5-VL-3B-Instruct/INT4 --weight-format int4 --trust-remote-code