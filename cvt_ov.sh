source venv/bin/activate

model_id='./Qwen/Qwen2-VL-2B-Instruct'
# optimum-cli export openvino --model $model_id --task image-text-to-text ./out_img_dir
# optimum-cli export openvino --model $model_id --task video-text-to-text ./out_video_dir

# compress to int4
optimum-cli export openvino --model $model_id --task image-text-to-text Qwen2-VL-2B-Instruct/INT4 --weight-format int4
# optimum-cli export openvino --model $model_id --task video-text-to-text Qwen2-VL-2B-Instruct/INT4 --weight-format int4
