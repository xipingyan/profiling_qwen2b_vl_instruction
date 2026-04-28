source py_export_main_model/bin/activate
# uv pip install "optimum[openvino,nncf]"

main_model_dir=../../models/Qwen/Qwen3-4B/
output_dir=exported_main_model

optimum-cli export openvino --model $main_model_dir --task text-generation $output_dir/INT4 --weight-format int4 --trust-remote-code