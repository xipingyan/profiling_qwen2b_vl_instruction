app=./openvino/bin/intel64/Release/benchmark_app

# source ./download_ov/openvino_toolkit_ubuntu24_2025.1.0.18503.6fec06580ab_x86_64/setupvars.sh
# app=/mnt/xiping/openvino_cpp_samples_build/intel64/Release/benchmark_app

# model_ir=./ov_model_i8/openvino_vision_embeddings_model.xml
# $app -d GPU -m $model_ir -data_shape hidden_states[1440,1176] -nthreads 1 -nstreams 1 -hint none -infer_precision f16 -nireq 1 -niter 10

model_ir=./ov_model_i8/openvino_vision_embeddings_merger_model.xml
$app -d GPU -m $model_ir -data_shape hidden_states[1836,1280],attention_mask[1,1836,1836],rotary_pos_emb[1836,40] -nthreads 1 -nstreams 1 -hint none -infer_precision f16 -nireq 1 -niter 10