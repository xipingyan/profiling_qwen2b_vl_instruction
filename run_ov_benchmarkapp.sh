app=./openvino/bin/intel64/Release/benchmark_app

# source ./download_ov/openvino_toolkit_ubuntu24_2025.1.0.18503.6fec06580ab_x86_64/setupvars.sh
# app=/mnt/xiping/openvino_cpp_samples_build/intel64/Release/benchmark_app

# vl_1=================================
# model_ir=./ov_model_i8/openvino_vision_embeddings_model.xml
# $app -d GPU -m $model_ir -data_shape hidden_states[1440,1176] -nthreads 1 -nstreams 1 -hint none -infer_precision f16 -nireq 1 -niter 10

# vl_2=================================
# 1836
model_ir=./ov_model_i8/openvino_vision_embeddings_merger_model.xml
# OV_VERBOSE=4 
# mkdir dump_graphs_add_before
# OV_GPU_DUMP_GRAPHS_PATH="./dump_graphs_add_before/" 
$app -d GPU -m $model_ir -data_shape hidden_states[1440,1280],attention_mask[1,1440,1440],rotary_pos_emb[1440,40] \
    -nthreads 1 -nstreams 1 -hint none \
    -infer_precision f16 -nireq 1 -niter 10
    # -exec_graph_path exec_graph.xml  
    # > benchmakr_vl.log 

# # llm=================================
# model_ir=ov_model_i8/openvino_language_model.xml
# mkdir dump_llm_graphs_add_before
# OV_GPU_DUMP_GRAPHS_PATH="./dump_llm_graphs_add_before/" $app -d GPU -m $model_ir -data_shape inputs_embeds[1,409,1536],attention_mask[1,409],position_ids[3,409,1],beam_idx[1] \
#     -nthreads 1 -nstreams 1 -hint none \
#     -infer_precision f16 -nireq 1 -niter 10