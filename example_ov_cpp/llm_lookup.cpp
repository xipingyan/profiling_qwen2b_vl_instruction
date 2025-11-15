// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "load_image.hpp"
#include <openvino/genai/visual_language/pipeline.hpp>
#include <filesystem>
#include <chrono>

int test_llm_lookup(int argc, char* argv[]) {
    ov::AnyMap enable_compile_cache;
    std::string model_path = "/mnt/xiping/gpu_profiling/profiling_qwen2b_vl_instruction/OpenVINO/Qwen2-0.5B-int8-ov/";
    std::string device = "GPU.0";
    if (device == "GPU") {
        // Cache compiled models on disk for GPU to save time on the
        // next run. It's not beneficial for CPU.
        enable_compile_cache.insert({ov::cache_dir("vlm_cache")});
        std::cout << "    enable_compile_cache = " << "vlm_cache" << std::endl;
    }

    bool enable_look_up = false;
    if (std::getenv("LOOKUP") && std::string(std::getenv("LOOKUP")) == std::string("1")) {
        enable_look_up = true;
    }
    std::cout << "== enable_look_up = " << enable_look_up << std::endl;

    ov::genai::LLMPipeline pipe(model_path, device, ov::genai::prompt_lookup(enable_look_up));

    ov::genai::GenerationConfig config;
    config.max_new_tokens = 20;
    // config.num_beam_groups = 3;
    // config.num_beams = 15;
    // config.diversity_penalty = 1.0f;
    // config.num_return_sequences = config.num_beams;
    config.do_sample=false;
    config.temperature=0.1;

    if (enable_look_up)
    {
        // add parameter to enable prompt lookup decoding to generate `num_assistant_tokens` candidates per iteration
        config.num_assistant_tokens = 5;
        // Define max_ngram_size
        config.max_ngram_size = 3;
    }

    // config.apply_chat_template=true;
    // pipe.get_tokenizer().set_chat_template("{\"enable_thinking\": False}");

    // ov::genai::ChatHistory history({{{"role", "user"}, 
    //     {"content", "You are an AI assistant, please answer with json format."}}});
    // pipe.get_tokenizer().apply_chat_template(history, true);

    for (size_t i = 0;i < 3; i++) {
        pipe.start_chat();
        auto t1 = std::chrono::high_resolution_clock::now();
        auto outputs = pipe.generate("What is the capital of China?", config);
        auto t2 = std::chrono::high_resolution_clock::now();
        pipe.finish_chat();
        std::cout << "time:" << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms, " << outputs << '\n';
    }
    return 0;
}

int test_vllm_lookup(int argc, char* argv[]) {
    std::cout << "== test_vllm_lookup ..." << std::endl;
    ov::AnyMap enable_compile_cache;
    std::string model_path = "../models/ov/Qwen2.5-VL-3B-Instruct/INT4/";
    std::string device = "GPU.0";
    if (device == "GPU") {
        enable_compile_cache.insert({ov::cache_dir("vlm_cache")});
        std::cout << "    enable_compile_cache = " << "vlm_cache" << std::endl;
    }

    std::cout << "== Set ENV: LOOKUP=1 to enable LOOKUP ..." << std::endl;
    bool enable_look_up = false;
    if (std::getenv("LOOKUP") && std::string(std::getenv("LOOKUP")) == std::string("1")) {
        enable_look_up = true;
    }
    std::cout << "  == enable_look_up = " << enable_look_up << std::endl;

    ov::genai::GenerationConfig config;
    config.max_new_tokens = 20;
    config.do_sample=false;
    config.temperature=0.1;
    if (enable_look_up)
    {
        // add parameter to enable prompt lookup decoding to generate `num_assistant_tokens` candidates per iteration
        config.num_assistant_tokens = 5;
        // Define max_ngram_size
        config.max_ngram_size = 3;
    }

    ov::AnyMap cfg;
    // cfg["ATTENTION_BACKEND"] = "SDPA";
    cfg["ATTENTION_BACKEND"] = "PA";
    // cfg["prompt_lookup"] = enable_look_up;

    ov::genai::VLMPipeline pipe(model_path, device, cfg);

    auto images = utils::load_images("../test_video/rsz_0.png");
    std::string prompts = "Is there animal in this image? please answer like: \"There is 2 ducks in this image.\"";

    for (size_t i = 0;i < 1; i++) {
        // pipe.start_chat();
        auto t1 = std::chrono::high_resolution_clock::now();
        auto outputs = pipe.generate(prompts, ov::genai::image(images[0]), ov::genai::generation_config(config));
        auto t2 = std::chrono::high_resolution_clock::now();
        // pipe.finish_chat();
        std::cout << "time:" << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms, " << outputs << '\n';
    }
    return 0;
}