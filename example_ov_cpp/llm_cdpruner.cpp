// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "load_image.hpp"
#include <openvino/genai/visual_language/pipeline.hpp>
#include <filesystem>
#include <chrono>

int test_llm_cdpruner(int argc, char* argv[]) {
    std::cout << "== test_llm_cdpruner ..." << std::endl;
    ov::AnyMap enable_compile_cache;
    std::string model_path = "../models/ov/Qwen2.5-VL-3B-Instruct/INT4/";
    std::string device = "GPU.0";
    if (device == "GPU") {
        enable_compile_cache.insert({ov::cache_dir("vlm_cache")});
        std::cout << "    enable_compile_cache = " << "vlm_cache" << std::endl;
    }

    ov::genai::GenerationConfig config;
    config.max_new_tokens = 20;
    config.do_sample=false;
    config.temperature=0.1;
    config.pruning_ratio=30;  // enable CDPruner with 30% pruning ratio
    config.relevance_weight=0.7f;

    ov::AnyMap cfg;
    cfg["ATTENTION_BACKEND"] = "PA";

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