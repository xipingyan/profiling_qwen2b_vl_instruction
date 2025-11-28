// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "load_image.hpp"
#include <openvino/genai/visual_language/pipeline.hpp>
#include <filesystem>
#include <chrono>

int test_vllm_eagle3(int argc, char* argv[]) {
    std::cout << "== test_vllm_eagle3 ..." << std::endl;
    ov::AnyMap enable_compile_cache;
    std::string model_path = "../models/eagle3/qwen2.5-vl-7b-ov-int4";
    std::string draft_path = "../models/eagle3/qwen2.5-vl-7b-eagle3-ov-int4";
    std::string img_path="../test_video/IMG_20250723_145708_008_IN.jpg";

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
    cfg["prompt_lookup"] = enable_look_up;

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