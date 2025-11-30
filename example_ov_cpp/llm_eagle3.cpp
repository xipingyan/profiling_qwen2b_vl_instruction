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
    if (std::getenv("OPT_MODEL") && std::getenv("OPT_MODEL") == std::string("1"))
    {
        model_path = "../models/eagle3/Qwen2.5-VL-7B-Instruct-int4-opt";
    }
    std::cout << "  Main model: " << model_path << std::endl;

    std::string draft_path = "../models/eagle3/qwen2.5-vl-7b-eagle3-ov-int4";
    std::string img_path="../test_video/IMG_20250723_145708_008_IN.jpg";

    std::string device = "GPU";
    if (device == "GPU") {
        enable_compile_cache.insert({ov::cache_dir("vlm_cache")});
        std::cout << "    enable_compile_cache = " << "vlm_cache" << std::endl;
    }
    std::cout << "  == device = " << device << std::endl;

    ov::genai::GenerationConfig config;
    config.max_new_tokens = 256;
    config.do_sample=false;
    config.temperature=0.1;

    auto draft_model = ov::genai::draft_model(draft_path, device);

    ov::AnyMap cfg;
    // cfg["ATTENTION_BACKEND"] = "SDPA";
    cfg["ATTENTION_BACKEND"] = "PA";
    if (std::getenv("EAGLE3") && std::getenv("EAGLE3") == std::string("0"))
    {
        std::cout << " **** Disable eagle3" << std::endl;
    }
    else
    {
        cfg[draft_model.first] = draft_model.second;
        std::cout << " **** Enable eagle3" << std::endl;
    }

    auto pipe = ov::genai::VLMPipeline(model_path, device, cfg);

    auto images = utils::load_images(img_path);
    std::string prompts = "Please describe this image.";
    prompts = "简单描述一下这张图";

    for (size_t i = 0;i < 4; i++) {
        std::cout << "== Loop: " << i << std::endl;
        // pipe.start_chat();
        auto t1 = std::chrono::high_resolution_clock::now();
        auto outputs = pipe.generate(prompts, ov::genai::image(images[0]), ov::genai::generation_config(config));
        auto t2 = std::chrono::high_resolution_clock::now();
        // pipe.finish_chat();
        std::cout << " time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms\n";
        std::cout << " outputs: " << outputs << '\n';
        std::cout << " TTFT: " << outputs.perf_metrics.get_ttft().mean << '\n';
        std::cout << " TPOT: " << outputs.perf_metrics.get_tpot().mean << '\n';
    }
    return 0;
}
