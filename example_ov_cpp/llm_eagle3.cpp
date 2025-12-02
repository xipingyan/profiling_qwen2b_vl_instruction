// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "load_image.hpp"
#include <openvino/genai/visual_language/pipeline.hpp>
#include <filesystem>
#include <chrono>
#include "openvino/genai/speculative_decoding/perf_metrics.hpp"
template <typename T>
void print_perf_metrics(T& perf_metrics, std::string model_name) {
    std::cout << "\n" << model_name << std::endl;
    auto generation_duration = perf_metrics.get_generate_duration().mean;
    std::cout << "  Generate time: " << generation_duration << " ms" << std::endl;
    std::cout << "  TTFT: " << perf_metrics.get_ttft().mean << " ± " << perf_metrics.get_ttft().std << " ms"
              << std::endl;
    std::cout << "  TPOT: " << perf_metrics.get_tpot().mean << " ± " << perf_metrics.get_tpot().std << " ms/token"
              << std::endl;
    std::cout << "  Num generated token: " << perf_metrics.get_num_generated_tokens() << " tokens" << std::endl;
    if (model_name == "Total") {
        std::cout << "  Total iteration number: " << perf_metrics.raw_metrics.m_new_token_times.size() << std::endl;
    } else {
        std::cout << "  Total iteration number: " << perf_metrics.raw_metrics.m_durations.size() << std::endl;
    }
    if (perf_metrics.get_num_input_tokens() > 0) {
        std::cout << "  Input token size: " << perf_metrics.get_num_input_tokens() << std::endl;
    }
}
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
    config.num_assistant_tokens = 5; // eagle3 param.

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

    for (size_t i = 0;i < 3; i++) {
        std::cout << "== Loop: " << i << std::endl;
        // pipe.start_chat();
        auto t1 = std::chrono::high_resolution_clock::now();
        auto outputs = pipe.generate(prompts, ov::genai::image(images[0]), ov::genai::generation_config(config));
        auto t2 = std::chrono::high_resolution_clock::now();
        // pipe.finish_chat();
        std::cout << " time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms\n";
        std::cout << " outputs: " << outputs << '\n';
        std::cout << " TTFT: " << outputs.perf_metrics.get_ttft().mean << " ± " << outputs.perf_metrics.get_ttft().mean << std::endl;
        std::cout << " TPOT: " << outputs.perf_metrics.get_tpot().mean << " ± " << outputs.perf_metrics.get_tpot().mean << std::endl;

        auto sd_perf_metrics = std::dynamic_pointer_cast<ov::genai::SDPerModelsPerfMetrics>(outputs.extended_perf_metrics);
        print_perf_metrics(outputs.perf_metrics, "Total");
        if (sd_perf_metrics) {
            print_perf_metrics(sd_perf_metrics->main_model_metrics, "MAIN MODEL");
            std::cout << "  accepted token: " << sd_perf_metrics->get_num_accepted_tokens() << " tokens" << std::endl;
            std::cout << "  compress rate: "
                    << sd_perf_metrics->main_model_metrics.get_num_generated_tokens() * 1.0f /
                            sd_perf_metrics->main_model_metrics.raw_metrics.m_durations.size()
                    << std::endl;
            print_perf_metrics(sd_perf_metrics->draft_model_metrics, "DRAFT MODEL");
        }
    }
    return 0;
}
