// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "load_image.hpp"
#include <openvino/genai/visual_language/pipeline.hpp>
#include <filesystem>
#include <chrono>

bool print_subword(std::string&& subword) {
    return !(std::cout << subword << std::flush);
}

int main(int argc, char* argv[]) try {    
    std::string img_fn = "../../cat_1.jpg";
    std::string model_path = "../../ov_model_i8/";

    if (2 == argc && std::string(argv[1]) == std::string("-h")) {
        throw std::runtime_error(std::string{"Usage "} + argv[0] + " <MODEL_DIR> <IMAGE_FILE OR DIR_WITH_IMAGES>");
    }
    else if (3 == argc)
    {
        img_fn = argv[2];
        model_path = argv[1];
    }
    
    std::vector<ov::Tensor> rgbs = utils::load_images(img_fn);

    // GPU and NPU can be used as well.
    // Note: If NPU selected, only language model will be run on NPU
    std::string device = "GPU";
    ov::AnyMap enable_compile_cache;
    if (device == "GPU") {
        // Cache compiled models on disk for GPU to save time on the
        // next run. It's not beneficial for CPU.
        enable_compile_cache.insert({ov::cache_dir("vlm_cache")});
    }
    std::cout << "Start to load model: " << model_path << std::endl;
    ov::genai::VLMPipeline pipe(model_path, device, enable_compile_cache);

    ov::genai::GenerationConfig generation_config;
    generation_config.max_new_tokens = 100;

    std::string prompt = "你是一位图像内容理解专家，能够理解图像内容和文字描述的关系，并输出图片和文字描述的相似度。\
            请遵守以下规则：\
            1、输出得分值范围[0,1)。\
            2、只输出得分值。\
            3、最多输出3位小数。\
        文字内容为：小猫。";
    prompt = "请回答以下问题，务必只能回复一个词 \"Y\"或 \"N\"：图片和\"小猫。\"是否相关？";
    std::string prompt2 = "请回答以下问题，务必只能回复一个词 \"Y\"或 \"N\"：图片和\"小狗。\"是否相关？";

    // std::cout << "question:\n";
    // std::getline(std::cin, prompt);
    std::cout << "prompt: " << prompt << std::endl;
    for (int i = 0; i < 5; i++) {
        pipe.start_chat();
        std::cout << "Loop: [" << i << "] ";
        auto t1 = std::chrono::high_resolution_clock::now();
        auto real_prompt = i % 2 == 0 ? prompt : prompt2;
        auto aa = pipe.generate(real_prompt,
                      ov::genai::images(rgbs),
                      ov::genai::generation_config(generation_config),
                      ov::genai::streamer(print_subword));
        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << ", result: text =" << aa.texts[0].c_str() << ", score=" << aa.scores[0] << \
            ", tm=" << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << " ms" << std::endl;
        pipe.finish_chat();
    }

    // std::cout << "\n----------\n"
    //     "question:\n";
    // while (std::getline(std::cin, prompt)) {
    //     pipe.generate(prompt,
    //                   ov::genai::generation_config(generation_config),
    //                   ov::genai::streamer(print_subword));
    //     std::cout << "\n----------\n"
    //         "question:\n";
    // }
} catch (const std::exception& error) {
    try {
        std::cerr << error.what() << '\n';
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
} catch (...) {
    try {
        std::cerr << "Non-exception object thrown\n";
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
}
