// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "load_image.hpp"
#include <openvino/genai/visual_language/pipeline.hpp>
#include <filesystem>
#include <chrono>

bool print_subword(std::string&& subword) {
    return !(std::cout << subword << std::flush);
}

void pasre_params(int argc, char* argv[], std::string &model_path, bool &input_video, std::string &img_video_path)
{
    auto help_fun = std::runtime_error(std::string{"Usage "} + argv[0] + " <MODEL_DIR> <video/img> <IMAGE_FILE OR DIR_WITH_IMAGES>");

    if (argc == 1)
    {
        throw help_fun;
    }

    if (2 == argc && std::string(argv[1]) == std::string("-h"))
    {
        throw help_fun;
    }
    
    if (3 <= argc)
    {
        input_video = std::string(argv[2]) == "video";
    }
    
    if (4 <= argc)
    {
        img_video_path = argv[3];
    }

    model_path = argv[1];
    std::cout << "== Params:" << std::endl;
    std::cout << "    model_path = " << model_path << std::endl;
    std::cout << "    input_video = " << input_video << std::endl;
    std::cout << "    img_video_path = " << img_video_path << std::endl;
}

void test_images(ov::genai::VLMPipeline& pipe, std::vector<ov::Tensor> rgbs)
{
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
    for (int i = 0; i < 5; i++)
    {
        pipe.start_chat();
        std::cout << "Loop: [" << i << "] ";
        auto t1 = std::chrono::high_resolution_clock::now();
        auto real_prompt = i % 2 == 0 ? prompt : prompt2;
        auto aa = pipe.generate(real_prompt,
                                ov::genai::images(rgbs),
                                ov::genai::generation_config(generation_config),
                                ov::genai::streamer(print_subword));
        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << ", result: text =" << aa.texts[0].c_str() << ", score=" << aa.scores[0] << ", tm=" << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << std::endl;
        pipe.finish_chat();
    }
}

void test_video(ov::genai::VLMPipeline& pipe, std::vector<ov::Tensor> rgbs)
{
    ov::genai::GenerationConfig generation_config;
    generation_config.max_new_tokens = 100;

    std::string prompt = "请描述这个视频：";
    std::cout << "  prompt: " << prompt << std::endl;
    for (int i = 0; i < 1; i++)
    {
        pipe.start_chat();
        std::cout << "  Loop: [" << i << "] ";
        auto t1 = std::chrono::high_resolution_clock::now();
        auto aa = pipe.generate(prompt,
                                ov::genai::images(rgbs),
                                ov::genai::generation_config(generation_config),
                                ov::genai::streamer(print_subword));
        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << ", result: text =" << aa.texts[0].c_str() << ", score=" << aa.scores[0] << ", tm=" << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << std::endl;
        pipe.finish_chat();
    }
    // python video result:
    // 视频展示了一群学生在操场上进行接力赛跑的场景。画面中，几位穿着统一蓝色运动服的学生正在起跑线前准备开始比赛。他们站在跑道上，背景是绿色的树木和远处的建筑物。\n\n随着裁判员的一声哨响，学生们迅速起跑，向前冲刺。他们的动作协调一致，显示出良好的团队合作精神。观众们站在一旁观看，为参赛者加油助威。\n\n整个场景充满了活力与激情，展现了青春体育的精神。
    // python images result:
    // 视频展示了一群学生在操场上进行跑步比赛的场景。背景中可以看到一些树木和建筑物，天气看起来有些阴沉。跑道上有一条白色的线条标记着起跑线。\n\n画面中有几位穿着浅蓝色运动服的学生正在起跑，他们身体前倾，双手握拳放在胸前，准备向前冲刺。旁边有几位观众在观看比赛，其中一位穿着白色T恤和黑色短裤的学生正在用手机记录比赛。其他观众则站在一旁，有的在交谈，有的在鼓掌
    // ov images result:
    // 视频展示了一次田径比赛的场景。画面中，跑道上有一群穿着蓝色运动服的女生正在参加接力赛跑。她们排成一排，依次向前跑去，显示出她们的专注和努力。在跑道的旁边，有几位观众在观看比赛，他们有的站着，有的坐着，有的拿着手机拍照或录像。背景中可以看到一些树木和建筑物，为整个场景增添了一种校园的氛围。
}

int main(int argc, char* argv[]) try {    
    std::string img_video_path = "../../cat_1.jpg";
    std::string model_path = "../../ov_model_i8/";
    bool input_video = false;

    pasre_params(argc, argv, model_path, input_video, img_video_path);
    // GPU and NPU can be used as well.
    // Note: If NPU selected, only language model will be run on NPU
    std::string device = "GPU";
    ov::AnyMap enable_compile_cache;
    if (device == "GPU") {
        // Cache compiled models on disk for GPU to save time on the
        // next run. It's not beneficial for CPU.
        enable_compile_cache.insert({ov::cache_dir("vlm_cache")});
        std::cout << "    enable_compile_cache = " << "vlm_cache" << std::endl;
    }
    std::cout << "    device = " << device << std::endl;

    std::vector<ov::Tensor> rgbs = utils::load_images(img_video_path);

    std::cout << "== Start to load model: " << model_path << std::endl;
    ov::genai::VLMPipeline pipe(model_path, device, enable_compile_cache);

    if (input_video) {
        test_video(pipe, rgbs);
    }
    else {
        test_images(pipe, rgbs);
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
