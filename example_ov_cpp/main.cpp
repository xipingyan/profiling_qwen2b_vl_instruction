// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "load_image.hpp"
#include <openvino/genai/visual_language/pipeline.hpp>
#include <filesystem>
#include <chrono>

bool print_subword(std::string &&subword)
{
    return !(std::cout << subword << std::flush);
}

void pasre_params(int argc, char *argv[], std::string &model_path, bool &input_video, std::string &img_video_path, std::string &device)
{
    auto help_fun = std::runtime_error(std::string{"Usage "} + argv[0] + " <MODEL_DIR> <video/img> <IMAGE_FILE OR DIR_WITH_IMAGES> <device>");

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
    if (5 <= argc)
    {
        device = argv[4];
    }

    model_path = argv[1];
    std::cout << "== Params:" << std::endl;
    std::cout << "    model_path = " << model_path << std::endl;
    std::cout << "    input_video = " << input_video << std::endl;
    std::cout << "    img_video_path = " << img_video_path << std::endl;
    std::cout << "    device = " << device << std::endl;
}

void test_images(ov::genai::VLMPipeline &pipe, std::vector<ov::Tensor> rgbs)
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

void test_video(ov::genai::VLMPipeline &pipe, std::vector<ov::Tensor> &rgbs, ov::Tensor &video)
{
    ov::genai::GenerationConfig generation_config;
    generation_config.max_new_tokens = 2048;

    std::string prompt = "请描述这个视频：";
    prompt = ""
             "请基于输入的视频信息（包含视觉画面中的车辆相关元素，如车辆状态、人物与车辆的交互动作、车辆所处场景的道路特征等），完成以下分析任务：首先，精准识别视频中与车辆相关的核心事件或关键现象，包括但不限于人物对车辆的操作行为（如驾驶、停靠、触碰、撞击、移动车辆部件）、车辆自身的状态变化（如剐蹭、变形、故障、位移）、车辆所处场景中的异常情况（如车辆占用应急通道、车辆周边出现障碍物影响行驶、车辆违规停放等）；其次，对识别到的核心内容进行定性判断，明确行为性质（如合规操作 / 违规操作）、风险等级（如无风险 / 低风险 / 高风险）、影响程度（如无影响 / 轻微影响 / 严重影响）或必要措施（如无需提示 / 需提醒用户注意 / 需紧急处理）；最后，将分析结果浓缩为 30 个 token 以内的总结性结论，要求语言简洁、判断明确，避免模糊表述。示例 1：输入视频信息为 “行车记录仪画面显示，一辆黑色 SUV 在路边停靠，一名女子打开右后车门时，车门边缘轻微碰到旁边的白色轿车，白色轿车车身未出现明显划痕，女子未察觉便关上车门离开”，输出结论为 “SUV 车门轻碰白色轿车，无明显损伤，属低风险接触”。示例 2：输入视频信息为 “监控画面中，一名男子站在一辆红色跑车旁，用钥匙在车门表面划动，跑车车门出现一道明显的划痕”，输出结论为 “男子用钥匙划伤跑车车门，造成明显损伤，属高风险破坏，需提醒用户”。示例 3：输入视频信息为 “停车场监控显示，一辆蓝色轿车在倒车时，车尾撞到后方的水泥柱，撞击力度较轻，轿车后保险杠出现轻微变形，无零件脱落”，输出结论为 “蓝色轿车倒车轻撞水泥柱，保险杠微变形，属低风险损伤”。示例 4：输入视频信息为 “路口监控画面中，一辆白色面包车违规停在消防通道上，车身完全占用通道宽度，后司机返回将车开走”，输出结论为 “面包车占用消防通道，无即时影响，需提醒挪车”。示例 5：输入视频信息为 “画面中，一辆电动车在非机动车道行驶，突然变道进入机动车道，与后方驶来的轿车发生轻微剐蹭，电动车后视镜被撞歪，无人员受伤”），输出结论为 “电动车违规变道致剐蹭，轻微损伤，属中风险事故”。示例 7：输入视频信息为 “小区停车场内，一辆银色轿车的左前轮停在绿化带上，碾压了草坪”，输出结论为 “轿车碾压绿化带停车，轻微破坏，需提醒驶离”）。请严格遵循上述要求，基于输入的车辆相关视频信息，必须输出 30token以内的精准结论。结论严格按照json格式{\"事件\"：xxx,\"严重等级\"：xxx ，\"提醒用户\":是或者否}"
             "";
    std::cout << "  prompt: " << prompt << std::endl;

    // std::cout << "  test_video pass 'video' with multiple tensors " << std::endl;
    // for (int i = 0; i < 3; i++)
    // {
    //     pipe.start_chat();
    //     std::cout << "  Loop: [" << i << "] ";
    //     auto t1 = std::chrono::high_resolution_clock::now();
    //     auto aa = pipe.generate(prompt,
    //                             ov::genai::videos(rgbs),
    //                             ov::genai::generation_config(generation_config));
    //     // ov::genai::streamer(print_subword));
    //     auto t2 = std::chrono::high_resolution_clock::now();
    //     std::cout << ", result: text =" << aa.texts[0].c_str() << ", score=" << aa.scores[0] << ", tm=" << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << std::endl;
    //     pipe.finish_chat();
    // }

    // std::cout << "  test_video pass 'video' with one tensor " << std::endl;
    // for (int i = 0; i < 3; i++)
    // {
    //     pipe.start_chat();
    //     std::cout << "  Loop: [" << i << "] ";
    //     auto t1 = std::chrono::high_resolution_clock::now();
    //     auto aa = pipe.generate(prompt,
    //                             ov::genai::videos(std::vector<ov::Tensor>{video}),
    //                             ov::genai::generation_config(generation_config));
    //     // ov::genai::streamer(print_subword));
    //     auto t2 = std::chrono::high_resolution_clock::now();
    //     std::cout << ", result: text =" << aa.texts[0].c_str() << ", score=" << aa.scores[0] << ", tm=" << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << std::endl;
    //     pipe.finish_chat();
    // }

    std::cout << "  test_video pass 'video' + 'images' " << std::endl;

    std::vector<ov::Tensor> images = std::vector<ov::Tensor>{rgbs[0], rgbs[1]};
    auto shape = rgbs[0].get_shape();
    ov::Tensor video1 = ov::Tensor(ov::element::u8, ov::Shape({3, shape[1], shape[2], shape[3]}));
    ov::Tensor video2 = ov::Tensor(ov::element::u8, ov::Shape({6, shape[1], shape[2], shape[3]}));
    for (int i = 0; i < 9; i++)
    {
        if (i < 3)
            std::memcpy((char *)video1.data() + i * rgbs[0].get_byte_size(), (char *)rgbs[i].data(), rgbs[i].get_byte_size());
        else
            std::memcpy((char *)video2.data() + (i - 3) * rgbs[0].get_byte_size(), (char *)rgbs[i].data(), rgbs[i].get_byte_size());
    }
    std::vector<ov::Tensor> videos = {video1, video2};

    for (int i = 0; i < 3; i++)
    {
        pipe.start_chat();
        std::cout << "  Loop: [" << i << "] " << std::endl;
        auto t1 = std::chrono::high_resolution_clock::now();
        auto aa = pipe.generate(prompt,
                                ov::genai::images(images),
                                ov::genai::videos(videos),
                                ov::genai::generation_config(generation_config));
        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << "      result: text =" << aa.texts[0].c_str() << ", score=" << aa.scores[0] << ", tm=" << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << std::endl;
        pipe.finish_chat();
    }
}

int main(int argc, char *argv[])
{
    try
    {
        // return test_llm_lookup(argc, argv);

        std::string img_video_path = "../../cat_1.jpg";
        std::string model_path = "../../ov_model_i8/";
        bool input_video = true;
        std::string device = "GPU";

        pasre_params(argc, argv, model_path, input_video, img_video_path, device);
        ov::AnyMap cfg;
        if (device == "GPU")
        {
            cfg.insert({ov::cache_dir("vlm_cache")});
            std::cout << "    cfg vlm_cache = " << "vlm_cache" << std::endl;
        }
        std::cout << "    device = " << device << std::endl;

        std::vector<ov::Tensor> rgbs = utils::load_images(img_video_path);
        ov::Tensor video = utils::load_video(img_video_path);

        std::cout << "== Start to load model: " << model_path << std::endl;
        cfg["ATTENTION_BACKEND"] = "PA";
        ov::genai::VLMPipeline pipe(model_path, device, cfg);

        if (input_video)
        {
            test_video(pipe, rgbs, video);
        }
        else
        {
            test_images(pipe, rgbs);
        }
    }
    catch (const std::exception &error)
    {
        try
        {
            std::cerr << error.what() << '\n';
        }
        catch (const std::ios_base::failure &)
        {
        }
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}