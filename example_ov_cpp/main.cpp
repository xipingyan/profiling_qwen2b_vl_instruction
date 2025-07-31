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
    generation_config.max_new_tokens = 2048;
    // generation_config.is_video = true;

    std::string prompt = "请描述这个视频：";
    prompt = """输入的多张时间连续的视频画面是由车内的摄像头拍摄得到的，所以本视频的主体是拍摄视频的车辆。 下面所有的分析都是基于视频车相关的元素（如车辆状态、人车交互、物体和车之家的交互，周围环境特征等），完成以下任务：1.精准识别与车辆发生关系的核心事件或关键现象，包括但不限于：人物对车辆的交互类型（剐蹭、停靠、触碰、撞击、扒充电枪，附近移动，踢车，拍打等）；周围环境情况（周边有障碍物、违规停放，环境正常，环境复杂等），以及与车辆发生交互关系的人物的描述（外卖员，普通人，警察， 小孩， 老年人等）、与车辆发生交互关系的车辆类型（两轮车，三轮车，摩托车，自行车，轿车， SUV，房产，大货车等）； 简短事件描述(比如行人拍打车窗，外卖员剐蹭后视镜等)。2.经过仔细分析后将分析结果浓缩为 30 个 token 以内的结论，要求语言简洁、判断明确、无模糊表述。3.必须严格按照这个格式输出最后的结果：人物对车辆的交互类型 - 人物的描述 - 车辆类型 - 环境情况 简短事件描述。下面给出来多个示例：例1：视频为外卖员骑车剐蹭汽车后视镜，输出：剐蹭 - 外卖员 - 两轮车 - 环境正常 穿黄衣外卖员骑车疑似剐蹭后视镜。 例2：视频为可疑人员拔自车充电枪，输出：扒充电枪 - 普通人 - 无 - 环境正常 可疑人员疑似拔充电枪例。 例3：视频为行人和汽车在自车附近移动未接触，输出：附近移动 - 多种 - 多种 - 环境正常 多行人与汽车在自车附近移动未接触。 例4：视频为行人踢车后轮，输出：踢车 - 普通人 - 无 - 环境正常 行人疑似用脚踢自车轮胎。 例5：视频为黑色 SUV 停靠时，女子开右后车门轻碰旁边白色轿车（无明显划痕，女子未察觉离开），输出：碰撞 - 女子 - 白色轿车 - 环境正常 车门边缘轻微碰到旁边白色轿车门。 例6：视频为一辆红色大货车在人流复杂的路口拐弯的时候疑似剐蹭到了汽车的前保险杠，输出：剐蹭-无-大货车-环境复杂 大货车剐蹭了前保险杠。例7：视频为一个穿着黑色外套的男子在不停的拍打前挡风玻璃，周围是一个停车场，输出：拍打-男子-无-环境正常 男子拍打前挡风玻璃。请严格遵循上述要求，基于输入的车辆相关视频信息，输出 30token 以内的精准结论。""";
    prompt = """请基于输入的视频信息（包含视觉画面中的车辆相关元素，如车辆状态、人物与车辆的交互动作、车辆所处场景的道路特征等），完成以下分析任务：首先，精准识别视频中与车辆相关的核心事件或关键现象，包括但不限于人物对车辆的操作行为（如驾驶、停靠、触碰、撞击、移动车辆部件）、车辆自身的状态变化（如剐蹭、变形、故障、位移）、车辆所处场景中的异常情况（如车辆占用应急通道、车辆周边出现障碍物影响行驶、车辆违规停放等）；其次，对识别到的核心内容进行定性判断，明确行为性质（如合规操作 / 违规操作）、风险等级（如无风险 / 低风险 / 高风险）、影响程度（如无影响 / 轻微影响 / 严重影响）或必要措施（如无需提示 / 需提醒用户注意 / 需紧急处理）；最后，将分析结果浓缩为 30 个 token 以内的总结性结论，要求语言简洁、判断明确，避免模糊表述。示例 1：输入视频信息为 “行车记录仪画面显示，一辆黑色 SUV 在路边停靠，一名女子打开右后车门时，车门边缘轻微碰到旁边的白色轿车，白色轿车车身未出现明显划痕，女子未察觉便关上车门离开”，输出结论为 “SUV 车门轻碰白色轿车，无明显损伤，属低风险接触”。示例 2：输入视频信息为 “监控画面中，一名男子站在一辆红色跑车旁，用钥匙在车门表面划动，跑车车门出现一道明显的划痕”，输出结论为 “男子用钥匙划伤跑车车门，造成明显损伤，属高风险破坏，需提醒用户”。示例 3：输入视频信息为 “停车场监控显示，一辆蓝色轿车在倒车时，车尾撞到后方的水泥柱，撞击力度较轻，轿车后保险杠出现轻微变形，无零件脱落”，输出结论为 “蓝色轿车倒车轻撞水泥柱，保险杠微变形，属低风险损伤”。示例 4：输入视频信息为 “路口监控画面中，一辆白色面包车违规停在消防通道上，车身完全占用通道宽度，后司机返回将车开走”，输出结论为 “面包车占用消防通道，无即时影响，需提醒挪车”。示例 5：输入视频信息为 “画面中，一辆电动车在非机动车道行驶，突然变道进入机动车道，与后方驶来的轿车发生轻微剐蹭，电动车后视镜被撞歪，无人员受伤”），输出结论为 “电动车违规变道致剐蹭，轻微损伤，属中风险事故”。示例 7：输入视频信息为 “小区停车场内，一辆银色轿车的左前轮停在绿化带上，碾压了草坪”，输出结论为 “轿车碾压绿化带停车，轻微破坏，需提醒驶离”）。请严格遵循上述要求，基于输入的车辆相关视频信息，必须输出 30token以内的精准结论。结论严格按照json格式{\"事件\"：xxx,\"严重等级\"：xxx ，\"提醒用户\":是或者否}""";
    std::cout << "  prompt: " << prompt << std::endl;
    std::cout << "  generation_config.is_video = " << generation_config.is_video << std::endl;
    for (int i = 0; i < 3; i++)
    {
        pipe.start_chat();
        std::cout << "  Loop: [" << i << "] ";
        auto t1 = std::chrono::high_resolution_clock::now();
        auto aa = pipe.generate(prompt,
                                ov::genai::images(rgbs),
                                ov::genai::generation_config(generation_config));
                                // ov::genai::streamer(print_subword));
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
