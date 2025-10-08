import openvino_genai as ov_genai
from pathlib import Path
import requests
from PIL import Image
from io import BytesIO
import numpy as np
import openvino as ov
import time
import openvino.properties.hint as hints
from openvino_genai import (
    VLMPipeline,
    GenerationConfig,
    SchedulerConfig,
    ContinuousBatchingPipeline,
    GenerationStatus,
    StreamingStatus,
    GenerationFinishReason,
)

ov_model='../models/ov/Qwen2.5-VL-3B-Instruct/INT4/'
prompt = "请描述这个视频："

def load_image(image_url_or_file):
    if str(image_url_or_file).startswith("http") or str(image_url_or_file).startswith("https"):
        response = requests.get(image_url_or_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_url_or_file).convert("RGB")
    resized_image = image.resize((364, 448), Image.Resampling.BICUBIC)
    return image, ov.Tensor(resized_image)

def get_default_llm_properties():
    return {
        hints.inference_precision: ov.Type.f32,
        hints.kv_cache_precision: ov.Type.f16,
    }

def get_video():
    imgs = []
    frames = []
    for idx in range(9):
        image, image_tensor = load_image(f'../test_video/rsz_video/img_{idx}.png')
        imgs.append(image_tensor)
        frames.append(image_tensor.data)
    video = np.stack(frames, axis=0)
    return imgs, ov.Tensor(video)

def vlm_pipeline_result():
    scheduler_config = SchedulerConfig()
    print("== ov_model=", ov_model)
    device = 'CPU'
    print("== device = ", device)

    ov_pipe = ov_genai.VLMPipeline(ov_model, device=device, ATTENTION_BACKEND="SDPA", **get_default_llm_properties())

    generation_config = ov_genai.GenerationConfig()
    generation_config.max_new_tokens = 40

    imgs, video= get_video()
    print("type imgs: ", imgs[0].data.shape)
    print("type video: ", video.data.shape)

    res_generate = []
    res_generate.append(
                ov_pipe.generate(
                    prompt, videos=[video], generation_config=generation_config
                )
            )

    print('res_generate of vlm pipeline = ', res_generate[0].texts)
    return res_generate

def cb_add_request_pipeline_result(vlm_pipeline_res):
    scheduler_config = SchedulerConfig()
    cb_pipe = ContinuousBatchingPipeline(
        ov_model,
        scheduler_config=scheduler_config,
        device="CPU",
        properties=get_default_llm_properties(),
    )
    tokenizer = cb_pipe.get_tokenizer()

    imgs, video = get_video()
    generation_config = ov_genai.GenerationConfig()
    generation_config.max_new_tokens = 40
    eps = 0.001

    idx = 0
    handle = cb_pipe.add_request(idx, prompt, [], [video], generation_config)
    while handle.get_status() != GenerationStatus.FINISHED:
        cb_pipe.step()
    outputs = handle.read_all()
    for out_idx, output in enumerate(outputs):
        text = tokenizer.decode(output.generated_ids)
        print("text = ", text)
        assert text == vlm_pipeline_res[idx].texts[out_idx]
        assert abs(output.score - vlm_pipeline_res[idx].scores[out_idx]) < eps
        assert (
            output.finish_reason == GenerationFinishReason.STOP
            or output.finish_reason == GenerationFinishReason.LENGTH
        )

if __name__ == "__main__":
    print("OV Version:", ov.get_version())
    vlm_pipeline_res = vlm_pipeline_result()
    cb_add_request_pipeline_result(vlm_pipeline_res)
