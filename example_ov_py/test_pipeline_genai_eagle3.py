import openvino_genai as ov_genai
import requests
from PIL import Image
import numpy as np
import openvino as ov
import time
import sys
import os


def load_image(image_url_or_file):
    if str(image_url_or_file).startswith("http") or str(image_url_or_file).startswith("https"):
        image = Image.open(requests.get(image_url_or_file, stream=True).raw)
    else:
        image = Image.open(image_url_or_file).convert("RGB")
    return image

def streamer(subword: str) -> bool:
    """
    Args:
        subword: sub-word of the generated text.
    Returns: Return flag corresponds whether generation should be stopped.
    """
    print(subword, end="", flush=True)

def test_eagle3():
    model_path = "../models/eagle3/qwen2.5-vl-7b-ov-int4"
    model_path = "../models/eagle3/Qwen2.5-VL-7B-Instruct-int4-opt"
    print(f"  main model: {model_path}")

    draft_path = "../models/eagle3/qwen2.5-vl-7b-eagle3-ov-int4"
    img_path="../test_video/IMG_20250723_145708_008_IN.jpg"

    device = "GPU"
    prompts = "简单描述一下这张图"
    image = load_image(img_path)
    ov_image = ov.Tensor(image)

    scheduler_config = ov_genai.SchedulerConfig()
    scheduler_config.enable_prefix_caching = False
    scheduler_config.max_num_batched_tokens = sys.maxsize

    config = ov_genai.GenerationConfig()
    config.max_new_tokens = 256
    config.do_sample = False
    config.temperature = 0.1

    DISABLE_EAGLE3 = os.getenv('EAGLE3') == "0"
    if DISABLE_EAGLE3:
        print(f"** Disable eagle3.")
        pipe = ov_genai.VLMPipeline(model_path, device=device, scheduler_config=scheduler_config, ATTENTION_BACKEND="PA")
    else:
        print(f"** Enable eagle3.")
        draft_model = ov_genai.draft_model(draft_path, device)
        pipe = ov_genai.VLMPipeline(model_path, device=device, scheduler_config=scheduler_config, draft_model=draft_model, ATTENTION_BACKEND="PA")

    for i in range(4):
        print(f"== loop: {i}")
        output = pipe.generate(prompts, image=ov_image, generation_config=config)
        print(f"     output: {output}")
        print(f"     TTFT: {output.perf_metrics.get_ttft().mean:.2f} ± {output.perf_metrics.get_ttft().std:.2f} ms")
        print(f"     TPOT: {output.perf_metrics.get_tpot().mean:.2f} ± {output.perf_metrics.get_tpot().std:.2f} ms")

if __name__ == "__main__":
    print("OV Version:", ov.get_version())
    print("ov_genai Version:", ov_genai.__version__)
    test_eagle3()