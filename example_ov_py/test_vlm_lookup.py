import openvino_genai as ov_genai
from pathlib import Path
import requests
from PIL import Image
from io import BytesIO
import numpy as np
import openvino as ov
import time
import openvino_genai

def streamer(subword):
    print(subword, end='', flush=True)
    # Return flag corresponds whether generation should be stopped. 
    return openvino_genai.StreamingStatus.RUNNING

def test_vllm_lookup():
    model_path = "../models/ov/Qwen2.5-VL-3B-Instruct/INT4/"
    device = "CPU"
    enable_look_up = True

    pipe = ov_genai.VLMPipeline(model_path, device=device, prompt_lookup=enable_look_up)

    config = ov_genai.GenerationConfig()
    config.max_new_tokens = 64
    config.do_sample=False
    config.temperature=0.1
    if enable_look_up:
        # add parameter to enable prompt lookup decoding to generate `num_assistant_tokens` candidates per iteration
        config.num_assistant_tokens = 3
        # Define max_ngram_size
        config.max_ngram_size = 3

    img_fn = "../test_video/cat_120_100.png"
    image = ov.Tensor(np.array(Image.open(img_fn).convert("RGB")))

    prompts = "Is there animal in this image? please answer like: \"There is 2 ducks in this image.\""
    # prompts = "请描述图片"
    for i in range(1):
        pipe.start_chat()
        t1 = time.time()
        outputs = pipe.generate(prompts, image=image, generation_config=config)
        t2 = time.time()
        pipe.finish_chat()
        print(f"== time: {t2-t1} ms outputs = {outputs}")
        print(f" TTFT: {outputs.perf_metrics.get_ttft().mean} ± {outputs.perf_metrics.get_ttft().std} ms")
        print(f" TPOT: {outputs.perf_metrics.get_tpot().mean} ± {outputs.perf_metrics.get_tpot().std} ms")
        print(f" generated num: {outputs.perf_metrics.get_num_generated_tokens()}")
        print(f" infer num: {len(outputs.perf_metrics.raw_metrics.token_infer_durations)}")

if __name__ == "__main__":
    print("OV Version:", ov.get_version())
    test_vllm_lookup()