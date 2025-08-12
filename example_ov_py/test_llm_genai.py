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

def test_llm_lookup():
    model_path = "/mnt/xiping/gpu_profiling/profiling_qwen2b_vl_instruction/OpenVINO/Qwen2-0.5B-int8-ov/"
    device = "GPU.0"
    pipe = ov_genai.LLMPipeline(model_path, device=device, prompt_lookup=True)

    config = ov_genai.GenerationConfig()
    config.max_new_tokens = 20
    # config.is_video=True
    config.do_sample=False
    config.temperature=0.1

    # add parameter to enable prompt lookup decoding to generate `num_assistant_tokens` candidates per iteration
    config.num_assistant_tokens = 5
    # Define max_ngram_size
    config.max_ngram_size = 3
    
    for i in range(3):
        pipe.start_chat()
        t1 = time.time()
        outputs = pipe.generate("What is the capital of China?", config)
        t2 = time.time()
        pipe.finish_chat()
        print(f"== time: {t2-t1} ms outputs = {outputs}")

if __name__ == "__main__":
    print("OV Version:", ov.get_version())
    test_llm_lookup()
