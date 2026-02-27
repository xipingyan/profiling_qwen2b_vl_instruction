import base64
import mimetypes
import os
import sys

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# local_model_id = os.getenv("MODEL_ID", "../models/Qwen/Qwen3.5-35B-A3B-Base")
local_model_id = os.getenv("MODEL_ID", "../models/Qwen/Qwen3.5-35B-A3B-FP8")


def _parse_max_memory() -> dict:
    """Build accelerate-style max_memory dict from env vars.

    Examples:
      MAX_MEMORY_GPU0=20GiB MAX_MEMORY_GPU1=28GiB MAX_MEMORY_CPU=96GiB
    """
    max_memory: dict = {}
    for i in range(torch.cuda.device_count()):
        v = os.getenv(f"MAX_MEMORY_GPU{i}")
        if v:
            max_memory[i] = v
    v = os.getenv("MAX_MEMORY_CPU")
    if v:
        max_memory["cpu"] = v
    return max_memory


def test_qwen3_5_image():
    model_name = local_model_id
    
    # 1. Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Large checkpoints (e.g. FP8/MoE) may OOM during Transformers "CONVERSION"
    # where it merges weights via torch.stack (extra temporary memory).
    # If you hit that, try enabling offload/max_memory via env vars:
    #   OFFLOAD_FOLDER=./offload MAX_MEMORY_GPU0=20GiB MAX_MEMORY_GPU1=28GiB MAX_MEMORY_CPU=96GiB
    device_map = os.getenv("DEVICE_MAP", "balanced")
    offload_folder = os.getenv("OFFLOAD_FOLDER")
    max_memory = _parse_max_memory()

    dtype_env = os.getenv("TORCH_DTYPE", "auto").lower()
    if dtype_env in {"fp16", "float16"}:
        torch_dtype = torch.float16
    elif dtype_env in {"bf16", "bfloat16"}:
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = "auto"

    load_kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": device_map,
        "low_cpu_mem_usage": True,
    }
    if offload_folder:
        load_kwargs.update(
            {
                "offload_folder": offload_folder,
                "offload_state_dict": True,
            }
        )
    if max_memory:
        load_kwargs["max_memory"] = max_memory

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    
    # 2. Prepare the input
    messages = [
        {"role": "user", "content": "How do I run Qwen3.5 with PyTorch?"}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # 3. Generate response
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.8
    )
    
    # 4. Decode output
    response = tokenizer.batch_decode(
        generated_ids[:, model_inputs.input_ids.shape[1]:], 
        skip_special_tokens=True
    )[0]
    
    print(response)

if __name__ == "__main__":
    try:
        test_qwen3_5_image()
    except Exception as e:
        print(str(e), file=sys.stderr)
        raise