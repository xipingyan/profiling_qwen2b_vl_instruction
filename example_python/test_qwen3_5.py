import base64
import mimetypes
import os
import sys

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

local_model_id = os.getenv("MODEL_ID", "../models/Qwen/Qwen3.5-35B-A3B-Base")
# local_model_id = os.getenv("MODEL_ID", "../models/Qwen/Qwen3.5-35B-A3B-FP8")


def test_qwen3_5_image():
    model_name = local_model_id
    
    # 1. Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # torch_dtype="auto",     # Automatically selects BF16 if supported, else FP16
        # device_map="auto"       # Automatically distributes across available GPUs
        torch_dtype=torch.float16,  # Force loading in FP16 (adjust if your GPU doesn't support it)
        device_map="cpu"
    )
    
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