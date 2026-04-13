import os
import sys

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

DRAFT_MODEL_DIR = os.getenv("DRAFT_MODEL_DIR", "../../models/z-lab/Qwen3-4B-DFlash-b16/")
MAIN_MODEL_DIR = os.getenv("MAIN_MODEL_DIR", "../../models/Qwen/Qwen3-4B/")


def load_draft_model():
    device_map = os.getenv("DEVICE_MAP", "cpu")
    dtype = torch.bfloat16 if device_map != "cpu" else torch.float32

    try:
        return AutoModel.from_pretrained(
            DRAFT_MODEL_DIR,
            trust_remote_code=True,
            dtype=dtype,
            device_map=device_map,
        ).eval()
    except RuntimeError as exc:
        message = str(exc)
        if "NVIDIA driver on your system is too old" in message or "CUDA" in message:
            print("GPU load failed; retrying on CPU.", file=sys.stderr)
            return AutoModel.from_pretrained(
                DRAFT_MODEL_DIR,
                trust_remote_code=True,
                dtype=torch.float32,
                device_map="cpu",
            ).eval()
        raise

def main():
    draft = load_draft_model()
    print(f"Loaded draft model on {next(draft.parameters()).device}")

    target = AutoModelForCausalLM.from_pretrained(
        MAIN_MODEL_DIR,
        trust_remote_code=True,
        dtype=torch.bfloat16 if next(draft.parameters()).device.type == "cuda" else torch.float32,
        device_map=os.getenv("TARGET_DEVICE_MAP", os.getenv("DEVICE_MAP", "cpu")),
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(MAIN_MODEL_DIR)

    messages = [{"role": "user", "content": "How many positive whole-number divisors does 196 have?"}]
    input_ids = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=False,
    ).to(draft.device)

    output = draft.spec_generate(
        input_ids=input_ids,
        max_new_tokens=2048,
        temperature=0.0,
        target=target,
        stop_token_ids=[tokenizer.eos_token_id],
    )
    print(tokenizer.decode(output[0], skip_special_tokens=False))


if __name__ == "__main__":
    main()