import os
import sys

import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

import openvino as ov

def export_embedding_model(target):
    # target.model.embed_tokens
    print("== Start to export embeds OV model.")
    batch, seq = 1, 5
    input_ids = torch.randint(0, 1000, (batch, seq), dtype=torch.long)

    class TraceableEmbedsWrapper(nn.Module):
        def __init__(self, embed_tokens: nn.Module):
            super().__init__()
            self.embed_tokens = embed_tokens

        def forward(self, input_ids):
            outputs = self.embed_tokens(input_ids)
            return outputs
    wrapped = TraceableEmbedsWrapper(target.model.embed_tokens)
    example_input = (input_ids,)
    ov_model = ov.convert_model(
        wrapped,
        example_input=example_input,
    )
    print("== Start to save embeds OV model.")
    export_embeds_dir = os.getenv("EXPORT_DRAFT_DIR", "exported_draft_model")
    os.makedirs(export_embeds_dir, exist_ok=True)
    export_embeds_model_name = os.path.join(export_embeds_dir, "openvino_text_embeddings_model.xml")
    ov.save_model(ov_model, export_embeds_model_name)
    print(f"== Embeds OV model exported and saved successfully as {export_embeds_model_name}.")

def export_draft_model(draft, target):
    print("== Start to export OV model.")
    batch, seq = 1, 5
    target_hidden = torch.randn(batch, seq, 12800)
    noise_embedding = torch.randn(batch, seq, 2560)
    # DFlash draft internally combines two streams (target/noise), so mask/positions
    # need length 2 * seq for tracing.
    draft_seq = seq * 2
    position_ids = torch.arange(draft_seq).unsqueeze(0)
    attention_mask = torch.ones(batch, draft_seq, dtype=torch.bool)

    class TraceableDraftWrapper(nn.Module):
        def __init__(self, model: nn.Module, lm_head: nn.Module):
            super().__init__()
            self.model = model
            self.lm_head = lm_head

        def forward(self, position_ids, attention_mask, noise_embedding, target_hidden):
            outputs = self.model(
                position_ids=position_ids,
                attention_mask=attention_mask,
                noise_embedding=noise_embedding,
                target_hidden=target_hidden,
                past_key_values=None,
                use_cache=False,
                return_dict=True,
            )
            if torch.is_tensor(outputs):
                hidden_states = outputs
                block_size=16
                hidden_states = hidden_states[:, -block_size+1:, :]
                return self.lm_head(hidden_states)
            if hasattr(outputs, "logits"):
                hidden_states = outputs.logits
                block_size=16
                hidden_states = hidden_states[:, -block_size+1:, :]
                return self.lm_head(hidden_states)
            if isinstance(outputs, (tuple, list)):
                hidden_states = outputs[0]
                block_size=16
                hidden_states = hidden_states[:, -block_size+1:, :]
                return self.lm_head(hidden_states)
            raise TypeError(f"Unsupported output type for export: {type(outputs)}")

    wrapped = TraceableDraftWrapper(draft, target.lm_head)
    example_input = (position_ids, attention_mask, noise_embedding, target_hidden)

    ov_model = ov.convert_model(
        wrapped,
        example_input=example_input,
    )

    # print("== Start to compress OV weights.")
    # compress_weights is memory inplace.
    # ov_model_4bit = compress_weights(ov_model, mode=CompressWeightsMode.INT4_SYM)

    print("== Start to save OV model.")
    export_draft_dir = os.getenv("EXPORT_DRAFT_DIR", "exported_draft_model")
    os.makedirs(export_draft_dir, exist_ok=True)
    export_draft_model_name = os.path.join(export_draft_dir, "openvino_model.xml")
    ov.save_model(ov_model, export_draft_model_name)
    print(f"== OV model exported and saved  successfully as {export_draft_model_name}.")

    # 导出为 ONNX 格式
    # dummy_input = {
    #     "target_hidden": torch.randn(1, 5, 12800),
    #     "noise_embedding": torch.randn(1, 5, 2560),
    #     "position_ids": torch.arange(5).unsqueeze(0),
    #     "past_key_values": None,
    #     "use_cache": True,
    #     "is_causal": False
    # }
    # torch.onnx.export(
    #     draft,
    #     (dummy_input["target_hidden"], dummy_input["noise_embedding"], dummy_input["position_ids"], dummy_input["past_key_values"], dummy_input["use_cache"], dummy_input["is_causal"]),
    #     "draft_model.onnx",
    #     input_names=["target_hidden", "noise_embedding", "position_ids", "past_key_values", "use_cache", "is_causal"],
    #     output_names=["output"],
    #     opset_version=13,
    # )

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

    if os.getenv("EXPORT_OV", "0") == "1":
        try:
            export_draft_model(draft, target)
            export_embedding_model(target)
        except Exception as exc:
            print(f"OV export failed: {exc}", file=sys.stderr)

    if os.getenv("EXPORT_ONLY", "0") == "1":
        print("EXPORT_ONLY=1, skip spec_generate.")
        return

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