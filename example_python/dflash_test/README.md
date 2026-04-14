# Profiling DFlash.

Refer original [DFlash](https://github.com/z-lab/dflash)

# Reproduce 

Take Qwen3-4B as example.

#### ENV

```
git clone https://github.com/z-lab/dflash
python -m venv env_dflash
cd dflash
source ../env_dflash/bin/activate
uv pip install -e .
```

#### Run sample

```
./run.sh
```


## Convert draft model and verify

verify code: in the function: spec_generate

```
models/z-lab/Qwen3-4B-DFlash-b16/dflash.py:230
models/z-lab/Qwen3-4B-DFlash-b16/dflash.py:250
```

Ov infer's codes:

```
        use_ov = False
        if use_ov:
            import os
            import openvino as ov
            ov_core = ov.Core()
            ov_xml_path = os.getenv("OV_DRAFT_XML", "draft_model.xml")
            ov_draft = ov_core.read_model(ov_xml_path)
            ov_device = "GPU" if target.device.type == "cuda" else "CPU"
            ov_compiled = ov_core.compile_model(ov_draft, ov_device)



            if use_ov:
                print(f"== OV inference for block starting at position {start} ...")
                # The exported OV draft model is traced without KV cache.
                # It expects rotary/mask length to cover concatenated [target_hidden, noise] tokens.
                ov_ctx_len = target_hidden.shape[1]
                ov_q_len = noise_embedding.shape[1]
                ov_seq_len = ov_ctx_len + ov_q_len
                ov_position_ids = torch.arange(ov_seq_len, device=target.device).unsqueeze(0)
                ov_attention_mask = torch.ones((1, ov_seq_len), dtype=torch.bool, device=target.device)

                ov_outputs = ov_compiled(
                    [
                        ov_position_ids.cpu().numpy(),
                        ov_attention_mask.cpu().numpy(),
                        noise_embedding.cpu().numpy(),
                        target_hidden.cpu().numpy(),
                    ]
                )
                draft_result = torch.from_numpy(next(iter(ov_outputs.values()))).to(target.device)
                draft_result = draft_result[:, -block_size + 1 :, :]
            else:
                draft_result = self(
                    target_hidden=target_hidden,
                    noise_embedding=noise_embedding,
                    position_ids=position_ids[:, past_key_values_draft.get_seq_length(): start + block_size],
                    past_key_values=past_key_values_draft,
                    use_cache=True,
                    is_causal=False,
                )[:, -block_size+1:, :]
```