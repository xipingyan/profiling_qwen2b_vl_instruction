# HF model example

Prepare model: Qwen/Qwen2-VL-2B-Instruct
Refer ../README.md to download.

```
    python3 -m venv qwen_env
    source qwen_env/bin/activate

    pip install qwen-vl-utils transformers torch accelerate torchvision 
    python ./example.py

```

# Model structure

1. apply_chat_template

    texts = ['<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_s...cat"是否相关？<|im_end|>\n<|im_start|>assistant\n']

2. process_vision_info

    image_inputs[pillow image, 756*476], video_inputs[None]

3. processor (Qwen2VLProcessor)
    <!-- Note, 'image_pad' -->
    texts='<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>请回答以下问题，务必只能回复一个词 "Y"或 "N"：\n                        图片和"a cat"是否相关？<|im_end|>\n<|im_start|>assistant\n'
    image_inputs=PIL.Image.Image image mode=RGB size=756x476 at 0x7F55544C59C0

    inputs = self.__processor( # Qwen2VLProcessor
            text=texts, 
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt")
        内部会修改text，在问题前面，添加'image_pad', 根据会产生的image token数目。后面，做完再把image feature替换进去。
        text tokenizer的时候, 会把'image_pad'替换为151655。

    inputs={
        'input_ids' = [1, 409]
        'attention_mask' = [1, 409]
        'pixel_values' = [1440, 1176]
        'image_grid_thw' = [1,4] = tensor([[ 1, 30, 48]])
    }

3.1 image_processor

    min,max pixels: 3136, 307200
    patch_size * merge_size = 14 * 2
    resized_height, resized_width = 420 * 672
    patches = [1, 3, 420, 672]
    repeats pathches=[2, 3, 420, 672]
    grid_t, grid_h, grid_w = 1,30,48
    1, 15, 24, 2, 2, 3, 2, 14, 14
    (1440, 1176)

4. Qwen/Qwen2-VL-2B-Instruct

    dtype = torch.bfloat16

4.1 VL model
    inputs_embeds[1,409,1536] = self.model.embed_tokens(input_ids[1,409]) # F.embedding(x, weight[151936, 1536],)

    image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw) -> [360,1536]
    Qwen2VisionTransformerPretrainedModel
        hidden_states = patch_embed(hidden_states)->[1440,1280]
            ->[-1,3,2,14,14]
            self.proj(conv3d), kernel:[2,14,14],stride:[2,14,14],out_channel:1280 - > [1440,1280]
            example_python/qwen_env/lib/python3.10/site-packages/transformers/models/qwen2_vl/modeling_qwen2_vl.py:230
        rotary_pos_emb=self.rot_pos_emb(grid_thw[[1,30,48]])->torch.Size([1440, 40])

        hidden_states = loop 33:Qwen2VLVisionBlock(hidden_states[1440, 1280])
        example_python/qwen_env/lib/python3.10/site-packages/transformers/models/qwen2_vl/modeling_qwen2_vl.py:1004
            hidden_states = norm1(hidden_states)
            hidden_states = hidden_states + self.attn:VisionSdpaAttention(hidden_states)
                q[1440, 16, 80], k[1440, 16, 80], v[1440, 16, 80] = self.qkv(hidden_states)
                q[1440, 16, 80], k[1440, 16, 80] = apply_rotary_pos_emb_vision(q, k, cos, sin)
                q[16, 1440, 80] = q.transpose(0, 1)
                k[16, 1440, 80] = k.transpose(0, 1)
                v[16, 1440, 80] = v.transpose(0, 1)
                attn_output[1, 16, 1440, 80] = F.scaled_dot_product_attention(q,k,v, attention_mask[1, 1440, 1440])
                attn_output[1440, 16, 80] = attn_output.squeeze(0).transpose(0, 1)
                attn_output[1440, 1280] = attn_output.reshape(seq_length, -1)
                attn_output[1440, 1280] = self.proj(attn_output) # proj = Linear(in_features=1280, out_features=1280, bias=True)
                #qkv = Linear(in_features=1280, out_features=3840, bias=True)

            example_python/qwen_env/lib/python3.10/site-packages/transformers/models/qwen2_vl/modeling_qwen2_vl.py:424
            hidden_states[1440, 1280]=norm2(hidden_states[1440, 1280]) # F.layer_norm, weights=1280,
            hidden_states[1440, 1280] = mlp(hidden_states[1440, 1280])->[1440, 1280]
                fc1 = Linear(in_features=1280, out_features=5120, bias=True)
                fc1 = fc1 * torch.sigmoid(1.702 * fc1[1440, 5120])
                fc2 = Linear(in_features=5120, out_features=1280, bias=True)
        
        self.merger(hidden_states[1440, 1280])->[360, 1536]
            x = self.mlp(self.ln_q(x[1440, 1280]).view(-1, self.hidden_size:5120))
                x[1440, 1280]=F.layer_norm(x[1440, 1280], weights[1280])
                x[360,5120]=x.view(-1, self.hidden_size:5120)
                x[360,5120]=F.linear(x[360,5120], weight[5120, 5120])
                x[360,5120]=F.gelu(x[360,5120])
                x[360,1536]=F.linear(x[360,5120], weights[1536, 5120], bias[1536])

            # self.ln_q = LayerNorm(context_dim, eps=1e-6)
            # self.mlp = nn.Sequential(
            #     nn.Linear(self.hidden_size, self.hidden_size),
            #     nn.GELU(),
            #     nn.Linear(self.hidden_size, dim),
            # )

    inputs_embeds = inputs_embeds.masked_scatter(image_embeds[360, 1536])->[1, 409, 1536]

4.2 base_model: Qwen2VLModel
    Qwen2VLModel(inputs_embeds[1,409,1536])->
    example_python/qwen_env/lib/python3.10/site-packages/transformers/models/qwen2_vl/modeling_qwen2_vl.py:1056
        position_embeddings = self.rotary_emb->[3, 1, 409, 128]*2

        for decoder_layer in self.layers: loop 29 ->hidden_states[1, 409, 1536]
            layer_outputs:hidden_states[1, 409, 1536] = decoder_layer(
                    hidden_states[1, 409, 1536],
                    attention_mask=causal_mask,
                    position_ids=position_ids[3, 1, 409],
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )
                example_python/qwen_env/lib/python3.10/site-packages/transformers/models/qwen2_vl/modeling_qwen2_vl.py:849
                hidden_states[1, 409, 1536] = self.input_layernorm(hidden_states[1, 409, 1536]) # Qwen2RMSNorm

                hidden_states, self_attn_weights, present_key_value = self.self_attn(hidden_states,...) # ```Qwen2VLSdpaAttention```
                example_python/qwen_env/lib/python3.10/site-packages/transformers/models/qwen2_vl/modeling_qwen2_vl.py:737
                    query_states[1, 409, 1536] = self.q_proj(hidden_states[1, 409, 1536])
                    key_states[1, 409, 256] = self.k_proj(hidden_states)
                    value_states[1, 409, 256] = self.v_proj(hidden_states)

                    query_states[1, 12, 409, 128] = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
                    key_states[1, 2, 409, 128] = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
                    value_states[1, 2, 409, 128] = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

                    query_states[1, 12, 409, 128], key_states[1, 2, 409, 128] = apply_multimodal_rotary_pos_emb(query_states, key_states)

                    key_states[1, 12, 409, 128] = repeat_kv(key_states, self.num_key_value_groups)
                    value_states[1, 12, 409, 128] = repeat_kv(value_states, self.num_key_value_groups)
                    
                    attn_output[1, 12, 409, 128] = torch.nn.functional.scaled_dot_product_attention(q,k,v)

                    attn_output = attn_output.transpose(1, 2).contiguous()
                    attn_output[1, 409, 1536] = attn_output.view(bsz, q_len, self.hidden_size)

                    attn_output[1, 409, 1536] = self.o_proj(attn_output[1, 409, 1536])
                
                hidden_states[1, 409, 1536] = residual + hidden_states[1, 409, 1536]
                
                # Fully Connected
                residual = hidden_states
                hidden_states = self.post_attention_layernorm(hidden_states)
                hidden_states[1, 409, 1536] = self.mlp(hidden_states) # 3 linear
                hidden_states = residual + hidden_states

        hidden_states[1, 409, 1536] = self.norm(hidden_states[1, 409, 1536])

        logits[1, 409, 151936]=lm_head(hidden_states[[1, 409, 1536]]) # linear(,weight[151936, 1536])

        Qwen2VLCausalLMOutputWithPast()