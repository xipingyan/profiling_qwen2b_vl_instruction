import os
import sys
CUR_FILE_PATH=os.path.dirname(os.path.abspath(__file__))
sys.path.append(CUR_FILE_PATH + "/../")

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import time

# OpenVINO
import openvino as ov
from openvino.runtime import Core, Model, Tensor, PartialShape, Type, Shape, op, serialize
import openvino.runtime as ov
import openvino.properties.hint as hints
from openvino.runtime.op import util as op_util
from openvino.runtime import opset8 as opset

EXPORT_OV = False

def torch_type_to_ov_type(ttype:torch.dtype):
    if ttype is torch.float32:
        return Type.f32
    elif ttype is torch.float16:
        return Type.f16
    else:
        print(f"== Fail: not supported type {ttype}")
        exit()

def qwen2VLVisionBlock(blk, hidden_states, ):
    # att
    #   norm1

    # blk.norm1.
    opset.mvn(hidden_states, )
    #  F.layer_norm(
    #         input, self.normalized_shape, self.weight, self.bias, self.eps
    #     )
    
    hidden_states
    # mlp

def new_vl_model(pt_model:Qwen2VLForConditionalGeneration):
    conv3d=pt_model.visual.patch_embed.proj
    input = opset.parameter([-1, -1], Type.f32, name='hidden_states')

    # step 1: conv3d =============================
    input_reshape = opset.reshape(input, [-1, 3, 2, 14, 14], True)
    numpy_weight = conv3d.weight.to(torch.float).cpu().detach().numpy()
    conv3d_weight = opset.constant(numpy_weight, torch_type_to_ov_type(torch.float32), "conv3d_weights")
    strides=[2,14,14]
    pads_begin = [0, 0, 0]
    pads_end = [0, 0, 0]
    dilations = [1, 1, 1]
    conv3d = opset.convolution(input_reshape, conv3d_weight, strides, pads_begin, pads_end, dilations)
    conv3d_reshape = opset.reshape(conv3d, [-1, 1280], special_zero=True)

    # step 2: qwen2vl =============================
    hidden_state = conv3d_reshape
    for blk in pt_model.visual.blocks:
        hidden_state = qwen2VLVisionBlock(blk, hidden_state)

    Result = opset.result(conv3d_reshape, name='vl_result')
    return Model([Result], [input], 'Model_VL')

def pipeline_vl_ov(ov_model, pt_model:Qwen2VLForConditionalGeneration, torch_input):
    core = Core()
    ov.save_model(ov_model, "my_vl.xml")
    cm = core.compile_model(model=ov_model, device_name='CPU')

    # input = torch.load("hidden_states_conv3d_inp.pt").to(torch.float32).cpu().numpy()
    grid_thw = torch_input["image_grid_thw"]
    hidden_state = torch_input["pixel_values"].to(torch.float32).cpu().numpy()
    # 'input_ids'
    # 'attention_mask'
    # pt_model.

    rotary_pos_emb = pt_model.visual.rot_pos_emb(grid_thw)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())
    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        dim=0,
        # Select dtype based on the following factors:
        #  - FA2 requires that cu_seqlens_q must have dtype int32
        #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
        # See https://github.com/huggingface/transformers/pull/34852 for more information
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
    print("cu_seqlens =", cu_seqlens)

    result = cm([hidden_state])[cm.output(0)]

    return result

def export_vl_to_ov(model:Qwen2VLForConditionalGeneration, torch_input):
    print("== Start to export VL model to OV.")

    ov_model = new_vl_model(model)
    result = pipeline_vl_ov(ov_model, model, torch_input)

    print('Result shape:', result.shape)
    outp_ov_pt = torch.tensor(result.tolist(), dtype=torch.float32)
    # torch.save(outp_ov_pt, "hidden_states_conv3d_output_ov.pt")
    outp_pt = torch.load("hidden_states_conv3d_outp.pt").to(torch.float32).cpu()

    print("== Start to compare result:")
    r = torch.isclose(outp_pt, outp_ov_pt, rtol=0.05, atol=0.05)
    # for i in range(outp_ov_pt.shape[0]):
    #     for j in range(outp_ov_pt.shape[1]):
    #         if outp_ov_pt[i][j] - outp_pt[i][j] > 0.01:
    #             print(f"== [{i}, {j}]diff {outp_ov_pt[i][j] - outp_pt[i][j]}")
    print(f"== torch.isclose(OV and PT), result = {r.all()}")

class Model_Qwen2_VL_2B():
    def __init__(self):
        self.__model_id = "../Qwen/Qwen2-VL-2B-Instruct"
        model_path=self.__model_id
        print(f"== '{self.__model_id}' path: {model_path}")
    
        self.__tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.__model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype="auto", # Use auto for appropriate precision (e.g., bfloat16 if supported)
                device_map="auto",  # Automatically distribute across available GPUs/CPU
                trust_remote_code=True
            ).eval() # Set to evaluation mode
        print("Model and tokenizer loaded.")
        max_pixels = 640*480
        self.__processor = AutoProcessor.from_pretrained(model_path,max_pixels=max_pixels)

    def infer_rerank(self, desc_list:list[str], img_list:list[str]):
        # --- 1. Load Model and Tokenizer ---
        messages_list = []
        for idx, img_path in enumerate(img_list):
            prompt_text = '''请回答以下问题，务必只能回复一个词 "Y"或 "N"：图片和"'''+ desc_list[idx] +'''"是否相关？'''
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": img_path,
                        },
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
            messages_list.append(messages)

        # Use tokenizer to prepare input - this might involve specific processing steps
        # for images combined with text for Qwen2-VL.
        # Check the documentation/examples for the canonical way.
        # A common pattern: tokenize text, get image features, combine.
        # Simplified approach using the tokenizer directly IF it handles images this way:
        texts = [
            self.__processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages_list
        ]
        image_inputs, video_inputs = process_vision_info(messages_list)
        inputs = self.__processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda" if torch.cuda.is_available() else 'cpu')
        # print(inputs)
        
        # --- 4. Model Forward Pass & Get Logits ---
        #print("Performing forward pass...")
        with torch.no_grad():
            outputs = self.__model(**inputs) # No labels needed, just forward pass
            logits = outputs.logits

        
        if EXPORT_OV:
            export_vl_to_ov(self.__model, inputs)

        #print("Forward pass complete.")
        #print(f"Logits shape: {logits.shape}") # Should be (batch_size, sequence_length, vocab_size)

        # --- 5. Locate Target Logits ('Y'/'N' variants) ---
        # We need the logits for the *next* token prediction after the input sequence ends
        # batch_size is likely 1 here
        last_token_logits = logits[:, -1, :] # Shape: (vocab_size,)
        #print(f"Logits for the next token prediction shape: {last_token_logits.shape}")

        # !! CRITICAL STEP: Identify the EXACT token IDs for 'Yes' and 'No' !!
        # This is highly model/tokenizer specific. Common candidates:
        # ' Yes', ' No' (with leading space)
        # ' Y', ' N' (with leading space)
        # 'Yes', 'No'
        # 'yes', 'no'
        # ' True', ' False'
        # You MUST verify which ones the model actually uses or prefers.
        # How to verify:
        # 1. Generate some answers to Y/N questions and see the output.
        # 2. Inspect the tokenizer's vocabulary.
        # 3. Check logits for multiple candidates.

        target_token_strings = [' Y', ' N'] # <--- *** ASSUMPTION - VERIFY THIS ***
        #print(f"Attempting to find logits for tokens: {target_token_strings}")
        sim = []

        try:
            token_ids = {}
            problematic_tokens = {}
            for token_str in target_token_strings:
                # Use encode, ensuring it adds neither BOS nor EOS unless intended,
                # and that it returns a single ID for the target token string.
                encoded = self.__tokenizer.encode(token_str, add_special_tokens=False)
                if len(encoded) == 1:
                    token_ids[token_str] = encoded[0]
                else:
                    problematic_tokens[token_str] = encoded
                    print(f"Warning: Token '{token_str}' did not encode to a single ID: {encoded}. Check the exact token string (spaces matter!).")

            if not token_ids:
                raise ValueError(f"Could not find single token IDs for any of the target tokens: {target_token_strings}. Please verify the exact token strings (e.g., with leading spaces) used by the Qwen2-VL-2B-Instruct tokenizer.")

            #print(f"Found token IDs: {token_ids}")

            # Extract the logit values for these specific token IDs
            extracted_logits = {}
            i = 0
            while i < last_token_logits.shape[0]:
                last_token_logit_per_batch = last_token_logits[i,:]
                i = i+ 1
                vocab_size = last_token_logit_per_batch.shape[0]
                for token_str, token_id in token_ids.items():
                    if 0 <= token_id < vocab_size:
                        extracted_logits[token_str] = last_token_logit_per_batch[token_id].item() # Get Python float
                    else:
                        print(f"Warning: Token ID {token_id} for '{token_str}' is out of vocab bounds ({vocab_size}).")
                #print(extracted_logits)
                # Print the extracted logits
                #print("\n--- Extracted Logits ---")
                #for token_str, logit_value in extracted_logits.items():
                    #print(f"Logit for token '{token_str}' (ID: {token_ids[token_str]}): {logit_value:.4f}")

                # (Optional) Compare and calculate relative probabilities
                if len(extracted_logits) == 2: # Assuming two target tokens like Yes/No
                    logit_vals = list(extracted_logits.values())
                    token_strs = list(extracted_logits.keys())
                    #print(f"\nComparison: Logit('{token_strs[0]}') {' > ' if logit_vals[0] > logit_vals[1] else ' < ' if logit_vals[0] < logit_vals[1] else ' = '} Logit('{token_strs[1]}')")

                    # Calculate relative probability using Softmax over these two logits
                    prob_tensor = torch.softmax(torch.tensor(logit_vals), dim=0)
                    #print(f"Relative Probability (considering only these two tokens):")
                    #print(f"  P('{token_strs[0]}') = {prob_tensor[0].item():.4f}")
                    #print(f"  P('{token_strs[1]}') = {prob_tensor[1].item():.4f}")
                    sim.append(prob_tensor[0].item())
                    #print(sim)

        except ValueError as e:
            print(f"\nError finding token IDs: {e}")
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            import traceback
            traceback.print_exc()

        return sim

def unit_test_qwen2_vl_2b():
    print("EXPORT_OV=", EXPORT_OV)
    if EXPORT_OV:
        print("== OV version:", ov.get_version())

    torch.cuda.set_device(1)
    print("== device: ", torch.cuda.get_device_name(torch.cuda.current_device()))

    model = Model_Qwen2_VL_2B()
    img_root='../'
    inp_imgs = [img_root+"cat_1.jpg"]
    inp_dess = ["a cat"]
    similarity = model.infer_rerank(inp_dess, inp_imgs)

    # inp_imgs = [img_root+"cat_1.jpg"]*2
    # inp_dess = ["a cat", "a dog"]
    # for id in range(20):
    #     t1= time.time()
    #     similarity = model.infer_rerank(inp_dess, inp_imgs)
    #     t2= time.time()
    #     print(f"== Infer [{id}] tm: {t2-t1:.3f} s")

    print(f"== similarity shape: {np.array(similarity).size}, type: {type(similarity)}, data type: {np.array(similarity).dtype}")

    print(f"== inputs:\n{inp_dess}\nVS\n{[os.path.split(fn)[-1] for fn in inp_imgs]}")
    print(f"== rerank similarity: {similarity}")
    print("== Done.")

if __name__ == "__main__":
    os.environ['EXPORT_OV']="1"
    EXPORT_OV = os.getenv('EXPORT_OV') == '1'
    unit_test_qwen2_vl_2b()