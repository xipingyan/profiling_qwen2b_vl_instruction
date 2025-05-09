import os
import sys
CUR_FILE_PATH=os.path.dirname(os.path.abspath(__file__))
sys.path.append(CUR_FILE_PATH + "/../")

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import time

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
            prompt_text = '''请回答以下问题，务必只能回复一个词 "Y"或 "N"：
                        图片和"'''+ desc_list[idx] +'''"是否相关？'''

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

unit_test_qwen2_vl_2b()