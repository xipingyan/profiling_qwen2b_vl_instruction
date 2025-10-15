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

# # bpe_tokenizer accelerate.
# from bpe_qwen import AutoLinearTokenizer
# tokenizer = AutoLinearTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")

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

def resize_img(image_url_or_file):
    fn = image_url_or_file.split("img_")[1]
    rsz_fn = "../test_video/rsz_"+fn
    if os.path.exists(rsz_fn):
        return rsz_fn

    image = Image.open(image_url_or_file).convert("RGB")
    resized_image = image.resize((448, 364), Image.Resampling.BICUBIC)
    resized_image.save(rsz_fn)
    return rsz_fn

def test_video(device):
    model_id = "../models/Qwen/Qwen2.5-VL-3B-Instruct/"
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
    from qwen_vl_utils import process_vision_info
    
    prompt = """输入的多张时间连续的视频画面是由车内的摄像头拍摄得到的，所以本视频的主体是拍摄视频的车辆。 下面所有的分析都是基于视频车相关的元素（如车辆状态、人车交互、物体和车之家的交互，周围环境特征等），完成以下任务：1.精准识别与车辆发生关系的核心事件或关键现象，包括但不限于：人物对车辆的交互类型（剐蹭、停靠、触碰、撞击、扒充电枪，附近移动，踢车，拍打等）；周围环境情况（周边有障碍物、违规停放，环境正常，环境复杂等），以及与车辆发生交互关系的人物的描述（外卖员，普通人，警察， 小孩， 老年人等）、与车辆发生交互关系的车辆类型（两轮车，三轮车，摩托车，自行车，轿车， SUV，房产，大货车等）； 简短事件描述(比如行人拍打车窗，外卖员剐蹭后视镜等)。2.经过仔细分析后将分析结果浓缩为 30 个 token 以内的结论，要求语言简洁、判断明确、无模糊表述。3.必须严格按照这个格式输出最后的结果：人物对车辆的交互类型 - 人物的描述 - 车辆类型 - 环境情况 简短事件描述。下面给出来多个示例：例1：视频为外卖员骑车剐蹭汽车后视镜，输出：剐蹭 - 外卖员 - 两轮车 - 环境正常 穿黄衣外卖员骑车疑似剐蹭后视镜。 例2：视频为可疑人员拔自车充电枪，输出：扒充电枪 - 普通人 - 无 - 环境正常 可疑人员疑似拔充电枪例。 例3：视频为行人和汽车在自车附近移动未接触，输出：附近移动 - 多种 - 多种 - 环境正常 多行人与汽车在自车附近移动未接触。 例4：视频为行人踢车后轮，输出：踢车 - 普通人 - 无 - 环境正常 行人疑似用脚踢自车轮胎。 例5：视频为黑色 SUV 停靠时，女子开右后车门轻碰旁边白色轿车（无明显划痕，女子未察觉离开），输出：碰撞 - 女子 - 白色轿车 - 环境正常 车门边缘轻微碰到旁边白色轿车门。 例6：视频为一辆红色大货车在人流复杂的路口拐弯的时候疑似剐蹭到了汽车的前保险杠，输出：剐蹭-无-大货车-环境复杂 大货车剐蹭了前保险杠。例7：视频为一个穿着黑色外套的男子在不停的拍打前挡风玻璃，周围是一个停车场，输出：拍打-男子-无-环境正常 男子拍打前挡风玻璃。请严格遵循上述要求，基于输入的车辆相关视频信息，输出 30token 以内的精准结论。"""

    print("== Test video:")
    messages = [{
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": [
                    resize_img('../test_video/rsz_video/img_0.png'),
                    resize_img('../test_video/rsz_video/img_1.png'),
                    resize_img('../test_video/rsz_video/img_2.png'),
                    resize_img('../test_video/rsz_video/img_3.png'),
                    resize_img('../test_video/rsz_video/img_4.png'),
                    resize_img('../test_video/rsz_video/img_5.png'),
                    resize_img('../test_video/rsz_video/img_6.png'),
                    resize_img('../test_video/rsz_video/img_7.png'),
                    resize_img('../test_video/rsz_video/img_8.png')
                ],
            },
            {"type": "image", "image": resize_img('../test_video/rsz_video/img_0.png')},
            # {"type": "text", "text": "请描述这个视频："},
            {"type": "text", "text": prompt},
        ],
    }]

    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

    processor = AutoProcessor.from_pretrained(model_id)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        # fps=1.0,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )
    inputs = inputs.to(device)

    # default: Load the model on the available device(s)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=device
    )

    # Inference
    t1 = time.time()
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    t2 = time.time()
    print(f"== infer time: {t2-t1:.3f} s")

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)

def test_imgages(device):
    model_id = "../Qwen/Qwen2.5-VL-3B-Instruct/"
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
    from qwen_vl_utils import process_vision_info

    # default: Load the model on the available device(s)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=device
    )
    processor = AutoProcessor.from_pretrained(model_id)

    messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": resize_img('../test_video/rsz_video/img_0.png')},
            {"type": "image", "image": resize_img('../test_video/rsz_video/img_1.png')},
            {"type": "image", "image": resize_img('../test_video/rsz_video/img_2.png')},
            {"type": "image", "image": resize_img('../test_video/rsz_video/img_3.png')},
            {"type": "image", "image": resize_img('../test_video/rsz_video/img_4.png')},
            {"type": "image", "image": resize_img('../test_video/rsz_video/img_5.png')},
            {"type": "image", "image": resize_img('../test_video/rsz_video/img_6.png')},
            {"type": "image", "image": resize_img('../test_video/rsz_video/img_7.png')},
            {"type": "image", "image": resize_img('../test_video/rsz_video/img_8.png')},
            {"type": "text", "text": "请描述这个视频："},
        ],
    }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    # Inference
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # unit_test_qwen2_vl_2b()
    # test_imgages(device)
    test_video(device)