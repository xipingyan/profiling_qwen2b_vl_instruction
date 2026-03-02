import base64
import mimetypes
import os
import sys

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import cv2

local_model_id = os.getenv("MODEL_ID", "../models/Qwen/Qwen3.5-35B-A3B-Base")

def test_qwen3_5_llm():
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
    # messages = [
    #     {"role": "user", "content": "How do I run Qwen3.5 with PyTorch?"}
    # ]
    messages = [
        {"role": "user", "content": "如何跑Qwen3.5，基于PyTorch，需要处理一张图像，请只给出示例代码？"}
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
        max_new_tokens=1024,
        temperature=0.7,
        top_p=0.8
    )
    
    # 4. Decode output
    response = tokenizer.batch_decode(
        generated_ids[:, model_inputs.input_ids.shape[1]:], 
        skip_special_tokens=True
    )[0]
    
    print(response)

from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, Qwen3_5MoeForConditionalGeneration
# Depeandencies:
# 
def test_qwen3_5_image():
    print(f"Start testing Qwen3.5 with image input using model: {local_model_id}")        

    model = Qwen3_5MoeForConditionalGeneration.from_pretrained(local_model_id, dtype=torch.float16, device_map="cpu")
    tokenizer = AutoTokenizer.from_pretrained(local_model_id)
    processor = AutoProcessor.from_pretrained(local_model_id)

    if getattr(processor, "chat_template", None) is None and getattr(tokenizer, "chat_template", None):
        processor.chat_template = tokenizer.chat_template

    image_path = os.getenv("IMAGE_PATH", "demo.jpeg")
    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                # {"type": "text", "text": "Describe this image in short."},
                {"type": "text", "text": "请简要描述这张图片的内容。"},
            ],
        }
    ]

    # Preparation for inference
    inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
    inputs = inputs.to(model.device)

    # Generate
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print("Response:\n", response)

def load_video_frames(video_path, max_frames=10):
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    
    # Get total frame count to calculate the interval
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # Ensure we don't try to grab more frames than exist
    max_frames = min(max_frames, total_frames)
    
    # Calculate step size to get evenly distributed frames
    interval = total_frames // max_frames
    
    frames = []
    for i in range(max_frames):
        # Set the reader to the specific frame index
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
        ret, frame = cap.read()
        
        if not ret:
            break
            
        # OpenCV uses BGR; convert to RGB for PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
        
    cap.release()
    return frames, fps

def test_qwen3_5_video():
    print(f"Start testing Qwen3.5 with video input using model: {local_model_id}")
    model = Qwen3_5MoeForConditionalGeneration.from_pretrained(local_model_id, dtype=torch.float16, device_map="cpu")
    tokenizer = AutoTokenizer.from_pretrained(local_model_id)
    processor = AutoProcessor.from_pretrained(local_model_id)

    if getattr(processor, "chat_template", None) is None and getattr(tokenizer, "chat_template", None):
        processor.chat_template = tokenizer.chat_template

    video_path = os.getenv("VIDEO_PATH", "spinning-earth-480.mp4")
    # Load the video frames as a list of PIL Images. only keep 10 frames for testing.

    frames, fps = load_video_frames(video_path, max_frames=10)
    print(f"Loaded {len(frames)} frames from the video at {fps} FPS.")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": frames,
                },
                {"type": "text", "text": "请简要描述视频的内容。"},
            ],
        }
    ]

    # Preparation for inference
    inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            fps=fps
        )
    inputs = inputs.to(model.device)

    # Generate
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print("Response:\n", response)

if __name__ == "__main__":
    try:
        # test_qwen3_5_llm()
        # test_qwen3_5_image()
        test_qwen3_5_video()
    except Exception as e:
        print(str(e), file=sys.stderr)
        raise