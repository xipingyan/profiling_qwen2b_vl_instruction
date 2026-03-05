import base64
import mimetypes
import os
import sys

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import cv2
from PIL import Image
from example_utils import load_video_frames

# from transformers import AutoProcessor, AutoTokenizer, Qwen3_5MoeForConditionalGeneration

import soundfile as sf
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info


local_model_id = os.getenv("MODEL_ID", "../models/Qwen/Qwen3-Omni-30B-A3B-Instruct/")
# Dependencies
# uv pip install git+https://github.com/huggingface/transformers
# uv pip install accelerate torchvision opencv-python soundfile qwen-omni-utils -U
# pip install -U flash-attn --no-build-isolation

def test_qwen3_omni():
    print(f"Start testing Qwen3 Omni with image input using model: {local_model_id}")

    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        local_model_id,
        # dtype="auto",
        # device_map="auto",
        # attn_implementation="flash_attention_2",
        dtype=torch.float16, device_map="cpu"        
    ) 

    processor = Qwen3OmniMoeProcessor.from_pretrained(local_model_id)

    conversation = [
        {
            "role": "user",
            "content": [
                # {"type": "image", "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cars.jpg"},
                {"type": "image", "image": "./cars.jpg"},
                # {"type": "audio", "audio": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cough.wav"},
                {"type": "audio", "audio": "./cough.wav"},
                {"type": "text", "text": "What can you see and hear? Answer in one short sentence."}
            ],
        },
    ]

    # Set whether to use audio in video
    USE_AUDIO_IN_VIDEO = True

    # Preparation for inference
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = processor(text=text, 
                    audio=audios, 
                    images=images, 
                    videos=videos, 
                    return_tensors="pt", 
                    padding=True, 
                    use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = inputs.to(model.device).to(model.dtype)

    # Inference: Generation of the output text and audio
    text_ids, audio = model.generate(**inputs, 
                                    speaker="Ethan", 
                                    thinker_return_dict_in_generate=True,
                                    use_audio_in_video=USE_AUDIO_IN_VIDEO)

    text = processor.batch_decode(text_ids.sequences[:, inputs["input_ids"].shape[1] :],
                                skip_special_tokens=True,
                                clean_up_tokenization_spaces=False)
    print(text)
    if audio is not None:
        output_audio_path = "output_qwen3_omni.wav"
        sf.write(
            output_audio_path,
            audio.reshape(-1).detach().cpu().numpy(),
            samplerate=24000,
        )
        print(f"Audio response saved to {output_audio_path}")


if __name__ == "__main__":
    try:
        test_qwen3_omni()
    except Exception as e:
        print(str(e), file=sys.stderr)
        raise