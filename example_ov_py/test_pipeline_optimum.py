from PIL import Image 
import requests 
from optimum.intel.openvino import OVModelForVisualCausalLM
from transformers import AutoProcessor, TextStreamer
import openvino as ov

import transformers
from transformers.video_utils import load_video

input_video, _ = load_video("../../profiling_qwen2b_vl_instruction/test_video/01d7eb3dc6b737efecb3bcfd62b06508.mp4", num_frames=2, backend="opencv")

print(type(input_video))
print(input_video.shape)

exit(0)

# model_id = '../models/ov/Qwen2.5-VL-3B-Instruct/INT4/'
# processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
# ov_model = OVModelForVisualCausalLM.from_pretrained(model_id, trust_remote_code=True)

def load_image(image_url_or_file):
    if str(image_url_or_file).startswith("http") or str(image_url_or_file).startswith("https"):
        image = Image.open(requests.get(image_url_or_file, stream=True).raw)
    else:
        image = Image.open(image_url_or_file).convert("RGB")
    resized_image = image.resize((364, 448), Image.Resampling.BICUBIC)
    return resized_image

def load_all_images():
    imgs = []
    for idx in range(9):
        image = load_image(f'../test_video/img_{idx}.png')
        imgs.append(image)
    return imgs

def test_image():
    prompt = "<|image_1|>\n请描述这个视频："

    # image = load_all_images()[0]
    images = load_all_images()

    inputs = ov_model.preprocess_inputs(text=prompt, image=images, processor=processor)

    generation_args = { 
        "max_new_tokens": 50, 
        "temperature": 0.0, 
        "do_sample": False,
        # "streamer": TextStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
    } 

    generate_ids = ov_model.generate(**inputs, 
        eos_token_id=processor.tokenizer.eos_token_id, 
        **generation_args
    )

    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(generate_ids, 
    skip_special_tokens=True, 
    clean_up_tokenization_spaces=False)[0]

    print("== response =", response)

def test_video():
    prompt = "请描述这个视频："
    print("== Test video:")

    images = load_all_images()

    inputs = ov_model.preprocess_inputs(text=prompt, video=images, processor=processor)

    generation_args = { 
        "max_new_tokens": 50, 
        "temperature": 0.0, 
        "do_sample": False,
        # "streamer": TextStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
    } 

    generate_ids = ov_model.generate(**inputs, 
        eos_token_id=processor.tokenizer.eos_token_id, 
        **generation_args
    )

    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(generate_ids, 
    skip_special_tokens=True, 
    clean_up_tokenization_spaces=False)[0]

    print("== response =", response)

if __name__ == "__main__":
    print("OV Version:", ov.get_version())
    # test_image()
    test_video()