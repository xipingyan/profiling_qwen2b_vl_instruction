import openvino_genai as ov_genai
from pathlib import Path
import requests
from PIL import Image
from io import BytesIO
import numpy as np
import openvino as ov
import time

# ov_model='./ov_model_i8/'
# ov_model='../Qwen2-VL-2B-Instruct/INT4'
# ov_model='../Qwen2-VL-2B-Instruct_video/INT4'
ov_model='../Qwen2.5-VL-3B-Instruct/INT4'

print("== ov_model=", ov_model)
device = 'GPU.1'
print("== device = ", device)
pipe = ov_genai.VLMPipeline(ov_model, device=device)

config = ov_genai.GenerationConfig()
config.max_new_tokens = 100
# config.is_video=True
print("== config.is_video = ", config.is_video)

def load_image(image_url_or_file):
    if str(image_url_or_file).startswith("http") or str(image_url_or_file).startswith("https"):
        response = requests.get(image_url_or_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_url_or_file).convert("RGB")
    resized_image = image.resize((364, 448), Image.Resampling.BICUBIC)
    image_data = np.array(resized_image.getdata()).reshape(resized_image.size[1], resized_image.size[0], 3).astype(np.byte)
    return image, ov.Tensor(image_data)

def streamer(subword: str) -> bool:
    """
    Args:
        subword: sub-word of the generated text.
    Returns: Return flag corresponds whether generation should be stopped.
    """
    print(subword, end="", flush=True)

def test_image():
    image, image_tensor = load_image('../cat_1.jpg')
    prompt = "请回答以下问题，务必只能回复一个词 \"Y\"或 \"N\"：图片和\"小狗。\"是否相关？"

    print(f"Question:\n  {prompt}")

    for id in range(5):
        t1 = time.time()
        # print("sssss=", type(image_tensor))
        # output = pipe.generate([prompt]*2, image=[image_tensor]*2, generation_config=config)
        output = pipe.generate(prompt, image=image_tensor, generation_config=config)
        t2 = time.time()
        print(f'== {id} time = {t2-t1:.3f} s')
    print('output = ', output)
    # print('output.texts = ', output[0].texts, output[1].texts)
    # print('output[0].scores = ', output[0].scores)
    # print('output[1].scores = ', output[1].scores)

def test_video():
    print("== test video.")
    imgs = []
    for idx in range(9):
        image, image_tensor = load_image(f'../test_video/img_{idx}.png')
        imgs.append(image_tensor)

    prompt = "请描述这个视频："

    print(f"Question:\n  {prompt}")

    for id in range(1):
        t1 = time.time()
        output = pipe.generate(prompt, images=imgs, generation_config=config)
        t2 = time.time()
        print(f'== {id} time = {t2-t1:.3f} s')
    print('output = ', output)

if __name__ == "__main__":
    print("OV Version:", ov.get_version())
    # test_image()
    test_video()
