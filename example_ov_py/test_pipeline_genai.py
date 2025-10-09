import openvino_genai as ov_genai
from pathlib import Path
import requests
from PIL import Image
from io import BytesIO
import numpy as np
import openvino as ov
import time

ov_model='../models/ov/Qwen2.5-VL-3B-Instruct/INT4/'

print("== ov_model=", ov_model)
device = 'GPU'
# device = 'CPU'
print("== device = ", device)
pipe = ov_genai.VLMPipeline(ov_model, device=device, ATTENTION_BACKEND='SDPA')

config = ov_genai.GenerationConfig()
config.max_new_tokens = 50

def load_image(image_url_or_file):
    if str(image_url_or_file).startswith("http") or str(image_url_or_file).startswith("https"):
        image = Image.open(requests.get(image_url_or_file, stream=True).raw)
    else:
        image = Image.open(image_url_or_file).convert("RGB")
    resized_image = image.resize((448, 364), Image.Resampling.BICUBIC)
    return resized_image

def load_all_images():
    imgs = []
    for idx in range(9):
        image = load_image(f'../test_video/rsz_video/img_{idx}.png')
        imgs.append(image)
    return imgs

def streamer(subword: str) -> bool:
    """
    Args:
        subword: sub-word of the generated text.
    Returns: Return flag corresponds whether generation should be stopped.
    """
    print(subword, end="", flush=True)

def test_image():
    image = load_image('../test_video/img_0.png')
    ov_image = ov.Tensor(image)
    prompt = "请回答以下问题，务必只能回复一个词 \"Y\"或 \"N\"：图片和\"小狗。\"是否相关？"

    print(f"Question:\n  {prompt}")

    for id in range(5):
        t1 = time.time()
        # print("sssss=", type(image_tensor))
        # output = pipe.generate([prompt]*2, image=[image_tensor]*2, generation_config=config)
        output = pipe.generate(prompt, image=ov_image, generation_config=config)
        t2 = time.time()
        print(f'== {id} time = {t2-t1:.3f} s')
    print('output = ', output)

def test_video(as_video=True):
    print(f"== test video. as_video={as_video}")

    images = load_all_images()
    video = np.stack(images, axis=0)

    print(f"== video shape = {video.shape}")
    ov_video = ov.Tensor(video)
    ov_imgs = [ov.Tensor(np.stack([img], axis=0)) for img in images]
    print(f"== type(ov_imgs) = {type(ov_imgs)}, {ov_imgs[0].data.shape}")
    

    if as_video:
        prompt = "请描述这个视频："
    else:
        prompt = "请描述这些图像："
    print(f"Question:\n  {prompt}")

    for id in range(1):
        t1 = time.time()
        # pipe.start_chat()
        if as_video:
            output = pipe.generate(prompt, videos=[ov_video], generation_config=config)
        else:
            output = pipe.generate(prompt, images=ov_imgs, generation_config=config)
        # pipe.finish_chat()
        t2 = time.time()
        print(f'== {id} time = {t2-t1:.3f} s')
    print('output = ', output)

if __name__ == "__main__":
    print("OV Version:", ov.get_version())
    # test_image()
    test_video(as_video=True)
    # test_video(as_video=False)
