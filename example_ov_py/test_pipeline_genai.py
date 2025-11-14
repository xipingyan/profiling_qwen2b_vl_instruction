import openvino_genai as ov_genai
import requests
from PIL import Image
import numpy as np
import openvino as ov
import time

def get_pipeline():
    ov_model='../models/ov/Qwen2.5-VL-3B-Instruct/INT4/'

    print("== ov_model=", ov_model)
    device = 'GPU'
    # device = 'CPU'
    print("== device =", device)
    ATTENTION_BACKEND='SDPA'
    # ATTENTION_BACKEND='PA'
    print("== ATTENTION_BACKEND =", ATTENTION_BACKEND)
    pipe = ov_genai.VLMPipeline(ov_model, device=device, ATTENTION_BACKEND=ATTENTION_BACKEND)

    config = ov_genai.GenerationConfig()
    config.max_new_tokens = 50
    config.set_eos_token_id(pipe.get_tokenizer().get_eos_token_id())
    return pipe, config

def load_image_preresize(image_url_or_file):
    if str(image_url_or_file).startswith("http") or str(image_url_or_file).startswith("https"):
        image = Image.open(requests.get(image_url_or_file, stream=True).raw)
    else:
        image = Image.open(image_url_or_file).convert("RGB")
    resized_image = image.resize((448, 364), Image.Resampling.BICUBIC)
    return resized_image

def load_image(image_url_or_file):
    if str(image_url_or_file).startswith("http") or str(image_url_or_file).startswith("https"):
        image = Image.open(requests.get(image_url_or_file, stream=True).raw)
    else:
        image = Image.open(image_url_or_file).convert("RGB")
    return image

def load_all_images():
    imgs = []
    for idx in range(9):
        image = load_image_preresize(f'../test_video/rsz_video/img_{idx}.png')
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
    pipeline, config = get_pipeline()
    image = load_image_preresize('../test_video/rsz_0.png')
    ov_image = ov.Tensor(image)
    prompt = "请回答以下问题，务必只能回复一个词 \"Y\"或 \"N\"：图片和\"小狗。\"是否相关？"

    print(f"Question:\n  {prompt}")

    result_from_streamer = []
    def streamer(word: str) -> bool:
        nonlocal result_from_streamer
        result_from_streamer.append(word)
        return False

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
    pipe, config = get_pipeline()

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

def test_images_videos():
    print(f"== test_images_videos")
    pipe, config = get_pipeline()

    images = load_all_images()
    video = np.stack(images, axis=0)

    ov_video = ov.Tensor(video)
    image = load_image(f'../openvino.genai/tests/python_tests/.pytest_cache/d/images/cat.jpg')
    ov_imgs = [ov.Tensor(np.stack([image], axis=0))]

    print(f"== video shape = {video.shape}")
    print(f"== ov_imgs shape = {ov_imgs[0].data.shape}")
    
    prompt = "请描述这个视频："
    print(f"Question:\n  {prompt}")

    result_from_streamer=[]
    def streamer(word: str) -> bool:
        nonlocal result_from_streamer
        result_from_streamer.append(word)
        return False

    pipe.start_chat("you are a helpfull assitant.")
    for id in range(2):
        t1 = time.time()
        output = pipe.generate(prompt, images=ov_imgs, videos=ov_video, generation_config=config, streamer=streamer,)
        # pipe.finish_chat()
        t2 = time.time()
        print(f'== {id} time = {t2-t1:.3f} s')
    pipe.finish_chat()
    print('output = ', output)

import cv2
def test_ci_case():
    print(f"== test_ci_case")
    pipeline, config = get_pipeline()
    num_frames = 10
    video = cv2.VideoCapture("../test_video/spinning-earth-480.mp4")
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = 25
    frames = []
    for i in range(total_frames):
        ret, frame = video.read()
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not ret:
            continue
        if i % interval == 0:
            frames.append(pil_img)
    video.release()

    ov_video = ov.Tensor(np.stack(frames, axis=0))
    print(f"== video shape = {ov_video.get_shape()}")
    prompt = "What is Earth's spin and which continents are visible over time in the video? Which of them are shown on the beginning and which of them are presented on the end of the clip?"
    print(f"Question:\n  {prompt}")

    pipe.start_chat()
    for id in range(1):
        t1 = time.time()
        output = pipe.generate(prompt, videos=ov_video, generation_config=config, streamer=streamer,)
        t2 = time.time()
        print(f'== {id} time = {t2-t1:.3f} s')
    pipe.finish_chat()
    print('output = ', output)

def test_add_extension():
    ov_model='../models/ov/Qwen2.5-VL-3B-Instruct/INT4/'
    print("== ov_model=", ov_model)

    import platform, os, openvino_tokenizers
    os_name = platform.system()
    if os_name == "Windows":
        ov_tokenizer_path = os.path.dirname(openvino_tokenizers.__file__) + "\\lib\\openvino_tokenizers.dll"
    elif os_name == "Linux":
        ov_tokenizer_path = os.path.dirname(openvino_tokenizers.__file__) + "/lib/libopenvino_tokenizers.so"
    else:
        print(f"Skipped. Current test only support Windows and Linux")
        return

    pipe = ov_genai.VLMPipeline(ov_model, "CPU", {{"EXTENSIONS": ["/mnt/xiping/gpu_profiling/ov_self_build_model_example/python/custom_op/1_register_kernel/cpu/build/libopenvino_custom_add_extension.so"]}})

if __name__ == "__main__":
    print("OV Version:", ov.get_version())
    # test_image()
    # test_images_videos()
    test_video(as_video=True)
    # test_video(as_video=False)
    # test_add_extension()
