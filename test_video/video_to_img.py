import uuid
import requests
import cv2
from PIL import Image

def sample_frames(url, num_frames):
    if 0: # download from url
        response = requests.get(url)
        path_id = str(uuid.uuid4())
        path = f"./{path_id}.mp4" 
        with open(path, "wb") as f:
            f.write(response.content)
    else:
        # copy from windows
        path = "./01d7eb3dc6b737efecb3bcfd62b06508.mp4"

    video = cv2.VideoCapture(path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = total_frames // num_frames
    frames = []
    for i in range(total_frames):
        ret, frame = video.read()
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not ret:
            continue
        if i % interval == 0:
            frames.append(pil_img)
    video.release()
    return frames[:num_frames]

video_1 = "https://huggingface.co/spaces/merve/llava-interleave/resolve/main/cats_1.mp4"
video_1 = sample_frames(video_1, 9)
for idx,img in enumerate(video_1):
    img.save(f"img_{idx}.png")