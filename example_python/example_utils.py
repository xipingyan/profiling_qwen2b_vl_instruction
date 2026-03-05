import cv2
from PIL import Image

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