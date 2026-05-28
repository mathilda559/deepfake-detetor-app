import cv2
import numpy as np

def extract_frames(video_path, max_frames=30):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // max_frames)

    count = 0
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % step == 0:
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)

            count += 1
            if count >= max_frames:
                break

        frame_id += 1

    cap.release()
    return frames
