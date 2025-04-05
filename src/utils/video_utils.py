import cv2

def read_video(path):
    video = cv2.VideoCapture(path)
    frames = []
    while True:
        flag, frame = video.read()
        if not flag:
            break
        frames.append(frame)
    return frames

def save_video(frames, path):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(path, fourcc, 30, (w, h))
    for frame in frames:
        out.write(frame)
    out.release()