import cv2
import numpy as np
import os
import json
from tqdm import tqdm
from ultralytics import YOLO
from src.utils.io import save_json_data
from src.utils.img_utils import load_mask, foot_pos

# constants for easy configuration
DEFAULT_FRAME_WINDOW = 5  # number of frames to consider for action prediction


class ActionPredictor:
    def __init__(self, model_path, video_path,
                 frame_window=DEFAULT_FRAME_WINDOW, action_classes=None, mask_path=None):
        # load action prediction model and set basic attributes
        self.model = YOLO(model_path) # placeholder for model loading
        self.mask = load_mask(mask_path)
        self.video_path = video_path
        self.output_dir = "outputs/action_data"
        self.action_data = {'action': {}}  # store action predictions
        self.frame_window = frame_window  # window size for action context
        self.action_classes = action_classes  # list of possible actions
        self.frame_width = 640

        # create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

    def process_video(self, read_from_json=True, json_path=None):
        # load tracking data from JSON if specified
        if read_from_json and json_path is not None and os.path.isfile(json_path):
            with open(json_path, 'r') as f:
                return json.load(f)

        # open video file
        cap = cv2.VideoCapture(self.video_path)
        frame_id = 0
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # process each frame with progress bar
        with tqdm(total=num_frames, desc='Action | Processing video...', colour='cyan') as pg_barr:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # detect and track players in current frame
                action_info = self.predictor(frame, self.frame_width)
                self.action_data["action"][frame_id] = action_info

                frame_id += 1
                pg_barr.update(1)

        cap.release()

        # save tracking data to file
        output_path = os.path.join(self.output_dir, 'action.json')

        save_json_data(self.action_data, output_path)
        return self.action_data

    def predictor(self, frame, frame_width):
        # detect and track players using YOLO with ByteTrack
        result = self.model(source=frame, imgsz=frame_width, verbose=False)[0]

        # store tracking info for each player
        boxes_list, classes_list = [], []
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            for i, (bbox, cls) in enumerate(zip(boxes, classes)):
                bbox = bbox.tolist()

                # foot position
                foot_pos_x, foot_pos_y = foot_pos(bbox)
                if self.mask[int(foot_pos_y-1), int(foot_pos_x-1)] == 0:
                    continue

                cls = cls.tolist()
                center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
                boxes_list.append(bbox)
                classes_list.append(cls)

        action_info = {
            "bbox": boxes_list,
            "class": classes_list
        }

        return action_info