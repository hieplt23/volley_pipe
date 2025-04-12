import cv2
from src.utils.io import save_json_data
from src.utils.img_utils import foot_pos, load_mask
import os
import json
from ultralytics import YOLO
from tqdm import tqdm

class PlayerTracker:
    def __init__(self, model_path, video_path, mask_path=None):
        # load YOLO model and set basic attributes
        self.model = YOLO(model_path)
        self.mask = load_mask(mask_path)
        self.video_path = video_path
        self.output_dir = "outputs/tracking_data"
        self.tracking_data = {"player": {}}  # store tracking data for players
        self.frame_width = 640

        # create output directory if it doesn't exist
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

    def process_video(self, read_from_json=True, json_path=None):
        # load tracking data from JSON if specified
        if read_from_json and json_path is not None and os.path.isfile(json_path):
            with open(json_path, 'r') as f:
                return json.load(f)

        # open video file
        cap = cv2.VideoCapture(self.video_path)
        frame_id = 0
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        # process each frame with progress bar
        with tqdm(total=num_frames, desc='Player tracking | Processing video...', colour='cyan') as pg_barr:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # detect and track players in current frame
                player_info = self.detect_and_track_players(frame, self.frame_width)
                self.tracking_data["player"][frame_id] = player_info

                frame_id += 1
                pg_barr.update(1)

        cap.release()

        # save tracking data to file
        output_path = os.path.join(self.output_dir, 'player.json')

        save_json_data(self.tracking_data, output_path)
        return self.tracking_data

    def detect_and_track_players(self, frame, frame_width):
        # detect and track players using YOLO with ByteTrack
        result = self.model.track(
            source=frame,
            tracker='bytetrack.yaml',
            imgsz=frame_width,
            verbose=False,
            classes=0  # class 0 for 'person'
        )[0]

        # store tracking info for each player
        player_info = {}
        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()

            for bbox, track_id, confidence in zip(boxes, track_ids, confidences):
                bbox = bbox.tolist()
                center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
                track_id = track_id.item()

                # foot position
                foot_pos_x, foot_pos_y = foot_pos(bbox)
                if self.mask[int(foot_pos_y-1), int(foot_pos_x-1)] == 0:  # check if player not in field
                    continue

                player_info[int(track_id)] = {
                    "bbox": bbox,
                    "center": center
                }

        return player_info