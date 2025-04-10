from ultralytics import YOLO
from tqdm import tqdm
import cv2
import os
import json
from src.utils.io import save_json_data

# constants for easy configuration
DEFAULT_FRAME_SIZE = 640
MAX_HISTORY = 5
MAX_MISSED_THRESHOLD = 5

class BallTracker:
    def __init__(self, model_path, video_path, mask_path,
                 max_history=MAX_HISTORY, max_missed_threshold=MAX_MISSED_THRESHOLD):
        # load YOLO model and set basic attributes
        self.model = YOLO(model_path)
        self.video_path = video_path
        self.output_dir = "outputs/tracking_data"
        self.tracking_data = {"ball": {}}
        self.history = []  # list to store ball position history
        self.max_history = max_history
        self.missed_frame_count = 0  # count frames where ball is not detected
        self.max_missed_threshold = max_missed_threshold
        self.frame_width = DEFAULT_FRAME_SIZE  # default video width
        self.frame_height = DEFAULT_FRAME_SIZE  # default video height

        # create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

    def predict_missing_frame(self):
        # predict ball position when not detected
        if self.missed_frame_count >= self.max_missed_threshold or len(self.history) < 1:
            self.missed_frame_count += 1
            return {"bbox": None, "center": None}

        # use last position if only one point exists
        if len(self.history) == 1:
            return {"bbox": None, "center": self.history[-1]}

        # calculate velocity and predict new position using linear interpolation
        last_pos, prev_pos = self.history[-1], self.history[-2]
        velocity = (last_pos[0] - prev_pos[0], last_pos[1] - prev_pos[1])
        predicted_center = (last_pos[0] + velocity[0], last_pos[1] + velocity[1])

        # check if predicted position is within frame
        if (0 <= predicted_center[0] <= self.frame_width and
            0 <= predicted_center[1] <= self.frame_height):
            if len(self.history) >= self.max_history:
                self.history.pop(0)  # remove oldest position
            self.history.append(predicted_center)
            return {"bbox": None, "center": predicted_center}

        self.missed_frame_count += 1
        return {"bbox": None, "center": None}

    def detect_ball(self, frame):
        # detect ball in the frame using YOLO
        results = self.model.predict(source=frame, imgsz=640, conf=0.30, verbose=False)

        if results and len(results[0].boxes) > 0:
            box = results[0].boxes[0]
            bbox = box.xyxy[0].tolist()
            center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]

            # reset missed count and update history when ball is detected
            self.missed_frame_count = 0
            if len(self.history) >= self.max_history:
                self.history.pop(0)
            self.history.append(center)
            return {"bbox": bbox, "center": center}

        # if no detection, predict position
        self.missed_frame_count += 1
        return self.predict_missing_frame()

    def process_video(self, read_from_json=True, json_path=None):
        # load tracking data from JSON if specified
        if read_from_json and json_path and os.path.isfile(json_path):
            with open(json_path, 'r') as f:
                return json.load(f)

        # open video file
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")

        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # set frame dimensions from video
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # process each frame with progress bar
        with tqdm(total=num_frames, desc='Ball tracking | Processing video...', colour='cyan') as pbar:
            frame_id = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                self.tracking_data["ball"][frame_id] = self.detect_ball(frame)
                frame_id += 1
                pbar.update(1)

        cap.release()

        # path of json data
        output_path = os.path.join(self.output_dir, 'ball.json')
        save_json_data(self.tracking_data, output_path) # save results after processing

        return self.tracking_data