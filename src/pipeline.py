from src.trackers.player_tracker import PlayerTracker
from src.trackers.ball_tracker import BallTracker
from src.predictors.action_predictor import ActionPredictor
import cv2
import os
import numpy as np


class VolleyballPipeline:
    def __init__(self, config):
        self.config = config
        self.ball_tracker = BallTracker(self.config["ball_model_path"], self.config["video_path"],
                                        self.config['mask_path'])
        self.player_tracker = PlayerTracker(self.config["player_model_path"], self.config["video_path"],
                                        self.config['mask_path'])
        self.action_predictor = ActionPredictor(self.config["action_model_path"], self.config["video_path"],
                                                action_classes=self.config["action_classes"], mask_path=self.config['mask_path'])

    def run(self):
        """Run the full pipeline: track ball, track players, predict actions."""
        ball_data = self.ball_tracker.process_video(json_path=self.config['ball_data'])
        player_data = self.player_tracker.process_video(json_path=self.config['player_data'])
        action_data = self.action_predictor.process_video(json_path=self.config['action_data'])
        self.visualize(ball_data, player_data, action_data, save=False)
        # return ball_data, player_data, action_data

    def visualize(self, ball_data, player_data, action_data, save=False):
        # visualize ball, players, and actions on video
        cap = cv2.VideoCapture(self.config["video_path"])
        frame_id = 0
        trail_history = []  # store ball positions for trail
        max_trail_length = 10

        # setup video writer
        output_video_path = os.path.join(self.config["output_dir"], "output_visualized.mp4")
        if save:
            os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # create overlay for drawing
            overlay = frame.copy()
            frame_id_str = str(frame_id)

            # draw ball with comet trail
            if frame_id_str in ball_data["ball"] and ball_data["ball"][frame_id_str]["center"]:
                ball_info = ball_data["ball"][frame_id_str]
                xc, yc = map(int, ball_info["center"])
                trail_history.append((xc, yc))
                if len(trail_history) > max_trail_length:
                    trail_history.pop(0)

                for i in range(len(trail_history) - 1):
                    alpha = (i + 1) / len(trail_history)
                    color = (0, int(165 * alpha), int(255 * alpha))  # orange gradient
                    thickness = max(1, int(5 * alpha))
                    cv2.line(overlay, trail_history[i], trail_history[i + 1], color, thickness)

                cv2.circle(overlay, (xc, yc), 8, (0, 165, 255), -1)  # orange glow
                cv2.circle(overlay, (xc, yc), 10, (255, 255, 255), 2)  # white outline

            # draw players with ellipse and action
            if frame_id_str in player_data["player"]:
                players = player_data["player"][frame_id_str]
                for track_id, info in players.items():
                    bbox = info["bbox"]
                    x_min, y_min, x_max, y_max = map(int, bbox)
                    center_x = (x_min + x_max) // 2
                    feet_y = y_max - 10

                    # print track id for player
                    cv2.putText(overlay, f'ID: {track_id}', (x_min, y_min-5), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                color=(0, 255, 0), fontScale=1, thickness=3)

                    # draw ellipse at feet
                    axes = (int((x_max - x_min) * 0.4), int((x_max - x_min) * 0.2))
                    cv2.ellipse(overlay, (center_x, feet_y), axes, 5, -10, 224, (148, 0, 211), 8)

            # draw action boxes and classes
            # action class colors (BGR format)
            action_class_colors = [
                (0, 255, 255),  # yellow for "block"
                (0, 255, 0),  # green for "defense"
                (255, 255, 0),  # cyan for "serve"
                (255, 165, 0),  # orange for "set"
                (0, 0, 255)  # red for "spike"
            ]
            if frame_id_str in action_data['action']:
                actions = action_data['action'][frame_id_str]
                if "bbox" in actions and "class" in actions:
                    action_boxes = actions["bbox"]
                    action_classes = actions["class"]
                    for box, cls in zip(action_boxes, action_classes):
                        x_min, y_min, x_max, y_max = map(int, box)
                        cls_idx = int(cls) % len(action_class_colors)  # ensure index is valid
                        color = action_class_colors[cls_idx]
                        action_text = self.config["action_classes"][cls_idx]

                        # draw gradient bounding box
                        cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), color, 5)
                        cv2.rectangle(overlay, (x_min + 2, y_min + 2), (x_max - 2, y_max - 2),
                                      (int(color[0] * 0.7), int(color[1] * 0.7), int(color[2] * 0.7)), 2)

                        # draw action label with shadow and background
                        text_pos = (x_min, y_min - 10)
                        text_size, _ = cv2.getTextSize(action_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
                        bg_x_min, bg_y_min = text_pos[0] - 5, text_pos[1] - text_size[1] - 5
                        bg_x_max, bg_y_max = text_pos[0] + text_size[0] + 5, text_pos[1] + 5
                        cv2.rectangle(overlay, (bg_x_min, bg_y_min), (bg_x_max, bg_y_max),
                                      (50, 50, 50, 150), -1)  # semi-transparent gray background
                        cv2.putText(overlay, action_text, (text_pos[0] + 2, text_pos[1] + 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)  # black shadow
                        cv2.putText(overlay, action_text, text_pos,
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)  # colored text

            # blend overlay with frame
            alpha = 0.5  # 50% transparency
            frame = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)

            # write frame if save is True
            if save:
                out.write(frame)

            # display frame
            # cv2.imshow('Volleyball Visualization', cv2.resize(frame, (1200, 780)))
            cv2.imshow('Volleyball Visualization', cv2.resize(frame, (640, 640)))
            if cv2.waitKey(7) & 0xFF == ord('q'):
                break
            frame_id += 1

        cap.release()
        if save:
            out.release()
            print(f"Video saved to {output_video_path}")

        cv2.destroyAllWindows()
