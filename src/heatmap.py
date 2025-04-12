import cv2
from ultralytics.solutions import Heatmap

cap = cv2.VideoCapture("../inputs/input_video.mp4")
assert cap.isOpened(), "Error reading video file"

# Video writer
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("../outputs/videos/heatmap_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# For object counting with heatmap, you can pass region points.
region_points = [[53, 1042], [1914, 1027], [1916, 974], [1256, 630], [470, 627]]   # polygon points

# Initialize heatmap object
heatmap = Heatmap(model="../models/ball_best.pt", colormap=cv2.COLORMAP_JET)

# Process video
while cap.isOpened():
    success, im0 = cap.read()

    if not success:
        print("Video frame is empty or processing is complete.")
        break

    results = heatmap(im0)

    # print(results)  # access the output

    video_writer.write(results.plot_im)  # write the processed frame.

cap.release()
video_writer.release()
cv2.destroyAllWindows()  # destroy all opened windows
