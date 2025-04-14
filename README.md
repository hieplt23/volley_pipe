# Volleyball Tracking and Action Recognition Pipeline

## Overview
This project implements a **Volleyball Tracking and Action Recognition Pipeline** using state-of-the-art deep learning models. The system is designed to process volleyball match videos, track players and the ball, and predict player actions. The pipeline is modular, allowing for easy customization and extension.

Key features:
- **Player Tracking**: Detect and track players using YOLO and ByteTrack.
- **Ball Tracking**: Detect and track the volleyball.
- **Action Recognition**: Predict player actions such as "block", "defense", "serve", "set" and "spike".
- **Visualization**: Generate annotated videos with tracking and action predictions.

---

## Project Structure
```planintext
volleyball/
├── config/                # Configuration files
├── data/                  # Input data (e.g., videos, masks)
├── models/                # Pre-trained models
├── inputs/                # Additional input files
├── outputs/               # Output directory for results
│   ├── tracking_data/     # Player and ball tracking data
│   ├── videos/            # Annotated videos
├── src/                   # Source code
│   ├── trackers/          # Player and ball tracking modules
│   ├── predictors/        # Action prediction module
│   ├── utils/             # Utility functions
│   ├── pipeline.py        # Main pipeline implementation
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker configuration
├── main.py                # Entry point for the application
```
---

## Pipeline Description
The pipeline consists of three main components:
1. **Player Tracking**:
   - Detects and tracks players using YOLO and ByteTrack.
   - Outputs bounding boxes and tracking IDs for each player.

2. **Ball Tracking**:
   - Detects and tracks the volleyball.
   - Outputs the ball's position and trajectory.

3. **Action Recognition**:
   - Predicts player actions based on bounding boxes and video frames.
   - Supports actions like "block", "defense", "serve", "set" and "spike".

---

## Installation

### Prerequisites
- **Docker**: Ensure Docker is installed on your system.
- **Python 3.8+**: If running locally, Python is required.

### Clone the Repository
```bash
git clone https://github.com/hieplt23/volley_pipe.git
cd volley_pipe
```
### Install Dependencies
If running locally:
```bash
pip install -r requirements.txt
```
---

## Usage
Running the Pipeline
1. **Prepare Input Files:**  
   - Place your video file in the ``inputs/`` directory.
   - Ensure the configuration file (``config/config.yaml``) is correctly set up.
2. **Run the Pipeline**: Execute the following command:
```python
python main.py
```
3. **Output:**
   - Tracking data will be saved in ``outputs/tracking_data/``.
   - Annotated videos will be saved in ``outputs/videos/``.
---

## Running with Docker
**Build the Docker Image**
```bash
docker build -t volleyvision -f Dockerfile .
```
**Run the Docker Container**
```bash
docker run --rm -v $(pwd)/inputs:/workspace/inputs -v $(pwd)/outputs:/workspace/outputs volleyvision
```
---

## Demo
![Video demo](./outputs/videos/demo.gif)
<p align="center"><i>Demo video</i></p>

---

Below is an example of the pipeline in action:  
1. **Input Video**:  
   A volleyball match video is provided as input.
2. **Output**:  
   Annotated video with:
    - Player tracking (bounding boxes).
    - Ball tracking (trajectory and position).
    - Action predictions (highlighted bounding boxes with action labels).
---

## Visualization
The pipeline includes a visualization module to overlay tracking and action predictions on the video. Key features:  
- **Player Tracking:** Players are highlighted with bounding boxes.
- **Ball Tracking:** The ball is tracked with a comet trail.
- **Action Recognition:** Actions are displayed with bounding boxes and labels.
To enable visualization:  
- Set ``show=True`` and/or ``save=True`` in the ``visualize()`` method of the ``VolleyballPipeline`` class.
---

## Configuration
The pipeline is configured using a YAML file (``config/config.yaml``). Key parameters include:  
- ``video_path``: Path to the input video.
- ``player_model_path``: Path to the player detection model.
- ``ball_model_path``: Path to the ball detection model.
- ``action_model_path``: Path to the action recognition model.
- ``output_dir``: Directory to save outputs.
---

## Example Pipeline Flow
1. **Input Video**:  
   - A volleyball match video is loaded.
2. **Player Tracking**:  
   - Players are detected and tracked frame-by-frame.
3. **Ball Tracking**:  
   - The ball's position and trajectory are determined.
4. **Action Recognition**:  
   - Player actions are predicted based on their movements and positions.
5. **Visualization**:  
   - An annotated video is generated, showing tracking and action predictions.
---

## Future Work
- Add support for additional sports.
- Improve action recognition accuracy with advanced models.
- Optimize performance for real-time processing.
---

## Contact
For questions or contributions, please contact:
- **Name**: Hiep Le Thanh
- **Email**: lethanhhiep0220@gmail.com
