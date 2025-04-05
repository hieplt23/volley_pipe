# Volleyball Project

## Overview

The Volleyball project is designed to **track players, detect balls**, and predict player actions within volleyball matches using advanced computer vision and machine learning techniques.

This project leverages the following features:

- **Ball tracking**: Detects the volleyball's movement and trajectory throughout the game.
- **Player tracking**: Identifies players and distinguishes their roles in the game environment.
- **Action prediction**: Predicts player movements or actions using advanced predictive modeling.

---

## Features

- **Accurate Player and Ball Tracking**  
  Utilizing OpenCV and deep learning models, the system efficiently tracks players and the ball in real-time.

- **Action Prediction**  
  Employs machine learning models to make predictions on player actions such as passing, spiking, or defending.

- **Scalable Configuration**  
  Configurable through YAML files, allowing flexible adjustment of model parameters and system settings.

---

## Installation

To install and run the Volleyball project, follow these steps:

### Prerequisites
Make sure you have Python 3.12.9 installed along with the following libraries:

- Numpy
- YOLO
- OpenCV-Python (`opencv-python`)
- PyYAML

Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

(*Note: Make sure to have a `requirements.txt` file containing the required libraries.*)

---

## Running the Project

1. Clone this repository:

    ```bash
    git clone https://github.com/hieplt23/voley_pipe.git
    cd voley_pipe
    ```

2. Configure the system using `config/config.yaml`:

    - Adjust parameters for trackers, predictors, and general settings.

3. Execute the main program:

    ```bash
    python main.py
    ```

4. (Optional) Customize pipeline steps using `src/pipeline.py`.

---

## Demo

Include a GIF below showing the tracking and prediction outputs:

**Tracking and Prediction Visualization**  
![Demo GIF here](Path_to_your_GIF.gif)


---

## Contributing

Contributions are welcome! If you'd like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m "Added a new feature"`).
4. Push to the branch (`git push origin feature-name`).
5. Open a pull request.