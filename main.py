from argparse import ArgumentParser
from src.pipeline import VolleyballPipeline
from src.utils.io import load_config


if __name__ == '__main__':
    # Parse command-line arguments
    parser = ArgumentParser(description="Volleyball Tracking and Action Prediction")
    parser.add_argument('-i', '--video_path', type=str, default='inputs/input_video.mp4', help="Path to input video")
    parser.add_argument('-c', '--config', type=str, default="config/config.yaml", help="Path to config file")
    args = parser.parse_args()

    # Load config from YAML and update with command-line args
    config = load_config(args.config)
    config["video_path"] = args.video_path
    config["config_path"] = args.config

    # Run pipeline
    pipeline = VolleyballPipeline(config)
    pipeline.run()

