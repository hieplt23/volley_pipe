import json
import yaml

def save_json_data(data, output_path):
    if not data:
        raise ValueError("Empty data")

    # Save tracking data to JSON file
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Tracking data saved to {output_path}")

def load_config(config_path):
    if not config_path:
        raise ValueError('Empty config path')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)