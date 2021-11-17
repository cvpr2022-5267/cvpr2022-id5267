import os
import json


def save_config(dir_path, name, data):
    os.makedirs(dir_path, exist_ok=True)
    with open(os.path.join(dir_path, name), "w") as f:
        json.dump(data, f, sort_keys=True, indent=4, separators=(',', ':'))
