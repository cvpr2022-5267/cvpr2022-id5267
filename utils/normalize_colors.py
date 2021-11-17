import os
import numpy as np
import json


def normalize_colors(path: str()):

    file_path = os.path.join(path, "colors.json")
    with open(file_path, "r") as f:
        data = json.load(f)
        colors = np.array(list(data.values()))
        r_m, g_m, b_m = colors.mean(axis=0).round(3)
        r_std, g_std, b_std = colors.std(axis=0).round(3)
        color_norm = {"mean": [r_m, g_m, b_m], "std": [r_std, g_std, b_std]}
    with open(os.path.join(path, "colors_norm.json"), "w") as f:
        json.dump(color_norm, f)


if __name__ == "__main__":
    normalize_colors("/home/ning/toy_dexdexnet/data/train")
