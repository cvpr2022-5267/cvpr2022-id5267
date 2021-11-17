import os
import json
import numpy as np


def combine_label(path):
    # path = "/home/ning/toy_dexdexnet/data/val/labels"
    save_file = os.path.join(os.path.dirname(path), "labels.txt")
    result = []
    for file in sorted(os.listdir(path)):
        with open(os.path.join(path, file), "r") as f:
            r = json.load(f)
            result.append(list(r.values()))
    result = np.array(result).reshape(-1, 1)
    rows = np.arange(result.shape[0]).reshape(-1, 1)
    result = np.concatenate((rows, result), axis=1)
    np.savetxt(save_file, result, fmt="%03i %6.4f", delimiter="\t")
    return


if __name__ == "__main__":
    combine_label("/home/ning/toy_dexdexnet/data/test/labels")
