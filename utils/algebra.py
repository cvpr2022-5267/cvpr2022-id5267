import math
import numpy as np



def calc_function(x, y, angle):
    """given point (x, y) and angle, calculate linear function y = kx + b, return b"""

    k = math.tan(angle)
    b = y - k * x
    return b


def calc_mean_var(x):
    x = np.array(x)
    mean = np.mean(x)
    std = np.std(x)
    return mean, std


if __name__ == "__main__":
    x = [8.0853, 7.8592, 8.0334, 8.0864, 8.1560]
    mean, std = calc_mean_var(x)
    print("mean: ", mean)
    print("std: ", std)

