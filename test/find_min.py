import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin
import scipy.optimize as optimize


def f(x):
    a = np.arange(0, 360, 10)
    y = (a - x) ** 2 / 36
    y = np.sum(y)
    return y

# x = np.arange(-10, 10, 0.1)
# plt.plot(x, f(x))

# fmin(f, x0=0)
print(optimize.minimize(f, x0=0))

# def f(x):
#     return x**2 + 10*np.sin(x)
#
#
# x = np.arange(-10, 10, 0.1)
# plt.plot(x, f(x))
