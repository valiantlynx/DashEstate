from cProfile import label

import numpy as np

def mse_loss(theta, x, y):
    f_theta = np.full_like(x, theta)
    return np.mean((f_theta - y) ** 2)


def ftheta(a,b,x):
    return (a * x + b)


a = 1
x = np.arange(0,1,0.01)
y = np.full_like(x, 1)


print(ftheta(a,1,x))

