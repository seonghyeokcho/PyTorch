import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def ReLU(x):
    return np.maximum(0, x)


def LeakyReLU(x, alpha):
    return np.maximum(x*alpha, x)


# hyperbolic tangent, tanh(쌍곡 탄젠트)
def tanh(x):
    return (np.exp(x)-np.exp(-x)) / (np.exp(x)+np.exp(-x))


# hyperbolic cosine, cosh(쌍곡 코사인)
def cosh(x):
    return (np.exp(x)+np.exp(-x)) / 2


# hyperbolic sine, sinh(쌍곡 사인)
def sinh(x):
    return (np.exp(x)-np.exp(-x)) / 2