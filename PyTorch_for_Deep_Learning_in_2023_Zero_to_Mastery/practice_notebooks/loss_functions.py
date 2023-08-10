import numpy as np


# SE(Squared Error) & SSE(Sum of Squares for Error)
def SE(y, y_hat):
    return (y - y_hat)**2


def SSE(y, y_hat):
    return 0.5 * np.sum((y - y_hat)**2)


# MSE(Mean Squared Error)
def MSE(y, y_hat):
    N = len(y)
    return (1/N)*np.sum((y - y_hat)**2)


# MAE(Mean Absolute Error)
def MAE(y, y_hat):
    N = len(y)
    return (1/N)*np.sum(abs(y - y_hat))