import numpy as np


# SE(Squared Error)
def SE(y_hat: any, y: any) -> None:
    return (y - y_hat)**2


# SSE(Sum of Squares for Error) -> for a single class
def SSE(y_hat: any, y: any) -> None:
    return 0.5 * np.sum((y - y_hat)**2)


# MSE(Mean Squared Error) - L2 Loss
def MSE(y_hat: any, y: any) -> None:
    N = len(y)
    return (1/N) * np.sum((y - y_hat)**2)


# MAE(Mean Absolute Error) - L1 Loss
def MAE(y_hat: any, y: any) -> None:
    N = len(y)
    return (1/N) * np.sum(abs(y - y_hat))


# RMSE(Root Mean Squared Error)
def RMSE(y_hat: any, y: any) -> None:
    N = len(y)
    return np.sqrt((1/N) * np.sum((y - y_hat)**2))


# CEE(Cross Entropy Error)
def CEE(P: any, Q: any) -> None:
    '''
    "분류할 클래스의 수가 C > 2 인 '한 개의 데이터'에 대하여 사용"
    
    P: Label(Ground Truth)
    Q: Prediction
    delta: 로그의 성질에 따라 진수(x)가 0이 되어 로그값이 '-inf'로 발산하는 경우를 방지하기 위해 사용된다.
    Return a scalar 
    '''
    delta = 1e-7
    return -np.sum(P * np.log(Q + delta))


# BCEE(Binary Cross Entropy Error)
def BCEE(y_hat: any, y: any) -> None:
    '''
    "분류할 클래스 y가 0 또는 1 단 두가지인 '입력 데이터 전체'에 대하여 사용"
    
    y: Label(Ground Truth)
    y_hat: Prediction
    delta: 로그의 성질에 따라 진수(x)가 0이 되어 로그값이 '-inf'로 발산하는 경우를 방지하기 위해 사용된다.
    N: Number of Train Data
    Return a scalar
    '''
    delta = 1e-7
    N = len(y)
    return -(1/N) * np.sum(y*np.log(y_hat+delta) + (1-y)*np.log(1-y_hat+delta))


# CCEE(Categorical Cross Entropy Error)
def CCEE(y_hat: any, y: any) -> None:
    '''
    "분류할 클래스의 수가 C > 2 인 '입력 데이터 전체'에 대하여 사용"
    
    y: Label(Ground Truth)
    y_hat: Prediction
    delta: 로그의 성질에 따라 진수(x)가 0이 되어 로그값이 '-inf'로 발산하는 경우를 방지하기 위해 사용된다.
    Return a scalar
    '''
    delta = 1e-7
    N = y.shape[0]
    return -(1/N)*np.sum(np.sum(y*np.log(y_hat + delta), axis=1))