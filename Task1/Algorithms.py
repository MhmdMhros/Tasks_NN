import numpy as np
import pandas as pd
from helper_functions import *


def perceptron(x1, x2, y, eta, m, isBias):
    """
    :param x1: series feature 1 column
    :param x2: series feature 2 column
    :param y: series target column
    :param eta: int learning rate
    :param m: int epochs
    :param isBias: int(0/1) generate random bias?
    :return: weights(theta) & minimum number of epochs for convergence
    """

    w = pd.concat([pd.Series([0]), pd.Series(np.random.rand(2))], ignore_index=True)

    cnt_epochs = 0
    for epoch in range(m):
        error_found = False
        for i in range(len(x1)):
            net = w[0] + x1.iloc[i] * w[1] + x2.iloc[i] * w[2]
            y_hat = signum(net)
            error = y.iloc[i] - y_hat
            if error != 0:
                error_found = True
                if isBias:
                    w[0] = w[0] + eta * error
                w[1] = w[1] + eta * error * x1.iloc[i]
                w[2] = w[2] + eta * error * x2.iloc[i]
        if not error_found:
            break
        cnt_epochs += 1
    return w, cnt_epochs


def adaline(x1, x2, y, eta, m, mse, isBias):
    """
    :param x1: series feature 1 column
    :param x2: series feature 2 column
    :param y: series target column
    :param eta: int learning rate
    :param m: int epochs
    :param mse: int MSE threshold
    :param isBias: int(0/1) generate random bias?
    :return: weights(theta) & minimum number of epochs for convergence
    """

    w = pd.concat([pd.Series([0]), pd.Series(np.random.rand(2))], ignore_index=True)

    cnt_epochs = 0
    for epoch in range(m):
        for i in range(len(x1)):
            net = w[0] + x1.iloc[i] * w[1] + x2.iloc[i] * w[2]
            y_hat = net  # Linear Activation Function
            error = y.iloc[i] - y_hat
            if isBias:
                w[0] = w[0] + eta * error
            w[1] = w[1] + eta * error * x1.iloc[i]
            w[2] = w[2] + eta * error * x2.iloc[i]
        total_mse = calc_mse(x1, x2, y, w)
        if total_mse < mse:
            break
        cnt_epochs += 1
    return w, cnt_epochs
