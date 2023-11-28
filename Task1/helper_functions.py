import numpy as np
import pandas as pd


def signum(x):
    if x >= 0:
        return 1
    else:
        return -1


def calc_mse(x1, x2, y, w):
    total_mse = 0
    for i in range(len(x1)):
        net = w[0] + x1.iloc[i] * w[1] + x2.iloc[i] * w[2]
        error = y.iloc[i] - net
        total_mse += (error * error)
    total_mse = total_mse / (2 * len(x1))
    return total_mse


def feature_scaling(x, a, b):
    x = np.array(x)
    normalized_x = np.zeros((x.shape[0], x.shape[1]))
    for i in range(x.shape[1]):
        normalized_x[:, i] = ((x[:, i] - min(x[:, i])) / (max(x[:, i]) - min(x[:, i]))) * (b-a) + a
    normalized_x = pd.DataFrame(normalized_x)
    return normalized_x


def calculate_confusion_matrix(d, y):
    tp = sum((true == 1) and (pred == 1) for true, pred in zip(d, y))
    tn = sum((true == -1) and (pred == -1) for true, pred in zip(d, y))
    fp = sum((true == -1) and (pred == 1) for true, pred in zip(d, y))
    fn = sum((true == 1) and (pred == -1) for true, pred in zip(d, y))
    # print(d, y)
    # print(tp, tn, fp, fn)
    return tp, tn, fp, fn


def calculate_accuracy(tp, tn, fp, fn):
    total = tp + tn + fp + fn
    correct = tp + tn
    accuracy = correct / total
    return accuracy
