import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Algorithms import *
from helper_functions import *
from plotting import *


def train_model(algo_type, feature1, feature2, class1, class2, eta, m, mse, isBias):
    c = {1: 'BOMBAY', 2: 'CALI', 3: 'SIRA'}
    c1 = c[class1]
    c2 = c[class2]
    class_name_to_target = {c1: -1, c2: 1}
    # Loading data
    data = pd.read_excel('..//data//Dry_Bean_Dataset.xlsx')
    print(data)
    # Fill the nan values with mean value of a specific target class
    # print('Before:', data.iloc[:5, feature2 - 1]) # for f2 = 4
    mean_value = data[data['Class'] == c1].mean()
    data.iloc[:, feature1 - 1].fillna(mean_value[feature1 - 1], inplace=True)
    data.iloc[:, feature2 - 1].fillna(mean_value[feature2 - 1], inplace=True)

    mean_value = data[data['Class'] == c2].mean()
    data.iloc[:, feature1 - 1].fillna(mean_value[feature1 - 1], inplace=True)
    data.iloc[:, feature2 - 1].fillna(mean_value[feature2 - 1], inplace=True)
    # print('After', data.iloc[:5, feature2 - 1])

    # Only take rows that have targets of class 1 or 2
    filtered_data = data[(data['Class'] == c1) | (data['Class'] == c2)]

    # Take your 2 features and target to split them Randomly
    X = filtered_data.iloc[:, [feature1-1, feature2-1]]
    X = feature_scaling(X, -1, 1)  # Scaling X from 0 to 1
    Y = filtered_data['Class'].map(class_name_to_target)  # Convert target names to -1 or 1
    Y = pd.Series(np.array(Y))  # Convert Y to Numpy array then Series to make the rows have the same indices
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, shuffle=True, random_state=10)

    # Divide the features into variables to use them in the learning algorithm
    f1 = X_train.iloc[:, 0]
    f2 = X_train.iloc[:, 1]

    if algo_type == '1':
        w, cnt_epochs = perceptron(f1, f2, y_train, eta, m, isBias)
        y_hats = test(w, X_test, y_test)
        # plot_decision_boundary(f1, f2, y_train, w, c1, c2, 'Training')
        plot_decision_boundary(X_test.iloc[:, 0], X_test.iloc[:, 1], y_test, w, c1, c2, 'Testing')
        tp, tn, fp, fn = calculate_confusion_matrix(np.array(y_test), y_hats)
        accuracy = calculate_accuracy(tp, tn, fp, fn) * 100
        plot_confusion_matrix(tp, tn, fp, fn, c1, c2, accuracy, cnt_epochs)
    else:
        w, cnt_epochs = adaline(f1, f2, y_train, eta, m, mse, isBias)
        y_hats = test(w, X_test, y_test)
        # plot_decision_boundary(f1, f2, y_train, w, c1, c2, 'Training')
        plot_decision_boundary(X_test.iloc[:, 0], X_test.iloc[:, 1], y_test, w, c1, c2, 'Testing')
        tp, tn, fp, fn = calculate_confusion_matrix(np.array(y_test), y_hats)
        accuracy = calculate_accuracy(tp, tn, fp, fn) * 100
        plot_confusion_matrix(tp, tn, fp, fn, c1, c2, accuracy, cnt_epochs)


def test(w, X_test, y_test):
    # cnt_true = 0
    # cnt_false = 0
    y_hats = []
    for i in range(len(X_test)):
        net = w[0] + X_test.iloc[i, 0] * w[1] + X_test.iloc[i, 1] * w[2]
        y_hat = signum(net)
        y_hats.append(y_hat)
        # if y_hat == y_test.iloc[i]:
        #     cnt_true += 1
        # else:
        #     cnt_false += 1
    return y_hats
