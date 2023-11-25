import numpy as np

from algorithms import *
from helper_functions import *
from plotting import *
import pandas as pd
import openpyxl
from sklearn.model_selection import train_test_split

def train_model(function_type, num_layers, num_neurons, eta, m, isBias):
    # Loading data
    df = pd.read_excel('data//Dry_Bean_Dataset.xlsx')
    # Split the data to training and testing sets
    # =============================================
    target_map = {'BOMBAY': 0, 'CALI': 1, 'SIRA': 2}
    df['Class'] = df['Class'].map(target_map)
    X = df.iloc[:, 0:5]  # Features
    Y = df['Class']  # Label
    columns_to_scale = ['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'roundnes']
    for column in columns_to_scale:
        fillna_and_scale_and_replace(X, column, -1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, shuffle=True, random_state=10)
    if function_type == 'sigmoid':
        input_size = 5
        num_neurons = [int(num) for num in num_neurons.split(",")]
        print(num_neurons)
        output_size = 3
        # print(X_train.values)
        # print(np.array(y_train).reshape(-1, 1))
        hidden_layers = num_neurons
        # print(layers)
        # create a Multilayer Perceptron with one hidden layer
        mlp = MLP(input_size, hidden_layers, output_size)
        # train network
        mlp.train(X_train.values, np.array(y_train).reshape(-1, 1), m, eta)
        output = mlp.forward_propagate(X_test.values)
        print(X_test.iloc[0])
        print(y_test.iloc[0])
        # Define label mapping
        label_mapping = {0: 'BOMBAY', 1: 'CALI', 2: 'SIRA'}

        print(output[0])
        # Replace "1" with corresponding labels in each inner list
        result_labels = [label_mapping[np.argmax(inner_list)] for inner_list in output]

        print(result_labels[0])



    else:
        print(function_type)
