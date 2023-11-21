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
        print(function_type)
    else:
        print(function_type)
