import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import openpyxl
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
def scale_and_replace(df, column, a, b):
    values = np.array(df[column]).reshape(-1, 1)

    # Feature scaling
    scaler = MinMaxScaler(feature_range=(a, b))
    scaled_values = scaler.fit_transform(values)

    # Replace zero values with the median
    median_value = np.median(scaled_values[scaled_values != 0])
    scaled_values[scaled_values == 0] = median_value

    # Assign the scaled and replaced values back to the DataFrame
    df[column] = scaled_values.flatten()

def Preprocessing(filepath = '..//data//Dry_Bean_Dataset.xlsx'):
    # Loading data
    data = pd.read_excel(filepath)
    target_map = {'BOMBAY': 0, 'CALI': 1, 'SIRA': 2}
    data['Class'] = data['Class'].map(target_map)
    X = data.iloc[:, 0:5]  # Features
    Y = data['Class']  # Label

    # Calculate mean values for each class
    class_means = data.groupby('Class').mean()
    # Replace NaN values conditionally based on class means
    for class_name in class_means.index:
        class_data = data[data['Class'] == class_name]
        nan_indices = class_data.isnull().any(axis=1)
        nan_rows = class_data[nan_indices]
        data.loc[nan_rows.index] = nan_rows.fillna(class_means.loc[class_name])

    # Splitting data based on classes
    class_bombay = data[data['Class'] == 0]
    class_cali = data[data['Class'] == 1]
    class_sira = data[data['Class'] == 2]

    # Splitting each class into X and target Y
    X_bombay = class_bombay.drop('Class', axis=1)
    Y_bombay = class_bombay['Class']
    X_cali = class_cali.drop('Class', axis=1)
    Y_cali = class_cali['Class']
    X_sira = class_sira.drop('Class', axis=1)
    Y_sira = class_sira['Class']

    columns_to_scale = ['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'roundnes']
    for column in columns_to_scale:
        scale_and_replace(X_bombay, column, -1, 1)
        scale_and_replace(X_cali, column, -1, 1)
        scale_and_replace(X_sira, column, -1, 1)

    # Splitting each class into train and test sets (60-40 split)
    X_train_bombay, X_test_bombay, Y_train_bombay, Y_test_bombay = train_test_split(X_bombay, Y_bombay, test_size=0.4,
                                                                                    train_size=0.6)
    X_train_cali, X_test_cali, Y_train_cali, Y_test_cali = train_test_split(X_cali, Y_cali, test_size=0.4,
                                                                            train_size=0.6)
    X_train_sira, X_test_sira, Y_train_sira, Y_test_sira = train_test_split(X_sira, Y_sira, test_size=0.4,
                                                                            train_size=0.6)

    # Concatenating train and test sets
    X_train = pd.concat([X_train_bombay, X_train_cali, X_train_sira])
    X_test = pd.concat([X_test_bombay, X_test_cali, X_test_sira])
    Y_train = pd.concat([Y_train_bombay, Y_train_cali, Y_train_sira])
    Y_test = pd.concat([Y_test_bombay, Y_test_cali, Y_test_sira])

    X_combined = pd.concat([X_train, X_test])
    Y_combined = pd.concat([Y_train, Y_test])

    # Shuffle combined data
    X_combined_shuffled, Y_combined_shuffled = shuffle(X_combined, Y_combined, random_state=42)

    # Split the shuffled combined data back into X_train, X_test, Y_train, Y_test
    X_train_shuffled = X_combined_shuffled.iloc[:len(X_train)]
    X_test_shuffled = X_combined_shuffled.iloc[len(X_train):]
    Y_train_shuffled = Y_combined_shuffled.iloc[:len(Y_train)]
    Y_test_shuffled = Y_combined_shuffled.iloc[len(Y_train):]
    # Converting the data into numpy arrays
    trainSamples = X_train_shuffled.to_numpy()
    trainLabels = Y_train_shuffled.to_numpy()
    testSamples = X_test_shuffled.to_numpy()
    testLabels = Y_test_shuffled.to_numpy()
    return trainSamples, trainLabels, testSamples, testLabels

