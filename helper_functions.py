import numpy as np
from sklearn.preprocessing import MinMaxScaler

def fillna_and_scale_and_replace(df, column, a, b):
    # Fill NaN values with the mean
    df[column].fillna(df[column].mean(), inplace=True)

    values = np.array(df[column]).reshape(-1, 1)

    # Feature scaling
    scaler = MinMaxScaler(feature_range=(a, b))
    scaled_values = scaler.fit_transform(values)

    # Replace zero values with the median
    median_value = np.median(scaled_values[scaled_values != 0])
    scaled_values[scaled_values == 0] = median_value

    # Assign the scaled and replaced values back to the DataFrame
    df[column] = scaled_values.flatten()

def create_layers_neurons_list(num_layers, num_neurons):
    return [num_neurons] * num_layers

# Define the sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Define the tanh activation function and its derivative
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2