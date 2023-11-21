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