import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from collections import defaultdict

class CustomHumidityPredictor(BaseEstimator, RegressorMixin):
    def __init__(self, learning_rate=1e-4, n_iterations=10000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.feature_weights = None
        self.bias_term = None

    def fit(self, input_data, actual_humidity):
        # input_data is a 2D array: each row is a day, each column is a feature
        num_rows, num_features = input_data.shape

        # starts with a weight # close to 0 for each column
        # that changes based on how much of an impact it has on humidity
        self.feature_weights = np.random.randn(num_features) * 0.01

        self.bias_term = 0

        # trains the model n_iteration times
        for _ in range(self.n_iterations):
            # creates dot product to guess humidity based on current weight of column
            predicted_humidity = np.dot(input_data, self.feature_weights) + self.bias_term

            # finds error between predicted and real
            prediction_errors = predicted_humidity - actual_humidity

            # calculates how much the weight should be changed based on error
            gradient_weights = (2 / num_rows) * np.dot(input_data.T, prediction_errors)
            gradient_bias = (2 / num_rows) * np.sum(prediction_errors)

            # changes the weight
            self.feature_weights -= self.learning_rate * gradient_weights
            self.bias_term -= self.learning_rate * gradient_bias

        return self

    def predict(self, input_data):
        # uses the trained model weights to make a semi accurate prediction
        return np.dot(input_data, self.feature_weights) + self.bias_term

class CustomTemperaturePredictor(BaseEstimator, RegressorMixin):
    def __init__(self, learning_rate=1e-4, n_iterations=10000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.feature_weights = None
        self.bias_term = None

    def fit(self, input_data, actual_temperature):
        num_rows, num_features = input_data.shape
        self.feature_weights = np.random.randn(num_features) * 0.01
        self.bias_term = 0

        for _ in range(self.n_iterations):
            predicted_temperature = np.dot(input_data, self.feature_weights) + self.bias_term
            prediction_errors = predicted_temperature - actual_temperature
            gradient_weights = (2 / num_rows) * np.dot(input_data.T, prediction_errors)
            gradient_bias = (2 / num_rows) * np.sum(prediction_errors)

            self.feature_weights -= self.learning_rate * gradient_weights
            self.bias_term -= self.learning_rate * gradient_bias

        return self

    def predict(self, input_data):
        return np.dot(input_data, self.feature_weights) + self.bias_term

def Clustering(data_points, num_clusters=3, max_iterations=100):
    # picks random starting points for the clusters
    starting_indices = np.random.choice(len(data_points), num_clusters, replace=False)
    cluster_centers = data_points[starting_indices]

    for _ in range(max_iterations):
        clusters_dict = defaultdict(list)

        # goes through and assigns point to nearest cluster center based on avg temp
        for point_index, current_point in enumerate(data_points):
            distances_to_centers = [np.linalg.norm(current_point - center) for center in cluster_centers]
            nearest_center_index = np.argmin(distances_to_centers)
            clusters_dict[nearest_center_index].append(point_index)

        # updates cluster centers based on new data
        new_cluster_centers = []
        for cluster_points in clusters_dict.values():
            average_point = np.mean(data_points[cluster_points], axis=0)
            new_cluster_centers.append(average_point)

        new_cluster_centers = np.array(new_cluster_centers)

        # stop once through all data and centers arent changing
        if np.allclose(cluster_centers, new_cluster_centers):
            break

        cluster_centers = new_cluster_centers

    return clusters_dict, cluster_centers

def detect_anomalies(df, temp_column="temp", temp_diff_threshold=15):
    # creates DATE and finds all with each MONTH_NUM
    df["DATE"] = pd.to_datetime(df[["YEAR", "MONTH", "DAY"]])
    df["MONTH_NUM"] = df["DATE"].dt.month

    # finds average of all dates within that month along dataset
    month_averages = df.groupby("MONTH_NUM")[temp_column].mean().to_dict()

    anomalies = []
    for _, row in df.iterrows():
        month = row["MONTH_NUM"]
        temp = row[temp_column]
        avg = month_averages[month]
        diff = temp - avg

        if abs(diff) >= temp_diff_threshold:
            anomalies.append((row["DATE"].strftime("%Y-%m-%d"), temp, diff, avg))

    return anomalies