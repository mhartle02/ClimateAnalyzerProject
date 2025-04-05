import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

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
