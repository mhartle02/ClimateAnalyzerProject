import unittest
import numpy as np
import pandas as pd
import time
#import the algorithms methods
from src.algorithms import (
    CustomHumidityPredictor,
    CustomTemperaturePredictor,
    Clustering,
    Anomaly
)

class TestAlgorithms(unittest.TestCase):
    #random seeding for the data
    def setUp(self):
        np.random.seed(42)

    def test_humidity_predictor(self):
        start_time = time.time()

        # Generate linear humidity data
        X = np.random.rand(100, 2)
        y = 3 * X[:, 0] + 5 * X[:, 1] + 2  # y = 3x1 + 5x2 + 2
        #generate the model and fit it with the random data x,y
        model = CustomHumidityPredictor(learning_rate=1e-2, n_iterations=1000)
        model.fit(X, y)
        predictions = model.predict(X)

        # Check that predictions are close +- 1
        self.assertTrue(np.allclose(predictions, y, atol=1.0))
        print(f"test_humidity_predictor took {time.time() - start_time:.4f} seconds")

    def test_temperature_predictor(self):
        start_time = time.time()

        X = np.random.rand(100, 2)
        y = -2 * X[:, 0] + 4 * X[:, 1] + 1  # y = -2x1 + 4x2 + 1
        model = CustomTemperaturePredictor(learning_rate=1e-2, n_iterations=1000)
        model.fit(X, y)
        predictions = model.predict(X)

        self.assertTrue(np.allclose(predictions, y, atol=1.0))

        print(f"test_temperature_predictor took {time.time() - start_time:.4f} seconds")

    def test_clustering(self):
        start_time = time.time()

        # Generate 3 clusters around 3 different centers
        cluster_1 = np.random.normal(0, 0.5, (10, 2))
        cluster_2 = np.random.normal(5, 0.5, (10, 2))
        cluster_3 = np.random.normal(10, 0.5, (10, 2))
        data = np.vstack((cluster_1, cluster_2, cluster_3))

        clusters, centers = Clustering(data, num_clusters=3)
        total_points = sum(len(indices) for indices in clusters.values())

        self.assertEqual(total_points, 30)
        self.assertEqual(len(centers), 3)

        print(f"test_clustering took {time.time() - start_time:.4f} seconds")

    def test_anomaly(self):
        start_time = time.time()

        data = {
            "YEAR": [2024]*6,
            "MONTH": [1]*6,
            "DAY": [1, 2, 3, 4, 5, 6],
            "temp": [30, 31, 32, 100, 29, 28]  # 100 is  an anomaly and checks for it
        }
        df = pd.DataFrame(data)
        anomalies = Anomaly(df, temp_column="temp", temp_diff_threshold=15)

        # Expect 1 anomaly
        self.assertEqual(len(anomalies), 1)
        self.assertIn("2024-01-04", anomalies[0][0])

        print(f"test_anomaly took {time.time() - start_time:.4f} seconds")

if __name__ == "__main__":
    unittest.main()
