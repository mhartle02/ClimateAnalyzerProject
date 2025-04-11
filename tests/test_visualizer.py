import unittest
import time
from src.visualizer import Visualizer


class TestVisualizer(unittest.TestCase):
    def test_plot_clustered_data(self):
        # Record the start time
        start_time = time.time()

        # sample data used to run the visualizer
        temperatures = [75, 80, 85, 70, 78, 82, 88]
        cluster_numbers = [1, 1, 2, 1, 2, 2, 2]
        city_month_labels = ['Chicago Jan', 'New York Jan',
                             'Tallahassee Jan', 'Chicago Feb',
                             'New York Feb',
                             'Tallahassee Feb', 'Chicago Mar']
        # clusters centers
        cluster_centers = [(72,), (80,)]

        # calls the plotting method with our sample data
        Visualizer.plot_clustered_data(temperatures, cluster_numbers,
                                       city_month_labels, cluster_centers)

        # Print time
        end_time = time.time()
        time_taken = end_time - start_time
        print(f"Time taken for test_plot_clustered_data: "
              f"{time_taken:.4f} seconds")

    def test_plot_humidity_predictions(self):
        # start time
        start_time = time.time()

        # samples
        actual = [60, 65, 70, 75, 80]
        predicted = [62, 64, 69, 74, 81]
        x_labels = ['2024-01-01', '2024-01-02', '2024-01-03',
                    '2024-01-04', '2024-01-05']

        # call for humidity
        Visualizer.plot_humidity_predictions(actual, predicted, x_labels)

        # get final time
        end_time = time.time()
        time_taken = end_time - start_time
        print(f"Time taken for test_plot_humidity_predictions: "
              f"{time_taken:.4f} seconds")

    def test_plot_anomaly_bars(self):
        # start time
        start_time = time.time()

        # sample data
        anomalies = [('2024-01-01', 'Chicago', 5),
                     ('2024-01-02', 'New York', -3),
                     ('2024-01-03', 'Tallahassee', 2)]

        # method for plot
        Visualizer.plot_anomaly_bars(anomalies)

        # print time
        end_time = time.time()
        time_taken = end_time - start_time
        print(f"Time taken for test_plot_anomaly_bars: "
              f"{time_taken:.4f} seconds")


# Run the tests
if __name__ == '__main__':
    unittest.main()
