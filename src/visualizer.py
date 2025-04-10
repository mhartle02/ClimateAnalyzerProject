import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class Visualizer:
    @staticmethod
    def plot_clustered_data(temperatures, cluster_numbers, city_month_labels, cluster_centers):
        # sorts data so appears on graph in order
        sort_order = np.argsort(temperatures)
        sorted_temps = [temperatures[i] for i in sort_order]
        sorted_clusters = [cluster_numbers[i] for i in sort_order]
        sorted_labels = [city_month_labels[i] for i in sort_order]

        # creates the graph, making the size, axis' and colors
        plt.figure(figsize=(20, 6))
        sns.scatterplot(
            x=range(len(sorted_temps)),
            y=sorted_temps,
            hue=sorted_clusters,
        )

        # adds in the lines to represent the different clusters
        center_temperatures = sorted([center[0] for center in cluster_centers])
        for i in range(len(center_temperatures) - 1):
            boundary_temp = (center_temperatures[i] + center_temperatures[i + 1]) / 2
            plt.axhline(y=boundary_temp, linestyle='--', color='gray', alpha=0.5)

        # labels the x-axis with each point showing the date & city
        plt.xticks(ticks=range(len(sorted_labels)), labels=sorted_labels, rotation=90)

        # adds titles the axis'
        plt.ylabel("Average Temperature (°F)")
        plt.title("Clustered Monthly Average Temperatures by City")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_humidity_predictions(actual, predicted, x_labels):
        plt.figure(figsize=(12, 6))
        x = range(len(actual)) if x_labels is None else x_labels

        # creates the lines based on the data
        plt.plot(x, actual, label='Actual Humidity', marker='o')
        plt.plot(x, predicted, label='Predicted Humidity', marker='x')

        # creates the labels & point titles
        plt.xlabel("Date" if x_labels else "Sample Index")
        plt.ylabel("Humidity (%)")
        plt.title("Actual vs Predicted Humidity")
        plt.legend()
        if x_labels:
            plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_anomaly_bars(anomalies):
        # gets dates and temp diffs to use in graph
        dates = [a[0] for a in anomalies]
        differences = [a[2] for a in anomalies]

        # uses red for hotter temp, blue for colder
        colors = ['red' if diff > 0 else 'blue' for diff in differences]

        plt.figure(figsize=(14, 6))
        plt.bar(dates, differences, color=colors)
        plt.axhline(0, color='black', linewidth=0.8)
        plt.xticks(rotation=90)
        plt.ylabel('Temperature Difference from Monthly Avg (°F)')
        plt.title('Temperature Anomalies Compared to Monthly Averages')
        plt.tight_layout()
        plt.show()
