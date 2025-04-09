import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Tuple, Union

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
            x=range(len(sorted_temps)),       # X-axis: just a position for each point
            y=sorted_temps,                   # Y-axis: the sorted temperature values
            hue=sorted_clusters,              # Color by cluster
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
