import argparse
from data_processor import DataProcessor
from algorithms import CustomHumidityPredictor, Clustering, Anomaly
from visualizer import Visualizer
import pandas as pd
import numpy as np

def load_clean_data():
    processor = DataProcessor("data/tallahassee_data.json")
    data = processor.load_data()
    if data is not None:
        return processor.clean_data()
    else:
        print("Could not load data.")
        return None

def run_humidity():
    data = load_clean_data()
    if data is None:
        return

    X = data[["temp", "dew", "precip", "windspeed"]].values
    y = data["humidity"].values

    valid = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X = X[valid]
    y = y[valid]
    data = data[valid]

    model = CustomHumidityPredictor()
    model.fit(X, y)
    predictions = model.predict(X)

    # Ask user which month they want to visualize
    try:
        month = int(input("Enter month number (1-12) to visualize: "))
        if month < 1 or month > 12:
            raise ValueError
    except ValueError:
        print("Invalid month number. Must be 1-12.")
        return

    month_indices = data["MONTH"] == month
    dates = pd.to_datetime(data[month_indices][["YEAR", "MONTH", "DAY"]]).dt.strftime("%Y-%m-%d").tolist()

    Visualizer.plot_humidity_predictions(
        actual=y[month_indices],
        predicted=predictions[month_indices],
        x_labels=dates
    )


def run_clustering():
    city_files = {
        "Tallahassee": "data/tallahassee_data.json",
        "New York City": "data/nyc_data.json",
        "Chicago": "data/chicago_data.json"
    }
    all_data = []
    for city, path in city_files.items():
        df = DataProcessor.load_and_clean_with_city(path, city)
        if df is not None:
            all_data.append(df)

    full_data = pd.concat(all_data)
    grouped = full_data.groupby(["city", "YEAR", "MONTH"]).agg({"temp": "mean"}).reset_index()

    temps = grouped[["temp"]].values
    groups, centers = Clustering(temps, num_clusters=3)

    labels, values, descriptions = [], [], []
    for group_id, indices in groups.items():
        for idx in indices:
            row = grouped.iloc[idx]
            labels.append(group_id)
            values.append(row["temp"])
            descriptions.append(f"{row['city']} - {int(row['MONTH'])}/{int(row['YEAR'])}")

    Visualizer.plot_clustered_data(values, labels, descriptions, centers)

def run_anomalies():
    data = load_clean_data()
    anomalies = Anomaly(data)
    if anomalies:
        print("\nDetected anomalies:")
        for date, temp, diff, avg in anomalies:
            label = "hotter" if diff > 0 else "colder"
            print(f"{date} → {temp:.2f}°F ({abs(diff):.2f}°F {label} than {avg:.2f}°F)")
        Visualizer.plot_anomaly_bars(anomalies)
    else:
        print("No anomalies found.")

def main():
    """
    Makes it so you can run the graph functions from the command line, without
    having to go through the menu

    [OPTION] can be humidity, cluster or anomaly
    Run:
    python3 src/cli.py --graph [OPTION]
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", choices=["humidity", "cluster", "anomaly"],
                        help="Choose a task: humidity, cluster, anomalies")

    args = parser.parse_args()

    if args.graph == "humidity":
        run_humidity()
    elif args.graph == "cluster":
        run_clustering()
    elif args.graph == "anomaly":
        run_anomalies()
    else:
        print("Invalid task. Use --graph with one of: humidity, cluster, anomaly")

if __name__ == "__main__":
    main()
