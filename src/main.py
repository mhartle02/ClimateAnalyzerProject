from data_processor import DataProcessor
from visualizer import Visualizer
from algorithms import CustomHumidityPredictor, CustomTemperaturePredictor, Clustering, detect_anomalies
import pandas as pd
import numpy as np
from time import *

def cleanData():
    file_path = "data/tallahassee_data.json"

    #loads in the data from data_processor.py file
    dp = DataProcessor(file_path)
    data = dp.load_data()

    if data is not None:
        print("Loaded data successfully from file")
        print("First few rows:\n", data.head())

        #cleans data from data_processor.py file
        cleaned_data = dp.clean_data()
        print("Data cleaned successfully")
        print("Columns after cleaning:\n", cleaned_data.columns.tolist())
        print("\nFirst few cleaned rows:\n", cleaned_data.head())
        return cleaned_data
    else:
        print("Data not loaded")

def predict_humidity():
        cleaned_data = cleanData()
        # X = columns we want to look at to see how it effects Y
        X = cleaned_data[["temp", "dew", "precip", "windspeed"]].values
        y = cleaned_data["humidity"].values

        # checks and removes any invalid rows
        valid_indices = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[valid_indices]
        y = y[valid_indices]

        # creates model that learns from the data set to predict humidity
        model = CustomHumidityPredictor(learning_rate=1e-4, n_iterations=10000)
        model.fit(X, y)
        print("\nFinished training model")

        print("\nEnter data to predict humidity:")

        try:
            print("\nEnter values for humidity prediction below.")
            print(
                "Temperature (°F): The air temperature. In Tallahassee, it typically ranges from 35°F (winter) to 95°F (summer).")
            temp = float(input("Enter Temperature (°F): "))

            print("\nDew Point (°F): The temperature at which dew forms. Higher dew points feel more humid.")
            print("In Tallahassee, dew points usually range from 50°F to 75°F.")
            dew = float(input("Enter Dew Point (°F): "))

            print("\nPrecipitation (inches): Total rainfall for the day.")
            print("Tallahassee typically sees 0.1–1 inch on rainy days. Over 2 inches indicates heavy rain.")
            precip = float(input("Enter Precipitation (inches): "))

            print("\nWindspeed (mph): Average windspeed. Higher winds usually reduce humidity.")
            print(
                "In Tallahassee, average daily windspeed ranges from 5 to 15 mph, with occasional gusts above 20 mph.")
            windspeed = float(input("Enter Windspeed (mph): "))

            user_input = pd.DataFrame({
                "temp": [temp],
                "dew": [dew],
                "precip": [precip],
                "windspeed": [windspeed]
            })

            # sends user data into the model and predicts humidity
            predicted_humidity = model.predict(user_input.values)[0]
            print(f"\nPredicted Humidity: {predicted_humidity:.2f}%")

        except ValueError:
            print("Enter numbers only. Try again.")


def predict_temperature():
    # Step 1: Clean data
    data = cleanData()  # Assuming cleanData() cleans and returns the DataFrame

    # Step 2: Group data by year and month
    monthly = data.groupby(["YEAR", "MONTH"]).agg({
        'temp': 'mean',
        'dew': 'mean',
        'humidity': 'mean',
        'precip': 'sum',
        'windspeed': 'mean'
    }).reset_index()

    # Step 3: Create feature 'prev_temp' (previous month's temperature)
    monthly['prev_temp'] = monthly['temp'].shift(1)

    # Remove rows with NaN values (after the shift)
    monthly = monthly.dropna()

    # Step 4: Prepare features (X) and target (y)
    X = monthly[['MONTH', 'dew', 'humidity', 'precip', 'windspeed', 'prev_temp']].values
    y = monthly['temp'].values

    # Step 5: Train the custom temperature predictor
    model = CustomTemperaturePredictor(learning_rate=1e-5, n_iterations=10000)
    model.fit(X, y)

    # Step 6: Predict temperature for the next months
    # Start with the last month in the dataset
    last_month_data = monthly.iloc[-1].copy()  # Make a copy of the last row

    last_month_features = np.array([[last_month_data['MONTH'],
                                     last_month_data['dew'],
                                     last_month_data['humidity'],
                                     last_month_data['precip'],
                                     last_month_data['windspeed'],
                                     last_month_data['temp']]])

    future_months = 12  # Predicting for the next 12 months
    predicted_temperatures = []

    for _ in range(future_months):
        predicted_temp = model.predict(last_month_features)[0]
        predicted_temperatures.append(predicted_temp)

        # Update the last month's data for the next prediction
        # We update the `prev_temp` and `month` for the next prediction
        next_month = (last_month_data['MONTH'] % 12) + 1
        last_month_features = np.array([[next_month,
                                         last_month_data['dew'],
                                         last_month_data['humidity'],
                                         last_month_data['precip'],
                                         last_month_data['windspeed'],
                                         predicted_temp]])

        # Update last month data for the next iteration
        last_month_data['MONTH'] = next_month
        last_month_data['temp'] = predicted_temp  # Use predicted temp as previous month's temp for next month

    # Step 7: Print out the predicted temperatures for the next 12 months
    print("Predicted Average Temperatures for the next 12 months:")
    for i, temp in enumerate(predicted_temperatures, 1):
        print(f"Month {i}: {temp:.2f}°F")


def cluster_temperatures():
    city_files = {
        "Tallahassee": "data/tallahassee_data.json",
        "New York City": "data/nyc_data.json",
        "Chicago": "data/chicago_data.json"
    }

    # puts all the data for each city into one table
    all_city_data = []
    for city_name, file_path in city_files.items():
        city_data = DataProcessor.load_and_clean_with_city(file_path, city_name)
        if city_data is not None:
            all_city_data.append(city_data)

    full_table = pd.concat(all_city_data)

    # getting average temperatures for each month
    monthly_averages = full_table.groupby(["city", "YEAR", "MONTH"]).agg({
        "temp": "mean"
    }).reset_index()

    # runs cluster function based on the avg temp values
    temperature_values = monthly_averages[["temp"]].values
    cluster_groups, cluster_centers = Clustering(temperature_values, num_clusters=3)

    # prints results
    print("\nGrouped Months by Temperature:\n")
    for group_number, month_indices in cluster_groups.items():
        center_temp = cluster_centers[group_number][0]
        print(f"Group {group_number + 1} (Average Temp: {center_temp:.2f}°F):")
        for i in month_indices:
            row = monthly_averages.iloc[i]
            print(f"  {row['city']} - {int(row['MONTH'])}/{int(row['YEAR'])} (Avg Temp: {row['temp']:.2f}°F)")
        print()

    #creates empty lists so data can be appended to send to graph
    labels = []
    temps = []
    descriptions = []

    # fills lists from data
    for group_id, indices in cluster_groups.items():
        for i in indices:
            row = monthly_averages.iloc[i]
            labels.append(group_id)
            temps.append(row["temp"])
            descriptions.append(f"{row['city']} - {int(row['MONTH'])}/{int(row['YEAR'])}")

    # sends to graph from visualizer.py
    Visualizer.plot_clustered_data(temps, labels, descriptions, cluster_centers)



def detect_daily_anomalies():
    cleaned_data = cleanData()
    if cleaned_data is None:
        print("No data found.")
        return

    # passes cleaned data into anomaly function from algorithms.py
    anomalies = detect_anomalies(cleaned_data)

    # takes anomalies and prints them all out
    if anomalies:
        Visualizer.plot_anomaly_bars(anomalies)
        print("\nTallahassee Daily Temperature Anomalies:\n")
        for date, temp, diff, month_avg in anomalies:
            if diff > 0:
                direction = "hotter"
            else:
                direction = "colder"
            print(f"{date} → {temp:.2f}°F ({abs(diff):.2f}°F {direction} than monthly average of {month_avg:.2f}°F)")
    else:
        print("\nNo anomalies detected.")

def humidity_graph():
    cleaned_data = cleanData()
    X = cleaned_data[["temp", "dew", "precip", "windspeed"]].values
    y = cleaned_data["humidity"].values

    # makes sure all data is valid
    valid_indices = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X = X[valid_indices]
    y = y[valid_indices]
    cleaned_data = cleaned_data[valid_indices]

    # creates DATE column to use for the x-axis
    cleaned_data["DATE"] = pd.to_datetime(cleaned_data[["YEAR", "MONTH", "DAY"]])

    # trains model (same as humidity predictor function)
    model = CustomHumidityPredictor(learning_rate=1e-4, n_iterations=10000)
    model.fit(X, y)
    print("\nFinished training model")
    # gets predictions
    predicted = model.predict(X)

    # asks user what month they want to see predictions vs actual graph
    try:
        selected_month = int(input("Enter a month number (1-12) to visualize humidity: "))
        if not 1 <= selected_month <= 12:
            raise ValueError("Invalid month.")
    except ValueError:
        print("Invalid input. Please enter a number between 1 and 12.")
        return

    # gets selected month
    month_mask = cleaned_data["MONTH"] == selected_month
    actual = y[month_mask]
    predicted_for_month = predicted[month_mask]
    dates = cleaned_data.loc[month_mask, "DATE"].dt.strftime("%m-%d-%Y").tolist()

    # sends data to visualizer.py function to graph
    if len(actual) == 0:
        print(f"No data available for month {selected_month}.")
    else:
        Visualizer.plot_humidity_predictions(actual, predicted_for_month, x_labels=dates)


def show_menu():
    print("\nClimate Analyzer\n"
          "----------------\n"
          "1) Predict Humidity in Tallahassee, Fl\n"
          "2) Predict Average Monthly Temperature in Tallahassee, Fl\n"
          "3) Cluster Monthly Temperatures from Tallahasssee, Chicago & NYC\n"
          "4) Detect Temperature Anomalies in Tallahassee, Fl\n"
          "5) Graph Predicted Humidity vs Real Humidity in Tallahassee, Fl\n"
          "exit To exit program")

if __name__ == "__main__":
    running = True
    while running:
        show_menu()
        selection = input("> ")
        if selection == "1":
            predict_humidity()
        elif selection == "2":
            predict_temperature()
        elif selection == "3":
            cluster_temperatures()
        elif selection == "4":
            detect_daily_anomalies()
        elif selection == "5":
            humidity_graph()
        elif selection == "exit" or selection == "EXIT":
            print("Goodbye...")
            running = False
        else:
            print("\nInvalid selection.\n")

