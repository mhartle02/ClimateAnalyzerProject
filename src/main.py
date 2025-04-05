from data_processor import DataProcessor
from algorithms import CustomHumidityPredictor, CustomTemperaturePredictor
import pandas as pd
import numpy as np
from time import *

def cleanData():
    file_path = "data/climate_data.json"

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




def show_menu():
    print("\nClimate Analyzer\n"
          "----------------\n"
          "1) Predict Humidity\n"
          "2) Predict Average Monthly Temperature")
if __name__ == "__main__":
    running = True
    while running:
        show_menu()
        selection = input(">")
        if selection == "1": predict_humidity()
        elif selection == "2":
            predict_temperature()
        elif selection == "exit":
            print("Goodbye...")
            running = False
        else:
            print("\nInvalid selection.\n")

