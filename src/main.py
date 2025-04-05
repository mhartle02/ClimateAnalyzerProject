from data_processor import DataProcessor
from algorithms import CustomHumidityPredictor
import pandas as pd
import numpy as np

def main():
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

    else:
        print("ERROR LOADING DATA")

if __name__ == "__main__":
    main()
