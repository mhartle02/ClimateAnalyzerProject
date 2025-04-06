import numpy as np
import pandas as pd
import json

class DataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        try:
            with open(self.file_path, 'r') as f:
                raw_json = json.load(f)
                self.data = pd.DataFrame(raw_json.get("days", []))
            return self.data
        except (ValueError, FileNotFoundError) as e:
            print(f"Error loading JSON file: {e}")
            return None

    def clean_data(self):
        if self.data is None:
            print("Data hasn't been loaded in.")
            return None

        # changes datetime to separate day month and year
        if "datetime" in self.data.columns:
            self.data["datetime"] = pd.to_datetime(self.data["datetime"], errors='coerce')
            self.data["YEAR"] = self.data["datetime"].dt.year
            self.data["MONTH"] = self.data["datetime"].dt.month
            self.data["DAY"] = self.data["datetime"].dt.day
            self.data.drop(columns=["datetime"], inplace=True)

        # only the columns that make sense with what we need
        desired_columns = ["DAY", "MONTH", "YEAR", "temp", "dew", "humidity", "precip", "windspeed"]

        existing_columns = [col for col in desired_columns if col in self.data.columns]
        self.data = self.data[existing_columns]

        # getting rid of any rows that may contain null values (incase)
        self.data = self.data.dropna(subset=existing_columns)

        return self.data

    @staticmethod
    def load_and_clean_with_city(file_path, city_name):
        processor = DataProcessor(file_path)
        data = processor.load_data()
        if data is not None:
            cleaned = processor.clean_data()
            cleaned["city"] = city_name
            return cleaned
        return None
