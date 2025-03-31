import numpy as np
import pandas as pd

class DataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        self.data = pd.read_json(self.file_path, lines=True)
        return self.data

    def clean_data(self):
        self.data.replace("", np.nan, inplace=True)

        #checks if ALL data within any columns have no data or are zero & deletes them
        self.data = self.data.loc[:, ~self.data.isin([0, np.nan]).all()]

        #splits "DATE" into separate "YEAR" and "MONTH" to make it easier to search
        if "DATE" in self.data:
            self.data["YEAR"] = pd.to_datetime(self.data["DATE"], format='%Y-%m').dt.year
            self.data["MONTH"] = pd.to_datetime(self.data["DATE"], format='%Y-%m').dt.month

        return self.data