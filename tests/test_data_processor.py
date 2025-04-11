import unittest
import pandas as pd
import json
from unittest.mock import patch, mock_open
from src.data_processor import DataProcessor


class TestDataProcessor(unittest.TestCase):

    def setUp(self):
        """creates fake sample data to run the tests with"""
        self.sample_data = {
            "days": [
                {
                    "datetime": "2023-07-01",
                    "temp": 85.0,
                    "dew": 70.0,
                    "humidity": 75,
                    "precip": 0.2,
                    "windspeed": 10
                },
                {
                    "datetime": "2023-07-02",
                    "temp": 88.0,
                    "dew": 71.0,
                    "humidity": 80,
                    "precip": 0.0,
                    "windspeed": 12
                }
            ]
        }
        self.mock_path = "fake_file.json"
        print(f"\nRunning {self._testMethodName}")

    def tearDown(self):
        """prints when the test ends"""
        print(f"Finished running {self._testMethodName}")

    @patch("builtins.open", new_callable=mock_open)
    @patch("json.load")
    def test_load_data(self, mock_json, mock_file):
        """tests load_data() with fake data to make sure it all loaded in"""
        mock_json.return_value = self.sample_data
        processor = DataProcessor(self.mock_path)
        df = processor.load_data()

        self.assertEqual(len(df), 2)
        self.assertIn("temp", df.columns)

    def test_clean_data(self):
        """tests to make sure data gets cleaned (making sure year exists)"""
        processor = DataProcessor(self.mock_path)
        processor.data = pd.DataFrame(self.sample_data["days"])
        cleaned = processor.clean_data()

        self.assertIn("YEAR", cleaned.columns)
        self.assertEqual(len(cleaned), 2)

    def test_clean_data_invalid_rows(self):
        """tests to make sure bad data gets removed when cleaned"""
        bad_data = {
            "days": [
                {"datetime": "2023-07-01", "temp": None},
                {"datetime": "2023-07-02"}
            ]
        }
        processor = DataProcessor(self.mock_path)
        processor.data = pd.DataFrame(bad_data["days"])
        cleaned = processor.clean_data()

        self.assertEqual(len(cleaned), 0)

    def test_load_and_clean_with_city(self):
        """tests load/clean command that includes the city """
        with patch("builtins.open",
                   mock_open(read_data=json.dumps(self.sample_data))), \
             patch("json.load", return_value=self.sample_data):

            df = DataProcessor.load_and_clean_with_city(self.mock_path,
                                                        "Tallahassee")

            self.assertIn("city", df.columns)
            self.assertTrue((df["city"] == "Tallahassee").all())
            self.assertEqual(len(df), 2)


if __name__ == '__main__':
    unittest.main()
