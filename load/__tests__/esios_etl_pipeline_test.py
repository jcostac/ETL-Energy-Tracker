import unittest
from datetime import datetime, timedelta
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from extract.esios_extractor import ESIOSPreciosExtractor
from transform.esios_transform import TransformadorESIOS
from load.local_data_lake_loader import LocalDataLakeLoader

class TestEsiosPipeline(unittest.TestCase):
    TEST_DATES = [
        (datetime.now() - timedelta(days=93)).strftime('%Y-%m-%d'),  # Recent date
        '2024-03-31',  # DST change date in Spain
    ]

    def test_full_etl(self):
        for test_date in self.TEST_DATES:
            with self.subTest(phase="Extraction", test_date=test_date):
                extractor = ESIOSPreciosExtractor()
                extract_result = extractor.extract_data_for_all_markets(fecha_inicio=test_date, fecha_fin=test_date)
                self.assertIsInstance(extract_result, dict)
                self.assertIn("success", extract_result)
                self.assertIn("details", extract_result)
                details = extract_result["details"]
                self.assertIn("markets_downloaded", details)
                self.assertIn("markets_failed", details)
                self.assertIn("date_range", details)
                self.assertFalse(details["markets_failed"], f"Extraction had failures: {details['markets_failed']}")
                self.assertTrue(extract_result['success'], f"ESIOS extraction failed for {test_date}")

            with self.subTest(phase="Transformation", test_date=test_date):
                transformer = TransformadorESIOS()
                transform_result = transformer.transform_data_for_all_markets(fecha_inicio=test_date, fecha_fin=test_date)
                self.assertIsInstance(transform_result, dict)
                self.assertIn("data", transform_result)
                self.assertIn("status", transform_result)
                status = transform_result["status"]
                self.assertIn("success", status)
                self.assertIn("details", status)
                details = status["details"]
                self.assertIn("markets_processed", details)
                self.assertIn("markets_failed", details)
                self.assertIn("mode", details)
                self.assertIn("date_range", details)
                self.assertFalse(details["markets_failed"], f"Transformation had failures: {details['markets_failed']}")
                self.assertTrue(status['success'], f"ESIOS transformation failed for {test_date}")
                for market, df in transform_result["data"].items():
                    self.assertIsInstance(df, pd.DataFrame, f"Transformed data for {market} is not a DataFrame")
                    if not df.empty:
                        expected_columns = {'datetime_utc', 'precio', 'id_mercado'}
                        self.assertTrue(expected_columns.issubset(df.columns), f"Missing columns in {market} DF: {expected_columns - set(df.columns)}")

            with self.subTest(phase="Loading", test_date=test_date):
                loader = LocalDataLakeLoader()
                load_result = loader.load_transformed_data_esios(transform_result)
                self.assertIsInstance(load_result, dict)
                self.assertIn("success", load_result)
                self.assertTrue(load_result['success'], f"ESIOS load failed for {test_date}")

if __name__ == "__main__":
    unittest.main()

