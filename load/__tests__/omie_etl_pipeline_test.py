import unittest
from datetime import datetime, timedelta
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from extract.omie_extractor import OMIEExtractor
from transform.omie_transform import TransformadorOMIE
from load.data_lake_loader import DataLakeLoader

class TestOMIEPipeline(unittest.TestCase):
    TEST_DATES = [
        (datetime.now() - timedelta(days=93)).strftime('%Y-%m-%d'),  # Recent date
        '2024-03-31',  # DST change date in Spain
    ]

    def test_full_etl(self):
        for test_date in self.TEST_DATES:
            with self.subTest(phase="Extraction", test_date=test_date):
                    extractor = OMIEExtractor()
                    extract_result = extractor.extract_data_for_all_markets(fecha_inicio=test_date, fecha_fin=test_date)
                    self.assertIsInstance(extract_result, dict)
                    self.assertIn("success", extract_result)
                    self.assertIn("details", extract_result)
                    details = extract_result["details"]
                    self.assertIn("markets_downloaded", details)
                    self.assertIn("markets_failed", details)
                    self.assertIn("date_range", details)
                    self.assertFalse(details["markets_failed"], f"Extraction had failures: {details['markets_failed']}")
                    self.assertTrue(extract_result['success'], f"OMIE extraction failed for {test_date}")


            with self.subTest(phase = "Tansform volumenes"):
                transformer = TransformadorOMIE()
                transform_result_volumenes = transformer.transform_data_for_all_markets(fecha_inicio=test_date, fecha_fin=test_date)
                self.assertIsInstance(transform_result_volumenes, dict)
                self.assertIn("status", transform_result_volumenes)
                self.assertTrue(transform_result_volumenes['status']['success'], f"OMIE transform failed for {test_date} for volumenes")

            with self.subTest(phase="Load Volumenes", test_date=test_date):
                loader = DataLakeLoader()
                load_result_volumenes = loader.load_transformed_data_volumenes_omie(transform_result_volumenes)
                self.assertIsInstance(load_result_volumenes, dict)
                self.assertIn("success", load_result_volumenes)
                self.assertTrue(load_result_volumenes['success'], f"OMIE load failed for {test_date} for volumenes")

if __name__ == "__main__":
    unittest.main()
