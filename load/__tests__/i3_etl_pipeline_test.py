import unittest
from datetime import datetime, timedelta
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from extract.i3_extractor import I3VolumenesExtractor
from transform.i3_transform import TransformadorI3
from transform.curtailment_transform import TransformadorCurtailment
from load.data_lake_loader import DataLakeLoader

class TestI3Pipeline(unittest.TestCase):
    TEST_DATES = [
        (datetime.now() - timedelta(days=4)).strftime('%Y-%m-%d'),  # Recent date (I3 has ~4 day lag)
        '2024-10-27',  # End of DST (fall back to standard time)
    ]

    def test_full_etl(self):
        for test_date in self.TEST_DATES:
            with self.subTest(phase="Extraction Volumenes", test_date=test_date):
                extractor = I3VolumenesExtractor()
                extract_result = extractor.extract_data_for_all_markets(fecha_inicio=test_date, fecha_fin=test_date)
                self.assertIsInstance(extract_result, dict)
                self.assertIn("success", extract_result)
                self.assertIn("details", extract_result)
                details = extract_result["details"]
                self.assertIn("markets_downloaded", details)
                self.assertIn("markets_failed", details)
                self.assertIn("date_range", details)
                self.assertFalse(details["markets_failed"], f"Extraction had failures: {details['markets_failed']}")
                self.assertTrue(extract_result['success'], f"I3 volumenes extraction failed for {test_date}")

            with self.subTest(phase="Transformation Volumenes", test_date=test_date):
                transformer = TransformadorI3()
                transform_result_volumenes = transformer.transform_data_for_all_markets(
                    fecha_inicio=test_date, fecha_fin=test_date
                )
                self.assertIsInstance(transform_result_volumenes, dict)
                self.assertIn("data", transform_result_volumenes)
                self.assertIn("status", transform_result_volumenes)
                status = transform_result_volumenes["status"]
                self.assertIn("success", status)
                self.assertIn("details", status)
                details = status["details"]
                self.assertIn("markets_processed", details)
                self.assertIn("markets_failed", details)
                self.assertIn("mode", details)
                self.assertIn("date_range", details)
                self.assertFalse(details["markets_failed"], f"Volumenes transformation had failures: {details['markets_failed']}")
                self.assertTrue(status['success'], f"I3 volumenes transformation failed for {test_date}")

            with self.subTest(phase="Load Volumenes", test_date=test_date):
                loader = DataLakeLoader()
                load_result_volumenes = loader.load_transformed_data_volumenes_i3(transform_result_volumenes)
                self.assertIsInstance(load_result_volumenes, dict)
                self.assertIn("success", load_result_volumenes)
                self.assertTrue(load_result_volumenes['success'], f"I3 volumenes load failed for {test_date}")


if __name__ == "__main__":
    unittest.main()
