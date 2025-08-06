import unittest
from datetime import datetime, timedelta
import pandas as pd
import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from extract.i3_extractor import I3VolumenesExtractor
from transform.i3_transform import TransformadorI3
from load.data_lake_loader import DataLakeLoader

class TestI3Pipeline(unittest.TestCase):
    TEST_DATES = [
        (datetime.now() - timedelta(days=93)).strftime('%Y-%m-%d'),
    ]

    def test_full_etl(self):
        for test_date in self.TEST_DATES:
            with self.subTest(phase="Extraction", test_date=test_date):
                extractor = I3VolumenesExtractor()
                extract_result = extractor.extract_data_for_all_markets(fecha_inicio=test_date, fecha_fin=test_date, mercados_lst=None)
                self.assertIsInstance(extract_result, dict)
                self.assertIn("success", extract_result)
                self.assertIn("details", extract_result)
                details = extract_result["details"]
                self.assertIn("markets_downloaded", details)
                self.assertIn("markets_failed", details)
                self.assertIn("date_range", details)
                self.assertFalse(details["markets_failed"], f"Extraction had failures: {details['markets_failed']}")
                self.assertTrue(extract_result['success'], f"I3 volumenes extraction failed for {test_date}")

            with self.subTest(phase="Transformation", test_date=test_date):
                transformer = TransformadorI3()
                transform_result = transformer.transform_data_for_all_markets(
                    fecha_inicio=test_date, fecha_fin=test_date, mercados_lst=None)
                self.assertIsInstance(transform_result, dict)
                self.assertIn("data", transform_result)
                self.assertIn("status", transform_result)
                self.assertTrue(transform_result['status']['success'], f"I3 transformation failed for {test_date}")
                self.assertIn('diario', transform_result['data'])
                self.assertIn('intra', transform_result['data'])

                # secundaria is not available after SRS change date
                srs_change_date = datetime(2024, 11, 20)
                test_date_dt = datetime.strptime(test_date, "%Y-%m-%d")
                if test_date_dt < srs_change_date:
                    self.assertIn('secundaria', transform_result['data'])
                    self.assertFalse(transform_result['data']['secundaria'].empty, f"I3 transformation returned empty data for {test_date}")
                else:
                    self.assertTrue(transform_result['data']['secundaria'].empty, f"No secundaria data for {test_date}")

                self.assertIn('terciaria', transform_result['data'])
                self.assertIn('rr', transform_result['data'])
                self.assertIn('p48', transform_result['data'])
                self.assertIn('indisponibilidades', transform_result['data'])
                self.assertIn('restricciones_md', transform_result['data'])
                self.assertIn('restricciones_tr', transform_result['data'])
                self.assertIn('desvios', transform_result['data'])
                self.assertFalse(transform_result['data']['diario'].empty, f"I3 transformation returned empty data for {test_date}")
                self.assertFalse(transform_result['data']['intra'].empty, f"I3 transformation returned empty data for {test_date}")
                self.assertFalse(transform_result['data']['terciaria'].empty, f"I3 transformation returned empty data for {test_date}")
                self.assertFalse(transform_result['data']['rr'].empty, f"I3 transformation returned empty data for {test_date}")
                self.assertFalse(transform_result['data']['p48'].empty, f"I3 transformation returned empty data for {test_date}")
                self.assertFalse(transform_result['data']['indisponibilidades'].empty, f"I3 transformation returned empty data for {test_date}")
                self.assertFalse(transform_result['data']['restricciones_md'].empty, f"I3 transformation returned empty data for {test_date}")
                self.assertFalse(transform_result['data']['restricciones_tr'].empty, f"I3 transformation returned empty data for {test_date}")
                self.assertFalse(transform_result['data']['desvios'].empty, f"I3 transformation returned empty data for {test_date}")

            with self.subTest(phase="Load", test_date=test_date):
                loader = DataLakeLoader()
                load_result = loader.load_transformed_data_volumenes_i3(transform_result)
                self.assertIsInstance(load_result, dict)
                self.assertIn("success", load_result)
                self.assertTrue(load_result['success'], f"I3 volumenes load failed for {test_date}")


if __name__ == "__main__":
    unittest.main()
