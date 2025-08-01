import unittest
from datetime import datetime, timedelta
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from extract.i90_extractor import I90PreciosExtractor, I90VolumenesExtractor
from transform.i90_transform import TransformadorI90
from transform.curtailment_transform import TransformadorCurtailment
from load.data_lake_loader import DataLakeLoader

class TestI90Pipeline(unittest.TestCase):
    TEST_DATES = [
        "2024-01-01",
        "2024-11-01",
        "2025-01-01",
        "2025-03-31",
        "2025-04-01"
    ]

    def test_full_etl(self):
        for test_date in self.TEST_DATES:
            with self.subTest(phase="Extraction Precios", test_date=test_date):
                extractor = I90PreciosExtractor()
                extract_result = extractor.extract_data_for_all_markets(fecha_inicio=test_date, fecha_fin=test_date, mercados_lst=["terciaria"])
                self.assertIsInstance(extract_result, dict)
                self.assertIn("success", extract_result)
                self.assertIn("details", extract_result)
                details = extract_result["details"]
                self.assertIn("markets_downloaded", details)
                self.assertIn("markets_failed", details)
                self.assertIn("date_range", details)
                self.assertFalse(details["markets_failed"], f"Extraction had failures: {details['markets_failed']}")
                self.assertTrue(extract_result['success'], f"I90 precios extraction failed for {test_date}")

            with self.subTest(phase="Transformation Precios", test_date=test_date):
                # Precios transformation
                transformer = TransformadorI90()
                transform_result_precios = transformer.transform_data_for_all_markets(
                    fecha_inicio=test_date, fecha_fin=test_date, dataset_type='precios_i90'
                )
                self.assertIsInstance(transform_result_precios, dict)
                self.assertIn("data", transform_result_precios)
                self.assertIn("status", transform_result_precios)
                status = transform_result_precios["status"]
                self.assertIn("success", status)
                self.assertIn("details", status)
                details = status["details"]
                self.assertIn("markets_processed", details)
                self.assertIn("markets_failed", details)
                self.assertIn("mode", details)
                self.assertIn("date_range", details)
                self.assertFalse(details["markets_failed"], f"Precios transformation had failures: {details['markets_failed']}")
                self.assertTrue(status['success'], f"I90 precios transformation failed for {test_date}")


            with self.subTest(phase="Load Precios", test_date=test_date):
                loader = DataLakeLoader()
                # Load precios
                load_result_precios = loader.load_transformed_data_precios_i90(transform_result_precios)
                self.assertIsInstance(load_result_precios, dict)
                self.assertIn("success", load_result_precios)
                self.assertTrue(load_result_precios['success'], f"I90 precios load failed for {test_date}")

        
            with self.subTest(phase="Extraction Volumenes", test_date=test_date):
                extractor = I90VolumenesExtractor()
                extract_result = extractor.extract_data_for_all_markets(fecha_inicio=test_date, fecha_fin=test_date, mercados_lst=["terciaria"])
                self.assertIsInstance(extract_result, dict)
                self.assertIn("success", extract_result)
                self.assertIn("details", extract_result)
                details = extract_result["details"]
                self.assertIn("markets_downloaded", details)
                self.assertIn("markets_failed", details)
                self.assertIn("date_range", details)
                self.assertFalse(details["markets_failed"], f"Extraction had failures: {details['markets_failed']}")
                self.assertTrue(extract_result['success'], f"I90 volumenes extraction failed for {test_date}")


            with self.subTest(phase="Transformation Volumenes", test_date=test_date):
                # Volumenes transformation
                transform_result_volumenes = transformer.transform_data_for_all_markets(
                    fecha_inicio=test_date, fecha_fin=test_date, dataset_type='volumenes_i90', mercados_lst=["terciaria"]
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
                self.assertTrue(status['success'], f"I90 volumenes transformation failed for {test_date}")

            with self.subTest(phase="Load Volumenes", test_date=test_date):
                # Load volumenes
                load_result_volumenes = loader.load_transformed_data_volumenes_i90(transform_result_volumenes)
                self.assertIsInstance(load_result_volumenes, dict)
                self.assertIn("success", load_result_volumenes)
                self.assertTrue(load_result_volumenes['success'], f"I90 volumenes load failed for {test_date}")
                
if __name__ == "__main__":
    unittest.main()
