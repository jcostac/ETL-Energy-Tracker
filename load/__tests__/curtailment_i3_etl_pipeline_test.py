import unittest
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from extract.i3_extractor import I3VolumenesExtractor
from transform.curtailment_transform import TransformadorCurtailment
from load.local_data_lake_loader import LocalDataLakeLoader
from transform.i3_transform import TransformadorI3

class TestI3CurtailmentPipeline(unittest.TestCase):
    TEST_DATES = [
        (datetime.now() - timedelta(days=4)).strftime('%Y-%m-%d'),  # Recent date (I3 has ~4 day lag)
        '2024-10-27',  # End of DST (fall back to standard time)
    ]

    def test_full_etl(self):
        for test_date in self.TEST_DATES:
            with self.subTest(phase="Extraction Volumenes for Curtailment", test_date=test_date):
                extractor = I3VolumenesExtractor()
                extract_result = extractor.extract_data_for_all_markets(fecha_inicio=test_date, fecha_fin=test_date, mercados_lst=['restricciones', 'curtailment_demanda'])
                self.assertIsInstance(extract_result, dict)
                self.assertIn("success", extract_result)
                self.assertTrue(extract_result['success'], f"I3 volumes extraction for curtailment failed for {test_date}")

            with self.subTest(phase="Transformation Curtailment Generacion", test_date=test_date):
                curtailment_transformer = TransformadorCurtailment()
                transform_result_curtailment = curtailment_transformer.transform_curtailment_i3(
                    fecha_inicio=test_date, fecha_fin=test_date
                )
                self.assertIsInstance(transform_result_curtailment, dict)
                self.assertIn("status", transform_result_curtailment)
                status = transform_result_curtailment["status"]
                self.assertTrue(status['success'], f"I3 curtailment transformation failed for {test_date}")

            with self.subTest(phase="Transformation Curtailment Demanda", test_date=test_date):
                transformer = TransformadorI3()
                transform_result_volumenes = transformer.transform_data_for_all_markets(
                    fecha_inicio=test_date, fecha_fin=test_date, dataset_type='volumenes_i3', mercados_lst=["curtailment_demanda"]
                )
                self.assertIsInstance(transform_result_volumenes, dict)
                self.assertIn("data", transform_result_volumenes)
                self.assertIn("status", transform_result_volumenes)

            with self.subTest(phase="Load Curtailment Demanda", test_date=test_date):
                loader = LocalDataLakeLoader()
                load_result_volumenes = loader.load_transformed_data_volumenes_i3(transform_result_volumenes)
                self.assertIsInstance(load_result_volumenes, dict)
                self.assertIn("success", load_result_volumenes)
                self.assertTrue(load_result_volumenes['success'], f"I3 volumenes load failed for {test_date}")

            with self.subTest(phase="Load Curtailment Generacion", test_date=test_date):
                loader = LocalDataLakeLoader()
                load_result_curtailment = loader.load_transformed_data_curtailments_i3(transform_result_curtailment)
                self.assertIsInstance(load_result_curtailment, dict)
                self.assertIn("success", load_result_curtailment)
                self.assertTrue(load_result_curtailment['success'], f"I3 curtailment load failed for {test_date}")

if __name__ == "__main__":
    unittest.main() 