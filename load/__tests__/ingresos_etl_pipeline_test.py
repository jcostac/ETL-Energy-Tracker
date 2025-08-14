import unittest
from datetime import datetime, timedelta
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from transform.ingresos_transform import TransformadorIngresos
from load.__tests__.esios_etl_pipeline_test import TestEsiosPipeline
from load.__tests__.i90_etl_pipeline_test import TestI90Pipeline
from load.__tests__.omie_etl_pipeline_test import TestOMIEPipeline
from configs.ingresos_config import IngresosConfig

class TestIngresosPipeline(unittest.TestCase):
    TEST_DATES = [
        (datetime.now() - timedelta(days=120)).strftime('%Y-%m-%d'),
    ]

    @classmethod
    def setUpClass(cls):
        """Run prerequisite ETL pipelines before running the ingresos tests."""
        print("\n--- Setting up prerequisite data for Ingresos tests ---")
        
        # Run ESIOS pipeline
        esios_tester = TestEsiosPipeline()
        esios_tester.test_full_etl()
        print("\n--- ESIOS prerequisite data generated ---")

        # Run I90 pipeline
        i90_tester = TestI90Pipeline()
        i90_tester.test_full_etl()
        print("\n--- I90 prerequisite data generated ---")


    def test_ingresos_calculation(self):
        for test_date in self.TEST_DATES:
            with self.subTest(test_date=test_date):
                transformer = TransformadorIngresos()
                
                # Get all valid markets from the config
                config = IngresosConfig()
                all_markets = list(config.mercado_name_id_map.keys())

                transform_result = transformer.calculate_ingresos_for_all_markets(
                    fecha_inicio=test_date, 
                    fecha_fin=test_date,
                    mercados_lst=all_markets
                )

                self.assertIsInstance(transform_result, dict)
                self.assertIn("data", transform_result)
                self.assertIn("status", transform_result)

                status = transform_result["status"]
                self.assertTrue(status['success'], f"Ingresos calculation failed for {test_date}")

                details = status["details"]
                self.assertFalse(details["markets_failed"], f"Ingresos calculation had failures: {details['markets_failed']}")

                # Verify the output for each market
                for market, df in transform_result["data"].items():
                    if df is not None:
                        self.assertIsInstance(df, pd.DataFrame, f"Transformed data for {market} is not a DataFrame")
                        if not df.empty:
                            expected_columns = {'datetime_utc', 'up', 'ingresos', 'id_mercado'}
                            self.assertTrue(expected_columns.issubset(df.columns), f"Missing columns in {market} DF: {expected_columns - set(df.columns)}")

if __name__ == "__main__":
    unittest.main()
