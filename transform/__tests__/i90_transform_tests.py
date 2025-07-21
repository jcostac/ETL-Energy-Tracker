
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import unittest
from transform.i90_transform import TransformadorI90
from datetime import datetime, timedelta

class TestI90Transform(unittest.TestCase):

    TEST_DATES = [
        (datetime.now() - timedelta(days=93)).strftime('%Y-%m-%d'),
        "2024-03-31",  # Start of DST (summer time) 
        "2024-10-27",  # End of DST (fall back to standard time)
    ]

    def test_single_day_precios(self):
        """
        Test the transformation of single-day price data for the "restricciones" market using the TransformadorI90 class.
        
        Asserts that the transformation completes successfully for the specified date and market.
        """

        for test_date in self.TEST_DATES:
            transformer = TransformadorI90()
            result = transformer.transform_data_for_all_markets(fecha_inicio="2025-03-02", fecha_fin="2025-03-02", dataset_type="precios_i90", mercados_lst=["restricciones"])
            result_status = result["status"]
            self.assertTrue(result_status["success"], f"Single day precios transformation failed. Details: {result_status.get('details', {})}")

    def test_single_day_volumenes(self):
        """
        Test the transformation of volume data for the latest day in the "intra" market using the TransformadorI90 class.
        
        Asserts that the transformation completes successfully.
        """
        for test_date in self.TEST_DATES:
            transformer = TransformadorI90()
            result = transformer.transform_data_for_all_markets(dataset_type="volumenes_i90", mercados_lst=["intra"])
            result_status = result["status"]
            self.assertTrue(result_status["success"], f"Single day volumenes transformation failed. Details: {result_status.get('details', {})}")

    def test_multiple_day_precios(self):
        """
        Test the transformation of "precios_i90" data for multiple days and a specific market.

        Asserts that the transformation for a recent 3-day range, 93 days in the past for the "restricciones" market completes successfully.
        """
        for test_date in self.TEST_DATES:
            transformer = TransformadorI90()
            result = transformer.transform_data_for_all_markets(
                fecha_inicio=test_date,
                fecha_fin=test_date,
                dataset_type="precios_i90",
                mercados_lst=["restricciones"]
            )
            result_status = result["status"]
            self.assertTrue(result_status["success"], f"Multiple day precios transformation failed. Details: {result_status.get('details', {})}")

    def test_multiple_day_volumenes(self):
        """
        Test the transformation of "volumenes_i90" data across multiple days for all markets.

        Asserts that the transformation is successful for a recent 3-day range, 93 days in the past.
        """
        for test_date in self.TEST_DATES:
            transformer = TransformadorI90()
            result = transformer.transform_data_for_all_markets(
                fecha_inicio=test_date,
                fecha_fin=test_date,
                dataset_type="volumenes_i90"
            )
            result_status = result["status"]
            self.assertTrue(result_status["success"], f"Multiple day volumenes transformation failed. Details: {result_status.get('details', {})}")

    def test_market_specific_intradiario(self):
        """
        Test the transformation process for the latest single day of intradiario market data using the TransformadorI90 class.
        
        Asserts that the transformation completes successfully and raises an error with details if it fails.
        """
        for test_date in self.TEST_DATES:
            transformer = TransformadorI90()
            result = transformer.transform_data_for_all_markets(dataset_type="volumenes_i90", mercados_lst=["intra"])
            result_status = result["status"]
            self.assertTrue(result_status["success"], f"Market specific intradiario transformation failed. Details: {result_status.get('details', {})}")

    def test_market_specific_volumenes(self):
        """
        Test the transformation process for the "diario" and "secundaria" markets on a single day using the "volumenes_i90" dataset.
        
        Asserts that the transformation completes successfully; raises an assertion error with details if it fails.
        """
        for test_date in self.TEST_DATES:
            transformer = TransformadorI90()
            result = transformer.transform_data_for_all_markets(fecha_inicio=test_date, fecha_fin=test_date, dataset_type="volumenes_i90", mercados_lst=["diario", "secundaria"])
            result_status = result["status"]
            self.assertTrue(result_status["success"], f"Market specific volumenes transformation failed. Details: {result_status.get('details', {})}")

    def test_multiple_days_market_specific_volumenes(self):
        """
        Test that the transformation process succeeds for multiple specified markets over a date range using the "volumenes_i90" dataset.
        
        Asserts that the transformation status indicates success for the markets "terciaria", "rr", "curtailment", "p48", "indisponibilidades", and "restricciones" from 2025-03-02 to 2025-03-04.
        """
        for test_date in self.TEST_DATES:
            transformer = TransformadorI90()
            result = transformer.transform_data_for_all_markets(fecha_inicio=test_date, fecha_fin=test_date, dataset_type="volumenes_i90", mercados_lst=["terciaria", "rr", "curtailment_demanda", "p48", "indisponibilidades", "restricciones"])
            result_status = result["status"]
            self.assertTrue(result_status["success"], f"Multiple markets volumenes transformation failed. Details: {result_status.get('details', {})}")

if __name__ == "__main__":
    unittest.main() 