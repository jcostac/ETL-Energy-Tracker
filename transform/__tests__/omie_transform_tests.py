#Add the project root to Python path
import sys
import os
from datetime import datetime, timedelta
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import unittest
from transform.omie_transform import TransformadorOMIE

class TestOMIETransform(unittest.TestCase):

    TEST_DATES = [
        (datetime.now() - timedelta(days=93)).strftime('%Y-%m-%d'),
        "2024-03-31",  # Start of DST (summer time)
        "2024-10-27",  # End of DST (fall back to standard time)
    ]

    def test_omie_transform_single_day(self):
        """
        Tests the OMIE data transformation process for a single day across all markets.
        
        Asserts that the transformation completes successfully and saves the resulting data to a CSV file for further inspection.
        """
        for test_date in self.TEST_DATES:
            transformer = TransformadorOMIE()
            result = transformer.transform_data_for_all_markets(fecha_inicio=test_date, fecha_fin=test_date)
            result_status = result["status"]
            self.assertTrue(result_status["success"], f"Single day transformation failed. Details: {result_status.get('details', {})}")

    def test_omie_transform_multiple_days(self):
        """
        Test the OMIE data transformation process for multiple consecutive days across all markets.
        
        Asserts that the transformation completes successfully for the specified date range.
        """
        for test_date in self.TEST_DATES:
            transformer = TransformadorOMIE()
            result = transformer.transform_data_for_all_markets(fecha_inicio=test_date, fecha_fin=test_date)
            self.assertTrue(result["success"], f"Multiple days transformation failed. Details: {result.get('details', {})}")

    def test_omie_transform_market_specific(self):
        """
        Test the OMIE data transformation for a specific subset of markets on a single day.
        
        Runs the transformation for the markets "intradiario" and "continuo" on "2025-01-01" and asserts that the process completes successfully.
        """
        for test_date in self.TEST_DATES:
            transformer = TransformadorOMIE()
            result = transformer.transform_data_for_all_markets(fecha_inicio=test_date, fecha_fin=test_date, mercados_lst=["intradiario", "continuo"])
            self.assertTrue(result["success"], f"Market specific transformation failed. Details: {result.get('details', {})}")

if __name__ == "__main__":
    unittest.main()
