import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import unittest
from transform.ingresos_transform import TransformadorIngresos
from datetime import datetime, timedelta

class TestIngresosTransform(unittest.TestCase):

    TEST_DATES = [
        (datetime.now() - timedelta(days=93)).strftime('%Y-%m-%d'),
        "2024-03-31"
    ]

    def test_single_day_calculation(self):
        """
        Test the income calculation for a single day for all markets.
        This test ensures that the transformation completes successfully for specific dates,
        including those around daylight saving time changes.
        """
        for test_date in self.TEST_DATES:
            with self.subTest(date=test_date):
                transformer = TransformadorIngresos()
                result = transformer.calculate_ingresos_for_all_markets(
                    fecha_inicio=test_date, fecha_fin=test_date
                )
                result_status = result["status"]
                self.assertTrue(result_status["success"], f"Single day calculation for 'continuo' failed on {test_date}. Details: {result_status.get('details', {})}")

    def test_latest_day_calculation(self):
        """
        Test the income calculation for the latest available data for the 'diario' market.
        This verifies that the system can process recent data without explicit date inputs.
        """
        transformer = TransformadorIngresos()
        result = transformer.calculate_ingresos_for_all_markets(mercados_lst=["diario"])
        result_status = result["status"]
        self.assertTrue(result_status["success"], f"Latest day calculation for 'diario' failed. Details: {result_status.get('details', {})}")

    def test_market_specific_calculation(self):
        """
        Test the income calculation for a single day for the 'restricciones' market.
        This ensures that market-specific logic is correctly applied during transformation.
        """
        for test_date in self.TEST_DATES:
            with self.subTest(date=test_date):
                transformer = TransformadorIngresos()
                result = transformer.calculate_ingresos_for_all_markets(
                    fecha_inicio=test_date, fecha_fin=test_date, mercados_lst=["restricciones"]
                )
                result_status = result["status"]
                self.assertTrue(result_status["success"], f"Market-specific calculation for 'restricciones' failed on {test_date}. Details: {result_status.get('details', {})}")

    def test_multiple_markets_calculation(self):
        """
        Test the income calculation for a single day across multiple markets.
        This test validates the system's ability to handle combined transformations for
        'diario' and 'intra' markets.
        """
        for test_date in self.TEST_DATES:
            with self.subTest(date=test_date):
                transformer = TransformadorIngresos()
                result = transformer.calculate_ingresos_for_all_markets(
                    fecha_inicio=test_date, fecha_fin=test_date, mercados_lst=["diario", "intra"]
                )
                result_status = result["status"]
                self.assertTrue(result_status["success"], f"Multiple markets calculation failed on {test_date}. Details: {result_status.get('details', {})}")

    def test_multiple_days_and_markets_calculation(self):
        """
        Test the income calculation over a range of dates and for multiple markets.
        This ensures the transformation is successful for a 3-day period for the 'continuo' market.
        """
        for test_date in self.TEST_DATES:
            with self.subTest(date=test_date):
                start_date = (datetime.strptime(test_date, '%Y-%m-%d') - timedelta(days=94)).strftime('%Y-%m-%d')
                transformer = TransformadorIngresos()
                result = transformer.calculate_ingresos_for_all_markets(
                    fecha_inicio=start_date, fecha_fin=test_date, mercados_lst=["continuo", "secundaria", "terciaria"]
                )
                result_status = result["status"]
                self.assertTrue(result_status["success"], f"Multiple days and markets calculation failed for range {start_date} to {test_date}. Details: {result_status.get('details', {})}")

if __name__ == "__main__":
    unittest.main()