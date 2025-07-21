
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import unittest
from extract.i90_extractor import I90Extractor, I90VolumenesExtractor, I90PreciosExtractor
from datetime import datetime, timedelta

class TestI90Extract(unittest.TestCase):

    TEST_DATES = [
        (datetime.now() - timedelta(days=93)).strftime('%Y-%m-%d')
    ]

    def test_volumenes_download_single_day(self):
        """
        Tests downloading volume data for all markets for a single day using the I90VolumenesExtractor.
        
        Raises an AssertionError if the extraction is unsuccessful.
        """
        for test_date in self.TEST_DATES:
            extractor = I90VolumenesExtractor()
            result = extractor.extract_data_for_all_markets(fecha_inicio=test_date, fecha_fin=test_date)
            self.assertTrue(result["success"], f"Single day volumenes download failed. Details: {result.get('details', {})}")

    def test_precios_download_single_day(self):
        """
        Tests downloading price data for all markets for a single day using the I90PreciosExtractor.
        
        Asserts that the extraction is successful and raises an error with details if it fails.
        """
        for test_date in self.TEST_DATES:
            extractor = I90PreciosExtractor()
            result = extractor.extract_data_for_all_markets(fecha_inicio=test_date, fecha_fin=test_date)
            self.assertTrue(result["success"], f"Single day precios download failed. Details: {result.get('details', {})}")

    def test_precios_download_multiple_days(self):
        """
        Tests downloading "precios" data for all markets over a specified multi-day range using the I90PreciosExtractor.
        
        Raises an assertion error if the extraction is not successful, including details from the result.
        """
        for test_date in self.TEST_DATES:
            extractor = I90PreciosExtractor()
            result = extractor.extract_data_for_all_markets(fecha_inicio=test_date, fecha_fin=test_date)
            self.assertTrue(result["success"], f"Multiple days precios download failed. Details: {result.get('details', {})}")

    def test_volumenes_download_multiple_days(self):
        """
        Tests downloading volume data for all markets over a specified multi-day range using the I90VolumenesExtractor.
        
        Raises an assertion error if the extraction is not successful, including details from the result.
        """
        for test_date in self.TEST_DATES:
            extractor = I90VolumenesExtractor()
            result = extractor.extract_data_for_all_markets(fecha_inicio=test_date, fecha_fin=test_date)
            self.assertTrue(result["success"], f"Multiple days volumenes download failed. Details: {result.get('details', {})}")

    def test_volumenes_download_market_specific(self):
        """
        Tests downloading volume data for a specified list of markets on a given date using the I90VolumenesExtractor.
        
        Raises an AssertionError if the extraction is not successful, including error details.
        """
        for test_date in self.TEST_DATES:
            extractor = I90VolumenesExtractor()
            result = extractor.extract_data_for_all_markets(fecha_inicio=test_date, fecha_fin=test_date, mercados_lst=["terciaria", "rr", "curtailment", "p48", "indisponibilidades", "restricciones"])
            self.assertTrue(result["success"], f"Volumenes download failed. Details: {result.get('details', {})}")

    def test_precios_download_market_specific(self):
        """
        Tests downloading price data for the "restricciones" market on a specific date using I90PreciosExtractor.
        
        Raises an AssertionError if the extraction is not successful.
        """
        for test_date in self.TEST_DATES:
            extractor = I90PreciosExtractor()
            result = extractor.extract_data_for_all_markets(fecha_inicio=test_date, fecha_fin=test_date, mercados_lst=["restricciones"])
            self.assertTrue(result["success"], f"Precios download failed. Details: {result.get('details', {})}")

if __name__ == "__main__":
    unittest.main()

