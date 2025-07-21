#Add the project root to Python path
import sys
import os
from datetime import datetime, timedelta

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import unittest
from extract.omie_extractor import OMIEExtractor


class TestOMIEExtract(unittest.TestCase):

    TEST_DATES = [
        (datetime.now() - timedelta(days=93)).strftime('%Y-%m-%d')
    ]

    def test_omie_download_single_day(self):
        """
        Test OMIE data extraction for a single day across all markets.
        
        Asserts that extraction is successful and prints a confirmation message.
        """
        for test_date in self.TEST_DATES:
            extractor = OMIEExtractor()
            result = extractor.extract_data_for_all_markets(fecha_inicio=test_date, fecha_fin=test_date)
            self.assertTrue(result["success"], f"Single day download failed. Details: {result.get('details', {})}")

    def test_omie_download_multiple_days(self):
        """
        Test extraction of OMIE data for all markets over a specified date range.
        
        Asserts that data extraction from "2025-03-02" to "2025-03-08" completes successfully.
        """
        for test_date in self.TEST_DATES:
            extractor = OMIEExtractor()
            result = extractor.extract_data_for_all_markets(fecha_inicio=test_date, fecha_fin=test_date)
            self.assertTrue(result["success"], f"Multiple days download failed. Details: {result.get('details', {})}")

    def test_omie_download_market_specific(self):
        """
        Test OMIE data extraction for a specific market on a single day.
        
        Runs the extraction for the "continuo" market on "2025-01-01" and asserts that the operation succeeds.
        """
        for test_date in self.TEST_DATES:
            extractor = OMIEExtractor()
            result = extractor.extract_data_for_all_markets(fecha_inicio=test_date, fecha_fin=test_date, mercados_lst=["continuo"])
            self.assertTrue(result["success"], f"Market specific download failed. Details: {result.get('details', {})}")

if __name__ == "__main__":
    unittest.main()

