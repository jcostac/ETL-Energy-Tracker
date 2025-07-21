#Add the project root to Python path
import sys
import os
from datetime import datetime, timedelta

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import unittest
from extract.i3_extractor import I3VolumenesExtractor

class TestI3Extract(unittest.TestCase):

    TEST_DATES = [
        (datetime.now() - timedelta(days=93)).strftime('%Y-%m-%d')
    ]

    def test_i3_download_multiple_days(self):
        """
        Demonstrates how to use the I3VolumenesExtractor class to extract volume data for a specified date range and market.
        """
        for test_date in self.TEST_DATES:
            i3_volumenes_extractor = I3VolumenesExtractor()
            results = i3_volumenes_extractor.extract_data_for_all_markets(fecha_inicio=test_date, fecha_fin=test_date, mercados_lst=["intra", "diario"])
            self.assertTrue(results["success"], "i3 multiple days download failed")

    def test_i3_download_latest_day(self):
        """
        Demonstrates how to use the I3VolumenesExtractor class to extract volume data for a specified date range.
        """
        for test_date in self.TEST_DATES:
            i3_volumenes_extractor = I3VolumenesExtractor()
            results = i3_volumenes_extractor.extract_data_for_all_markets(fecha_inicio=test_date, fecha_fin=test_date)
            self.assertTrue(results["success"], "i3 latest day download failed")

if __name__ == "__main__":
    unittest.main() 

