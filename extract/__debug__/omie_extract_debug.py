import sys
import os
from datetime import datetime, timedelta

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from extract.omie_extractor import OMIEExtractor

class OMIEExtractDebug:
    """
    A class for manually debugging the OMIE data extraction process.
    This class is intended to be run as a script for debugging purposes.
    """

    def __init__(self):
        self.TEST_DATES = [
            (datetime.now() - timedelta(days=93)).strftime('%Y-%m-%d')
        ]

    def _run_and_print_result(self, operation_name, result):
        """Prints the result of an operation."""
        if result["success"]:
            print(f"✅ {operation_name} successful.")
        else:
            print(f"❌ {operation_name} failed. Details: {result.get('details', {})}")

    def download_single_day(self, mercados_lst: list[str] = None):
        """
        Test OMIE data extraction for a single day.
        """
        for test_date in self.TEST_DATES:
            print(f"\n--- Running: Download OMIE for single day: {test_date} ---")
            extractor = OMIEExtractor()
            result = extractor.extract_data_for_all_markets(fecha_inicio=test_date, fecha_fin=test_date, mercados_lst=mercados_lst)
            self._run_and_print_result(f"Single day OMIE download on {test_date}", result)

    def download_market_specific(self, mercados_lst: list[str] = None):
        """
        Test OMIE data extraction for a specific market on a single day.
        """
        for test_date in self.TEST_DATES:
            print(f"\n--- Running: Download OMIE for markets {mercados_lst} on {test_date} ---")
            extractor = OMIEExtractor()
            result = extractor.extract_data_for_all_markets(fecha_inicio=test_date, fecha_fin=test_date, mercados_lst=mercados_lst)
            self._run_and_print_result(f"Market-specific OMIE download on {test_date}", result)
    
    def run_all(self):
        """Runs all debug methods."""
        self.download_single_day()
        self.download_market_specific()

if __name__ == "__main__":
    debugger = OMIEExtractDebug()

    # Call specific methods to debug, for example:
    # debugger.download_single_day()
    # debugger.download_market_specific(mercados_lst=["diario"])
    
    # Or run all of them:
    debugger.run_all()
