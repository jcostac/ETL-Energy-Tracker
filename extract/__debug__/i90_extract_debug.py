import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from extract.i90_extractor import I90VolumenesExtractor, I90PreciosExtractor
from datetime import datetime, timedelta

class I90ExtractDebug:
    """
    A class for manually debugging the I90 data extraction process.
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

    def download_volumenes_single_day(self, mercados_lst: list[str] = None):
        """
        Downloads volume data for all markets for a single day using the I90VolumenesExtractor.
        """
        for test_date in self.TEST_DATES:
            print(f"\n--- Running: Download volumenes for single day: {test_date} ---")
            extractor = I90VolumenesExtractor()
            result = extractor.extract_data_for_all_markets(fecha_inicio=test_date, fecha_fin=test_date)
            self._run_and_print_result(f"Single day volumenes download on {test_date}", result)

    def download_precios_single_day(self, mercados_lst: list[str] = None):
        """
        Downloads price data for all markets for a single day using the I90PreciosExtractor.
        """
        for test_date in self.TEST_DATES:
            print(f"\n--- Running: Download precios for single day: {test_date} ---")
            extractor = I90PreciosExtractor()
            result = extractor.extract_data_for_all_markets(fecha_inicio=test_date, fecha_fin=test_date)
            self._run_and_print_result(f"Single day precios download on {test_date}", result)

    def download_precios_multiple_days(self, mercados_lst: list[str] = None):
        """
        Downloads "precios" data for all markets over a specified multi-day range using the I90PreciosExtractor.
        """
        for test_date in self.TEST_DATES:
            print(f"\n--- Running: Download precios for multiple days (currently single day): {test_date} ---")
            extractor = I90PreciosExtractor()
            result = extractor.extract_data_for_all_markets(fecha_inicio=test_date, fecha_fin=test_date)
            self._run_and_print_result(f"Multiple days precios download on {test_date}", result)

    def download_volumenes_multiple_days(self, mercados_lst: list[str] = None):
        """
        Downloads volume data for all markets over a specified multi-day range using the I90VolumenesExtractor.
        """
        for test_date in self.TEST_DATES:
            print(f"\n--- Running: Download volumenes for multiple days (currently single day): {test_date} ---")
            extractor = I90VolumenesExtractor()
            result = extractor.extract_data_for_all_markets(fecha_inicio=test_date, fecha_fin=test_date)
            self._run_and_print_result(f"Multiple days volumenes download on {test_date}", result)

    def download_volumenes_market_specific(self, mercados_lst: list[str] = None):
        """
        Downloads volume data for a specified list of markets on a given date using the I90VolumenesExtractor.
        """
        for test_date in self.TEST_DATES:
            print(f"\n--- Running: Download volumenes for specific markets: {test_date} ---")
            extractor = I90VolumenesExtractor()
            mercados = ["terciaria", "rr", "curtailment", "p48", "indisponibilidades", "restricciones"]
            result = extractor.extract_data_for_all_markets(fecha_inicio=test_date, fecha_fin=test_date, mercados_lst=mercados)
            self._run_and_print_result(f"Market-specific volumenes download on {test_date}", result)

    def download_precios_market_specific(self):
        """
        Downloads price data for the "restricciones" market on a specific date using I90PreciosExtractor.
        """
        for test_date in self.TEST_DATES:
            print(f"\n--- Running: Download precios for specific markets: {test_date} ---")
            extractor = I90PreciosExtractor()
            mercados = ["restricciones"]
            result = extractor.extract_data_for_all_markets(fecha_inicio=test_date, fecha_fin=test_date, mercados_lst=mercados)
            self._run_and_print_result(f"Market-specific precios download on {test_date}", result)

    def run_all(self):
        """Runs all debug methods."""
        self.download_volumenes_single_day()
        self.download_precios_single_day()
        self.download_precios_multiple_days()
        self.download_volumenes_multiple_days()
        self.download_volumenes_market_specific()
        self.download_precios_market_specific()

if __name__ == "__main__":
    debugger = I90ExtractDebug()
    
    # Call specific methods to debug, for example:
    # debug_runner.download_volumenes_single_day()
    debugger.download_volumenes_single_day(mercados_lst=["restricciones_tr", "restricciones_md", "desvios"])
    
    # Or run all of them:
    # debugger.run_all()
