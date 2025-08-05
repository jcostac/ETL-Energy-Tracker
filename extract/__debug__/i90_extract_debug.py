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
            (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
        ]

    def _run_and_print_result(self, operation_name, result):
        """Prints the result of an operation."""
        if result["success"]:
            print(f"✅ {operation_name} successful.")
        else:
            print(f"❌ {operation_name} failed. Details: {result.get('details', {})}")

    def download_volumenes_single_day(self, fecha_inicio: str = None, fecha_fin: str = None, mercados_lst: list[str] = None):
        """
        Downloads volume data for all markets for a single day using the I90VolumenesExtractor.
        """
        extractor = I90VolumenesExtractor()

        if fecha_inicio is None and fecha_fin is None:
            for test_date in self.TEST_DATES:
                result = extractor.extract_data_for_all_markets(fecha_inicio=test_date, fecha_fin=test_date, mercados_lst=mercados_lst)
                self._run_and_print_result(f"Single day volumenes download on {test_date}", result)
        else:
            result = extractor.extract_data_for_all_markets(fecha_inicio=fecha_inicio, fecha_fin=fecha_fin, mercados_lst=mercados_lst)
            self._run_and_print_result(f"Single day volumenes download on {fecha_inicio}", result)

    def download_precios_single_day(self, fecha_inicio: str = None, fecha_fin: str = None, mercados_lst: list[str] = None):
        """
        Downloads price data for all markets for a single day using the I90PreciosExtractor.
        """
        extractor = I90PreciosExtractor()
        if fecha_inicio is None and fecha_fin is None:
            for test_date in self.TEST_DATES:
                result = extractor.extract_data_for_all_markets(fecha_inicio=test_date, fecha_fin=test_date, mercados_lst=mercados_lst)
                self._run_and_print_result(f"Single day precios download on {test_date}", result)
        else:
            result = extractor.extract_data_for_all_markets(fecha_inicio=fecha_inicio, fecha_fin=fecha_fin, mercados_lst=mercados_lst)
            self._run_and_print_result(f"Single day precios download on {fecha_inicio}", result)

    def run_all(self):
        self.download_volumenes_single_day(fecha_inicio=None, fecha_fin=None)
        self.download_precios_single_day(fecha_inicio=None, fecha_fin=None)


if __name__ == "__main__":
    debugger = I90ExtractDebug()
    
    # Call specific methods to debug, for example:
    debugger.download_volumenes_single_day(fecha_inicio=None, fecha_fin=None, mercados_lst=["restricciones_tr", "restricciones_md", "desvios"])
    debugger.download_precios_single_day(fecha_inicio=None, fecha_fin=None)

    # Or run all of them:
    # debugger.run_all()
