import sys
import os
from datetime import datetime, timedelta

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from extract.i3_extractor import I3VolumenesExtractor

class I3ExtractDebug:
    """
    A class for manually debugging the I3 data extraction process.
    This class is intended to be run as a script for debugging purposes.
    """


    TEST_DATES = [
        (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
    ]

    def _run_and_print_result(self, operation_name, result):
        """Prints the result of an operation."""
        if result["success"]:
            print(f"✅ {operation_name} successful.")
        else:
            print(f"❌ {operation_name} failed. Details: {result.get('details', {})}")


    def download_all_markets(self, fecha_inicio: str = None, fecha_fin: str = None, mercados_lst: list[str] = None):
        """
        Extracts volume data for a specified date range for all markets.
        """
        extractor = I3VolumenesExtractor()
        if fecha_inicio is None and fecha_fin is None:
            for test_date in self.TEST_DATES:
                print(f"\n--- Running: Download I3 for all markets on {test_date} ---")
                results = extractor.extract_data_for_all_markets(fecha_inicio=test_date, fecha_fin=test_date, mercados_lst=mercados_lst)
                self._run_and_print_result(f"I3 all-markets download on {test_date}", results)
        else:
            print(f"\n--- Running: Download I3 for all markets on {fecha_inicio} ---")
            results = extractor.extract_data_for_all_markets(fecha_inicio=fecha_inicio, fecha_fin=fecha_fin, mercados_lst=mercados_lst)
            self._run_and_print_result(f"I3 all-markets download on {test_date}", results)

if __name__ == "__main__":
    debugger = I3ExtractDebug()
    
    # Call specific methods to debug, for example:
    debugger.download_all_markets(fecha_inicio=None, fecha_fin=None, mercados_lst=["desvios"])
 
