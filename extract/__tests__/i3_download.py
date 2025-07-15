#Add the project root to Python path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from extract.i3_extractor import I3VolumenesExtractor
import time

def test_i3_download_multiple_days():
    """
    Demonstrates how to use the I3VolumenesExtractor class to extract volume data for a specified date range and market.
    """
    try:

        i3_volumenes_extractor = I3VolumenesExtractor()
        results = i3_volumenes_extractor.extract_data_for_all_markets(fecha_inicio_carga="2024-12-01", fecha_fin_carga="2024-12-02", mercados_lst=["intra"])
        assert results["success"] == True, "i3 multiple days download failed"

    except Exception as e:
        print(f"Error: {e}")
        raise e


def test_i3_download_latest_day():
    """
    Demonstrates how to use the I3VolumenesExtractor class to extract volume data for a specified date range.
    """
    try:
        i3_volumenes_extractor = I3VolumenesExtractor()
        results = i3_volumenes_extractor.extract_data_for_all_markets(fecha_inicio_carga=None, fecha_fin_carga=None)
        assert results["success"] == True, "i3 latest day download failed"
        
    except Exception as e:
        print(f"Error: {e}")
        raise e

if __name__ == "__main__":
    test_i3_download_latest_day()
    time.sleep(5)
    test_i3_download_multiple_days() 

