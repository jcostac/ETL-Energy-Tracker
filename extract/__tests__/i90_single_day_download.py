#Add the project root to Python path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from extract.i90_extractor import I90Extractor, I90VolumenesExtractor, I90PreciosExtractor
import time

def volumenes_download_single_day():
    """
    Test the download of the volumenes for a single day for all markets
    """
    extractor = I90VolumenesExtractor()
    result = extractor.extract_data_for_all_markets(fecha_inicio_carga="2025-03-01", fecha_fin_carga="2025-03-01")
    assert result["success"] == True, f"Single day volumenes download failed. Details: {result.get('details', {})}"

def precios_download_single_day():
    """
    Test the download of the precios for a single day for all markets
    """
    extractor = I90PreciosExtractor()
    result = extractor.extract_data_for_all_markets(fecha_inicio_carga="2025-03-01", fecha_fin_carga="2025-03-01")
    assert result["success"] == True, f"Single day precios download failed. Details: {result.get('details', {})}"

if __name__ == "__main__":
    volumenes_download_single_day()
    time.sleep(10)
    precios_download_single_day()
    print("🎉 All tests passed successfully!")

