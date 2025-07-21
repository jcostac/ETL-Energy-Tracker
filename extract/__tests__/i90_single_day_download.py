#Add the project root to Python path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from extract.i90_extractor import I90Extractor, I90VolumenesExtractor, I90PreciosExtractor
import time

def volumenes_download_single_day():
    """
    Tests downloading volume data for all markets for a single day using the I90VolumenesExtractor.
    
    Raises an AssertionError if the extraction is unsuccessful.
    """
    extractor = I90VolumenesExtractor()
    result = extractor.extract_data_for_all_markets(fecha_inicio="2025-03-01", fecha_fin="2025-03-01")
    assert result["success"] == True, f"Single day volumenes download failed. Details: {result.get('details', {})}"

def precios_download_single_day():
    """
    Tests downloading price data for all markets for a single day using the I90PreciosExtractor.
    
    Asserts that the extraction is successful and raises an error with details if it fails.
    """
    extractor = I90PreciosExtractor()
    result = extractor.extract_data_for_all_markets(fecha_inicio="2025-03-01", fecha_fin="2025-03-01")
    assert result["success"] == True, f"Single day precios download failed. Details: {result.get('details', {})}"

if __name__ == "__main__":
    volumenes_download_single_day()
    time.sleep(5)
    precios_download_single_day()
    print("🎉 All tests passed successfully!")

