#Add the project root to Python path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from extract.i90_extractor import I90PreciosExtractor, I90VolumenesExtractor
import time

def precios_download_multiple_days():
    """
    Tests downloading "precios" data for all markets over a specified multi-day range using the I90PreciosExtractor.
    
    Raises an assertion error if the extraction is not successful, including details from the result.
    """
    extractor = I90PreciosExtractor()
    result = extractor.extract_data_for_all_markets(fecha_inicio_carga="2025-03-02", fecha_fin_carga="2025-03-08")

    assert result["success"] == True, f"Multiple days precios download failed. Details: {result.get('details', {})}"


def volumenes_download_multiple_days():
    """
    Tests downloading volume data for all markets over a specified multi-day range using the I90VolumenesExtractor.
    
    Raises an assertion error if the extraction is not successful, including details from the result.
    """
    extractor = I90VolumenesExtractor()
    result = extractor.extract_data_for_all_markets(fecha_inicio_carga="2025-03-02", fecha_fin_carga="2025-03-08")

    assert result["success"] == True, f"Multiple days volumenes download failed. Details: {result.get('details', {})}"


if __name__ == "__main__":
    volumenes_download_multiple_days()
    time.sleep(5)
    precios_download_multiple_days()
    print("🎉 All tests passed successfully!")

