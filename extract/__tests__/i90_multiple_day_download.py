#Add the project root to Python path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from extract.i90_extractor import I90PreciosExtractor, I90VolumenesExtractor
import time

def precios_download_multiple_days():
        """
        Test the download of the precios for multiple days for all markets
        """
        extractor = I90PreciosExtractor()
        result = extractor.extract_data_for_all_markets(fecha_inicio_carga="2025-03-02", fecha_fin_carga="2025-03-08")

        assert result["success"] == True, f"Multiple days precios download failed. Details: {result.get('details', {})}"


def volumenes_download_multiple_days():
    """
    Test the download of the volumenes for multiple days for all markets
    """
    extractor = I90VolumenesExtractor()
    result = extractor.extract_data_for_all_markets(fecha_inicio_carga="2025-03-02", fecha_fin_carga="2025-03-08")
    
    assert result["success"] == True, f"Multiple days volumenes download failed. Details: {result.get('details', {})}"


if __name__ == "__main__":
    precios_download_multiple_days()
    time.sleep(10)
    volumenes_download_multiple_days()
    print("🎉 All tests passed successfully!")

