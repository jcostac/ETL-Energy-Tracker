#Add the project root to Python path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from extract.i90_extractor import I90PreciosExtractor, I90VolumenesExtractor
import time

def volumenes_download_market_specific():
    """
    Test the download of the volumenes for a specific market
    """
    extractor = I90VolumenesExtractor()
    result = extractor.extract_data_for_all_markets(fecha_inicio_carga="2025-01-01", fecha_fin_carga="2025-01-01", mercados_lst=["terciaria", "rr", "curtailment", "p48", "indisponibilidades", "restricciones"])
    assert result["success"] == True, f"Volumenes download failed. Details: {result.get('details', {})}"

def precios_download_market_specific():
    """
    Test the download of the precios for a specific market
    """
    extractor = I90PreciosExtractor()
    result = extractor.extract_data_for_all_markets(fecha_inicio_carga="2025-01-01", fecha_fin_carga="2025-01-01", mercados_lst=["restricciones"])
    assert result["success"] == True, f"Precios download failed. Details: {result.get('details', {})}"

if __name__ == "__main__":
    volumenes_download_market_specific()
    time.sleep(10)
    precios_download_market_specific()  
    print("🎉 All tests passed successfully!")