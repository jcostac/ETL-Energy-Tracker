#Add the project root to Python path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))


from extract.omie_extractor import OMIEExtractor
import time

def omie_download_single_day():
    """
    Test the download of the omie for a single day for all markets
    """
    extractor = OMIEExtractor()
    extractor.extract_data_for_all_markets(fecha_inicio_carga="2025-03-01", fecha_fin_carga="2023-01-01")

def omie_download_multiple_days():
    """
    Test the download of the omie for multiple days for all markets
    """
    extractor = OMIEExtractor()
    extractor.extract_data_for_all_markets(fecha_inicio_carga="2025-03-02", fecha_fin_carga="2025-03-08")

def omie_download_market_specific():
    """
    Test the download of the omie for a specific market
    """
    extractor = OMIEExtractor()
    extractor.extract_data_for_all_markets(fecha_inicio_carga="2025-01-01", fecha_fin_carga="2025-01-01", mercados_lst=["continuo"])


if __name__ == "__main__":
    omie_download_single_day()
    time.sleep(10)
    omie_download_multiple_days()
    time.sleep(10)
    omie_download_market_specific()

