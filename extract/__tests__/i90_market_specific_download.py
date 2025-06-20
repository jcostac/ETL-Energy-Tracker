from extract.i90_extractor import I90PreciosExtractor, I90VolumenesExtractor
import time

def volumenes_download_market_specific():
    """
    Test the download of the volumenes for a specific market
    """
    extractor = I90VolumenesExtractor()
    extractor.extract_data_for_all_markets(fecha_inicio_carga="2025-01-01", fecha_fin_carga="2025-01-01", mercados_lst=["diario", "intradiario", "terciaria", "rr", "curtailment", "p48", "indisponibilidades", "restricciones"])

def precios_download_market_specific():
    """
    Test the download of the precios for a specific market
    """
    extractor = I90PreciosExtractor()
    extractor.extract_data_for_all_markets(fecha_inicio_carga="2025-01-01", fecha_fin_carga="2025-01-01", mercados_lst=["restricciones"])


if __name__ == "__main__":
    volumenes_download_market_specific()
    time.sleep(10)
    precios_download_market_specific()