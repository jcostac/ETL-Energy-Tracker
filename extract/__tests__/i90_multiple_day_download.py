from extract.i90_extractor import I90PreciosExtractor, I90VolumenesExtractor
import time

def precios_download_multiple_days():
        """
        Test the download of the precios for multiple days for all markets
        """
        extractor = I90PreciosExtractor()
        extractor.extract_data_for_all_markets(fecha_inicio_carga="2025-03-02", fecha_fin_carga="2025-03-08")


def volumenes_download_multiple_days():
    """
    Test the download of the volumenes for multiple days for all markets
    """
    extractor = I90VolumenesExtractor()
    extractor.extract_data_for_all_markets(fecha_inicio_carga="2025-03-02", fecha_fin_carga="2025-03-08")


if __name__ == "__main__":
    precios_download_multiple_days()
    time.sleep(10)
    volumenes_download_multiple_days()