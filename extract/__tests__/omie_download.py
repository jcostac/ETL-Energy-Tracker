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
    result = extractor.extract_data_for_all_markets(fecha_inicio_carga="2025-03-01", fecha_fin_carga="2025-03-01")
    
    # Assert that the extraction was successful
    assert result["success"] == True, f"Single day download failed. Details: {result.get('details', {})}"
    print("✅ Single day download test passed")


def omie_download_multiple_days():
    """
    Test the download of the omie for multiple days for all markets
    """
    extractor = OMIEExtractor()
    result = extractor.extract_data_for_all_markets(fecha_inicio_carga="2025-03-02", fecha_fin_carga="2025-03-08")
    
    # Assert that the extraction was successful
    assert result["success"] == True, f"Multiple days download failed. Details: {result.get('details', {})}"
    print("✅ Multiple days download test passed")

def omie_download_market_specific():
    """
    Test the download of the omie for a specific market
    """
    extractor = OMIEExtractor()
    result = extractor.extract_data_for_all_markets(fecha_inicio_carga="2025-01-01", fecha_fin_carga="2025-01-01", mercados_lst=["continuo"])

    breakpoint()
    
    # Assert that the extraction was successful
    assert result["success"] == True, f"Market specific download failed. Details: {result.get('details', {})}"
    print("✅ Market specific download test passed")


if __name__ == "__main__":
    #omie_download_single_day()
    #time.sleep(5)
    #omie_download_multiple_days()
    #time.sleep(5)
    omie_download_market_specific()
    print("🎉 All tests passed successfully!")

