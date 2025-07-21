#Add the project root to Python path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))


from extract.omie_extractor import OMIEExtractor
import time

def omie_download_single_day():
    """
    Test OMIE data extraction for a single day across all markets.
    
    Asserts that extraction is successful and prints a confirmation message.
    """
    extractor = OMIEExtractor()
    result = extractor.extract_data_for_all_markets(fecha_inicio="2025-04-08", fecha_fin="2025-04-08")
    
    # Assert that the extraction was successful
    assert result["success"] == True, f"Single day download failed. Details: {result.get('details', {})}"
    print("✅ Single day download test passed")


def omie_download_multiple_days():
    """
    Test extraction of OMIE data for all markets over a specified date range.
    
    Asserts that data extraction from "2025-03-02" to "2025-03-08" completes successfully.
    """
    extractor = OMIEExtractor()
    result = extractor.extract_data_for_all_markets(fecha_inicio="2025-03-02", fecha_fin="2025-03-08")
    
    # Assert that the extraction was successful
    assert result["success"] == True, f"Multiple days download failed. Details: {result.get('details', {})}"
    print("✅ Multiple days download test passed")

def omie_download_market_specific():
    """
    Test OMIE data extraction for a specific market on a single day.
    
    Runs the extraction for the "continuo" market on "2025-01-01" and asserts that the operation succeeds.
    """
   
    extractor = OMIEExtractor()
    result = extractor.extract_data_for_all_markets(fecha_inicio="2025-01-01", fecha_fin="2025-01-01", mercados_lst=["continuo"])
    
    # Assert that the extraction was successful
    assert result["success"] == True, f"Market specific download failed. Details: {result.get('details', {})}"
    print("✅ Market specific download test passed")


if __name__ == "__main__":
    omie_download_single_day()
  
    omie_download_multiple_days()
 
    omie_download_market_specific()

    print("🎉 All tests passed successfully!")

