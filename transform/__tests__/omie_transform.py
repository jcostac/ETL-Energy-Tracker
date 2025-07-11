#Add the project root to Python path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from transform.omie_transform import TransformadorOMIE
import time

def omie_transform_single_day():
    """
    Tests the OMIE data transformation process for a single day across all markets.
    
    Asserts that the transformation completes successfully and saves the resulting data to a CSV file for further inspection.
    """
    transformer = TransformadorOMIE()
    result = transformer.transform_data_for_all_markets(fecha_inicio="2025-03-19", fecha_fin="2025-03-19")
    
    # Assert that the transformation was successful
    result_status = result["status"]
    assert result_status["success"] == True, f"Single day transformation failed. Details: {result_status.get('details', {})}"
    print("✅ Single day transformation test passed")

    result_data = result["data"]
    breakpoint()
    result_data.to_csv("omie_transform_single_day.csv")
    breakpoint()


def omie_transform_multiple_days():
    """
    Test the OMIE data transformation process for multiple consecutive days across all markets.
    
    Asserts that the transformation completes successfully for the specified date range.
    """
    transformer = TransformadorOMIE()
    result = transformer.transform_data_for_all_markets(fecha_inicio="2025-03-19", fecha_fin="2025-03-20")
    
    # Assert that the transformation was successful
    assert result["success"] == True, f"Multiple days transformation failed. Details: {result.get('details', {})}"
    print("✅ Multiple days transformation test passed")
   

def omie_transform_market_specific():
    """
    Test the OMIE data transformation for a specific subset of markets on a single day.
    
    Runs the transformation for the markets "intradiario" and "continuo" on "2025-01-01" and asserts that the process completes successfully.
    """
    transformer = TransformadorOMIE()
    result = transformer.transform_data_for_all_markets(fecha_inicio="2025-01-01", fecha_fin="2025-01-01", mercados_lst=["intradiario", "continuo"])
    
    # Assert that the transformation was successful
    assert result["success"] == True, f"Market specific transformation failed. Details: {result.get('details', {})}"
    print("✅ Market specific transformation test passed")


if __name__ == "__main__":
    omie_transform_single_day()
    time.sleep(5)
    omie_transform_multiple_days()
    time.sleep(5)
    omie_transform_market_specific()
    print("🎉 All tests passed successfully!")
