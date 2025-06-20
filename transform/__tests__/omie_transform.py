#Add the project root to Python path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from transform.omie_transform import TransformadorOMIE
import time

def omie_transform_single_day():
    """
    Test the transformation of the omie for a single day for all markets
    """
    transformer = TransformadorOMIE()
    result = transformer.transform_data_for_all_markets(fecha_inicio="2025-03-01", fecha_fin="2025-03-01")
    
    # Assert that the transformation was successful
    result_status = result["status"]
    assert result_status["success"] == True, f"Single day transformation failed. Details: {result_status.get('details', {})}"
    print("✅ Single day transformation test passed")


def omie_transform_multiple_days():
    """
    Test the transformation of the omie for multiple days for all markets
    """
    transformer = TransformadorOMIE()
    result = transformer.transform_data_for_all_markets(fecha_inicio="2025-03-02", fecha_fin="2025-03-08")
    
    # Assert that the transformation was successful
    assert result["success"] == True, f"Multiple days transformation failed. Details: {result.get('details', {})}"
    print("✅ Multiple days transformation test passed")

def omie_transform_market_specific():
    """
    Test the transformation of the omie for a specific market
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
