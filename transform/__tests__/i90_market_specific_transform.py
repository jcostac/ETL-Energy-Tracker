#Add the project root to Python path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from transform.i90_transform import TransformadorI90
import time

def test_market_specific_intradiario():
    """
    Test the transformation of a single day (latest day) for intradiario market
    """
    transformer = TransformadorI90()
    result = transformer.transform_data_for_all_markets(dataset_type="volumenes_i90", mercados_lst=["intra"])
    result_status = result["status"]

    assert result_status["success"] == True, f"Market specific intradiario transformation failed. Details: {result_status.get('details', {})}"
    breakpoint()

def test_market_specific_volumenes():
    """
    Test the transformation of a single day for specific markets
    """
    transformer = TransformadorI90()
    result = transformer.transform_data_for_all_markets(fecha_inicio="2025-03-03", fecha_fin="2025-03-03", dataset_type="volumenes_i90", mercados_lst=["diario", "secundaria"])
    result_status = result["status"]

    assert result_status["success"] == True, f"Market specific volumenes transformation failed. Details: {result_status.get('details', {})}"

def test_multiple_days_market_specific_volumenes():
    """
    Test the transformation with multiple specific markets for volumenes
    """
    transformer = TransformadorI90()
    result = transformer.transform_data_for_all_markets(fecha_inicio="2025-03-02", fecha_fin="2025-03-04", dataset_type="volumenes_i90", mercados_lst=["terciaria", "rr", "curtailment", "p48", "indisponibilidades", "restricciones"])
    result_status = result["status"]

    assert result_status["success"] == True, f"Multiple markets volumenes transformation failed. Details: {result_status.get('details', {})}"

if __name__ == "__main__":
    test_market_specific_intradiario()
    time.sleep(5)
    test_market_specific_volumenes()
    time.sleep(5)
    test_multiple_days_market_specific_volumenes()
    print("🎉 All market specific tests passed successfully!")
