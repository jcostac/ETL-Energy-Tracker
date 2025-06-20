#Add the project root to Python path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from transform.i90_transform import TransformadorI90
import time

def test_multiple_day_precios():
    """
    Test the transformation of multiple days for all markets
    """
    transformer = TransformadorI90()
    result = transformer.transform_data_for_all_markets(fecha_inicio="2025-03-01", fecha_fin="2025-03-03", dataset_type="precios_i90", mercados_lst=["restricciones"])
    result_status = result["status"]

    assert result_status["success"] == True, f"Multiple day precios transformation failed. Details: {result_status.get('details', {})}"

def test_multiple_day_volumenes():
    """
    Test the transformation of multiple days for all markets
    """
    transformer = TransformadorI90()
    result = transformer.transform_data_for_all_markets(fecha_inicio="2025-03-01", fecha_fin="2025-03-03", dataset_type="volumenes_i90")
    result_status = result["status"]
    
    assert result_status["success"] == True, f"Multiple day volumenes transformation failed. Details: {result_status.get('details', {})}"

if __name__ == "__main__":
    test_multiple_day_precios()
    time.sleep(5)
    test_multiple_day_volumenes()
    print("🎉 All tests passed successfully!")
