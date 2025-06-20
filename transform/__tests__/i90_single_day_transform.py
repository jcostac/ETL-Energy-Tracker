#Add the project root to Python path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from transform.i90_transform import TransformadorI90
import time

def test_single_day_precios():
    """
    Test the transformation of a single day  for all markets
    """
    transformer = TransformadorI90()
    result = transformer.transform_data_for_all_markets(fecha_inicio="2025-03-02", fecha_fin="2025-03-02", dataset_type="precios_i90", mercados_lst=["restricciones"])
    result_status = result["status"]

    assert result_status["success"] == True, f"Single day precios transformation failed. Details: {result_status.get('details', {})}"

def test_single_day_volumenes():
    """
    Test the transformation of a single day (latest day) for all markets
    """
    transformer = TransformadorI90()
    result = transformer.transform_data_for_all_markets(dataset_type="volumenes_i90", mercados_lst=["intra"])
    result_status = result["status"]


    assert result_status["success"] == True, f"Single day volumenes transformation failed. Details: {result_status.get('details', {})}"

if __name__ == "__main__":
    test_single_day_precios()
    time.sleep(5)
    test_single_day_volumenes()
    print("🎉 All tests passed successfully!")
