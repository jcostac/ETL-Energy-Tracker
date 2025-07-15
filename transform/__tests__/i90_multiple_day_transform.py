#Add the project root to Python path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from transform.i90_transform import TransformadorI90
import time

from datetime import datetime, timedelta

def test_multiple_day_precios():
    """
    Test the transformation of "precios_i90" data for multiple days and a specific market.

    Asserts that the transformation for a recent 2-day range, 93 days in the past for the "restricciones" market completes successfully.
    """
    # Use a recent 3-day window ending yesterday
    end_date = datetime.now() - timedelta(days=93)
    start_date = end_date - timedelta(days=95)
    fecha_inicio = start_date.strftime('%Y-%m-%d')
    fecha_fin = end_date.strftime('%Y-%m-%d')

    transformer = TransformadorI90()
    result = transformer.transform_data_for_all_markets(
        fecha_inicio=fecha_inicio,
        fecha_fin=fecha_fin,
        dataset_type="precios_i90",
        mercados_lst=["restricciones"]
    )
    result_status = result["status"]

    assert result_status["success"] == True, f"Multiple day precios transformation failed. Details: {result_status.get('details', {})}"

def test_multiple_day_volumenes():
    """
    Test the transformation of "volumenes_i90" data across multiple days for all markets.

    Asserts that the transformation is successful for a recent 2-day range, 93 days in the past.
    """
    end_date = datetime.now() - timedelta(days=93)
    start_date = end_date - timedelta(days=95)
    fecha_inicio = start_date.strftime('%Y-%m-%d')
    fecha_fin = end_date.strftime('%Y-%m-%d')

    transformer = TransformadorI90()
    result = transformer.transform_data_for_all_markets(
        fecha_inicio=fecha_inicio,
        fecha_fin=fecha_fin,
        dataset_type="volumenes_i90"
    )
    result_status = result["status"]

    assert result_status["success"] == True, f"Multiple day volumenes transformation failed. Details: {result_status.get('details', {})}"

if __name__ == "__main__":
    test_multiple_day_precios()
    time.sleep(5)
    test_multiple_day_volumenes()
    print("🎉 All tests passed successfully!")
