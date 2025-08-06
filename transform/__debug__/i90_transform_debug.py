from datetime import datetime, timedelta
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from transform.i90_transform import TransformadorI90

class DebugI90Transform:

    TEST_DATES = [
        (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
    ]

    def debug_transform_data_for_all_markets(self, fecha_inicio, fecha_fin, dataset_type, mercados_lst):
        """
        Test the transformation of single-day price data for the "restricciones" market using the TransformadorI90 class.
        
        Asserts that the transformation completes successfully for the specified date and market.
        """
        transformer = TransformadorI90()
        if fecha_inicio is None and fecha_fin is None:
            for test_date in self.TEST_DATES:
                result = transformer.transform_data_for_all_markets(fecha_inicio=test_date, fecha_fin=test_date, dataset_type=dataset_type, mercados_lst=mercados_lst)
        else:
            result = transformer.transform_data_for_all_markets(fecha_inicio=fecha_inicio, fecha_fin=fecha_fin, dataset_type=dataset_type, mercados_lst=mercados_lst)

        data = result["data"]
        breakpoint()
        result_status = result["status"]
        if result_status["success"]:
            print(f"✅ Single day transformation for {fecha_inicio} to {fecha_fin} PASSED.")
        else:
            print(f"❌ Single day transformation for {fecha_inicio} to {fecha_fin} FAILED. Details: {result_status.get('details', {})}")
        
        print(result["data"])        
        breakpoint()

  

if __name__ == "__main__":
    debugger = DebugI90Transform()
    print("Running I90 Transform Tests for debugging...")
    debugger.debug_transform_data_for_all_markets(fecha_inicio=None, fecha_fin=None, dataset_type="precios_i90", mercados_lst=None)
    print("Debugging script finished.") 
