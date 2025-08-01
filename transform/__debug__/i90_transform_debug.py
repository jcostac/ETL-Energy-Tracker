
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from transform.i90_transform import TransformadorI90

class DebugI90Transform:

    def debug_transform_data_for_all_markets(self, fecha_inicio, fecha_fin, dataset_type, mercados_lst):
        """
        Test the transformation of single-day price data for the "restricciones" market using the TransformadorI90 class.
        
        Asserts that the transformation completes successfully for the specified date and market.
        """
        transformer = TransformadorI90()
        result = transformer.transform_data_for_all_markets(fecha_inicio=fecha_inicio, fecha_fin=fecha_fin, dataset_type=dataset_type, mercados_lst=mercados_lst)
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
    debugger.debug_transform_data_for_all_markets("2025-03-31", "2025-04-01", dataset_type="volumenes_i90", mercados_lst=["terciaria"])
    print("Debugging script finished.") 