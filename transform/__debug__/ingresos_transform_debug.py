import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from transform.ingresos_transform import TransformadorIngresos

class DebugTransformTests():

    def debug_calculate_ingresos_for_all_markets(self, fecha_inicio, fecha_fin, mercados_lst):
        """
        Test the income calculation for a single day for all markets.
        This test ensures that the transformation completes successfully for specific dates,
        including those around daylight saving time changes.
        """
        transformer = TransformadorIngresos()
        result = transformer.calculate_ingresos_for_all_markets(
                    fecha_inicio=fecha_inicio, fecha_fin=fecha_fin, mercados_lst=mercados_lst)
        result_status = result["status"]
        if result_status["success"]:
            print(f"✅ Ingresos calculation for {fecha_inicio} to {fecha_fin} PASSED.")
        else:
            print(f"❌ Ingresos calculation for {fecha_inicio} to {fecha_fin} FAILED. Details: {result_status.get('details', {})}")
        
        print(result["data"])
        breakpoint()

   
if __name__ == "__main__":
    debugger = DebugTransformTests()
    TEST_DATES = [
        "2024-01-01",
        "2024-11-01", 
        "2025-01-01", 
        "2025-03-31", 
        "2025-04-01"
    ]
    debugger.debug_calculate_ingresos_for_all_markets("2025-03-31", "2025-04-01", mercados_lst=["diario"])
    breakpoint()

