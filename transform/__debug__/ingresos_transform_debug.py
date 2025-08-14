import sys
import os
from datetime import datetime, timedelta

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from transform.ingresos_transform import TransformadorIngresos

class DebugTransformTests():

    TEST_DATES = [(datetime.now() - timedelta(days=120)).strftime("%Y-%m-%d")]

    def debug_calculate_ingresos_for_all_markets(self, fecha_inicio, fecha_fin, mercados_lst, plot=False):
        """
        Test the income calculation for a single day for all markets.
        This test ensures that the transformation completes successfully for specific dates,
        including those around daylight saving time changes.
        """
        transformer = TransformadorIngresos()
        if fecha_inicio is None and fecha_fin is None:
            for test_date in self.TEST_DATES:
                result = transformer.calculate_ingresos_for_all_markets(fecha_inicio=test_date, fecha_fin=test_date, mercados_lst=mercados_lst, plot=plot)
        else:
            result = transformer.calculate_ingresos_for_all_markets(fecha_inicio=fecha_inicio, fecha_fin=fecha_fin, mercados_lst=mercados_lst, plot=plot)

        result_data = result["data"]
        result_status = result["status"]
        if result_status["success"]:
            print(f"✅ Ingresos calculation PASSED.")
        else:
            print(f"❌ Ingresos calculation FAILED. Details: {result_status.get('details', {})}")
        
        print(result_data)
        breakpoint()

   
if __name__ == "__main__":
    debugger = DebugTransformTests()
    debugger.debug_calculate_ingresos_for_all_markets(fecha_inicio=None, fecha_fin=None, mercados_lst=["desvios"], plot=False)
    breakpoint()

