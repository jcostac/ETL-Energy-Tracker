import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import unittest
from transform.i3_transform import TransformadorI3
from datetime import datetime, timedelta

class DebugI3Transform:

    TEST_DATES = [(datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")]

    def debug_transform_volumenes(self, fecha_inicio = None, fecha_fin = None, mercados_lst = None):
        """
        Debug the transformation of 'volumenes_i3' data for the latest available day for all markets.
        """
        transformer = TransformadorI3()

        if fecha_inicio is None and fecha_fin is None:
            for test_date in self.TEST_DATES:
                result = transformer.transform_data_for_all_markets(fecha_inicio=test_date, fecha_fin=test_date, mercados_lst=mercados_lst)
        else:
            result = transformer.transform_data_for_all_markets(fecha_inicio=fecha_inicio, fecha_fin=fecha_fin, mercados_lst=mercados_lst)

        result_status = result["status"]
        if result_status["success"]:
            print("✅ Volumenes_i3 transformation for latest day PASSED.")
        else:
            print(f"❌ Volumenes_i3 transformation for latest day FAILED. Details: {result_status.get('details', {})}")
        print(result["data"])
        breakpoint()

if __name__ == "__main__":
    debugger = DebugI3Transform()
    print("Running I3 Transform Debug for volumenes...")
    debugger.debug_transform_volumenes(fecha_inicio=None, fecha_fin=None, mercados_lst=["restricciones_md", "restricciones_tr", "desvios"])
    print("Debugging script finished.") 




