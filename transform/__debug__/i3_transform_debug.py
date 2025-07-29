import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import unittest
from transform.i3_transform import TransformadorI3

class DebugI3Transform:

    def debug_transform_specific_day_volumenes(self, fecha_inicio, fecha_fin):
        """
        Debug the transformation of 'volumenes_i3' data for a specific day and market.
        """
        transformer = TransformadorI3()
        result = transformer.transform_data_for_all_markets(
            fecha_inicio=fecha_inicio, 
            fecha_fin=fecha_fin, 
            mercados_lst=["secundaria"]
        )
        result_status = result["status"]
        if result_status["success"]:
            print(f"✅ Volumenes_i3 transformation for {fecha_inicio} to {fecha_fin} PASSED.")
        else:
            print(f"❌ Volumenes_i3 transformation for {fecha_inicio} to {fecha_fin} FAILED. Details: {result_status.get('details', {})}")
        print(result["data"])
        breakpoint()

    def debug_transform_latest_day_volumenes(self):
        """
        Debug the transformation of 'volumenes_i3' data for the latest available day for all markets.
        """
        transformer = TransformadorI3()
        result = transformer.transform_data_for_all_markets(fecha_inicio=None, fecha_fin=None)
        result_status = result["status"]
        if result_status["success"]:
            print("✅ Volumenes_i3 transformation for latest day PASSED.")
        else:
            print(f"❌ Volumenes_i3 transformation for latest day FAILED. Details: {result_status.get('details', {})}")
        print(result["data"])
        breakpoint()

if __name__ == "__main__":
    debugger = DebugI3Transform()
    print("Running I3 Transform Debug for specific day...")
    debugger.debug_transform_specific_day_volumenes("2024-10-27", "2024-10-27")
    print("Running I3 Transform Debug for latest day...")
    debugger.debug_transform_latest_day_volumenes()
    print("Debugging script finished.") 




