#Add the project root to Python path
import sys
import os
from datetime import datetime, timedelta
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import unittest
from transform.omie_transform import TransformadorOMIE

class DebugOMIETransform:

    TEST_DATES = [
        (datetime.now() - timedelta(days=93)).strftime('%Y-%m-%d'),
        "2024-03-31",  # Start of DST (summer time)
        "2024-10-27",  # End of DST (fall back to standard time)
    ]

    def debug_transform_single_day(self):
        """
        Debugs the OMIE data transformation process for a single day across all markets.
        Prints the result and status for inspection.
        """
        for test_date in self.TEST_DATES:
            print(f"\n--- Debugging OMIE transform for single day: {test_date} ---")
            transformer = TransformadorOMIE()
            result = transformer.transform_data_for_all_markets(fecha_inicio=test_date, fecha_fin=test_date)
            result_status = result.get("status", {})
            if result_status.get("success"):
                print(f"✅ Single day transformation for {test_date} PASSED.")
            else:
                print(f"❌ Single day transformation for {test_date} FAILED. Details: {result_status.get('details', {})}")
            print("Result data preview:")
            print(result.get("data"))
            breakpoint()

    def debug_transform_multiple_days(self, fecha_inicio, fecha_fin):
        """
        Debugs the OMIE data transformation process for a range of days across all markets.
        Prints the result and status for inspection.
        """
        print(f"\n--- Debugging OMIE transform for multiple days: {fecha_inicio} to {fecha_fin} ---")
        transformer = TransformadorOMIE()
        result = transformer.transform_data_for_all_markets(fecha_inicio=fecha_inicio, fecha_fin=fecha_fin)
        result_status = result.get("status", {})
        if result_status.get("success"):
            print(f"✅ Multiple days transformation from {fecha_inicio} to {fecha_fin} PASSED.")
        else:
            print(f"❌ Multiple days transformation from {fecha_inicio} to {fecha_fin} FAILED. Details: {result_status.get('details', {})}")
        print("Result data preview:")
        print(result.get("data"))
        breakpoint()

    def debug_transform_market_specific(self, mercados_lst=None):
        """
        Debugs the OMIE data transformation for a specific subset of markets on each test day.
        Prints the result and status for inspection.
        """
        if mercados_lst is None:
            mercados_lst = ["intradiario", "continuo"]
        for test_date in self.TEST_DATES:
            print(f"\n--- Debugging OMIE transform for markets {mercados_lst} on {test_date} ---")
            transformer = TransformadorOMIE()
            result = transformer.transform_data_for_all_markets(
                fecha_inicio=test_date, 
                fecha_fin=test_date, 
                mercados_lst=mercados_lst
            )
            result_status = result.get("status", {})
            if result_status.get("success"):
                print(f"✅ Market specific transformation for {test_date} ({mercados_lst}) PASSED.")
            else:
                print(f"❌ Market specific transformation for {test_date} ({mercados_lst}) FAILED. Details: {result_status.get('details', {})}")
            print("Result data preview:")
            print(result.get("data"))
            breakpoint()

if __name__ == "__main__":
    debugger = DebugOMIETransform()
    print("Running OMIE Transform Debug for single day...")
    debugger.debug_transform_single_day()
    print("Running OMIE Transform Debug for multiple days (2024-03-31 to 2024-04-02)...")
    debugger.debug_transform_multiple_days("2024-03-31", "2024-04-02")
    print("Running OMIE Transform Debug for market-specific days...")
    debugger.debug_transform_market_specific()
    print("Debugging script finished.") 
