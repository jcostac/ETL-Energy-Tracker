import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import unittest
from transform.i3_transform import TransformadorI3

class TestI3Transform(unittest.TestCase):

    def test_multiple_day_volumenes(self):
        """
        Test the transformation of "volumenes_i3" data across multiple days for all markets.
        """
        transformer = TransformadorI3()
        result = transformer.transform_data_for_all_markets(fecha_inicio="2024-12-01", fecha_fin="2024-12-02", mercados_lst=["intra", "diario"])
        result_status = result["status"]
        self.assertTrue(result_status["success"], f"Multiple day volumenes transformation failed. Details: {result_status.get('details', {})}")

    def test_latest_day_volumenes(self):
        """
        Test the transformation of "volumenes_i3" data for a single day for all markets.
        """
        transformer = TransformadorI3()
        result = transformer.transform_data_for_all_markets(fecha_inicio=None, fecha_fin=None)
        result_status = result["status"]
        self.assertTrue(result_status["success"], f"Single day volumenes transformation failed. Details: {result_status.get('details', {})}")

if __name__ == "__main__":
    unittest.main()






