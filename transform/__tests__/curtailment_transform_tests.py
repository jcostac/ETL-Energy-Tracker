
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import unittest
from transform.curtailment_transform import TransformadorCurtailment

class TestCurtailmentTransform(unittest.TestCase):

    def test_curtailment_i3_transform(self):
        transformer = TransformadorCurtailment()
        i3_result = transformer.transform_curtailment_i3() 
        self.assertTrue(i3_result['status']['success'], "i3 transformation failed")

    def test_curtailment_i90_transform(self):
        transformer = TransformadorCurtailment()
        i90_result = transformer.transform_curtailment_i90()
        self.assertTrue(i90_result['status']['success'], "i90 transformation failed")

if __name__ == "__main__":
    unittest.main() 