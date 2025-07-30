
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from transform.curtailment_transform import TransformadorCurtailment

class TestCurtailmentTransform:

    def test_curtailment_i3_transform(self):
        transformer = TransformadorCurtailment()
        i3_result = transformer.transform_curtailment_i3() 
        self.assertTrue(i3_result['status']['success'], "i3 transformation failed")
        breakpoint()

    def test_curtailment_i90_transform(self):
        transformer = TransformadorCurtailment()
        i90_result = transformer.transform_curtailment_i90()
        self.assertTrue(i90_result['status']['success'], "i90 transformation failed")
        breakpoint()

if __name__ == "__main__":
    debugger = TestCurtailmentTransform()
    print("Running Curtailment Transform Debug for i3...")
    debugger.test_curtailment_i3_transform()
    print("Running Curtailment Transform Debug for i90...")
    debugger.test_curtailment_i90_transform()
    print("Debugging script finished.") 