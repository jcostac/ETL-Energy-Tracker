import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import the module after adding the path
from transform.curtailment_transform import CurtailmentTransformer


def test_curtailment_i3_transform():
    transformer = CurtailmentTransformer()
    i3_result = transformer.transform_curtailment_i3() 
    assert i3_result['status']['success'] is True, "i3 transformation failed"


def test_curtailment_i90_transform():
    transformer = CurtailmentTransformer()
    i90_result = transformer.transform_curtailment_i90()
    assert i90_result['status']['success'] is True, "i90 transformation failed"

if __name__ == "__main__":
    test_curtailment_i3_transform()
    test_curtailment_i90_transform()


