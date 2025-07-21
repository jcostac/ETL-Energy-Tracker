import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import the module after adding the path
from transform.curtailment_transform import CurtailmentTransformer


def test_curtailment_transform():
    transformer = CurtailmentTransformer()
    i90_result = transformer.transform_curtailment_i90()
    i3_result = transformer.transform_curtailment_i3() 

    assert i90_result is not None, "i90 transformation failed"
    assert i3_result is not None, "i3 transformation failed"

if __name__ == "__main__":
    transformer = CurtailmentTransformer()
    # Example usage
    i90_result = transformer.transform_curtailment_i90()
    i3_result = transformer.transform_curtailment_i3() 