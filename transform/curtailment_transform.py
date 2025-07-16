import pandas as pd
from typing import Optional

from utilidades.raw_file_utils import RawFileUtils
from transform._procesador_curtailments import CurtailmentProcessor

class CurtailmentTransformer:
    """
    Transformer class for curtailment data from i90 and i3 sources.
    Handles reading the latest raw data and processing it using CurtailmentProcessor.
    """

    def __init__(self):
        self.raw_utils = RawFileUtils()
        self.processor = CurtailmentProcessor()

    def transform_curtailment_i90(self) -> Optional[pd.DataFrame]:
        """
        Reads the latest curtailment raw data from i90 and transforms it.

        Returns:
            Optional[pd.DataFrame]: Processed DataFrame or None if no data found.
        """
        df = self.raw_utils.read_latest_raw_file(mercado='curtailment', dataset_type='volumenes_i90')
        if df is None or df.empty:
            print("No raw data found for curtailment i90.")
            return None

        processed_df = self.processor.transform_raw_curtailment_data(df, 'curtailments_i90')
        return processed_df

    def transform_curtailment_i3(self) -> Optional[pd.DataFrame]:
        """
        Reads the latest curtailment raw data from i3 and transforms it.

        Returns:
            Optional[pd.DataFrame]: Processed DataFrame or None if no data found.
        """
        df = self.raw_utils.read_latest_raw_file(mercado='curtailment', dataset_type='volumenes_i3')
        if df is None or df.empty:
            print("No raw data found for curtailment i3.")
            return None

        processed_df = self.processor.transform_raw_curtailment_data(df, 'curtailments_i3')
        return processed_df

if __name__ == "__main__":
    transformer = CurtailmentTransformer()
    # Example usage
    i90_result = transformer.transform_curtailment_i90()
    i3_result = transformer.transform_curtailment_i3() 