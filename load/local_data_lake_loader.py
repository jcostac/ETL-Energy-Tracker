"""
LocalDataLakeLoader implementation for storing data in the local filesystem.
"""

from typing import Optional, Union
from pathlib import Path
import pandas as pd
import sys

# Get the absolute path to the scripts directory
SCRIPTS_DIR = Path(__file__).resolve().parent.parent
# Ensure the project root is added to sys.path if necessary
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

# Corrected import path assuming utilidades is at the project root
from utilidades.storage_file_utils import ProcessedFileUtils
# Import DATA_LAKE_BASE_PATH if needed for default base path
from configs.storage_config import DATA_LAKE_BASE_PATH


class LocalDataLakeLoader():
    """
    Implementation of DataLakeLoader for storing data in the local filesystem
    using ProcessedFileUtils.

    This class delegates the writing of processed parquet files (including partitioning)
    to the ProcessedFileUtils utility.
    """

    def __init__(self):
        """
        Initializes the LocalDataLakeLoader. For saving processed data locally.

        Args:
            base_path (Optional[Union[str, Path]]): The base path for the data lake.
                                                    If None, uses DATA_LAKE_BASE_PATH from config.
                                                    Defaults to None.
        """
        super().__init__()
        # ProcessedFileUtils constructor handles setting the correct base path (raw/processed)
        self.file_utils = ProcessedFileUtils(use_s3=False)
        # We might store the processed base path if needed elsewhere, but ProcessedFileUtils manages it internally
        # self.processed_base_path = self.file_utils.processed_path
        print(f"LocalDataLakeLoader initialized. Processed data path: {self.file_utils.processed_path}")


    def _save_processed_data(self, processed_df: pd.DataFrame, mercado: str, value_col: str, dataset_type: str) -> None:
        """
        Saves the processed DataFrame to the local data lake as a partitioned Parquet file.

        Delegates the writing operation to ProcessedFileUtils.

        Args:
            df (pd.DataFrame): The processed DataFrame to save.
            mercado (str): Market identifier (used for partitioning).
            value_col (str): The name of the main value column.
            dataset_type (str): The type of dataset (e.g., 'precios').
        """
        print(f"\n--- Initiating save operation for {dataset_type} ({mercado}) ---")
        if processed_df.empty:
            print("WARNING: Input DataFrame is empty. Skipping save.")
            return

        try:
            # 1. Save Processed Data
            print("\n💾 SAVING DATA")
            print("-"*50)
            print(f"Records to save: {len(processed_df)}")
            print(f"Target market: {mercado}")
            print(f"Dataset type: {dataset_type}")
            print("-"*50)
            print(f"Saving data in path: {self.file_utils.processed_path}")
            
            self.file_utils.write_processed_parquet(
                processed_df, 
                mercado, 
                value_col=value_col, 
                dataset_type=dataset_type
            )

            print(f"✅Data saved successfully in path: {self.file_utils.processed_path}")

        except Exception as e:
            print("\n❌ ERROR OCCURRED")
            print("-"*50)
            print(f"Operation: 'Save'")
            print(f"Market: {mercado}")
            print(f"Error: {str(e)}")
            print("="*80 + "\n")
            return

    def load_transformed_data_esios(self, transformed_data_dict, **kwargs):
        """
        Process the dictionary output from transform phase and load each market's data
        
        Args:
            transformed_data_dict: Dictionary with market names as keys and DataFrames as values
        """
        results = []
        
        for mercado, df in transformed_data_dict.items():
            if df is not None and not df.empty:
                # For each market, save its data
                self._save_processed_data(
                    processed_df=df,
                    mercado=mercado,
                    value_col='precio',  # Assuming the value column is 'precio'
                    dataset_type='precios'
                )
                results.append(f"Loaded {len(df)} records for market {mercado}")
            else:
                results.append(f"No data to load for market {mercado}")
    
        return results
    
    def load_transformed_data_precios_i90(self, transformed_data_dict, **kwargs):
        """
        Process the dictionary output from transform phase and load each market's i90 price data
        
        Args:
            transformed_data_dict: Dictionary with market names as keys and DataFrames as values
        """
        results = []
        
        for mercado, df in transformed_data_dict.items():
            if df is not None and not df.empty:
                # For each market, save its data
                self._save_processed_data(
                    processed_df=df,
                    mercado=mercado,
                    value_col='precio',  # Using precio_i90 as the value column
                    dataset_type='precios_i90'
                )
                results.append(f"Loaded {len(df)} records for i90 prices market {mercado}")
            else:
                results.append(f"No i90 price data to load for market {mercado}")

        return results

    def load_transformed_data_volumenes_i90(self, transformed_data_dict, **kwargs):
        """
        Process the dictionary output from transform phase and load each market's i90 volume data
        
        Args:
            transformed_data_dict: Dictionary with market names as keys and DataFrames as values
        """
        results = []
        
        for mercado, df in transformed_data_dict.items():
            if df is not None and not df.empty:
                # For each market, save its data
                self._save_processed_data(
                    processed_df=df,
                    mercado=mercado,
                    value_col='volumenes',  # Using volumen_i90 as the value column
                    dataset_type='volumenes_i90'
                )
                results.append(f"Loaded {len(df)} records for i90 volumes market {mercado}")
            else:
                results.append(f"No i90 volume data to load for market {mercado}")

        return results
    
    
if __name__ == "__main__":
    print("--- Running LocalDataLakeLoader Example ---")
    # Example usage: Create a dummy DataFrame
    data = {
        'datetime_utc': pd.to_datetime(['2023-01-01 00:00:00', '2023-01-01 01:00:00', '2023-02-01 00:00:00']),
        'precio': [50.5, 52.1, 60.0],
        'id_mercado': [1, 1, 1] # Example, ensure this exists in your data
        # Add other columns expected by ProcessedFileUtils if necessary
    }
    sample_df = pd.DataFrame(data)
    # Ensure datetime_utc is timezone-naive or UTC before passing, as ProcessedFileUtils expects
    if sample_df['datetime_utc'].dt.tz:
         sample_df['datetime_utc'] = sample_df['datetime_utc'].dt.tz_convert('UTC').dt.tz_localize(None)
    else:
         sample_df['datetime_utc'] = sample_df['datetime_utc'] # Assume naive is UTC


    # Instantiate the loader (uses default base path from config)
    local_loader = LocalDataLakeLoader()

    # Save the dummy data
    local_loader._save_processed_data(
        processed_df=sample_df,
        mercado='diario', # Example market
        value_col='precio',
        dataset_type='precios'
    )

    print("--- LocalDataLakeLoader Example Finished ---") 


