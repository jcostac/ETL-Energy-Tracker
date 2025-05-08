"""
LocalDataLakeLoader implementation for storing data in the local filesystem.
"""

__all__ = ['LocalDataLakeLoader']

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
            raise

    def load_transformed_data(self, transformed_data_dict: dict, dataset_type: str, value_col: str, **kwargs) -> dict:
        """
        Generic method to process and load transformed data for any dataset type.
        
        Args:
            transformed_data_dict: Dictionary containing:
                - "data": Dictionary with market names as keys and DataFrames as values
                - "status": Dictionary with transformation status information
            dataset_type: Type of dataset (e.g., 'precios', 'precios_i90', 'volumenes_i90')
            value_col: Name of the value column in the DataFrame

        Returns:
            Dictionary containing:
                - "success": Boolean indicating if the operation was successful
                - "messages": List of messages indicating the status of the operation
        """
        results = []
        market_success = {}  # Track success per market

        try:
            data_dict = transformed_data_dict.get("data")
            if data_dict is None or not data_dict:
                raise ValueError("The 'data' key is missing or contains no data.")
            
            for mercado, df in data_dict.items():
                market_success[mercado] = False  # Initialize market success as False
                
                if df is not None and not df.empty:
                    try:
                        self._save_processed_data(
                            processed_df=df,
                            mercado=mercado,
                            value_col=value_col,
                            dataset_type=dataset_type
                        )
                        results.append(f"✅ Successfully loaded {len(df)} records for {dataset_type} market {mercado}")
                        market_success[mercado] = True  # Mark market as successful
                    except Exception as e:
                        results.append(f"❌ Failed loading for market {mercado}: {str(e)}")
                        market_success[mercado] = False  # Explicitly set to False on error
                else:
                    results.append(f"ℹ️ No {dataset_type} data to load for market {mercado}")
                    market_success[mercado] = False  # Set to False for empty data

        except ValueError as ve:
            results.append(str(ve))
            # Mark all markets as failed if there's a data structure error
            for mercado in data_dict.keys():
                market_success[mercado] = False

        # Overall success is True only if all markets succeeded
        overall_success = all(market_success.values())
        
        return {
            "success": overall_success, 
            "messages": results,
            "market_status": market_success  # Include per-market status in response
        }

    def load_transformed_data_esios(self, transformed_data_dict, **kwargs):
        """
        Process the dictionary output from transform phase and load each market's data
        
        Args:
            transformed_data_dict: Dictionary containing:
                - "data": Dictionary with market names as keys and DataFrames as values
                - "status": Dictionary with transformation status information
        """
        return self.load_transformed_data(
            transformed_data_dict,
            dataset_type='precios',
            value_col='precio',
            **kwargs
        )
    
    def load_transformed_data_precios_i90(self, transformed_data_dict, **kwargs):
        """
        Process the dictionary output from transform phase and load each market's i90 price data
        
        Args:
            transformed_data_dict: Dictionary containing:
                - "data": Dictionary with market names as keys and DataFrames as values
                - "status": Dictionary with transformation status information
        """
        return self.load_transformed_data(
            transformed_data_dict,
            dataset_type='precios_i90',
            value_col='precio',
            **kwargs
        )

    def load_transformed_data_volumenes_i90(self, transformed_data_dict, **kwargs):
        """
        Process the dictionary output from transform phase and load each market's i90 volume data
        
        Args:
            transformed_data_dict: Dictionary containing:
                - "data": Dictionary with market names as keys and DataFrames as values
                - "status": Dictionary with transformation status information
        """
        return self.load_transformed_data(
            transformed_data_dict,
            dataset_type='volumenes_i90',
            value_col='volumenes',
            **kwargs
        )
    
    
if __name__ == "__main__":
    loader = LocalDataLakeLoader()
    transformed_data_dict = {
        'data': {
            'diario': "df",
            'secundaria': "df"
        },
        'status': {
            'diario': "xyz",
            'secundaria': "abc"
        }
    }
    loader.load_transformed_data_esios(transformed_data_dict)


