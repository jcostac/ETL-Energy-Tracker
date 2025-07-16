"""
LocalDataLakeLoader implementation for storing data in the local filesystem.
"""

__all__ = ['LocalDataLakeLoader']

from typing import Optional, Union
from pathlib import Path
import pandas as pd
import sys
from dotenv import load_dotenv

# Get the absolute path to the scripts directory
SCRIPTS_DIR = Path(__file__).resolve().parent.parent
# Ensure the project root is added to sys.path if necessary
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

# Corrected import path assuming utilidades is at the project root
from utilidades.processed_file_utils import ProcessedFileUtils


class LocalDataLakeLoader():
    """
    Implementation of DataLakeLoader for storing data in the local filesystem
    using ProcessedFileUtils.

    This class delegates the writing of processed parquet files (including partitioning)
    to the ProcessedFileUtils utility.
    """

    def __init__(self):
        """
        Initialize a LocalDataLakeLoader instance for saving processed data to the local filesystem.
        
        Loads environment variables and prepares file utilities for managing processed data storage.
        """
        load_dotenv()
        # ProcessedFileUtils constructor handles setting the correct base path (raw/processed)
        self.file_utils = ProcessedFileUtils()

        print(f"LocalDataLakeLoader initialized. Processed data path: {self.file_utils.processed_path}")


    def _save_processed_data(self, processed_df: pd.DataFrame, mercado: str, value_cols: list[str], dataset_type: str) -> None:
        """
        Save a processed DataFrame to the local data lake as a partitioned Parquet file.
        
        The DataFrame is saved using the specified market identifier, value columns, and dataset type. If the DataFrame is empty, the save operation is skipped. Raises any exceptions encountered during the save process.
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
                value_cols=value_cols, 
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

    def load_transformed_data(self, transformed_data_dict: dict, dataset_type: str, value_cols: list[str], **kwargs) -> dict:
        """
        Processes and loads transformed data for multiple markets, saving each as a partitioned Parquet file and reporting per-market status.
        
        Parameters:
            transformed_data_dict (dict): Dictionary with a "data" key mapping market names to DataFrames, and a "status" key with transformation status information.
            dataset_type (str): Identifier for the type of dataset being loaded.
            value_cols (list[str]): List of column names in the DataFrame to be treated as value columns.
        
        Returns:
            dict: Dictionary containing:
                - "success": True if all markets were loaded successfully, False otherwise.
                - "messages": List of status messages for each market.
                - "market_status": Dictionary mapping each market to a boolean indicating success or failure.
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
                            value_cols=value_cols,
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

    def load_transformed_data_esios(self, transformed_data_dict, **kwargs) -> dict:
        """
        Loads transformed "precios_esios" dataset for each market from the provided dictionary.
        
        Parameters:
            transformed_data_dict (dict): Dictionary with a "data" key mapping market names to DataFrames and a "status" key with transformation status.
        
        Returns:
            dict: Summary of the loading process, including overall success, messages, and per-market status.
        """
        return self.load_transformed_data(
            transformed_data_dict,
            dataset_type="precios_esios",
            value_cols= ['precio'],
            **kwargs
        )
    
    def load_transformed_data_precios_i90(self, transformed_data_dict, **kwargs) -> dict:
        """
        Loads transformed i90 price data for each market from the provided dictionary and stores it as partitioned Parquet files.
        
        Parameters:
            transformed_data_dict (dict): Dictionary with a "data" key mapping market names to DataFrames and a "status" key with transformation status.
        
        Returns:
            dict: Summary of the loading process, including overall success, messages, and per-market status.
        """
        return self.load_transformed_data(
            transformed_data_dict,
            dataset_type='precios_i90',
            value_cols= ['precio'],
            **kwargs
        )

    def load_transformed_data_volumenes_i90(self, transformed_data_dict, **kwargs) -> dict:
        """
        Loads transformed i90 volume data for each market from the provided dictionary and stores it in the local data lake.
        
        Parameters:
            transformed_data_dict (dict): Dictionary with a "data" key mapping market names to DataFrames and a "status" key with transformation status.
        
        Returns:
            dict: Summary containing overall success, status messages, and per-market success flags.
        """
        return self.load_transformed_data(
            transformed_data_dict,
            dataset_type='volumenes_i90',
            value_cols= ['volumenes'],
            **kwargs
        )
    
    def load_transformed_data_volumenes_omie(self, transformed_data_dict, **kwargs) -> dict:
        """
        Loads transformed OMIE volume data for each market from the provided dictionary and stores it in the local data lake.
        
        Parameters:
            transformed_data_dict (dict): Dictionary containing market names as keys and their corresponding DataFrames as values.
        
        Returns:
            dict: Summary of the loading process, including overall success, status messages, and per-market results.
        """
        return self.load_transformed_data(
            transformed_data_dict,
            dataset_type='volumenes_omie',
            value_cols= ['volumenes'],
            **kwargs
        )
    
    def load_transformed_data_volumenes_mic(self, transformed_data_dict, **kwargs) -> dict:
        """
        Loads and stores transformed MIC volume data for each market from the provided dictionary.
        
        Parameters:
            transformed_data_dict (dict): Dictionary containing market names as keys and their corresponding DataFrames as values.
        
        Returns:
            dict: Summary of the loading process, including overall success, status messages, and per-market results.
        """
        return self.load_transformed_data(
            transformed_data_dict,
            dataset_type='volumenes_mic',
            value_cols= ['volumenes', 'precio'],
            **kwargs
        )


    def load_transformed_data_volumenes_i3(self, transformed_data_dict, **kwargs) -> dict:
        """
        Loads transformed i3 volume data for each market from the provided dictionary and stores it in the local data lake.

        Parameters:
            transformed_data_dict (dict): Dictionary containing market names as keys and their corresponding DataFrames as values.
        
        Returns:
            dict: Summary of the loading process, including overall success, status messages, and per-market results.
        """
        return self.load_transformed_data(
            transformed_data_dict,
            dataset_type='volumenes_i3',
            value_cols= ['volumenes'],
            **kwargs
        )
    


def example_usage():
    """
    Demonstrates how to use the LocalDataLakeLoader to load transformed data for the 'esios' dataset type with sample empty DataFrames.
    """
    loader = LocalDataLakeLoader()
    transformed_data_dict = {
        'data': {
            'diario': pd.DataFrame(),
            'secundaria': pd.DataFrame()
        },
        'status': {
            'diario': "Success",
            'secundaria': "Success"
        }
    }
    loader.load_transformed_data_esios(transformed_data_dict)

if __name__ == "__main__":
    example_usage()


