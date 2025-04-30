from typing import Dict, List, Optional, Union
import pandas as pd
from datetime import datetime
import sys
import os

# Add necessary imports
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Ensure the project root is added to sys.path if necessary
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from configs.esios_config import ESIOSConfig
from utilidades.etl_date_utils import DateUtilsETL
from utilidades.data_validation_utils import DataValidationUtils

class ESIOSProcessor:
    """
    Processor class for ESIOS price data.
    Handles data cleaning, validation, and transformation operations based on notepad instructions.
    """
    def __init__(self):
        """
        Initialize the transformer with ESIOS configuration.

        Args:
            config (ESIOSConfig): An instance of ESIOSConfig containing market mappings.
        """
        self.config = ESIOSConfig()
        self.data_validatior = DataValidationUtils()
        self.indicators_to_filter = [600, 612, 613, 614, 615, 616, 617, 618, 1782]
        self.geo_names_in_raw_data = []# List of unique geo_name values in raw data
        self.geo_names_of_interest = ["España"]# List of unique geo_name values of interest


    @staticmethod
    def standardize_prices(df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize price values (e.g., convert to same unit, handle outliers).
        
        Args:
            df (pd.DataFrame): Input DataFrame with price data
            
        Returns:
            pd.DataFrame: DataFrame with standardized prices
        """
        # Remove extreme outliers (e.g., prices > 3 std from mean)
        mean_price = df['precio'].mean()
        std_price = df['precio'].std()
        df = df[abs(df['precio'] - mean_price) <= 3 * std_price]
        
        return df
    
    def unique_geo_names(self, df: pd.DataFrame) -> None:
        """
        Get unique geo_name values from ESIOSConfig. WIP: not implemented yet in piepleine might be useful for future use.
        
        Returns:
            List[str]: List of unique geo_name values"""
        
        unique_geo_names = df['geo_name'].unique()

        print(f"Geo names in dataset: {unique_geo_names}")

        self.geo_names_in_raw_data = unique_geo_names

        return

    def _filter_by_geo_name(self, df: pd.DataFrame, geo_name: str) -> pd.DataFrame:
        """
        Filter data based on geo_name for specific indicators.
        """
        self.unique_geo_names(df)
        
        # Convert indicators to filter into string format for comparison
        indicators_to_filter_str = [str(i) for i in self.indicators_to_filter]
        
        # Ensure 'indicador_id' is of string type
        if 'indicador_id' in df.columns:
            df['indicador_id'] = df['indicador_id'].astype(str)

        if 'geo_name' not in df.columns:
            print("Geo name column not found in dataframe. Skipping filter.")
            return df
        
        # Create and apply the filter only if the indicators and geo_names are in the dataframe
        if 'indicador_id' in indicators_to_filter_str and 'geo_name' in self.geo_names_of_interest:
            # Apply both filters together
            mask = (df['indicador_id'].isin(indicators_to_filter_str)) & (df['geo_name'].isin(self.geo_names_of_interest))
            df_filtered = df[mask]
            #print unique values of geo_name in filtered dataframe
            print(f"Unique geo_name values in filtered dataframe: {df_filtered['geo_name'].unique()}")
            return df_filtered
        
        return df

    def _rename_value_to_precio(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rename 'value' column to 'precio' if it exists.

        This method checks if the 'value' column is present in the DataFrame and renames it
        to 'precio'. If neither column is found, it raises a ValueError.

        Args:
            df (pd.DataFrame): The input DataFrame to be modified.

        Returns:
            pd.DataFrame: The modified DataFrame with 'value' renamed to 'precio'.
        
        Raises:
            ValueError: If neither 'value' nor 'precio' column is found in the DataFrame.
        """
        if 'value' in df.columns:
            df = df.rename(columns={'value': 'precio'})
            
        elif 'precio' not in df.columns:
            print(f"Debug - Columns in DataFrame: {df.columns}")
            raise ValueError("Neither 'value' nor 'precio' column found.")
        
        return df

    def _map_id_mercado(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Map 'indicador_id' to 'id_mercado' using config.

        This method maps the 'indicador_id' values in the DataFrame to 'id_mercado' using
        the mapping defined in the configuration. If any 'indicador_id' cannot be mapped,
        it raises a ValueError with a warning about the missing IDs.

        Args:
            df (pd.DataFrame): The input DataFrame containing 'indicador_id'.

        Returns:
            pd.DataFrame: The modified DataFrame with 'id_mercado' mapped.

        Raises:
            ValueError: If 'indicador_id' column is not found or if mapping results in null values.
        """
        if 'indicador_id' in df.columns:
            # Map 'indicador_id' to 'id_mercado' using config
            df['id_mercado'] = df['indicador_id'].map(self.config.market_id_map)
            # Check if any 'id_mercado' values are null
            if df['id_mercado'].isnull().any():
                # Get the 'indicador_id' values that are not mapped
                missing_ids = df[df['id_mercado'].isnull()]['indicador_id'].unique()
                #raise error if there are missing ids ie nulls
                raise ValueError(f"Warning: Could not map the following 'indicador_id' values to 'id_mercado': {list(missing_ids)}")
        else:
            raise ValueError("Error: 'indicador_id' column not found. Cannot map to 'id_mercado'.")
        
        return df

    def _handle_granularity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert hourly data to 15-minute intervals and combine with existing data if mixed granularities exist.

        This method checks for the presence of the 'granularidad' column in the DataFrame. 
        If the column exists, it separates the data into hourly and 15-minute intervals, 
        converting the hourly data to 15-minute intervals using a utility function. 
        If the 'granularidad' column is missing, it assumes uniform granularity.

        Args:
            df (pd.DataFrame): The input DataFrame containing the data to be processed.

        Returns:
            pd.DataFrame: The modified DataFrame with combined granularity data.
        """
        if 'granularidad' in df.columns:
            is_hourly_mask = df['granularidad'] == 'Hora'
            df_hourly = df[is_hourly_mask].copy()  # Extract hourly data
            df_15min = df[~is_hourly_mask].copy()  # Extract 15-minute data
            df_hourly_converted = pd.DataFrame()  # Initialize an empty DataFrame for converted data

            if not df_hourly.empty:
                print(f"Found {len(df_hourly)} rows with 'Hora' granularity out of {len(df)}. Converting to 15min...")
                df_hourly_converted = DateUtilsETL.convert_hourly_to_15min(df_hourly)
                print(f"Conversion resulted in {len(df_hourly_converted)} rows.")
            else: #print message if no hourly data found
                print(f"No rows with 'Hora' granularity found out of {len(df)}. Keeping existing {len(df_15min)} rows.")

            df = pd.concat([df_15min, df_hourly_converted], ignore_index=True)
            print(f"DataFrame now has {len(df)} rows after handling granularities.")
        else:
            # If 'granularidad' column is missing, assume uniform granularity
            print("Warning: 'granularidad' column not found. Assuming uniform granularity.")
        
        return df

    def _select_and_finalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select final columns, sort the DataFrame, set the index, and reorder columns.

        This method filters the DataFrame to keep only the necessary columns, sorts the data by 
        'datetime_utc', and resets the index. If no data remains after filtering, it returns 
        an empty DataFrame with the expected structure.

        Args:
            df (pd.DataFrame): The input DataFrame to be processed.

        Returns:
            pd.DataFrame: The final DataFrame with selected columns and sorted data.
        """
        final_cols = ['id_mercado', 'datetime_utc', 'precio']  # Define the final columns to keep
        cols_to_keep = [col for col in final_cols if col in df.columns]  # Filter columns to keep

        if not df.empty and cols_to_keep:
            df_final = df[cols_to_keep].copy()  # Create a copy of the DataFrame with selected columns
            df_final = df_final.sort_values(by='datetime_utc').reset_index(drop=True)  # Sort by datetime
            df_final.index.name = 'id'  # Set the index name
            final_ordered_cols = [col for col in final_cols if col in df_final.columns]  # Order final columns
            df_final = df_final[final_ordered_cols]  # Reorder the DataFrame
            return df_final
        else:
            # Return an empty DataFrame if no data or no columns to keep
            print("Warning: No data after filtering or required columns missing. Returning empty DataFrame.")
            empty_df = pd.DataFrame(columns=['id_mercado', 'datetime_utc', 'precio'])  # Create empty DataFrame
            empty_df.index.name = 'id'  # Set the index name
            return empty_df

    def _validate_data(self, df: pd.DataFrame, type: str) -> pd.DataFrame:
        """
        Validate the final DataFrame structure and content.

        This method checks if the DataFrame is not empty before performing validation. 
        If the DataFrame is empty, it skips the validation process.

        Args:
            df (pd.DataFrame): The DataFrame to be validated.

        Returns:
            pd.DataFrame: The original DataFrame after validation.
        """
        if not df.empty:  # Only validate if there's data
            if type == "processed":
                df = self.data_validatior.validate_processed_data(df, data="precios")  # Validate the data
            elif type == "raw":
                df = self.data_validatior.validate_raw_data(df, data="precios")  # Validate the data
        else:
            print("Skipping validation for empty DataFrame.")  # Skip validation for empty DataFrame
        return df

    def transform_price_data(self, df: pd.DataFrame, geo_name: Optional[str] = None) -> pd.DataFrame:
        """
        Apply a sequence of processing steps to transform ESIOS price data.

        Args:
            df (pd.DataFrame): Input DataFrame with raw/semi-processed ESIOS data.
                               Expected columns vary depending on the processing steps but
                               typically include 'indicador_id', 'geo_name', 'value'/'precio',
                               'datetime_utc', 'granularidad'.
            geo_name (Optional[str]): The geographical area name to filter by (e.g., "España").
                                      Defaults to "España" if None.

        Returns:
            pd.DataFrame: Transformed DataFrame with columns ['id_mercado', 'datetime_utc', 'precio']
                          and index named 'id'. Returns an empty DataFrame if processing fails
                          or results in no data.
        """
        if df.empty:
            print("Input DataFrame is empty. Skipping transformation.")
            # Return an empty DataFrame with the expected final structure
            empty_df = pd.DataFrame(columns=['id_mercado', 'datetime_utc', 'precio'])
            empty_df.index.name = 'id'
            return empty_df


        # Define the standard processing pipeline
        # In the future, this could be made configurable
        pipeline = [
            (self._filter_by_geo_name, {'geo_name': self.geo_names_of_interest}),
            (self._validate_data, {"type": "raw"}), #validate raw data
            (self._rename_value_to_precio, {}),
            (self._map_id_mercado, {}),
            (self._handle_granularity, {}),
            (self._select_and_finalize_columns, {}),
            (self._validate_data, {"type": "processed"}) #validate processed data
        ]

        # Execute the pipeline
        try:
            #iterate over pipeline steps
            i = 0
            df_processed = df.copy()
            for step_func, step_kwargs in pipeline:
                i += 1
                print("--------------------------------")
                print(f"Applying step {i} of {len(pipeline)}: {step_func.__name__}...")
                #use  function and necessary kwargs of the corresponding step
                df_processed = step_func(df_processed, **step_kwargs)
                print(f"DataFrame shape after step {step_func.__name__}:")
                print(f"Rows: {df_processed.shape[0]} Columns: {df_processed.shape[1]}")
                print("--------------------------------")

                #if the dataframe is empty and the step is not the final validation step, raise an error
                if df_processed.empty and step_func.__name__ != '_validate_final_data':
                    
                    print(f"DataFrame became empty after step: {step_func.__name__}. Stopping pipeline.")
                    raise ValueError(f"Error: DataFrame became empty after step: {step_func.__name__}. Stopping pipeline.")

            return df_processed

        except ValueError as e:
            # Return an empty DataFrame with the expected final structure on error
            empty_df = pd.DataFrame(columns=['id_mercado', 'datetime_utc', 'precio'])
            empty_df.index.name = 'id'
            return empty_df
        
        except Exception as e: # Catch any other unexpected error
            print(f"An unexpected error occurred during transformation: {e}")
            # Return an empty DataFrame with the expected final structure on error
            empty_df = pd.DataFrame(columns=['id_mercado', 'datetime_utc', 'precio'])
            empty_df.index.name = 'id'
            return empty_df
        
        finally:
            print("--------------------------------")
            print("Transformation pipeline ended.")
