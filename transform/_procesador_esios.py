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
        self.indicators_to_filter_by_geo_name = [600, 612, 613, 614, 615, 616, 617, 618, 1782]
        self.geo_names_in_raw_data = []# List of unique geo_name values in raw data
        self.geo_names_of_interest = ["España"]# List of unique geo_name values of interest


    def _standardize_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and check price values, with optional data quality warnings.
        
        Args:
            df (pd.DataFrame): Input DataFrame with price data
            
        Returns:
            pd.DataFrame: DataFrame with validated prices
        """
        print("\n📈 PRICE ANALYSIS")
        print("-"*30)
        
        # Round prices
        df['precio'] = df['precio'].round(2)
        
        # Validation checks
        if df['precio'].isnull().any():
            null_count = df['precio'].isnull().sum()
            print(f"⚠️  Found {null_count} null price values")
        
        # Check negative prices
        negative_prices = df[df['precio'] < 0]
        if not negative_prices.empty:
            print(f"ℹ️  Negative prices: {len(negative_prices)} records")
        
        # Check extreme prices
        mean_price = df['precio'].mean()
        std_price = df['precio'].std()
        extreme_prices = df[abs(df['precio'] - mean_price) > 3 * std_price]
        if not extreme_prices.empty:
            print(f"⚠️  Extreme prices: {len(extreme_prices)} records")
            print(f"   Range: {df['precio'].min():.2f} to {df['precio'].max():.2f}")
        
        # Check zero prices
        zero_prices = df[df['precio'] == 0]
        if not zero_prices.empty:
            print(f"ℹ️  Zero prices: {len(zero_prices)} records")
        
        # Statistics
        print("\n📊 Price Statistics")
        print(f"   Mean: {mean_price:.2f}")
        print(f"   Median: {df['precio'].median():.2f}")
        print(f"   Std Dev: {std_price:.2f}")
        print("-"*30)
        
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

    def _filter_by_geo_name(self, df: pd.DataFrame, geo_name: list[str]) -> pd.DataFrame:
        """
        Filter data based on geo_name for specific indicators.
        """
        self.unique_geo_names(df)
        
        # Convert indicators to filter into string format for comparison
        indicators_to_filter_str = [str(i) for i in self.indicators_to_filter_by_geo_name]
        print(f"Indicators to filter by geo_name: {indicators_to_filter_str}")
        
        # Ensure 'indicador_id' is of string type
        if 'indicador_id' in df.columns:
            df['indicador_id'] = df['indicador_id'].astype(str)
            
            # Create a mask for rows that need geo_name filtering
            # it creates a boolean mask needs_geo_filter that marks which rows have indicator IDs that need geographic filtering
            needs_geo_filter = df['indicador_id'].isin(indicators_to_filter_str) 
            print(f"Needs geo_name filter: {needs_geo_filter.any()}")
            
            if needs_geo_filter.any(): #if any rows need filtering

                #~needs_geo_filter: Keeps ALL rows that don't need geographic filtering (their indicator IDs are not in the list)
                #needs_geo_filter & df['geo_name'].isin(self.geo_names_of_interest): Keeps only the rows that need filtering AND have the specified geo_name
                mask = (~needs_geo_filter) | (needs_geo_filter & df['geo_name'].isin(geo_name))

                #mask: Combines the two conditions: keep all non-filtered rows OR keep only the filtered rows that match the specified geo_name
                df_filtered = df[mask]

                print(f"Unique geo_name values in filtered dataframe: {df_filtered['geo_name'].unique()}")
                print(f"Filtered from {len(df)} to {len(df_filtered)} rows")
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
        Handles mixed time granularities by converting hourly records to 15-minute intervals and combining them with existing 15-minute data.
        
        If the DataFrame contains a 'granularidad' column, separates hourly and 15-minute records, converts hourly data to 15-minute intervals, and merges the results. If the column is absent, assumes all data shares the same granularity and returns the DataFrame unchanged.
        
        Returns:
            pd.DataFrame: DataFrame with all records in 15-minute intervals if conversion was needed, otherwise the original DataFrame.
        """
        print("\n⏰ HANDLING TIME GRANULARITY")
        print("-"*30)

        if 'granularidad' not in df.columns:
            print("ℹ️  No granularity column found")
            print("   Assuming uniform granularity")
            return df

        # Split data by granularity
        is_hourly_mask = df['granularidad'] == 'Hora'
        df_hourly = df[is_hourly_mask].copy()
        df_15min = df[~is_hourly_mask].copy()
        
        print("📊 Initial Data:")
        print(f"   Total records: {len(df)}")
        print(f"   Hourly records: {len(df_hourly)}")
        print(f"   15-min records: {len(df_15min)}")

        # Convert hourly data if present
        if not df_hourly.empty:
            print("\n🔄 Converting hourly to 15-min...")
            df_hourly_converted = DateUtilsETL.convert_hourly_to_15min(df_hourly, "precios")
            print(f"✅ Conversion complete: {len(df_hourly)} → {len(df_hourly_converted)}")
            
            # Combine data
            df = pd.concat([df_15min, df_hourly_converted], ignore_index=True)
            print(f"\n📊 Final record count: {len(df)}")
        else:
            print("\nℹ️  No hourly data to convert")
            print(f"📊 Keeping {len(df_15min)} 15-min records")
        
        print("-"*30)
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
        print("\n🔍 DATA VALIDATION")
        print("-"*30)
        print(f"Type: {type.upper()}")
        
        if df.empty:
            print("⚠️  Empty DataFrame - Skipping validation")
            return df
        
        try:
            if type == "processed":
                print("Validating processed data structure...")
                df = self.data_validatior.validate_processed_data(df, validation_schema_type="precios")
                print("✅ Processed data validation passed")
            elif type == "raw":
                print("Validating raw data structure...")
                df = self.data_validatior.validate_raw_data(df, validation_schema_type="precios")
                print("✅ Raw data validation passed")
        except Exception as e:
            print(f"❌ Validation failed: {str(e)}")
            raise
        
        print("-"*30)
        return df

    def transform_price_data(self, df: pd.DataFrame, geo_name: Optional[str] = None) -> pd.DataFrame:
        """
        Apply a sequence of processing steps to transform ESIOS price data.
        """
        print("\n" + "="*80)
        print("🔄 STARTING ESIOS PRICE TRANSFORMATION")
        print("="*80)

        if df.empty:
            print("\n❌ EMPTY INPUT")
            print("Input DataFrame is empty. Skipping transformation.")
            empty_df = pd.DataFrame(columns=['id_mercado', 'datetime_utc', 'precio'])
            empty_df.index.name = 'id'
            return empty_df
        
        if geo_name is None:
            geo_name = self.geo_names_of_interest

        # Define the standard processing pipeline
        pipeline = [
            (self._filter_by_geo_name, {'geo_name': geo_name}),
            (self._validate_data, {"type": "raw"}),
            (self._rename_value_to_precio, {}),
            (self._map_id_mercado, {}),
            (self._standardize_prices, {}),
            (self._handle_granularity, {}),
            (self._select_and_finalize_columns, {}),
            (self._validate_data, {"type": "processed"})
        ]

        try:
            df_processed = df.copy()
            total_steps = len(pipeline)

            for i, (step_func, step_kwargs) in enumerate(pipeline, 1):
                print("\n" + "-"*50)
                print(f"📍 STEP {i}/{total_steps}: {step_func.__name__.replace('_', ' ').title()}")
                print("-"*50)
                
                df_processed = step_func(df_processed, **step_kwargs)
                
                print(f"\n📊 Data Status:")
                print(f"   Rows: {df_processed.shape[0]}")
                print(f"   Columns: {df_processed.shape[1]}")

                if df_processed.empty and step_func.__name__ != '_validate_final_data':
                    print("\n❌ PIPELINE HALTED")
                    print(f"DataFrame became empty after step: {step_func.__name__}")
                    raise ValueError(f"DataFrame empty after: {step_func.__name__}")

            print("\n✅ TRANSFORMATION COMPLETE")
            print(f"Final shape: {df_processed.shape}")
            print("="*80 + "\n")
            return df_processed

        except ValueError as e:
            print("\n❌ VALIDATION ERROR")
            print(f"Error: {str(e)}")
            print("="*80 + "\n")
            raise

        
        except Exception as e:
            print("\n❌ UNEXPECTED ERROR")
            print(f"Error: {str(e)}")
            print("="*80 + "\n")
            raise
