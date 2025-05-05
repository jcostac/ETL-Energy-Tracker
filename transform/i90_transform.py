import pandas as pd
from datetime import datetime
import pytz
import pretty_errors
import sys
from pathlib import Path
from typing import Optional, List, Type, Dict
import numpy as np # Add numpy import for np.where
import re

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utilidades.etl_date_utils import DateUtilsETL
from utilidades.db_utils import DatabaseUtils # If needed for configs or validation
from utilidades.storage_file_utils import RawFileUtils, ProcessedFileUtils
from transform._procesador_i90 import I90Processor # Import the new processor
from configs.i90_config import (
        I90Config, # Base class
        DiarioConfig, SecundariaConfig, TerciariaConfig, RRConfig,
        CurtailmentConfig, P48Config, IndisponibilidadesConfig, RestriccionesConfig
    )
import traceback


class TransformadorI90:
    def __init__(self):
        """
        Initialize the I90 transformer.
        """
        self.processor = I90Processor()
        self.raw_file_utils = RawFileUtils()
        self.processed_file_utils = ProcessedFileUtils()
        self.date_utils = DateUtilsETL()

        # Define dataset types and transformation modes
        self.dataset_types = ['volumenes_i90', 'precios_i90']
        self.transform_types = ['latest', 'batch', 'single', 'multiple']

        # Map market names (strings) to their actual config classes (Type[I90Config])
        self.market_config_map: Dict[str, Type[I90Config]] = {
            'diario': DiarioConfig,
            'secundaria': SecundariaConfig,
            'terciaria': TerciariaConfig,
            'rr': RRConfig,
            'curtailment': CurtailmentConfig,
            'p48': P48Config,
            'indisponibilidades': IndisponibilidadesConfig,
            'restricciones': RestriccionesConfig,
        }

        # Set markets as class attributes
        self.i90_volumenes_markets = self._compute_volumenes_markets()
        self.i90_precios_markets = self._compute_precios_markets()

    def _compute_volumenes_markets(self):
        """
        Computes the markets that have volumenes_sheets.
        """
        markets = []
        for config_cls in I90Config.__subclasses__():
            config = config_cls()
            market_name = config_cls.__name__.replace('Config', '').lower()
            if hasattr(config, 'volumenes_sheets') and config.volumenes_sheets and any(config.volumenes_sheets):
                markets.append(market_name)
        return markets

    def _compute_precios_markets(self):
        """
        Computes the markets that have precios_sheets.
        """
        markets = []
        for config_cls in I90Config.__subclasses__():
            config = config_cls()
            market_name = config_cls.__name__.replace('Config', '').lower()
            if hasattr(config, 'precios_sheets') and config.precios_sheets and any(config.precios_sheets):
                markets.append(market_name)
        return markets

    def get_config_for_market(self, mercado: str) -> I90Config:
        """Retrieves and instantiates the config object for a given market name."""
        config_class = self.market_config_map.get(mercado)
        if not config_class:
            # Check if it's a known market but just missing from the map
            all_known_i90_market_dl_names = self.i90_volumenes_markets + self.i90_precios_markets
            if mercado in all_known_i90_market_dl_names:
                 raise ValueError(f"Configuration class for known market '{mercado}' is missing from market_config_map.")
            else:
                 raise ValueError(f"Unknown market name: '{mercado}'. No configuration class found.")

        # Instantiate the config class
        try:
            return config_class()
        except Exception as e:
             # Catch errors during config instantiation (e.g., DB connection issues in __init__)
             print(f"Error instantiating config class {config_class.__name__} for market {mercado}: {e}")
             raise # Reraise the exception

    def transform_data_for_all_markets(self, start_date: Optional[str] = None, end_date: Optional[str] = None,
                                         mercados: Optional[List[str]] = None,
                                         dataset_types: Optional[List[str]] = None,
                                         transform_type: str = 'latest') -> dict:
        """
        Transforms data for specified markets and dataset types based on the transform_type.
        Returns: Dict[market, DataFrame or list]
        """
        if transform_type not in self.transform_types:
            raise ValueError(f"Invalid transform type: {transform_type}. Must be one of: {self.transform_types}")
        if transform_type in ['single', 'multiple'] and not start_date:
            raise ValueError(f"start_date is required for transform_type '{transform_type}'")
        if transform_type == 'multiple' and not end_date:
            raise ValueError(f"end_date is required for transform_type 'multiple'")

        if dataset_types is None:
            dataset_types = self.dataset_types
        else:
            invalid_types = [dt for dt in dataset_types if dt not in self.dataset_types]
            if invalid_types:
                raise ValueError(f"Invalid dataset_type(s) provided: {invalid_types}. Must be in {self.dataset_types}")

        # --- FLAT results dict ---
        results = {}

        print(f"\n===== STARTING TRANSFORMATION RUN (Mode: {transform_type.upper()}) =====")
        print(f"Dataset types to process: {', '.join(dataset_types)}")
        if start_date: print(f"Start Date: {start_date}")
        if end_date: print(f"End Date: {end_date}")
        print("="*60)

        # Only process one dataset_type at a time for flat output
        if len(dataset_types) > 1:
            print("Warning: Only the first dataset_type will be processed for flat output to match ESIOS pipeline.")
        dataset_type = dataset_types[0]

        # Determine relevant markets for this dataset type
        if mercados is None:
            relevant_markets = self.i90_volumenes_markets if dataset_type == 'volumenes_i90' else self.i90_precios_markets if dataset_type == 'precios_i90' else []
        else:
            known_markets_for_type = self.i90_volumenes_markets if dataset_type == 'volumenes_i90' else self.i90_precios_markets
            invalid_markets = [m for m in mercados if m not in known_markets_for_type]
            if invalid_markets:
                print(f"Warning: Market(s) {invalid_markets} are not typically associated with dataset type {dataset_type} or are unknown. Skipping them for this type.")
            relevant_markets = [m for m in mercados if m in known_markets_for_type]

        if not relevant_markets:
            print(f"No relevant markets specified or configured for dataset type: {dataset_type}")
            return results

        for mercado in relevant_markets:
            print(f"\n-- Market: {mercado} --")
            try:
                if transform_type == 'batch':
                    market_result = self._process_batch_mode(mercado, dataset_type)
                elif transform_type == 'single':
                    market_result = self._process_single_day(mercado, dataset_type, start_date)
                elif transform_type == 'latest':
                    market_result = self._process_latest_day(mercado, dataset_type)
                elif transform_type == 'multiple':
                    market_result = self._process_date_range(mercado, dataset_type, start_date, end_date)
                results[mercado] = market_result
            except Exception as e:
                results[mercado] = None
                print(f"❌ Failed to transform {dataset_type} for market {mercado} ({transform_type}): {e}")
                print(traceback.format_exc())
                continue

        print(f"\n===== TRANSFORMATION RUN FINISHED (Mode: {transform_type.upper()}) =====")
        # Optionally print summary of results
        print("Summary:")
        for market, result in results.items():
            status = "Failed/No Data"
            if isinstance(result, pd.DataFrame):
                if not result.empty:
                    status = f"Success ({len(result)} records)"
                else:
                    status = "Success (Empty DF)"
            elif isinstance(result, list):
                success_count = sum(1 for df in result if isinstance(df, pd.DataFrame) and not df.empty)
                total_items = len(result)
                status = f"Batch Success ({success_count}/{total_items} periods processed)"
            print(f"- {market}: {status}")
        print("="*60)

        return results

    def _transform_data(self, raw_df: pd.DataFrame, mercado: str, dataset_type: str) -> pd.DataFrame:
        """Transforms data using the processor and returns the processed DataFrame."""
        if raw_df.empty:
            print(f"Skipping transformation for {mercado} - {dataset_type}: Input DataFrame is empty.")
            return pd.DataFrame()

        try:
            market_config = self.get_config_for_market(mercado)
            print(f"Raw data loaded ({len(raw_df)} rows). Starting transformation for {mercado} - {dataset_type}...")
            processed_df = self.processor.transform_volumenes_or_precios_i90(raw_df, market_config, dataset_type)
            if processed_df is None or processed_df.empty:
                print(f"Transformation resulted in empty or None DataFrame for {mercado} - {dataset_type}.")
                return pd.DataFrame()
            
            print(f"Processed data:")
            print(processed_df.head())
            return processed_df
        except Exception as e:
            print(f"Error during transformation for {mercado} - {dataset_type}: {e}")
            print("Raw DF info before failed transform:")
            print(raw_df.info())
            return pd.DataFrame()

    def _process_df_based_on_transform_type(self, raw_df: pd.DataFrame, transform_type: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Filters the raw DataFrame based on the transform type and date range.
        Relies on 'datetime_utc' column being present and correctly parsed.
        """
        if raw_df.empty:
            return raw_df

        # --- Datetime Column Handling ---
        # The I90Processor._standardize_datetime should create 'datetime_utc', but do do this, we first hacve to filter by date, hence the necessesity of a fecha column
        # We rely on the "fecha" column for filtering by date to succeed.
        # or lack the column, leading to errors here or empty results.
        if "fecha" not in raw_df.columns:
            print("Error: 'fecha' column missing in DataFrame passed to _process_df_based_on_transform_type. Cannot apply date filters.")
            # Depending on mode, either return empty or the original df if batch
            return pd.DataFrame() 
        else:
            # Ensure fecha column is datetime type for filtering
            if not pd.api.types.is_datetime64_any_dtype(raw_df['fecha']):
                try:
                    raw_df['fecha'] = pd.to_datetime(raw_df['fecha'])
                except Exception as e:
                    print(f"Error converting 'fecha' in _process_df_based_on_transform_type: {e}. Returning empty DataFrame.")
                    return pd.DataFrame()
            

        # --- Filtering Logic ---
        try:
            if transform_type == 'latest':
                if raw_df.empty: return raw_df
                # Find the max date robustly
                last_day = raw_df['fecha'].dropna().dt.date.max()
                if pd.isna(last_day):
                    print("Warning: Could not determine the latest day (max date is NaT). Returning empty DataFrame for 'latest' mode.")
                    return pd.DataFrame()
                print(f"Filtering for latest mode: {last_day}")
               #return filtered df on last day
                return raw_df[raw_df['fecha'].dt.date == last_day].copy()

            elif transform_type == 'batch':
                unique_days = raw_df['fecha'].dropna().dt.date.nunique()
                print(f"Processing in batch mode with {unique_days} unique days")
                return raw_df # Process the entire dataframe

            elif transform_type == 'single':
                target_date = pd.to_datetime(start_date).date()
                print(f"Filtering for single mode: {target_date}")
                return raw_df[raw_df['fecha'].dt.date == target_date].copy()

            elif transform_type == 'multiple':
                start_dt = pd.to_datetime(start_date).date()
                end_dt = pd.to_datetime(end_date).date()
                if start_dt > end_dt: raise ValueError("Start date cannot be after end date.")
                print(f"Filtering for multiple mode: {start_dt} to {end_dt}")
                return raw_df[(raw_df['fecha'].dt.date >= start_dt) & (raw_df['fecha'].dt.date <= end_dt)].copy()

            else:
                # This case should technically be caught by the public method check
                raise ValueError(f"Invalid transform type for filtering: {transform_type}")

        except Exception as e:
             print(f"Error during date filtering ({transform_type} mode): {e}")
             return pd.DataFrame() # Return empty on filtering error

    def _process_batch_mode(self, mercado: str, dataset_type: str):
        """Process all available raw data for a market/dataset_type."""
        processed_dfs = []
        print(f"Starting BATCH transformation for {mercado} - {dataset_type}")
        try:
            # Use RawFileUtils to find all year/month combinations
            # Pass dataset_type if your raw storage differentiates files by it
            years = self.raw_file_utils.get_raw_folder_list(mercado, dataset_type=dataset_type)
            if not years: print(f"No years found for {mercado}/{dataset_type}. Skipping batch."); return processed_dfs

            for year in years:
                months = self.raw_file_utils.get_raw_folder_list(mercado, year, dataset_type=dataset_type)
                if not months: print(f"No months found for {mercado}/{dataset_type}/{year}. Skipping year."); continue

                for month in months:
                    print(f"Processing {mercado}/{dataset_type} for {year}-{month:02d}")
                    try:
                        raw_file_path = self.raw_file_utils.get_raw_file_path(year, month, mercado)
                        dataset_type = self._extract_dataset_type_from_filename(raw_file_path)
                        raw_df = self.raw_file_utils.read_raw_file(year, month, dataset_type, mercado)
                        processed_df = self._transform_data(raw_df, mercado, dataset_type)
                        if processed_df is not None and not processed_df.empty:
                            processed_dfs.append(processed_df)
                    except FileNotFoundError:
                        print(f"Raw file not found for {year}-{month:02d}. Skipping.")
                    except Exception as e:
                        print(f"Error processing file {year}-{month:02d} for {mercado}/{dataset_type}: {e}")
                        continue # Continue with next month/year
        except Exception as e:
            print(f"Error during batch processing setup for {mercado}/{dataset_type}: {e}")
        return processed_dfs

    def _process_single_day(self, mercado: str, dataset_type: str, date: str):
        """Process a single day's data."""
        print(f"Starting SINGLE transformation for {mercado} - {dataset_type} on {date}")
        try:
            target_date = pd.to_datetime(date)
            target_year = target_date.year
            target_month = target_date.month

            # Get the list of files for the target year and month
            files = self.raw_file_utils.get_raw_file_list(mercado, target_year, target_month)
            if not files:
                print(f"No files found for {mercado}/{dataset_type} for {target_year}-{target_month:02d}. Skipping.")
                return

            #read the files in the list of raw files
            raw_df = self.raw_file_utils.read_raw_file(target_year, target_month, dataset_type, mercado)

            # Filter for the specific day
            filtered_df = self._process_df_based_on_transform_type(raw_df, 'single', start_date=date)

            if filtered_df.empty:
                print(f"No data found for {mercado}/{dataset_type} on {date} within file {target_year}-{target_month:02d}.")
                return pd.DataFrame()

            processed_df = self._transform_data(filtered_df, mercado, dataset_type)
            return processed_df

        except FileNotFoundError:
             # It's possible the folder exists but not the specific file (e.g., parquet file name)
             print(f"Raw data file not found for {mercado}/{dataset_type} for {target_year}-{target_month:02d}.")
        except Exception as e:
            print(f"Error during single day processing for {mercado}/{dataset_type} on {date}: {e}")

    def _process_latest_day(self, mercado: str, dataset_type: str):
        """Process the latest day available in the raw data for the given dataset_type."""
        print(f"Starting LATEST transformation for {mercado} - {dataset_type}")
        try:
            # Find all years (descending)
            years = sorted(self.raw_file_utils.get_raw_folder_list(mercado=mercado), reverse=True)
            if not years:
                print(f"No data years found for {mercado}. Skipping latest.")
                return

            found = False 
            for year in years:
                # Find all months (descending) for this year (latest month first) -> this is useful bc i90 has a time lag o f90 days and latest raw folder for a market might not contain i90 data. 
                months = sorted(self.raw_file_utils.get_raw_folder_list(mercado=mercado, year=year), reverse=True)
                for month in months:
                    # Check if the file for this dataset_type exists
                    files = self.raw_file_utils.get_raw_file_list(mercado, year, month)
                    if not files:
                        continue
                    # Look for a file that matches the dataset_type
                    matching_files = [f for f in files if dataset_type in str(f)]
                    if matching_files:
                        print(f"Identified latest available file for {mercado}/{dataset_type}: {year}-{month:02d}")
                        # Read the file
                        raw_df = self.raw_file_utils.read_raw_file(year, month, dataset_type, mercado)
                        # Filter for the latest day within that file
                        filtered_df = self._process_df_based_on_transform_type(raw_df, 'latest')
                        if filtered_df.empty:
                            print(f"No data found for the latest day within file {year}-{month:02d}.")
                            return pd.DataFrame()
                        processed_df = self._transform_data(filtered_df, mercado, dataset_type)
                        found = True
                        return processed_df
                if found:
                    break
            if not found:
                print(f"No raw file found for {mercado}/{dataset_type} in any available year/month.")
        except Exception as e:
            print(f"Error during latest day processing for {mercado}/{dataset_type}: {e}")

    def _process_date_range(self, mercado: str, dataset_type: str, start_date: str, end_date: str):
        """Process a range of days, reading multiple monthly files if needed."""
        print(f"Starting MULTIPLE transformation for {mercado} - {dataset_type} from {start_date} to {end_date}")
        try:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)

            if start_dt > end_dt: raise ValueError("Start date cannot be after end date.")

            # Determine the range of year-month combinations needed
            # Generate all days in the range first
            all_days_in_range = pd.date_range(start_dt, end_dt, freq='D')
            if all_days_in_range.empty:
                print("Warning: Date range resulted in zero days. Nothing to process.")
                return
            
            # Get unique year-month tuples IE: [(2025, 1), (2025, 2), (2025, 3), (2025, 4)]
            year_months = sorted(list(set([(d.year, d.month) for d in all_days_in_range])))

            all_raw_dfs = []
            print(f"Reading files for year-months: {year_months}")
            for year, month in year_months:
                try:
                    # Pass dataset_type to reader if needed
                    raw_df = self.raw_file_utils.read_raw_file(year, month, dataset_type, mercado)
                    # Append the dataframe read from the file to the list
                    if raw_df is not None and not raw_df.empty:
                        all_raw_dfs.append(raw_df)
                    else:
                         print(f"Warning: No data returned from reading {mercado}/{dataset_type} for {year}-{month:02d}.")

                except FileNotFoundError:
                    print(f"Warning: Raw file not found for {mercado}/{dataset_type} for {year}-{month:02d}. Skipping.")
                except Exception as e:
                    print(f"Error reading or processing raw file for {year}-{month:02d}: {e}")
                    # Consider if you want to continue or stop on error
                    continue # Continue processing other months

            if not all_raw_dfs:
                 print(f"No raw data found for the specified date range {start_date} to {end_date}.")
                 return

            # Concatenate filtered monthly dataframes
            combined_raw_df = pd.concat(all_raw_dfs, ignore_index=True)

            print(f"Combined raw data ({len(combined_raw_df)} rows). Applying date range filtering.")
            # The final filtering step ensures the exact date range is met after concatenation.
            filtered_df = self._process_df_based_on_transform_type(combined_raw_df, 'multiple', start_date=start_date, end_date=end_date)


            if filtered_df.empty:
                print(f"No data found for {mercado}/{dataset_type} between {start_date} and {end_date} after final filtering.")
                return pd.DataFrame()

            print(f"Filtered data has {len(filtered_df)} rows. Proceeding with transformation...")
            processed_df = self._transform_data(filtered_df, mercado, dataset_type)
            return processed_df

        except ValueError as ve:
             print(f"Configuration or Value Error during multiple transform: {ve}")
        except Exception as e:
            print(f"An unexpected error occurred during multiple transform: {e}")
            import traceback
            print(traceback.format_exc())

    def _extract_dataset_type_from_filename(self, filename: str) -> str:
        """
        Extracts the dataset type (e.g., 'volumenes_i90' or 'precios_i90') from the raw file name.
        """
        match = re.search(r'(volumenes_i90|precios_i90)', filename)
        if match:
            return match.group(1)
        raise ValueError(f"Could not determine dataset_type from filename: {filename}")
    