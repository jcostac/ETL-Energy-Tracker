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
        markets = []
        for config_cls in I90Config.__subclasses__():
            config = config_cls()
            market_name = config_cls.__name__.replace('Config', '').lower()
            if hasattr(config, 'volumenes_sheets') and config.volumenes_sheets and any(config.volumenes_sheets):
                markets.append(market_name)
        return markets

    def _compute_precios_markets(self):
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
                                         transform_type: str = 'latest') -> None:
        """
        Transforms data for specified markets and dataset types based on the transform_type.

        Args:
            start_date: Start date for 'single' or 'multiple' modes (YYYY-MM-DD).
            end_date: End date for 'multiple' mode (YYYY-MM-DD).
            mercados: List of market names to process. If None, processes all relevant markets.
            dataset_types: List of dataset types ('volumenes_i90', 'precios_i90'). If None, processes all.
            transform_type: One of 'latest', 'batch', 'single', or 'multiple'.
        """
        if transform_type not in self.transform_types:
            raise ValueError(f"Invalid transform type: {transform_type}. Must be one of: {self.transform_types}")
        if transform_type in ['single', 'multiple'] and not start_date:
             raise ValueError(f"start_date is required for transform_type '{transform_type}'")
        if transform_type == 'multiple' and not end_date:
             raise ValueError(f"end_date is required for transform_type 'multiple'")

        if dataset_types is None:
            dataset_types = self.dataset_types
        else: # Validate provided dataset types
            invalid_types = [dt for dt in dataset_types if dt not in self.dataset_types]
            if invalid_types:
                raise ValueError(f"Invalid dataset_type(s) provided: {invalid_types}. Must be in {self.dataset_types}")


        for dataset_type in dataset_types:
            print(f"\n--- Processing Dataset Type: {dataset_type} ---")

            # Determine relevant markets for this dataset type
            if mercados is None:
                relevant_markets = self.i90_volumenes_markets if dataset_type == 'volumenes_i90' else self.i90_precios_markets if dataset_type == 'precios_i90' else []
            else:
                # Validate provided market names against known markets for the dataset type
                known_markets_for_type = self.i90_volumenes_markets if dataset_type == 'volumenes_i90' else self.i90_precios_markets
                invalid_markets = [m for m in mercados if m not in known_markets_for_type]
                if invalid_markets:
                    print(f"Warning: Market(s) {invalid_markets} are not typically associated with dataset type {dataset_type} or are unknown. Skipping them for this type.")
                relevant_markets = [m for m in mercados if m in known_markets_for_type]


            if not relevant_markets:
                 print(f"No relevant markets specified or configured for dataset type: {dataset_type}")
                 continue

            for mercado in relevant_markets:
                print(f"\n-- Market: {mercado} --")
                try:
                    if transform_type == 'batch':
                        self._transform_batch(mercado, dataset_type)
                    elif transform_type == 'single':
                        self._transform_single(mercado, dataset_type, start_date)
                    elif transform_type == 'latest':
                        self._transform_latest(mercado, dataset_type)
                    elif transform_type == 'multiple':
                        self._transform_multiple(mercado, dataset_type, start_date, end_date)
                except Exception as e:
                     # Catch errors here to allow processing of other markets/types
                     print(f"❌ Failed to transform {dataset_type} for market {mercado} ({transform_type}): {e}")
                     print(traceback.format_exc()) # Print stack trace for debugging
                     continue # Continue with the next market/dataset type

        print(f"\n✅ Successfully completed transformation run for specified markets and dataset types. ✅")


    def _transform_and_save(self, raw_df: pd.DataFrame, mercado: str, dataset_type: str):
        """Transforms data using the processor, adjusts schema for secundaria/date, and saves the result."""
        if raw_df.empty:
             print(f"Skipping transformation for {mercado} - {dataset_type}: Input DataFrame is empty.")
             return

        processed_df = pd.DataFrame() # Initialize empty df for error cases
        try:
            # 1. Get Market Configuration (Instantiated object)
            market_config = self.get_config_for_market(mercado)

            # 2. Transform Data using I90Processor
            print(f"Raw data loaded ({len(raw_df)} rows). Starting transformation for {mercado} - {dataset_type}...")
            # Processor now outputs 'up' column for unit identifier in volumenes
            processed_df = self.processor.transform_data(raw_df, market_config, dataset_type)

            if processed_df is None or processed_df.empty: # Check for None as well
                print(f"Transformation resulted in empty or None DataFrame for {mercado} - {dataset_type}. Nothing to save.")
                return

            # --- Schema Adjustment for Secundaria Market based on Date ---
            if mercado == 'secundaria' and dataset_type == 'volumenes_i90' and 'up' in processed_df.columns:
                print("Applying date-based schema adjustment for secundaria market...")
                threshold_date = pd.Timestamp('2024-11-20', tz='UTC')
                # Ensure datetime column is usable
                if not pd.api.types.is_datetime64_any_dtype(processed_df['datetime_utc']):
                     print("Warning: Cannot apply secundaria date logic, datetime_utc column has wrong type.")
                else:
                    min_date = processed_df['datetime_utc'].min()
                    max_date = processed_df['datetime_utc'].max()

                    # Check if the data chunk contains dates on or after the threshold
                    if max_date >= threshold_date:
                        print(f"Data chunk includes dates on/after {threshold_date.date()}. Creating 'zr' column.")
                        # Create 'zr' column with values from 'up' for dates >= threshold
                        processed_df['zr'] = np.where(
                            processed_df['datetime_utc'] >= threshold_date,
                            processed_df['up'],
                            pd.NA # Use pandas NA for missing values
                        )
                        # Modify 'up' column to keep values only for dates < threshold
                        processed_df['up'] = np.where(
                            processed_df['datetime_utc'] < threshold_date,
                            processed_df['up'],
                            pd.NA # Use pandas NA for missing values
                        )
                        # Ensure 'zr' column type matches 'up' if needed (e.g., string)
                        processed_df['zr'] = processed_df['zr'].astype(processed_df['up'].dtype)

                        print("Columns 'up' and 'zr' now reflect the threshold split.")
                    else:
                         # Data is entirely before the threshold, keep 'up' as is.
                         print(f"Data chunk is entirely before {threshold_date.date()}. Keeping 'up' column.")
            # ---------------------------------------------------------

        except Exception as e:
            print(f"Error during transformation or schema adjustment for {mercado} - {dataset_type}: {e}")
            # Optionally log the raw_df or intermediate processed_df head/info here
            # print("Raw DF info before failed transform/adjustment:")
            # print(raw_df.info())
            return # Stop processing

        try:
            # 3. Save Processed Data (with potentially adjusted schema)
            print(f"Final processing complete ({len(processed_df)} rows). Saving data...")

            # Ensure final columns are ordered logically if 'zr' was added
            final_columns_order = ['datetime_utc', 'id_mercado']
            if 'up' in processed_df.columns: final_columns_order.append('up')
            if 'zr' in processed_df.columns: final_columns_order.append('zr')
            # Determine value column name
            value_col = 'volumen' if dataset_type == 'volumenes_i90' else 'precio'
            if value_col in processed_df.columns: final_columns_order.append(value_col)
            else: print(f"Warning: Expected value column '{value_col}' not found before saving.")

            # Reorder columns for saving
            processed_df = processed_df[final_columns_order]

            # Saving logic remains the same, ProcessedFileUtils handles the actual write
            self.processed_file_utils.write_processed_parquet(processed_df, mercado) # Might need adjustment based on ProcessedFileUtils capabilities

            date_range_str = "N/A"
            if 'datetime_utc' in processed_df.columns and not processed_df.empty:
                 try:
                     min_date_save = processed_df['datetime_utc'].min().date()
                     max_date_save = processed_df['datetime_utc'].max().date()
                     date_range_str = f"{min_date_save} to {max_date_save}"
                 except Exception as date_err:
                      print(f"Could not determine date range from final data: {date_err}")

            print(f"Successfully saved processed {mercado} ({dataset_type}) data for date range: {date_range_str}.")

        except Exception as e:
            print(f"Error saving processed data for {mercado} - {dataset_type}: {e}")
            # Optionally log processed_df head/info here
            # print("Processed DF info before failed save:")
            # print(processed_df.info())
            return


    # --- Data Loading and Processing Logic (similar to TransformadorESIOS) ---

    def _process_df_based_on_transform_type(self, raw_df: pd.DataFrame, transform_type: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Filters the raw DataFrame based on the transform type and date range.
        Relies on 'datetime_utc' column being present and correctly parsed.
        """
        if raw_df.empty:
            return raw_df

        # --- Datetime Column Handling ---
        # The I90Processor._standardize_datetime should create 'datetime_utc'.
        # We rely on that step having succeeded. If it failed, the df might be empty
        # or lack the column, leading to errors here or empty results.
        if 'datetime_utc' not in raw_df.columns:
            print("Error: 'datetime_utc' column missing in DataFrame passed to _process_df_based_on_transform_type. Cannot apply date filters.")
            # Depending on mode, either return empty or the original df if batch
            return pd.DataFrame() if transform_type != 'batch' else raw_df
        else:
            # Ensure it's datetime type (might be redundant if processor guarantees it)
             if not pd.api.types.is_datetime64_any_dtype(raw_df['datetime_utc']):
                 try:
                     raw_df['datetime_utc'] = pd.to_datetime(raw_df['datetime_utc'], utc=True)
                 except Exception as e:
                     print(f"Error converting 'datetime_utc' in _process_df_based_on_transform_type: {e}. Returning empty DataFrame.")
                     return pd.DataFrame()

        # --- Filtering Logic ---
        try:
            if transform_type == 'latest':
                if raw_df.empty: return raw_df
                # Find the max date robustly
                last_day = raw_df['datetime_utc'].dropna().dt.date.max()
                if pd.isna(last_day):
                    print("Warning: Could not determine the latest day (max date is NaT). Returning empty DataFrame for 'latest' mode.")
                    return pd.DataFrame()
                print(f"Filtering for latest mode: {last_day}")
                # Ensure comparison works even if source has NaT datetimes
                return raw_df[raw_df['datetime_utc'].dt.date == last_day].copy()

            elif transform_type == 'batch':
                unique_days = raw_df['datetime_utc'].dropna().dt.date.nunique()
                print(f"Processing in batch mode with {unique_days} unique days")
                return raw_df # Process the entire dataframe

            elif transform_type == 'single':
                target_date = pd.to_datetime(start_date).date()
                print(f"Filtering for single mode: {target_date}")
                return raw_df[raw_df['datetime_utc'].dt.date == target_date].copy()

            elif transform_type == 'multiple':
                start_dt = pd.to_datetime(start_date).date()
                end_dt = pd.to_datetime(end_date).date()
                if start_dt > end_dt: raise ValueError("Start date cannot be after end date.")
                print(f"Filtering for multiple mode: {start_dt} to {end_dt}")
                return raw_df[(raw_df['datetime_utc'].dt.date >= start_dt) & (raw_df['datetime_utc'].dt.date <= end_dt)].copy()

            else:
                # This case should technically be caught by the public method check
                raise ValueError(f"Invalid transform type for filtering: {transform_type}")

        except Exception as e:
             print(f"Error during date filtering ({transform_type} mode): {e}")
             return pd.DataFrame() # Return empty on filtering error


    def _transform_batch(self, mercado: str, dataset_type: str):
        """Process all available raw data for a market/dataset_type."""
        print(f"Starting BATCH transformation for {mercado} - {dataset_type}")
        try:
            # Use RawFileUtils to find all year/month combinations
            # Pass dataset_type if your raw storage differentiates files by it
            years = self.raw_file_utils.get_raw_folder_list(mercado, dataset_type=dataset_type)
            if not years: print(f"No years found for {mercado}/{dataset_type}. Skipping batch."); return

            for year in years:
                months = self.raw_file_utils.get_raw_folder_list(mercado, year, dataset_type=dataset_type)
                if not months: print(f"No months found for {mercado}/{dataset_type}/{year}. Skipping year."); continue

                for month in months:
                    print(f"Processing {mercado}/{dataset_type} for {year}-{month:02d}")
                    try:
                        raw_file_path = self.raw_file_utils.get_raw_file_path(year, month, mercado)
                        dataset_type = self._extract_dataset_type_from_filename(raw_file_path)
                        raw_df = self.raw_file_utils.read_raw_file(year, month, dataset_type, mercado)
                        self._transform_and_save(raw_df, mercado, dataset_type)
                    except FileNotFoundError:
                        print(f"Raw file not found for {year}-{month:02d}. Skipping.")
                    except Exception as e:
                        print(f"Error processing file {year}-{month:02d} for {mercado}/{dataset_type}: {e}")
                        continue # Continue with next month/year
        except Exception as e:
            print(f"Error during batch processing setup for {mercado}/{dataset_type}: {e}")


    def _transform_single(self, mercado: str, dataset_type: str, date: str):
        """Process a single day's data."""
        print(f"Starting SINGLE transformation for {mercado} - {dataset_type} on {date}")
        try:
            target_date = pd.to_datetime(date)
            target_year = target_date.year
            target_month = target_date.month

            # Read the specific month's file, passing dataset_type if needed
            raw_file_path = self.raw_file_utils.get_raw_file_path(target_year, target_month, mercado)
            dataset_type = self._extract_dataset_type_from_filename(raw_file_path)
            raw_df = self.raw_file_utils.read_raw_file(target_year, target_month, dataset_type, mercado)

            # Filter for the specific day
            filtered_df = self._process_df_based_on_transform_type(raw_df, 'single', start_date=date)

            if filtered_df.empty:
                print(f"No data found for {mercado}/{dataset_type} on {date} within file {target_year}-{target_month:02d}.")
                return

            self._transform_and_save(filtered_df, mercado, dataset_type)

        except FileNotFoundError:
             # It's possible the folder exists but not the specific file (e.g., parquet file name)
             print(f"Raw data file not found for {mercado}/{dataset_type} for {target_year}-{target_month:02d}.")
        except Exception as e:
            print(f"Error during single day processing for {mercado}/{dataset_type} on {date}: {e}")


    def _transform_latest(self, mercado: str, dataset_type: str):
        """Process the latest day available in the raw data."""
        print(f"Starting LATEST transformation for {mercado} - {dataset_type}")
        latest_year, latest_month = None, None # Initialize for error message
        try:
            # Find the latest year/month available using RawFileUtils
            years = sorted(self.raw_file_utils.get_raw_folder_list(mercado, dataset_type=dataset_type), reverse=True)
            if not years: print(f"No data years found for {mercado}/{dataset_type}. Skipping latest."); return
            latest_year = years[0]

            months = sorted(self.raw_file_utils.get_raw_folder_list(mercado, latest_year, dataset_type=dataset_type), reverse=True)
            if not months: print(f"No data months found for latest year {latest_year}. Skipping latest."); return
            latest_month = months[0]

            print(f"Identified latest potential file location: {latest_year}-{latest_month:02d}")
            # Read the file, passing dataset_type if needed
            raw_file_path = self.raw_file_utils.get_raw_file_path(latest_year, latest_month, mercado)
            dataset_type = self._extract_dataset_type_from_filename(raw_file_path)
            raw_df = self.raw_file_utils.read_raw_file(latest_year, latest_month, dataset_type, mercado)

            # Filter for the latest day within that file
            filtered_df = self._process_df_based_on_transform_type(raw_df, 'latest')

            if filtered_df.empty:
                print(f"No data found for the latest day within file {latest_year}-{latest_month:02d}.")
                return

            self._transform_and_save(filtered_df, mercado, dataset_type)

        except FileNotFoundError:
             # Provide more context if year/month were determined
             location = f" for {mercado}/{dataset_type} at {latest_year}-{latest_month:02d}" if latest_year else ""
             print(f"Latest raw file not found{location}.")
        except Exception as e:
            print(f"Error during latest day processing for {mercado}/{dataset_type}: {e}")

    def _transform_multiple(self, mercado: str, dataset_type: str, start_date: str, end_date: str):
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
            # Get unique year-month tuples
            year_months = sorted(list(set([(d.year, d.month) for d in all_days_in_range])))


            all_raw_dfs = []
            print(f"Reading files for year-months: {year_months}")
            for year, month in year_months:
                try:
                    # Pass dataset_type to reader if needed
                    raw_file_path = self.raw_file_utils.get_raw_file_path(year, month, mercado)
                    dataset_type = self._extract_dataset_type_from_filename(raw_file_path)
                    df_month = self.raw_file_utils.read_raw_file(year, month, dataset_type, mercado)
                    if not df_month.empty:
                        # Pre-filter month data to reduce memory usage before concat
                        # Filter based on year/month is sufficient here before combining
                        month_start = pd.Timestamp(year=year, month=month, day=1)
                        # Handle month end carefully (days in month)
                        month_end = month_start + pd.offsets.MonthEnd(0)

                        # Ensure datetime_utc exists for filtering
                        if 'datetime_utc' not in df_month.columns:
                            # Attempt to derive or skip
                            if 'fecha' in df_month.columns:
                                df_month['datetime_utc'] = pd.to_datetime(df_month['fecha'], utc=True, errors='coerce')
                                df_month = df_month.dropna(subset=['datetime_utc'])
                            else:
                                print(f"Warning: Cannot pre-filter file {year}-{month:02d} as 'datetime_utc' or 'fecha' is missing.")
                                # Append without pre-filtering, relying on final filter
                                all_raw_dfs.append(df_month)
                                continue # Skip to next iteration

                        # Filter data within the month that falls into the overall start/end date range
                        df_filtered_month = df_month[
                            (df_month['datetime_utc'] >= pd.Timestamp(start_dt, tz='UTC')) &
                            (df_month['datetime_utc'] <= pd.Timestamp(end_dt, tz='UTC').replace(hour=23, minute=59, second=59)) # Ensure end date is inclusive
                        ]

                        if not df_filtered_month.empty:
                             all_raw_dfs.append(df_filtered_month)
                             print(f"Read and pre-filtered {len(df_filtered_month)} rows for {year}-{month:02d}")
                        else:
                            print(f"No relevant data found in {year}-{month:02d} for the specified date range.")

                    else:
                        print(f"File {year}-{month:02d} is empty or read failed.")
                except FileNotFoundError:
                    print(f"Warning: Raw file not found for {mercado}/{dataset_type} for {year}-{month:02d}. Skipping.")
                except Exception as e:
                    print(f"Error reading or pre-filtering raw file for {year}-{month:02d}: {e}")
                    continue # Continue processing other months

            if not all_raw_dfs:
                 print(f"No raw data found for the specified date range {start_date} to {end_date}.")
                 return

            # Concatenate filtered monthly dataframes
            combined_raw_df = pd.concat(all_raw_dfs, ignore_index=True)
            print(f"Combined raw data ({len(combined_raw_df)} rows).")
            # The final filtering step might be redundant if pre-filtering worked, but safe to keep.
            # It ensures the exact date range is met after concatenation.
            filtered_df = self._process_df_based_on_transform_type(combined_raw_df, 'multiple', start_date=start_date, end_date=end_date)


            if filtered_df.empty:
                print(f"No data found for {mercado}/{dataset_type} between {start_date} and {end_date} after final filtering.")
                return

            print(f"Filtered data has {len(filtered_df)} rows. Proceeding with transformation...")
            self._transform_and_save(filtered_df, mercado, dataset_type)

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
    