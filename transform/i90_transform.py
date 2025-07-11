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
        DiarioConfig, SecundariaConfig, TerciariaConfig, RRConfig, IntraConfig,
        CurtailmentConfig, P48Config, IndisponibilidadesConfig, RestriccionesConfig
    )
import traceback


class TransformadorI90:
    def __init__(self):
        """
        Initialize the TransformadorI90 instance with processor, file utilities, date utilities, dataset types, transform modes, and market configuration mappings.
        
        Sets up internal attributes for processing I90 market data, including available dataset types, transformation modes, and mappings from market names to their configuration classes. Also computes and stores lists of markets that support volumenes and precios datasets.
        """
        self.processor = I90Processor()
        self.raw_file_utils = RawFileUtils()
        self.processed_file_utils = ProcessedFileUtils()
        self.date_utils = DateUtilsETL()

        # Define dataset types and transformation modes
        self.dataset_types = ['volumenes_i90', 'precios_i90']
        self.transform_types = ['latest',  'single', 'multiple']

        # Map market names (strings) to their actual config classes (Type[I90Config])
        self.market_config_map: Dict[str, Type[I90Config]] = {
            'diario': DiarioConfig,
            'intra': IntraConfig,
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
        Return a list of market names that support volumenes sheets based on their configuration classes.
        """
        markets = []
        for config_cls in I90Config.__subclasses__():
            if config_cls.has_volumenes_sheets(): # Check if the market has volumenes_sheets ie True
                market_name = config_cls.__name__.replace('Config', '').lower() # parse the market name
                markets.append(market_name) # Add the market name to the list
        return markets

    def _compute_precios_markets(self):
        """
        Return a list of market names whose configuration classes indicate the presence of precios sheets.
        """
        markets = []
        for config_cls in I90Config.__subclasses__():
            if config_cls.has_precios_sheets(): # Check if the market has precios_sheets ie True
                market_name = config_cls.__name__.replace('Config', '').lower() # parse the market name
                markets.append(market_name) # Add the market name to the list
        return markets

    def get_config_for_market(self, mercado: str, fecha: Optional[datetime] = None) -> I90Config:
        """
        Return an instantiated configuration object for the specified market.
        
        If the market is 'intra', a `fecha` parameter must be provided for correct configuration instantiation. Raises a `ValueError` if the market is unknown or if required parameters are missing.
        
        Parameters:
        	mercado (str): The market name for which to retrieve the configuration.
        	fecha (Optional[datetime]): The date required for 'intra' market configuration.
        
        Returns:
        	I90Config: The instantiated configuration object for the specified market.
        """
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
            if mercado == 'intra' and fecha is not None:
                return config_class(fecha)
            elif mercado == 'intra' and fecha is None:
                raise ValueError("fecha parameter is required for IntraConfig")
            else:
                return config_class()
        except Exception as e:
             # Catch errors during config instantiation (e.g., DB connection issues in __init__)
             print(f"Error instantiating config class {config_class.__name__} for market {mercado}: {e}")
             raise # Reraise the exception

    def transform_data_for_all_markets(self, fecha_inicio: Optional[str] = None, fecha_fin: Optional[str] = None,
                                         mercados_lst: Optional[List[str]] = None,
                                         dataset_type: str = None) -> dict:
        """
                                         Transforms I90 market data for specified markets and dataset type over a given date range or mode.
                                         
                                         Automatically determines the transformation mode based on the provided date parameters:
                                         - If no dates are given, processes the latest available data ('latest' mode).
                                         - If only `fecha_inicio` is provided or both dates are equal, processes a single day ('single' mode).
                                         - If both dates are provided and different, processes the full date range ('multiple' mode).
                                         
                                         Validates the dataset type and filters markets to those relevant for the requested dataset. For each market, applies the appropriate transformation and collects results, tracking successes and failures.
                                         
                                         Parameters:
                                             fecha_inicio (str, optional): Start date in 'YYYY-MM-DD' format, or the single date to process.
                                             fecha_fin (str, optional): End date in 'YYYY-MM-DD' format. If omitted or equal to `fecha_inicio`, processes a single day.
                                             mercados_lst (List[str], optional): List of market names to process. If omitted, processes all markets relevant to the dataset type.
                                             dataset_type (str): The dataset type to process ('volumenes_i90' or 'precios_i90').
                                         
                                         Returns:
                                             dict: A dictionary with:
                                                 - 'data': Mapping of market names to processed DataFrames (or lists of DataFrames).
                                                 - 'status': Dictionary containing:
                                                     - 'success': Boolean indicating overall success.
                                                     - 'details': Dictionary with lists of processed and failed markets, the mode used, and the date range.
                                         """
        # Auto-infer transform type based on date parameters
        if fecha_inicio is None and fecha_fin is None:
            transform_type = 'latest'
        elif fecha_inicio is not None and (fecha_fin is None or fecha_inicio == fecha_fin):
            transform_type = 'single'
        elif fecha_inicio is not None and fecha_fin is not None and fecha_inicio != fecha_fin:
            transform_type = 'multiple'
            
        # Initialize status tracking
        status_details = {
            "markets_processed": [],
            "markets_failed": [],
            "mode": transform_type,
            "date_range": f"{fecha_inicio} to {fecha_fin}" if fecha_fin else fecha_inicio
        }
        
        overall_success = True
        results = {}

        try:
            # Validate dataset_type
            if dataset_type is None:
                raise ValueError(f"dataset_type must be provided. Must be one of {self.dataset_types}")
            else:
                if dataset_type not in self.dataset_types:
                    raise ValueError(f"Invalid dataset_type provided: {dataset_type}. Must be in {self.dataset_types}")

            print(f"\n===== STARTING TRANSFORMATION RUN (Mode: {transform_type.upper()}) =====")
            print(f"Dataset type to process: {dataset_type}")
            if fecha_inicio: print(f"Start Date: {fecha_inicio}")
            if fecha_fin: print(f"End Date: {fecha_fin}")
            print("="*60)

            # Determine relevant markets for this dataset type
            if mercados_lst is None:
                relevant_markets = self.i90_volumenes_markets if dataset_type == 'volumenes_i90' else self.i90_precios_markets
            else:
                known_markets_for_type = self.i90_volumenes_markets if dataset_type == 'volumenes_i90' else self.i90_precios_markets
                invalid_markets = [m for m in mercados_lst if m not in known_markets_for_type]
                if invalid_markets:
                    print(f"Warning: The following market(s)  are not associated with dataset type {dataset_type}. Skipping them.")
                    print(f"    - Invalid markets: {', '.join(invalid_markets)}")
                    print(f"    - Known markets for type {dataset_type} are: {', '.join(known_markets_for_type)}")
                relevant_markets = [m for m in mercados_lst if m in known_markets_for_type]

            if not relevant_markets:
                print(f"⚠️  No markets solicited for processing have data associated with dataset type:")
                print(f"    - {dataset_type}")
                return {"data": results, "status": {"success": False, "details": f"No relevant markets specified or configured for dataset type: {dataset_type}."}}

            for mercado in relevant_markets:
                print(f"\n-- Market: {mercado} --")
                try:
                    if transform_type == 'single':
                        market_result = self._process_single_day(mercado, dataset_type, fecha_inicio)
                    elif transform_type == 'latest':
                        market_result = self._process_latest_day(mercado, dataset_type)
                    elif transform_type == 'multiple':
                        market_result = self._process_date_range(mercado, dataset_type, fecha_inicio, fecha_fin)

                    # Check if transformation was successful
                    if market_result is not None:
                        if isinstance(market_result, pd.DataFrame):
                            if not market_result.empty:
                                status_details["markets_processed"].append(mercado)
                                results[mercado] = market_result
                            else:
                                print(f"⚠️ Transformation for market {mercado} resulted in an empty DataFrame. Continuing with other markets.")
                                results[mercado] = market_result  # Still store the empty DataFrame
                        elif isinstance(market_result, list):
                            if any(df is not None and not df.empty for df in market_result):
                                status_details["markets_processed"].append(mercado)
                                results[mercado] = market_result
                            else:
                                print(f"⚠️ Transformation for market {mercado} resulted in a list of empty DataFrames. Continuing with other markets.")
                                results[mercado] = market_result  # Store the list of empty DataFrames
                        else:
                            status_details["markets_failed"].append({
                                "market": mercado,
                                "error": "Transformation produced no valid data"
                            })
                            overall_success = False
                            results[mercado] = None
                            continue

                except Exception as e:
                    status_details["markets_failed"].append({
                        "market": mercado,
                        "error": str(e)
                    })
                    overall_success = False
                    results[mercado] = None
                    print(f"❌ Failed to transform {dataset_type} for market {mercado} ({transform_type}): {e}")
                    print(traceback.format_exc())
                    continue

            print(f"\n===== TRANSFORMATION RUN FINISHED (Mode: {transform_type.upper()}) =====")
            # Print summary of results
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

        except Exception as e:
            overall_success = False
            status_details["error"] = str(e)

        return {
            "data": results,
            "status": {
                "success": overall_success,
                "details": status_details
            }
        }

    def _transform_data(self, raw_df: pd.DataFrame, mercado: str, dataset_type: str, fecha: Optional[datetime] = None) -> pd.DataFrame:
        """
        Transforms a raw I90 market DataFrame into a processed DataFrame for the specified market and dataset type.
        
        If the input DataFrame is empty or an error occurs during transformation, returns an empty DataFrame.
        
        Parameters:
            raw_df (pd.DataFrame): The raw market data to be transformed.
            mercado (str): The market identifier.
            dataset_type (str): The type of dataset to transform ('volumenes_i90' or 'precios_i90').
            fecha (Optional[datetime]): The date to use for market configuration, required for certain markets.
        
        Returns:
            pd.DataFrame: The processed DataFrame, or an empty DataFrame if transformation fails.
        """
        if raw_df.empty:
            print(f"Skipping transformation for {mercado} - {dataset_type}: Input DataFrame is empty.")
            return pd.DataFrame()

        try:
            market_config = self.get_config_for_market(mercado, fecha)
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

    def _process_df_based_on_transform_type(self, raw_df: pd.DataFrame, transform_type: str, fecha_inicio: Optional[str] = None, fecha_fin: Optional[str] = None) -> pd.DataFrame:
        """
        Filter a raw DataFrame by date according to the specified transform type.
        
        Depending on the transform type ('latest', 'single', or 'multiple'), this method selects rows from the DataFrame based on the 'fecha' column. Raises an error if the 'fecha' column is missing or cannot be converted to datetime. Returns an empty DataFrame if filtering fails or no matching rows are found.
        
        Parameters:
            raw_df (pd.DataFrame): The input DataFrame containing a 'fecha' column.
            transform_type (str): The filtering mode ('latest', 'single', or 'multiple').
            fecha_inicio (Optional[str]): The start date for filtering (used for 'single' and 'multiple' modes).
            fecha_fin (Optional[str]): The end date for filtering (used for 'multiple' mode).
        
        Returns:
            pd.DataFrame: The filtered DataFrame according to the transform type.
        """
        if raw_df.empty:
            return raw_df

        # --- Datetime Column Handling ---
        # The I90Processor._standardize_datetime should create 'datetime_utc', but do do this, we first hacve to filter by date, hence the necessesity of a fecha column
        # We rely on the "fecha" column for filtering by date to succeed.
        # or lack the column, leading to errors here or empty results.
        if "fecha" not in raw_df.columns:
            print("Error: 'fecha' column missing in DataFrame passed to _process_df_based_on_transform_type. Cannot apply date filters.")
            raise ValueError("'fecha' column missing in DataFrame passed to _process_df_based_on_transform_type. Cannot apply date filters.")
        else:
            # Ensure fecha column is datetime type for filtering
            if not pd.api.types.is_datetime64_any_dtype(raw_df['fecha']):
                try:
                    raw_df['fecha'] = pd.to_datetime(raw_df['fecha'])
                except Exception as e:
                    print(f"Error converting 'fecha' in _process_df_based_on_transform_type: {e}. Returning empty DataFrame.")
                    raise e
            

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

            elif transform_type == 'single':
                target_date = pd.to_datetime(fecha_inicio).date()
                print(f"Filtering for single mode: {target_date}")
                return raw_df[raw_df['fecha'].dt.date == target_date].copy()

            elif transform_type == 'multiple':
                start_dt = pd.to_datetime(fecha_inicio).date()
                end_dt = pd.to_datetime(fecha_fin).date()
                if start_dt > end_dt: raise ValueError("Start date cannot be after end date.")
                print(f"Filtering for multiple mode: {start_dt} to {end_dt}")
                return raw_df[(raw_df['fecha'].dt.date >= start_dt) & (raw_df['fecha'].dt.date <= end_dt)].copy()

            else:
                # This case should technically be caught by the public method check
                raise ValueError(f"Invalid transform type for filtering: {transform_type}")

        except Exception as e:
             print(f"Error during date filtering ({transform_type} mode): {e}")
             return pd.DataFrame() # Return empty on filtering error

    def _process_single_day(self, mercado: str, dataset_type: str, date: str):
        """
        Processes and transforms I90 market data for a specific market, dataset type, and single date.
        
        Parameters:
        	mercado (str): The market name to process.
        	dataset_type (str): The type of dataset to process ('volumenes_i90' or 'precios_i90').
        	date (str): The target date in string format (YYYY-MM-DD).
        
        Returns:
        	pd.DataFrame: The processed DataFrame for the specified market, dataset type, and date, or an empty DataFrame if no data is found.
        """
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

            matching_files = [f for f in files if dataset_type in str(f)]
            
            if not matching_files:
                print(f"No files matching dataset_type '{dataset_type}' found for {mercado} for {target_year}-{target_month:02d}.")
                return

            # Read the files in the list of raw files
            raw_df = self.raw_file_utils.read_raw_file(target_year, target_month, dataset_type, mercado)

            # Filter for the specific day
            filtered_df = self._process_df_based_on_transform_type(raw_df, 'single', fecha_inicio=date)

            if filtered_df.empty:
                print(f"No data found for {mercado}/{dataset_type} on {date} within file {target_year}-{target_month:02d}.")
                return pd.DataFrame()

            processed_df = self._transform_data(filtered_df, mercado, dataset_type, fecha=target_date)
            return processed_df

        except FileNotFoundError:
             # It's possible the folder exists but not the specific file (e.g., parquet file name)
             print(f"Raw data file not found for {mercado}/{dataset_type} for {target_year}-{target_month:02d}.")
        except Exception as e:
            print(f"Error during single day processing for {mercado}/{dataset_type} on {date}: {e}")

    def _process_latest_day(self, mercado: str, dataset_type: str):
        """
        Processes and transforms the latest available day's data for the specified market and dataset type.
        
        Searches for the most recent raw data file containing the requested dataset type, filters the data to the latest day present, and applies the appropriate transformation. Returns the processed DataFrame or an empty DataFrame if no data is found.
        """
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
                        # Extract the actual date from the filtered data
                        if not filtered_df.empty and 'fecha' in filtered_df.columns:
                            # Get the actual date being processed
                            processing_date = filtered_df['fecha'].iloc[0]
                            if pd.notna(processing_date):
                                fecha_for_config = pd.to_datetime(processing_date)
                            else:
                                fecha_for_config = None
                        else:
                            fecha_for_config = None
                        
                        processed_df = self._transform_data(filtered_df, mercado, dataset_type, fecha=fecha_for_config)
                        found = True
                        return processed_df
                if found:
                    break
            if not found:
                print(f"No raw file found for {mercado}/{dataset_type} in any available year/month.")
        except Exception as e:
            print(f"Error during latest day processing for {mercado}/{dataset_type}: {e}")

    def _process_date_range(self, mercado: str, dataset_type: str, fecha_inicio: str, fecha_fin: str):
        """
        Processes and transforms I90 market data for a specified market and dataset type over a given date range.
        
        Reads and aggregates raw monthly files covering the date range, filters the combined data to the exact date interval, and applies the appropriate transformation. Handles missing files and errors gracefully, continuing processing for available months.
        
        Parameters:
            mercado (str): The market name to process.
            dataset_type (str): The type of dataset to process ('volumenes_i90' or 'precios_i90').
            fecha_inicio (str): Start date of the range in 'YYYY-MM-DD' format.
            fecha_fin (str): End date of the range in 'YYYY-MM-DD' format.
        
        Returns:
            pd.DataFrame: The processed DataFrame for the specified market and date range, or an empty DataFrame if no data is found.
        """
        print(f"Starting MULTIPLE transformation for {mercado} - {dataset_type} from {fecha_inicio} to {fecha_fin}")
        try:
            start_dt = pd.to_datetime(fecha_inicio)
            end_dt = pd.to_datetime(fecha_fin)

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
                 print(f"No raw data found for the specified date range {fecha_inicio} to {fecha_fin}.")
                 return

            # Concatenate filtered monthly dataframes
            combined_raw_df = pd.concat(all_raw_dfs, ignore_index=True)

            print(f"Combined raw data ({len(combined_raw_df)} rows). Applying date range filtering.")
            # The final filtering step ensures the exact date range is met after concatenation.
            filtered_df = self._process_df_based_on_transform_type(combined_raw_df, 'multiple', fecha_inicio=fecha_inicio, fecha_fin=fecha_fin)


            if filtered_df.empty:
                print(f"No data found for {mercado}/{dataset_type} between {fecha_inicio} and {fecha_fin} after final filtering.")
                return pd.DataFrame()

            print(f"Filtered data has {len(filtered_df)} rows. Proceeding with transformation...")
            # For date range, use the fecha_inicio as the reference for config
            start_dt = pd.to_datetime(fecha_inicio)
            processed_df = self._transform_data(filtered_df, mercado, dataset_type, fecha=start_dt)
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
    