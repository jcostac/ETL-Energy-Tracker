import pandas as pd
from datetime import datetime
import sys
from pathlib import Path
from typing import Optional, List
import re 
import traceback
import pretty_errors

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utilidades.etl_date_utils import DateUtilsETL
from utilidades.db_utils import DatabaseUtils
from utilidades.storage_file_utils import RawFileUtils, ProcessedFileUtils
from transform._procesador_i3 import I3Processor
from configs.i3_config import (
    I3Config,
    DiarioConfig, SecundariaConfig, TerciariaConfig, RRConfig, IntraConfig,
    CurtailmentConfig, P48Config, IndisponibilidadesConfig, RestriccionesConfig
)

class TransformadorI3:
    def __init__(self):
        """
        Initializes the TransformadorI3 instance with processor, file utilities, date utilities, dataset types, transform types, and market configuration mappings.
        
        Sets up internal utilities and configuration required for transforming raw I3 market data, and computes the list of markets supporting 'volumenes' datasets.
        """
        self.processor = I3Processor()
        self.raw_file_utils = RawFileUtils()
        self.processed_file_utils = ProcessedFileUtils()
        self.date_utils = DateUtilsETL()

        self.dataset_types = ['volumenes_i3']
        self.transform_types = ['latest', 'single', 'multiple']

        self.market_config_map = {
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

        self.i3_volumenes_markets = self._compute_volumenes_markets()

    def _compute_volumenes_markets(self):
        """
        Return a list of market names that have 'volumenes' sheets available in their configuration.
        
        Returns:
        	markets (list): List of market names (as lowercase strings) with 'volumenes' sheet support.
        """
        markets = []
        for config_cls in I3Config.__subclasses__():
            if config_cls.has_volumenes_sheets():
                market_name = config_cls.__name__.replace('Config', '').lower()
                markets.append(market_name)
        return markets

    def get_config_for_market(self, mercado: str, fecha: Optional[datetime] = None) -> I3Config:
        """
        Retrieve the configuration instance for a specified market.
        
        If the market is 'intra', a `fecha` (date) parameter must be provided to instantiate its configuration. Raises a `ValueError` if the market is unknown or if required parameters are missing.
        
        Parameters:
        	mercado (str): The market name for which to retrieve the configuration.
        	fecha (Optional[datetime]): The date required for 'intra' market configuration.
        
        Returns:
        	I3Config: The configuration instance for the specified market.
        """
        config_class = self.market_config_map.get(mercado)
        if not config_class:
            all_known_i3_market_dl_names = self.i3_volumenes_markets
            if mercado in all_known_i3_market_dl_names:
                raise ValueError(f"Configuration class for known market '{mercado}' is missing from market_config_map.")
            else:
                raise ValueError(f"Unknown market name: '{mercado}'. No configuration class found.")

        try:
            if mercado == 'intra' and fecha is not None:
                return config_class(fecha)
            elif mercado == 'intra' and fecha is None:
                raise ValueError("fecha parameter is required for IntraConfig")
            else:
                return config_class()
        except Exception as e:
            print(f"Error instantiating config class {config_class.__name__} for market {mercado}: {e}")
            raise

    def transform_data_for_all_markets(self, fecha_inicio: Optional[str] = None, fecha_fin: Optional[str] = None,
                                       mercados_lst: Optional[List[str]] = None) -> dict:
        """
                                       Transforms raw 'volumenes_i3' data for all specified or available markets over a given date range or mode.
                                       
                                       Depending on the provided date parameters, processes the latest available day, a single specified day, or a range of days for each relevant market. Returns a dictionary containing the processed data per market and a status summary indicating successes, failures, and processing mode.
                                       
                                       Parameters:
                                           fecha_inicio (str, optional): Start date in 'YYYY-MM-DD' format. Determines the transformation mode if provided.
                                           fecha_fin (str, optional): End date in 'YYYY-MM-DD' format. Used for range processing if different from `fecha_inicio`.
                                           mercados_lst (List[str], optional): List of market names to process. If None, processes all available markets.
                                       
                                       Returns:
                                           dict: A dictionary with keys 'data' (processed DataFrames per market) and 'status' (success flag and details).
                                       """
                                       if fecha_inicio is None and fecha_fin is None:
            transform_type = 'latest'
        elif fecha_inicio is not None and (fecha_fin is None or fecha_inicio == fecha_fin):
            transform_type = 'single'
        elif fecha_inicio is not None and fecha_fin is not None and fecha_inicio != fecha_fin:
            transform_type = 'multiple'

        status_details = {
            "markets_processed": [],
            "markets_failed": [],
            "mode": transform_type,
            "date_range": f"{fecha_inicio} to {fecha_fin}" if fecha_fin else fecha_inicio
        }

        overall_success = True
        results = {}
        dataset_type = 'volumenes_i3'

        try:
            print(f"\n===== STARTING TRANSFORMATION RUN (Mode: {transform_type.upper()}) =====")
            print(f"Dataset type: {dataset_type}")
            if fecha_inicio: print(f"Start Date: {fecha_inicio}")
            if fecha_fin: print(f"End Date: {fecha_fin}")
            print("="*60)

            if mercados_lst is None:
                relevant_markets = self.i3_volumenes_markets
            else:
                invalid_markets = [m for m in mercados_lst if m not in self.i3_volumenes_markets]
                if invalid_markets:
                    print(f"Warning: Invalid markets for {dataset_type}: {', '.join(invalid_markets)}")
                relevant_markets = [m for m in mercados_lst if m in self.i3_volumenes_markets]

            if not relevant_markets:
                return {"data": results, "status": {"success": False, "details": f"No relevant markets for {dataset_type}."}}

            for mercado in relevant_markets:
                print(f"\n-- Market: {mercado} --")
                try:
                    if transform_type == 'single':
                        market_result = self._process_single_day(mercado, dataset_type, fecha_inicio)
                    elif transform_type == 'latest':
                        market_result = self._process_latest_day(mercado, dataset_type)
                    elif transform_type == 'multiple':
                        market_result = self._process_date_range(mercado, dataset_type, fecha_inicio, fecha_fin)

                    if market_result is not None:
                        if isinstance(market_result, pd.DataFrame):
                            if not market_result.empty:
                                status_details["markets_processed"].append(mercado)
                                results[mercado] = market_result
                            else:
                                print(f"⚠️ Empty DataFrame for {mercado}. Continuing.")
                                results[mercado] = market_result
                        elif isinstance(market_result, list):
                            success_count = sum(1 for df in market_result if isinstance(df, pd.DataFrame) and not df.empty)
                            if success_count > 0:
                                status_details["markets_processed"].append(mercado)
                                results[mercado] = market_result
                            else:
                                print(f"⚠️ All empty DataFrames for {mercado}. Continuing.")
                                results[mercado] = market_result
                        else:
                            status_details["markets_failed"].append({
                                "market": mercado,
                                "error": "No valid data produced"
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
                    print(f"❌ Failed to transform {dataset_type} for {mercado} ({transform_type}): {e}")
                    print(traceback.format_exc())

            print(f"\n===== TRANSFORMATION RUN FINISHED (Mode: {transform_type.upper()}) =====")
            print("Summary:")
            for market, result in results.items():
                status = "Failed/No Data"
                if isinstance(result, pd.DataFrame):
                    status = f"Success ({len(result)} records)" if not result.empty else "Success (Empty DF)"
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
        Transforms raw market data for a specified market and dataset type using the appropriate configuration.
        
        If the input DataFrame is empty or the transformation fails, returns an empty DataFrame.
        
        Parameters:
            raw_df (pd.DataFrame): The raw market data to be transformed.
            mercado (str): The market identifier.
            dataset_type (str): The type of dataset to transform.
            fecha (Optional[datetime]): The date used for configuration selection, if required.
        
        Returns:
            pd.DataFrame: The transformed market data, or an empty DataFrame if transformation is not possible.
        """
        if raw_df.empty:
            print(f"Skipping transformation for {mercado} - {dataset_type}: Input DataFrame is empty.")
            return pd.DataFrame()

        try:
            market_config = self.get_config_for_market(mercado, fecha)
            print(f"Raw data loaded ({len(raw_df)} rows). Starting transformation for {mercado} - {dataset_type}...")
            processed_df = self.processor.transform_raw_i3_data(raw_df, market_config, dataset_type)
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
        Filter a DataFrame of raw market data based on the specified transformation type and date criteria.
        
        Depending on the `transform_type`, this method filters the DataFrame to include only rows matching the latest date, a single specified date, or a date range. Raises an error if the 'fecha' column is missing or cannot be converted to datetime.
        
        Parameters:
            raw_df (pd.DataFrame): The DataFrame containing raw market data with a 'fecha' column.
            transform_type (str): The type of transformation ('latest', 'single', or 'multiple').
            fecha_inicio (Optional[str]): The start date for filtering (used for 'single' and 'multiple' modes).
            fecha_fin (Optional[str]): The end date for filtering (used for 'multiple' mode).
        
        Returns:
            pd.DataFrame: The filtered DataFrame containing only the relevant rows for the specified transformation type.
        """
        if raw_df.empty:
            return raw_df

        if "fecha" not in raw_df.columns:
            print("Error: 'fecha' column missing in DataFrame. Cannot apply date filters.")
            raise ValueError("'fecha' column missing. Cannot apply date filters.")
        else:
            if not pd.api.types.is_datetime64_any_dtype(raw_df['fecha']):
                try:
                    raw_df['fecha'] = pd.to_datetime(raw_df['fecha'])
                except Exception as e:
                    print(f"Error converting 'fecha': {e}. Returning empty DataFrame.")
                    raise e

        try:
            if transform_type == 'latest':
                if raw_df.empty: return raw_df
                last_day = raw_df['fecha'].dropna().dt.date.max()
                if pd.isna(last_day):
                    print("Warning: Could not determine the latest day (max date is NaT). Returning empty DataFrame for 'latest' mode.")
                    return pd.DataFrame()
                print(f"Filtering for latest mode: {last_day}")
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
                raise ValueError(f"Invalid transform type for filtering: {transform_type}")

        except Exception as e:
            print(f"Error during date filtering ({transform_type} mode): {e}")
            return pd.DataFrame()

    def _process_single_day(self, mercado: str, dataset_type: str, date: str):
        """
        Processes and transforms raw data for a specific market and dataset type on a single given date.
        
        Reads the relevant raw data file for the specified market and month, filters the data for the target date, and applies the transformation logic. Returns the processed DataFrame, or an empty DataFrame if no data is found for the date. Handles missing files and logs errors encountered during processing.
        
        Parameters:
            mercado (str): The market identifier.
            dataset_type (str): The type of dataset to process (e.g., 'volumenes_i3').
            date (str): The target date in a format recognized by pandas (e.g., 'YYYY-MM-DD').
        
        Returns:
            pd.DataFrame: The processed DataFrame for the specified date, or an empty DataFrame if no data is found.
        """
        print(f"Starting SINGLE transformation for {mercado} - {dataset_type} on {date}")
        try:
            target_date = pd.to_datetime(date)
            target_year = target_date.year
            target_month = target_date.month

            files = self.raw_file_utils.get_raw_file_list(mercado, target_year, target_month)

            if not files:
                print(f"No files found for {mercado}/{dataset_type} for {target_year}-{target_month:02d}. Skipping.")
                return

            matching_files = [f for f in files if dataset_type in str(f)]

            if not matching_files:
                print(f"No files matching dataset_type '{dataset_type}' found for {mercado} for {target_year}-{target_month:02d}.")
                return

            raw_df = self.raw_file_utils.read_raw_file(target_year, target_month, dataset_type, mercado)

            filtered_df = self._process_df_based_on_transform_type(raw_df, 'single', fecha_inicio=date)

            if filtered_df.empty:
                print(f"No data found for {mercado}/{dataset_type} on {date} within file {target_year}-{target_month:02d}.")
                return pd.DataFrame()

            processed_df = self._transform_data(filtered_df, mercado, dataset_type, fecha=target_date)
            return processed_df

        except FileNotFoundError:
            print(f"Raw data file not found for {mercado}/{dataset_type} for {target_year}-{target_month:02d}.")
        except Exception as e:
            print(f"Error during single day processing for {mercado}/{dataset_type} on {date}: {e}")

    def _process_latest_day(self, mercado: str, dataset_type: str):
        """
        Processes and transforms the latest available day's data for a given market and dataset type.
        
        Searches for the most recent raw data file for the specified market and dataset type, filters the data to the latest date present, and applies the transformation logic. Returns the processed DataFrame or an empty DataFrame if no data is found.
        """
        print(f"Starting LATEST transformation for {mercado} - {dataset_type}")
        try:
            years = sorted(self.raw_file_utils.get_raw_folder_list(mercado=mercado), reverse=True)
            if not years:
                print(f"No data years found for {mercado}. Skipping latest.")
                return

            found = False
            for year in years:
                months = sorted(self.raw_file_utils.get_raw_folder_list(mercado=mercado, year=year), reverse=True)
                for month in months:
                    files = self.raw_file_utils.get_raw_file_list(mercado, year, month)
                    if not files:
                        continue
                    matching_files = [f for f in files if dataset_type in str(f)]
                    if matching_files:
                        print(f"Identified latest available file for {mercado}/{dataset_type}: {year}-{month:02d}")
                        raw_df = self.raw_file_utils.read_raw_file(year, month, dataset_type, mercado)
                        filtered_df = self._process_df_based_on_transform_type(raw_df, 'latest')
                        if filtered_df.empty:
                            print(f"No data found for the latest day within file {year}-{month:02d}.")
                            return pd.DataFrame()
                        if not filtered_df.empty and 'fecha' in filtered_df.columns:
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
        Processes and transforms raw market data for a specified market and dataset type over a given date range.
        
        Reads and combines raw data files for all months within the date range, filters the data to include only rows within the specified dates, and applies the transformation logic. Returns a processed DataFrame or an empty DataFrame if no data is found.
        """
        print(f"Starting MULTIPLE transformation for {mercado} - {dataset_type} from {fecha_inicio} to {fecha_fin}")
        try:
            start_dt = pd.to_datetime(fecha_inicio)
            end_dt = pd.to_datetime(fecha_fin)

            if start_dt > end_dt: raise ValueError("Start date cannot be after end date.")

            all_days_in_range = pd.date_range(start_dt, end_dt, freq='D')
            if all_days_in_range.empty:
                print("Warning: Date range resulted in zero days. Nothing to process.")
                return

            year_months = sorted(list(set([(d.year, d.month) for d in all_days_in_range])))

            all_raw_dfs = []
            print(f"Reading files for year-months: {year_months}")
            for year, month in year_months:
                try:
                    raw_df = self.raw_file_utils.read_raw_file(year, month, dataset_type, mercado)
                    if raw_df is not None and not raw_df.empty:
                        all_raw_dfs.append(raw_df)
                    else:
                        print(f"Warning: No data returned from reading {mercado}/{dataset_type} for {year}-{month:02d}.")

                except FileNotFoundError:
                    print(f"Warning: Raw file not found for {mercado}/{dataset_type} for {year}-{month:02d}. Skipping.")
                except Exception as e:
                    print(f"Error reading or processing raw file for {year}-{month:02d}: {e}")
                    continue

            if not all_raw_dfs:
                print(f"No raw data found for the specified date range {fecha_inicio} to {fecha_fin}.")
                return

            combined_raw_df = pd.concat(all_raw_dfs, ignore_index=True)

            print(f"Combined raw data ({len(combined_raw_df)} rows). Applying date range filtering.")
            filtered_df = self._process_df_based_on_transform_type(combined_raw_df, 'multiple', fecha_inicio=fecha_inicio, fecha_fin=fecha_fin)

            if filtered_df.empty:
                print(f"No data found for {mercado}/{dataset_type} between {fecha_inicio} and {fecha_fin} after final filtering.")
                return pd.DataFrame()

            print(f"Filtered data has {len(filtered_df)} rows. Proceeding with transformation...")
            start_dt = pd.to_datetime(fecha_inicio)
            processed_df = self._transform_data(filtered_df, mercado, dataset_type, fecha=start_dt)
            return processed_df

        except ValueError as ve:
            print(f"Configuration or Value Error during multiple transform: {ve}")
        except Exception as e:
            print(f"An unexpected error occurred during multiple transform: {e}")
            print(traceback.format_exc())

    def _extract_dataset_type_from_filename(self, filename: str) -> str:
        """
        Extracts the dataset type string 'volumenes_i3' from a filename.
        
        Raises:
            ValueError: If the dataset type cannot be determined from the filename.
        
        Returns:
            str: The extracted dataset type.
        """
        match = re.search(r'(volumenes_i3)', filename)
        if match:
            return match.group(1)
        raise ValueError(f"Could not determine dataset_type from filename: {filename}")