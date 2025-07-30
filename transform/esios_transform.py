import pandas as pd
from datetime import datetime
from utilidades.etl_date_utils import DateUtilsETL
import pytz
import pretty_errors
import sys
import traceback
from typing import Optional, Union
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utilidades.raw_file_utils import RawFileUtils
from transform.procesadores._procesador_esios import ESIOSProcessor
from configs.esios_config import ESIOSConfig
from utilidades.etl_date_utils import DateUtilsETL

class TransformadorESIOS:
    
    def __init__(self):

        # Initialize the processor
        """
        Initializes the TransformadorESIOS instance with processor, file utilities, and configuration for supported transformation types.
        """
        self.processor = ESIOSProcessor()

        # Initialize the file utils
        self.raw_file_utils = RawFileUtils()
        self.date_utils = DateUtilsETL()

        #global attributes
        self.dataset_type = "precios_esios"
        self.transform_types = ['latest', 'single', 'multiple']
    
    def _filter_data_by_transform_type(self, raw_df: pd.DataFrame, transform_type: str, fecha_inicio: str = None, fecha_fin: str = None) -> pd.DataFrame:
        """
        Filter the input DataFrame according to the specified transformation transform_type and date parameters.
        
        Depending on the transform_type, this method selects rows for the latest day, a single day, a date range, or returns the entire DataFrame. For 'single' and 'multiple' transform_types, the relevant date parameters must be provided. Returns an empty DataFrame if the input is empty.
        
        Parameters:
            raw_df (pd.DataFrame): Input DataFrame containing a 'datetime_utc' column.
            transform_type (str): Transformation transform_type; one of 'latest', 'single', or 'multiple'.
            fecha_inicio (str, optional): Start date in 'YYYY-MM-DD' format, required for 'single' and 'multiple' transform_types.
            fecha_fin (str, optional): End date in 'YYYY-MM-DD' format, required for 'multiple' transform_type.
        
        Returns:
            pd.DataFrame: Filtered DataFrame according to the selected transform_type and date(s).
        """

        # First ensure datetime_utc is properly converted
        if 'datetime_utc' in raw_df.columns:
            raw_df['datetime_utc'] = pd.to_datetime(raw_df['datetime_utc'])

        #check that fecha_inicio and fecha_fin are provided for single and multiple transform_types
        if transform_type == 'multiple' and (fecha_inicio is None or fecha_fin is None):
            raise ValueError("fecha_inicio and fecha_fin must be provided for multiple transform_type")
        if transform_type == 'single' and fecha_inicio is None:
            raise ValueError("fecha_inicio must be provided for single transform_type")

        if not raw_df.empty:
            print(f"ℹ️Processing in {transform_type.upper()} transform_type")
            #1. LATEST transform_type
            if transform_type == 'latest':
                # Process only the last day in the dataframe
                last_day = raw_df['datetime_utc'].dt.date.max()
                print(f"Latest day in data: {last_day}")
                # Compare date with date directly without converting to Timestamp
                filtered_df = raw_df[raw_df['datetime_utc'].dt.date == last_day]
                print(f"Records found after filtering for : {len(filtered_df)}")
                print(f"Dates in data: {sorted(raw_df['datetime_utc'].dt.date.unique())}")
                return filtered_df

            #2. SINGLE transform_type
            elif transform_type == 'single':  
                # Convert to pandas Timestamp for consistent comparison
                target_date = pd.to_datetime(fecha_inicio).date()

                # Process a single day
                print(f"Single day selected: {target_date}")
                
                # Ensure consistent date comparison
                filtered_df = raw_df[raw_df['datetime_utc'].dt.date == target_date]
                
                return filtered_df

            #4. MULTIPLE transform_type
            elif transform_type == 'multiple':
                #convert from naive to utc
                target_fecha_inicio = pd.to_datetime(fecha_inicio).date()
                target_fecha_fin = pd.to_datetime(fecha_fin).date()

                # Process a range of days
                print(f"Processing range of days: {target_fecha_inicio} to {target_fecha_fin}")

                # Use direct comparison instead of isin with date_range
                filtered_df = raw_df[
                    (raw_df['datetime_utc'].dt.date >= target_fecha_inicio) & 
                    (raw_df['datetime_utc'].dt.date <= target_fecha_fin)
                ]
                return filtered_df
            
            else:
                raise ValueError(f"Invalid transform type: {transform_type}")
        else:
            # Return empty DataFrame if input is empty
             print("Input DataFrame is empty, returning empty DataFrame.")
             return pd.DataFrame()

    def _transform_market_data(self, raw_df: pd.DataFrame, mercado: str) -> Optional[pd.DataFrame]:
        """Transforms data for a specific market and returns the processed DataFrame."""
        try:
            # 1. Transform Data
            print("\n" + "="*80)
            print(f"🔄 TRANSFORMING DATA FOR {mercado.upper()}")
            print("="*80)

            print("\n📊 INPUT DATA")
            print("-"*50)
            print(f"Raw records to process: {len(raw_df)}")

            processed_df = self.processor.transform_price_data(raw_df)

            if processed_df.empty:
                print("\n❌ TRANSFORMATION RESULTED IN EMPTY DATAFRAME")
                print(f"Market: {mercado}")
                print("="*80 + "\n")
                return None # Return None if empty

            print("\n📋 PROCESSED DATA PREVIEW")
            print("-"*50)
            print("First 5 records:")
            print(processed_df.head().to_string(
                index=True,
                justify='right',
                float_format=lambda x: f"{x:.6f}"
            ))
            print("-"*50)
            print("Last 5 records:")
            print(processed_df.tail().to_string(
                index=True,
                justify='right',
                float_format=lambda x: f"{x:.6f}"
            ))
            print("-"*50)

            # 2. REMOVED Save Processed Data section
            print(f"✅ Transformation successful for {mercado}. Records processed: {len(processed_df)}")
            print("="*80 + "\n")
            return processed_df # <- RETURN the processed DataFrame

        except Exception as e:
            print("\n❌ ERROR DURING TRANSFORMATION")
            print("-"*50)
            print(f"Market: {mercado}")
            print(f"Error: {str(e)}")
            print(traceback.format_exc())
            print("="*80 + "\n")
            return None # Return None on error

    def _route_to_market_transform(self, raw_df: pd.DataFrame, mercado: str) -> Optional[pd.DataFrame]:
        """Routes the data to appropriate market-specific transformation."""
        if mercado not in ESIOSConfig().esios_precios_markets:
            raise ValueError(f"Invalid market: {mercado}. Must be one of: {ESIOSConfig().esios_precios_markets}")

        # Call the market-specific methods which now return DataFrames
        if mercado == 'diario':
            return self.process_diario_market(raw_df)
        elif mercado == 'intra':
            return self.process_intra_market(raw_df)
        elif mercado == 'secundaria':
            return self.process_secundaria_market(raw_df)
        elif mercado == 'terciaria':
            return self.process_terciaria_market(raw_df)
        elif mercado == 'rr':
            return self.process_rr_market(raw_df)
        else:
             print(f"Warning: Market '{mercado}' does not have a specific process method in routing.")
             return None # Or handle default case if necessary

    def _process_single_day(self, mercado: str, date: str) -> Optional[pd.DataFrame]:
        """
        Process and transform market data for a single specified day.
        
        Attempts to locate, read, and filter the raw data file for the given market and date. If data is available for the specified day, applies the appropriate market transformation and returns the resulting DataFrame. Returns None if the data is missing, the date is invalid, or an error occurs.
        
        Parameters:
            mercado (str): The market identifier.
            date (str): The target date in 'YYYY-MM-DD' format.
        
        Returns:
            Optional[pd.DataFrame]: The processed DataFrame for the specified day, or None if unavailable or on error.
        """
        print("\n" + "="*80)
        print(f"🔄 STARTING SINGLE DAY TRANSFORM")
        print(f"Market: {mercado.upper()}")
        print(f"Date: {date}")
        print("="*80 + "\n")
        processed_df = None # Initialize return variable

        try:
            # Parse the date string to get year and month
            try:
                target_date = pd.to_datetime(date)
                target_year = target_date.year
                target_month = target_date.month
            except ValueError:
                print("❌ VALIDATION ERROR")
                print(f"Invalid date format: {date}")
                print("Please use YYYY-MM-DD format")
                print("="*80 + "\n")
                return None

            print("📂 CHECKING DATA AVAILABILITY")
            print("-"*50)

            # Check if the required year and month folders exist
            available_years = self.raw_file_utils.get_raw_folder_list(mercado)
            if target_year not in available_years:
                print("❌ DATA NOT FOUND")
                print(f"No raw data available for year {target_year}")
                print("="*80 + "\n")
                return None

            available_months = self.raw_file_utils.get_raw_folder_list(mercado, target_year)
            if target_month not in available_months:
                print("❌ DATA NOT FOUND")
                print(f"No raw data available for {target_year}-{target_month:02d}")
                print("="*80 + "\n")
                return None

            print("✅ Raw data file likely available")
            print("-"*50 + "\n")

            # Read and process data
            print("🔄 PROCESSING DATA")
            print("-"*50)
            print(f"1. Reading raw file for {target_year}-{target_month:02d}...")
            raw_df = self.raw_file_utils.read_raw_file(target_year, target_month, self.dataset_type, mercado)
            print(f"   Records found: {len(raw_df)}")

            print("\n2. Filtering for target date...")
            # Use 'single' transform_type for filtering
            filtered_df = self._filter_data_by_transform_type(raw_df, transform_type='single', fecha_inicio=date)

            if filtered_df.empty:
                print("\n❌ NO DATA FOUND FOR TARGET DATE")
                print(f"No records for {date} in the raw file.")
                if not raw_df.empty and 'datetime_utc' in raw_df.columns:
                     print(f"Available date range in file: {raw_df['datetime_utc'].dt.date.min()} to {raw_df['datetime_utc'].dt.date.max()}")
                print("="*80 + "\n")
                return None

            print(f"   Filtered records for {date}: {len(filtered_df)}")

            print("\n3. Applying market transformation...")
            processed_df = self._route_to_market_transform(filtered_df, mercado) # Get the processed DF
            print("-"*50)

            if processed_df is not None and not processed_df.empty:
                 print("\n✅ PROCESS COMPLETE")
                 print(f"Successfully transformed {mercado} data for {date}. Records: {len(processed_df)}")
            elif processed_df is None:
                 print("\n❌ TRANSFORMATION FAILED")
            else: # Empty DataFrame
                 print("\n⚠️ TRANSFORMATION RESULTED IN EMPTY DATAFRAME")

            print("="*80 + "\n")

        except FileNotFoundError:
            print("\n❌ FILE ERROR")
            print(f"Raw file not found for {target_year}-{target_month:02d}")
            print("="*80 + "\n")
            processed_df = None
        except Exception as e:
            print("\n❌ UNEXPECTED ERROR")
            print(f"Error processing single day data: {str(e)}")
            print(traceback.format_exc())
            print("="*80 + "\n")
            processed_df = None

        return processed_df # Return the result (DF or None)

    def _process_latest_day(self, mercado: str) -> Optional[pd.DataFrame]:
        """
        Processes and transforms the most recent day's data for the specified market.
        
        Returns:
            The processed DataFrame for the latest available day, or None if no data is found or an error occurs.
        """
        print("\n" + "="*80)
        print(f"🔄 STARTING LATEST DAY TRANSFORM")
        print(f"Market: {mercado.upper()}")
        print("="*80 + "\n")
        processed_df = None # Initialize return variable

        try:
            print("📂 LOCATING LATEST DATA")
            print("-"*50)

            # Get the latest year
            years = sorted(self.raw_file_utils.get_raw_folder_list(mercado), reverse=True)
            if not years:
                print("❌ NO RAW DATA FOUND")
                print(f"No data available for market {mercado}")
                print("="*80 + "\n")
                return None

            latest_year = years[0]
            print(f"✓ Latest year found: {latest_year}")

            # Get the latest month in the latest year
            months = sorted(self.raw_file_utils.get_raw_folder_list(mercado, latest_year), reverse=True)
            if not months:
                print("❌ NO RAW DATA FOUND")
                print(f"No data available for year {latest_year}")
                print("="*80 + "\n")
                return None

            latest_month = months[0]
            print(f"✓ Latest month found: {latest_month:02d}")
            print("-"*50 + "\n")

            # Process data
            print("🔄 PROCESSING DATA")
            print("-"*50)
            print(f"1. Reading latest raw file ({latest_year}-{latest_month:02d})...")
            raw_df = self.raw_file_utils.read_raw_file(latest_year, latest_month, self.dataset_type, mercado)
            print(f"   Records found: {len(raw_df)}")

            if raw_df.empty:
                 print("❌ Raw file is empty. Cannot process latest day.")
                 print("="*80 + "\n")
                 return None

            print("\n2. Filtering for latest day in the file...")
            filtered_df = self._filter_data_by_transform_type(raw_df, 'latest') # Use 'latest' transform_type filter

            if filtered_df.empty:
                print("❌ NO RECORDS FOUND FOR LATEST DAY")
                print("The latest raw file might not contain data for the expected latest date.")
                if 'datetime_utc' in raw_df.columns:
                     print(f"Latest date in file was: {raw_df['datetime_utc'].dt.date.max()}")
                print("="*80 + "\n")
                return None

            latest_date = filtered_df['datetime_utc'].dt.date.max() # Get date from filtered data
            print(f"   Latest date being processed: {latest_date}")
            print(f"   Filtered records for latest day: {len(filtered_df)}")

            print("\n3. Applying market transformation...")
            processed_df = self._route_to_market_transform(filtered_df, mercado) # Get the processed DF
            print("-"*50)

            if processed_df is not None and not processed_df.empty:
                 print("\n✅ PROCESS COMPLETE")
                 print(f"Successfully transformed latest {mercado} data.")
                 print(f"Date processed: {latest_date}. Records: {len(processed_df)}")
            elif processed_df is None:
                 print("\n❌ TRANSFORMATION FAILED")
            else: # Empty DataFrame
                 print("\n⚠️ TRANSFORMATION RESULTED IN EMPTY DATAFRAME")

            print("="*80 + "\n")

        except FileNotFoundError:
             print("\n❌ FILE ERROR")
             print(f"Could not find raw file for latest period ({latest_year}-{latest_month:02d}).")
             print("="*80 + "\n")
             processed_df = None
        except Exception as e:
            print("\n❌ UNEXPECTED ERROR")
            print(f"Error processing latest data: {str(e)}")
            print(traceback.format_exc())
            print("="*80 + "\n")
            processed_df = None

        return processed_df # Return the result (DF or None)

    def _process_date_range(self, mercado: str, fecha_inicio: str, fecha_fin: str) -> Optional[pd.DataFrame]:
        """
        Process and transform market data for a specified date range, returning a single combined DataFrame.
        
        Reads and concatenates all relevant raw data files for the given market and date range, filters the combined data to the specified period, and applies the appropriate market transformation. Returns the processed DataFrame if successful, or `None` if no data is found or an error occurs.
        
        Parameters:
            mercado (str): The market identifier to process.
            fecha_inicio (str): Start date of the range in 'YYYY-MM-DD' format.
            fecha_fin (str): End date of the range in 'YYYY-MM-DD' format.
        
        Returns:
            Optional[pd.DataFrame]: The processed DataFrame for the specified date range, or `None` if processing fails or no data is available.
        """
        print("\n" + "="*80)
        print(f"🔄 STARTING MULTIPLE DAY TRANSFORM (Date Range)")
        print(f"Market: {mercado.upper()}")
        print(f"Period: {fecha_inicio} to {fecha_fin}")
        print("="*80 + "\n")
        processed_df_final = None # Initialize return variable

        try:
            # Parse dates using DateUtilsETL for consistency
            fecha_inicio_dt = pd.to_datetime(fecha_inicio)
            fecha_fin_dt = pd.to_datetime(fecha_fin)

            if fecha_inicio_dt > fecha_fin_dt:
                 raise ValueError("Start date cannot be after end date.")

            # Determine the range of year-month combinations needed
            current_date = fecha_inicio_dt
            year_months = set()
            while current_date <= fecha_fin_dt:
                year_months.add((current_date.year, current_date.month))
                # Move to the first day of the next month
                if current_date.month == 12:
                    current_date = current_date.replace(year=current_date.year + 1, month=1, day=1)
                else:
                    current_date = current_date.replace(month=current_date.month + 1, day=1)

            year_months = sorted(list(year_months)) # Ensure chronological order

            all_raw_dfs = []
            processed_any_file = False

            # Read data for each relevant year-month
            print("\n📂 RAW FILE READING")
            print("-"*50)
            for year, month in year_months:
                try:
                    print(f"📌 Attempting to read {year}-{month:02d}...")
                    df = self.raw_file_utils.read_raw_file(year, month, self.dataset_type, mercado)

                    if not df.empty:
                        all_raw_dfs.append(df)
                        processed_any_file = True
                        print(f"✅ Successfully read {len(df)} records from: {year}-{month:02d}")
                    else:
                        print(f"⚠️  Empty raw file: {year}-{month:02d}")

                except FileNotFoundError:
                    print(f"❌ Raw file not found: {year}-{month:02d}")
                except Exception as e:
                    print(f"❌ Error reading {year}-{month:02d}: {str(e)}")
                    continue # Continue to next month/year file on error
            print("-"*50 + "\n")

            if not processed_any_file or not all_raw_dfs:
                 print("❌ ERROR: No raw data files could be read for the specified range.")
                 print(f"No data processed for {mercado} between {fecha_inicio} and {fecha_fin}")
                 print("="*80 + "\n")
                 return None

            # Data Processing Steps
            print("🔄 PROCESSING COMBINED DATA")
            print("-"*50)
            print("1. Concatenating raw dataframes...")
            combined_raw_df = pd.concat(all_raw_dfs, ignore_index=True)
            print(f"   Total raw records combined: {len(combined_raw_df)}")

            print("\n2. Filtering combined data for the specific date range...")
            # Use 'multiple' transform_type filter on the combined dataframe
            filtered_df = self._filter_data_by_transform_type(
                combined_raw_df,
                'multiple',
                fecha_inicio=fecha_inicio,
                fecha_fin=fecha_fin
            )

            if filtered_df.empty:
                print("❌ ERROR: No data found within the specified date range after filtering.")
                print(f"No records for {mercado} between {fecha_inicio} and {fecha_fin}")
                print("="*80 + "\n")
                return None

            print(f"   Filtered records within range: {len(filtered_df)}")

            print("\n3. Applying market transformation...")
            processed_df_final = self._route_to_market_transform(filtered_df, mercado) # Get the processed DF
            print("-"*50)

            if processed_df_final is not None and not processed_df_final.empty:
                 print("\n✅ PROCESS COMPLETE")
                 print(f"Successfully transformed {mercado} data for the period.")
                 print(f"Period: {fecha_inicio} to {fecha_fin}. Records: {len(processed_df_final)}")
            elif processed_df_final is None:
                 print("\n❌ TRANSFORMATION FAILED")
            else: # Empty DataFrame
                 print("\n⚠️ TRANSFORMATION RESULTED IN EMPTY DATAFRAME")

            print("="*80 + "\n")

        except ValueError as ve:
             print("\n❌ VALIDATION ERROR")
             print(f"Error processing date range for {mercado} ({fecha_inicio} to {fecha_fin})")
             print(f"Details: {str(ve)}")
             print("="*80 + "\n")
             processed_df_final = None
        except Exception as e:
            print("\n❌ UNEXPECTED ERROR")
            print(f"Error processing date range for {mercado} ({fecha_inicio} to {fecha_fin})")
            print(f"Details: {str(e)}")
            print(traceback.format_exc())
            print("="*80 + "\n")
            processed_df_final = None

        return processed_df_final # Return the single combined DF or None

    def process_diario_market(self, raw_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Transform raw market data for the 'diario' market.
        
        Returns:
            A processed DataFrame for the 'diario' market, or None if transformation fails or results in empty data.
        """
        return self._transform_market_data(raw_df, 'diario')

    def process_intra_market(self, raw_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Transforms data for the 'intra' market."""
        return self._transform_market_data(raw_df, 'intra')

    def process_secundaria_market(self, raw_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Transforms data for the 'secundaria' market."""
        return self._transform_market_data(raw_df, 'secundaria')

    def process_terciaria_market(self, raw_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Transforms data for the 'terciaria' market."""
        return self._transform_market_data(raw_df, 'terciaria')

    def process_rr_market(self, raw_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Transforms data for the 'rr' market."""
        return self._transform_market_data(raw_df, 'rr')
    
    def transform_data_for_all_markets(self, fecha_inicio: str = None, fecha_fin: str = None, 
                                      mercados_lst: list[str] = None):
        """
        Transform and aggregate market price data for one or more specified markets using the selected processing transform_type.
        
        Depending on the transform_type, processes data for the latest day, a single day, a date range, or all available historical data. Tracks and reports the success or failure of each market's transformation, returning both the processed data and a detailed status summary.
        
        Parameters:
            fecha_inicio (str, optional): Start date in 'YYYY-MM-DD' format. Required for 'single' and 'multiple' transform_types.
            fecha_fin (str, optional): End date in 'YYYY-MM-DD' format. Required for 'multiple' transform_type.
            mercados (list[str], optional): List of market names to process. Defaults to all available markets if not provided.
            transform_type (str): Processing transform_type. One of 'single', 'multiple', or 'latest'.
        
        Returns:
            dict: A dictionary with:
                - 'data': Mapping of each market to its processed DataFrame or list of DataFrames (if multiple days are processed).
                - 'status': Dictionary containing:
                    - 'success': Boolean indicating if at least one market was processed successfully.
                    - 'details': Dictionary with lists of processed and failed markets, the transform_type used, and the date range.
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
        
        # Validate inputs
        try:
                
            # Get list of all markets to process
            if mercados_lst is None:
                mercados_lst = ESIOSConfig().esios_precios_markets
            elif isinstance(mercados_lst, str): # Allow single market string
                mercados_lst = [mercados_lst]
                
            # Process each market and track status
            for mercado in mercados_lst:
                market_result = None
                try:
                    if transform_type == 'single':
                        market_result = self._process_single_day(mercado, fecha_inicio)
                    elif transform_type == 'latest':
                        market_result = self._process_latest_day(mercado, fecha_inicio)
                    elif transform_type == 'multiple':
                        market_result = self._process_date_range(mercado, fecha_inicio, fecha_fin)
                    
                    # Check if transformation was successful
                    if market_result is not None and (
                        (isinstance(market_result, pd.DataFrame) and not market_result.empty) or
                        (isinstance(market_result, list) and any(df is not None and not df.empty for df in market_result))
                    ):
                        status_details["markets_processed"].append(mercado)
                        results[mercado] = market_result
                    else:
                        status_details["markets_failed"].append({
                            "market": mercado,
                            "error": "Transformation produced no valid data"
                        })
                        overall_success = False
                        results[mercado] = None  # Store None for failed transformations
                        
                except Exception as e:
                    status_details["markets_failed"].append({
                        "market": mercado,
                        "error": str(e)
                    })
                    overall_success = False
                    results[mercado] = None
                    
            # If all markets failed, consider the entire transformation failed
            if not status_details["markets_processed"]:
                overall_success = False
                
        except Exception as e:
            overall_success = False
            status_details["error"] = str(e)
            
        # Return both the results and the status
        return {
            "data": results,
            "status": {
                "success": overall_success,
                "details": status_details
            }
        }