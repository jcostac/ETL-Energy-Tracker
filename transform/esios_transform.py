import pandas as pd
from datetime import datetime
from utilidades.etl_date_utils import DateUtilsETL
import pytz
import pretty_errors
import sys
import traceback
from typing import Optional, Union

# Add necessary imports
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Ensure the project root is added to sys.path if necessary
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utilidades.storage_file_utils import RawFileUtils
from transform._procesador_esios import ESIOSProcessor
from configs.esios_config import ESIOSConfig
from utilidades.etl_date_utils import DateUtilsETL

class TransformadorESIOS:
    
    def __init__(self):

        # Initialize the processor
        self.processor = ESIOSProcessor()

        # Initialize the file utils
        self.raw_file_utils = RawFileUtils()
        self.date_utils = DateUtilsETL()

        #global attributes
        self.dataset_type = 'precios'
        self.transform_types = ['latest', 'batch', 'single', 'multiple']
    
    def _filter_data_by_mode(self, raw_df: pd.DataFrame, mode: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
         Transform the dataframe based on the transform type. If single pick the last day in the dataframe.
         If batch process the entire dataframe.

         Args:
            raw_df (pd.DataFrame): The dataframe to transform.
            mode (str): The type of transformation to perform.
            start_date (str): The start date to process. Optional for single and multiple modes.
            end_date (str): The end date to process. Optional for multiple modes.

         Returns:
            pd.DataFrame: The transformed dataframe.

        Note:
            - If mode is 'single' or 'multiple', start_date and end_date must be provided.
            - For single mode, we only use the start_date to filter the dataframe.
            - For multiple mode, we take all days in the range start_date to end_date (inclusive).
        """

        # First ensure datetime_utc is properly converted
        if 'datetime_utc' in raw_df.columns:
            raw_df['datetime_utc'] = pd.to_datetime(raw_df['datetime_utc'])

        #check that start_date and end_date are provided for single and multiple modes
        if mode == 'multiple' and (start_date is None or end_date is None):
            raise ValueError("start_date and end_date must be provided for multiple mode")
        if mode == 'single' and start_date is None:
            raise ValueError("start_date must be provided for single mode")

        if not raw_df.empty:
            print(f"ℹ️Processing in {mode.upper()} mode")
            #1. LATEST mode
            if mode == 'latest':
                # Process only the last day in the dataframe
                last_day = raw_df['datetime_utc'].dt.date.max()
                print(f"Latest day in data: {last_day}")
                # Compare date with date directly without converting to Timestamp
                filtered_df = raw_df[raw_df['datetime_utc'].dt.date == last_day]
                print(f"Records found after filtering for : {len(filtered_df)}")
                print(f"Dates in data: {sorted(raw_df['datetime_utc'].dt.date.unique())}")
                return filtered_df

            #2. BATCH mode
            elif mode == 'batch':
                # Process the entire dataframe
                return raw_df

            #3. SINGLE mode
            elif mode == 'single':  
                # Convert to pandas Timestamp for consistent comparison
                target_date = pd.to_datetime(start_date).date()

                # Process a single day
                print(f"Single day selected: {target_date}")
                
                # Ensure consistent date comparison
                filtered_df = raw_df[raw_df['datetime_utc'].dt.date == target_date]
                
                return filtered_df

            #4. MULTIPLE mode
            elif mode == 'multiple':
                #convert from naive to utc
                target_start_date = pd.to_datetime(start_date).date()
                target_end_date = pd.to_datetime(end_date).date()

                # Process a range of days
                print(f"Processing range of days: {target_start_date} to {target_end_date}")

                # Use direct comparison instead of isin with date_range
                filtered_df = raw_df[
                    (raw_df['datetime_utc'].dt.date >= target_start_date) & 
                    (raw_df['datetime_utc'].dt.date <= target_end_date)
                ]
                return filtered_df
            
            else:
                raise ValueError(f"Invalid transform type: {mode}")
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

    def _process_batch_mode(self, mercado: str) -> list[Optional[pd.DataFrame]]:
        """Processes all historical data for a given market and returns a list of processed dataframes."""
        print("\n" + "="*80)
        print(f"🔄 STARTING BATCH TRANSFORM")
        print(f"Market: {mercado.upper()}")
        print("="*80 + "\n")

        years = self.raw_file_utils.get_raw_folder_list(mercado)
        processed_dfs = []

        for year in years:
            months = self.raw_file_utils.get_raw_folder_list(mercado, year)

            print(f"\n📅 Processing Year: {year}")
            print("-"*50)

            for month in months:
                processed_df_month = None # Initialize for this month
                try:
                    print(f"📌 Processing {year}-{month:02d}...")
                    raw_df = self.raw_file_utils.read_raw_file(year, month, self.dataset_type, mercado)

                    if raw_df.empty:
                        print(f"⚠️  Empty raw file for {year}-{month:02d}. Skipping.")
                        continue

                    print(f"   Raw records found: {len(raw_df)}")
                    # Filter is usually not needed for batch, but kept for consistency if logic changes
                    filtered_df = self._filter_data_by_mode(raw_df, 'batch')
                    # Route to transform (which returns the processed DF)
                    processed_df_month = self._route_to_market_transform(filtered_df, mercado)

                    if processed_df_month is not None and not processed_df_month.empty:
                        print(f"✅ Successfully transformed {year}-{month:02d}. Records: {len(processed_df_month)}")
                    elif processed_df_month is None:
                         print(f"❌ Transformation failed for {year}-{month:02d}.")
                    else: # Empty DataFrame returned
                         print(f"⚠️ Transformation resulted in empty DataFrame for {year}-{month:02d}.")

                except FileNotFoundError:
                    print(f"❌ Raw file not found for {year}-{month:02d}. Skipping.")
                except Exception as e:
                    print(f"❌ Error processing {year}-{month:02d}: {e}")
                    # Decide if you want to append None on error or just skip
                    # processed_df_month = None # Ensure it's None on error
                finally:
                    # Append the result for this month (could be DF or None)
                    processed_dfs.append(processed_df_month)

            print("-"*50)

        print(f"\n✅ BATCH TRANSFORM COMPLETE for {mercado.upper()}")
        print("="*80 + "\n")
        return processed_dfs # Return list of processed dataframes (or None)

    def _process_single_day(self, mercado: str, date: str) -> Optional[pd.DataFrame]:
        """Processes data for a single specified day and returns the processed DataFrame."""
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
            # Use 'single' mode for filtering
            filtered_df = self._filter_data_by_mode(raw_df, mode='single', start_date=date)

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
        """Processes the most recent day's data and returns the processed DataFrame."""
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
            filtered_df = self._filter_data_by_mode(raw_df, 'latest') # Use 'latest' mode filter

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

    def _process_date_range(self, mercado: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Processes data for a specified date range and returns a single combined processed DataFrame."""
        print("\n" + "="*80)
        print(f"🔄 STARTING MULTIPLE DAY TRANSFORM (Date Range)")
        print(f"Market: {mercado.upper()}")
        print(f"Period: {start_date} to {end_date}")
        print("="*80 + "\n")
        processed_df_final = None # Initialize return variable

        try:
            # Parse dates using DateUtilsETL for consistency
            start_date_dt = pd.to_datetime(start_date)
            end_date_dt = pd.to_datetime(end_date)

            if start_date_dt > end_date_dt:
                 raise ValueError("Start date cannot be after end date.")

            # Determine the range of year-month combinations needed
            current_date = start_date_dt
            year_months = set()
            while current_date <= end_date_dt:
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
                 print(f"No data processed for {mercado} between {start_date} and {end_date}")
                 print("="*80 + "\n")
                 return None

            # Data Processing Steps
            print("🔄 PROCESSING COMBINED DATA")
            print("-"*50)
            print("1. Concatenating raw dataframes...")
            combined_raw_df = pd.concat(all_raw_dfs, ignore_index=True)
            print(f"   Total raw records combined: {len(combined_raw_df)}")

            print("\n2. Filtering combined data for the specific date range...")
            # Use 'multiple' mode filter on the combined dataframe
            filtered_df = self._filter_data_by_mode(
                combined_raw_df,
                'multiple',
                start_date=start_date,
                end_date=end_date
            )

            if filtered_df.empty:
                print("❌ ERROR: No data found within the specified date range after filtering.")
                print(f"No records for {mercado} between {start_date} and {end_date}")
                print("="*80 + "\n")
                return None

            print(f"   Filtered records within range: {len(filtered_df)}")

            print("\n3. Applying market transformation...")
            processed_df_final = self._route_to_market_transform(filtered_df, mercado) # Get the processed DF
            print("-"*50)

            if processed_df_final is not None and not processed_df_final.empty:
                 print("\n✅ PROCESS COMPLETE")
                 print(f"Successfully transformed {mercado} data for the period.")
                 print(f"Period: {start_date} to {end_date}. Records: {len(processed_df_final)}")
            elif processed_df_final is None:
                 print("\n❌ TRANSFORMATION FAILED")
            else: # Empty DataFrame
                 print("\n⚠️ TRANSFORMATION RESULTED IN EMPTY DATAFRAME")

            print("="*80 + "\n")

        except ValueError as ve:
             print("\n❌ VALIDATION ERROR")
             print(f"Error processing date range for {mercado} ({start_date} to {end_date})")
             print(f"Details: {str(ve)}")
             print("="*80 + "\n")
             processed_df_final = None
        except Exception as e:
            print("\n❌ UNEXPECTED ERROR")
            print(f"Error processing date range for {mercado} ({start_date} to {end_date})")
            print(f"Details: {str(e)}")
            print(traceback.format_exc())
            print("="*80 + "\n")
            processed_df_final = None

        return processed_df_final # Return the single combined DF or None

    def transform_data_for_all_markets(self, start_date: str = None, end_date: str = None, 
                                      mercados: list[str] = None, mode: str = 'latest'):
        """
        Transforms data for specified markets and returns status along with results.
        """
        # Initialize status tracking
        status_details = {
            "markets_processed": [],
            "markets_failed": [],
            "mode": mode,
            "date_range": f"{start_date} to {end_date}" if end_date else start_date
        }
        
        overall_success = True
        results = {}
        
        # Validate inputs
        try:
            if mode not in self.transform_types:
                raise ValueError(f"Invalid transform type: {mode}. Must be one of: {self.transform_types}")
                
            # Add validation for required parameters
            if mode == 'single' and not start_date:
                raise ValueError("start_date must be provided for single mode")
            if mode == 'multiple' and (not start_date or not end_date):
                raise ValueError("Both start_date and end_date must be provided for multiple mode")
            
            # Get list of all markets to process
            if mercados is None:
                mercados = ESIOSConfig().esios_precios_markets
            elif isinstance(mercados, str): # Allow single market string
                mercados = [mercados]
                
            # Process each market and track status
            for mercado in mercados:
                market_result = None
                try:
                    if mode == 'batch':
                        market_result = self._process_batch_mode(mercado)
                    elif mode == 'single':
                        market_result = self._process_single_day(mercado, start_date)
                    elif mode == 'latest':
                        market_result = self._process_latest_day(mercado)
                    elif mode == 'multiple':
                        market_result = self._process_date_range(mercado, start_date, end_date)
                    
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

    def process_diario_market(self, raw_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Transforms data for the 'diario' market."""
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
    