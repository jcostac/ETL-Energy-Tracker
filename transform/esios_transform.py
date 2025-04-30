import pandas as pd
from datetime import datetime
from utilidades.etl_date_utils import DateUtilsETL
import pytz
import pretty_errors
import sys
import traceback

# Add necessary imports
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Ensure the project root is added to sys.path if necessary
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utilidades.storage_file_utils import RawFileUtils, ProcessedFileUtils
from transform._procesador_esios import ESIOSProcessor
from configs.esios_config import ESIOSConfig
from utilidades.etl_date_utils import DateUtilsETL

class TransformadorESIOS:
    
    def __init__(self):

        # Initialize the processor
        self.processor = ESIOSProcessor()

        # Initialize the file utils
        self.raw_file_utils = RawFileUtils()
        self.processed_file_utils = ProcessedFileUtils()
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

            #1. LATEST mode
            if mode == 'latest':
                # Process only the last day in the dataframe
                last_day = raw_df['datetime_utc'].dt.date.max()
              
                print(f"Processing in single mode for {last_day}")
                # Filter dataframe to only include the last day
                # Instead of comparing with date directly, convert to Timestamp
                filtered_df = raw_df[raw_df['datetime_utc'].dt.date == pd.Timestamp(last_day)]
                return filtered_df

            #2. BATCH mode
            elif mode == 'batch':
                # Process the entire dataframe
                print(f"Processing in batch mode with {len(raw_df['datetime_utc'].dt.date.unique())} days")
                return raw_df

            #3. SINGLE mode
            elif mode == 'single':  
                # Convert to pandas Timestamp for consistent comparison
                target_date = pd.to_datetime(start_date).date()

                # Process a single day
                print(f"Processing in single mode for {target_date}")
                
                # Ensure consistent date comparison
                filtered_df = raw_df[raw_df['datetime_utc'].dt.date == target_date]
                
                return filtered_df

            #4. MULTIPLE mode
            elif mode == 'multiple':
                #convert from naive to utc
                target_start_date = pd.to_datetime(start_date).date()
                target_end_date = pd.to_datetime(end_date).date()

                # Process a range of days
                print(f"Processing in multiple mode for {target_start_date} to {target_end_date}")

                # Use direct comparison instead of isin with date_range
                filtered_df = raw_df[
                    (raw_df['datetime_utc'].dt.date >= target_start_date) & 
                    (raw_df['datetime_utc'].dt.date <= target_end_date)
                ]
                return filtered_df
            
            else:
                raise ValueError(f"Invalid transform type: {mode}")

    def _save_transformed_data(self, raw_df: pd.DataFrame, mercado: str):
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
                print("\n❌ TRANSFORMATION FAILED")
                print(f"Transformation resulted in empty DataFrame for {mercado}")
                print("="*80 + "\n")
                return

            print("\n📋 PROCESSED DATA PREVIEW")
            print("-"*50)
            print("First 5 records:")
            print(processed_df.head().to_string(
                index=True,
                justify='right',
                float_format=lambda x: f"{x:.6f}"
            ))
            print("-"*50)

            # 2. Save Processed Data
            print("\n💾 SAVING DATA")
            print("-"*50)
            print(f"Records to save: {len(processed_df)}")
            print(f"Target market: {mercado}")
            print(f"Dataset type: {self.dataset_type}")
            
            self.processed_file_utils.write_processed_parquet(
                processed_df, 
                mercado, 
                value_col='precio', 
                dataset_type=self.dataset_type
            )

        except Exception as e:
            print("\n❌ ERROR OCCURRED")
            print("-"*50)
            print(f"Operation: {'Transformation' if 'transform' in str(e) else 'Save'}")
            print(f"Market: {mercado}")
            print(f"Error: {str(e)}")
            print("="*80 + "\n")
            return

    def _route_to_market_saver(self, raw_df: pd.DataFrame, mercado: str):
        """Routes the data to appropriate market-specific processor based on market type."""
        if mercado not in ESIOSConfig().esios_precios_markets:
            raise ValueError(f"Invalid market: {mercado}. Must be one of: {ESIOSConfig().esios_precios_markets}")

        if mercado == 'diario':
            self.process_diario_market(raw_df)
        elif mercado == 'intra':
            self.process_intra_market(raw_df)
        elif mercado == 'secundaria':
            self.process_secundaria_market(raw_df)
        elif mercado == 'terciaria':
            self.process_terciaria_market(raw_df)
        elif mercado == 'rr':
            self.process_rr_market(raw_df)

    def _process_batch_mode(self, mercado):
        """Processes all historical data for a given market."""
        print("\n" + "="*80)
        print(f"🔄 STARTING BATCH TRANSFORM")
        print(f"Market: {mercado.upper()}")
        print("="*80 + "\n")

        years = self.raw_file_utils.get_raw_folder_list(mercado)
        
        for year in years:
            months = self.raw_file_utils.get_raw_folder_list(mercado, year)
            
            print(f"\n📅 Processing Year: {year}")
            print("-"*50)
            
            for month in months:
                try:
                    print(f"📌 Processing {year}-{month:02d}...")
                    raw_df = self.raw_file_utils.read_raw_file(year, month, self.dataset_type, mercado)
                    
                    if raw_df.empty:
                        print(f"⚠️  Empty file for {year}-{month:02d}")
                        continue
                    
                    print(f"   Records found: {len(raw_df)}")
                    raw_df = self._filter_data_by_mode(raw_df, 'batch')
                    self._route_to_market_saver(raw_df, mercado)
                    print(f"✅ Successfully processed {year}-{month:02d}")

                except Exception as e:
                    print(f"❌ Error processing {year}-{month:02d}: {e}")
                    continue
            
            print("-"*50)
        
        print("\n✅ BATCH PROCESSING COMPLETE")
        print("="*80 + "\n")

    def _process_single_day(self, mercado, date: str):
        """Processes data for a single specified day."""
        print("\n" + "="*80)
        print(f"🔄 STARTING SINGLE DAY TRANSFORM")
        print(f"Market: {mercado.upper()}")
        print(f"Date: {date}")
        print("="*80 + "\n")

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
                return

            print("📂 CHECKING DATA AVAILABILITY")
            print("-"*50)
            
            # Check if the required year and month folders exist
            available_years = self.raw_file_utils.get_raw_folder_list(mercado)
            if target_year not in available_years:
                print("❌ DATA NOT FOUND")
                print(f"No data available for year {target_year}")
                print("="*80 + "\n")
                return

            available_months = self.raw_file_utils.get_raw_folder_list(mercado, target_year)
            if target_month not in available_months:
                print("❌ DATA NOT FOUND")
                print(f"No data available for {target_year}-{target_month:02d}")
                print("="*80 + "\n")
                return

            print("✅ Data files available")
            print("-"*50 + "\n")

            # Read and process data
            print("🔄 PROCESSING DATA")
            print("-"*50)
            print(f"1. Reading raw file for {target_year}-{target_month:02d}...")
            raw_df = self.raw_file_utils.read_raw_file(target_year, target_month, self.dataset_type, mercado)
            print(f"   Records found: {len(raw_df)}")
            
            print("\n2. Filtering for target date...")
            filtered_df = self._filter_data_by_mode(raw_df, mode='single', start_date=date, end_date=date)

            if filtered_df.empty:
                print("\n❌ NO DATA FOUND")
                print(f"No records for {date}")
                print(f"Available date range: {raw_df['datetime_utc'].dt.date.min()} to {raw_df['datetime_utc'].dt.date.max()}")
                print("="*80 + "\n")
                return
            
            print(f"   Filtered records: {len(filtered_df)}")
            
            print("\n3. Applying market transformation...")
            self._route_to_market_saver(filtered_df, mercado)
            print("-"*50)

            print("\n✅ PROCESS COMPLETE")
            print(f"Successfully processed {mercado} data for {date}")
            print("="*80 + "\n")

        except FileNotFoundError:
            print("\n❌ FILE ERROR")
            print(f"Raw file not found for {target_year}-{target_month:02d}")
            print("="*80 + "\n")
        except Exception as e:
            print("\n❌ UNEXPECTED ERROR")
            print(f"Error processing data: {str(e)}")
            print("="*80 + "\n")

    def _process_latest_day(self, mercado):
        """Processes the most recent day's data."""
        print("\n" + "="*80)
        print(f"🔄 STARTING LATEST DAY TRANSFORM")
        print(f"Market: {mercado.upper()}")
        print("="*80 + "\n")

        try:
            print("📂 LOCATING LATEST DATA")
            print("-"*50)
            
            # Get the latest year
            years = sorted(self.raw_file_utils.get_raw_folder_list(mercado), reverse=True)
            if not years:
                print("❌ NO DATA FOUND")
                print(f"No data available for market {mercado}")
                print("="*80 + "\n")
                return
            
            latest_year = years[0]
            print(f"✓ Latest year: {latest_year}")
            
            # Get the latest month
            months = sorted(self.raw_file_utils.get_raw_folder_list(mercado, latest_year), reverse=True)
            if not months:
                print("❌ NO DATA FOUND")
                print(f"No data available for year {latest_year}")
                print("="*80 + "\n")
                return
            
            latest_month = months[0]
            print(f"✓ Latest month: {latest_month:02d}")
            print("-"*50 + "\n")
            
            # Process data
            print("🔄 PROCESSING DATA")
            print("-"*50)
            print(f"1. Reading latest file ({latest_year}-{latest_month:02d})...")
            raw_df = self.raw_file_utils.read_raw_file(latest_year, latest_month, self.dataset_type, mercado)
            print(f"   Records found: {len(raw_df)}")

            print("\n2. Filtering for latest day...")
            raw_df = self._filter_data_by_mode(raw_df, 'latest')
            if raw_df.empty:
                print("❌ NO DATA FOUND")
                print("No records found in latest file")
                print("="*80 + "\n")
                return
            
            latest_date = raw_df['datetime_utc'].dt.date.max()
            print(f"   Latest date: {latest_date}")
            print(f"   Filtered records: {len(raw_df)}")

            print("\n3. Applying market transformation...")
            self._route_to_market_saver(raw_df, mercado)
            print("-"*50)

            print("\n✅ PROCESS COMPLETE")
            print(f"Successfully processed latest {mercado} data")
            print(f"Date processed: {latest_date}")
            print("="*80 + "\n")

        except Exception as e:
            print("\n❌ UNEXPECTED ERROR")
            print(f"Error processing latest data: {str(e)}")
            print("="*80 + "\n")
    
    def _process_date_range(self, mercado: str, start_date: str, end_date: str):
        """Processes data for a specified date range."""
        print("\n" + "="*80)
        print(f"🔄 STARTING MULTIPLE TRANSFORM")
        print(f"Market: {mercado.upper()}")
        print(f"Period: {start_date} to {end_date}")
        print("="*80 + "\n")

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
            print("\n📂 FILE PROCESSING")
            print("-"*50)
            for year, month in year_months:
                try:
                    print(f"📌 Reading {year}-{month:02d}...")
                    df = self.raw_file_utils.read_raw_file(year, month, self.dataset_type, mercado)

                    if not df.empty:
                        all_raw_dfs.append(df)
                        processed_any_file = True
                        print(f"✅ Successfully read: {year}-{month:02d}")
                    else:
                        print(f"⚠️  Empty file: {year}-{month:02d}")

                except FileNotFoundError:
                    print(f"❌ File not found: {year}-{month:02d}")
                except Exception as e:
                    print(f"❌ Error reading {year}-{month:02d}: {str(e)}")
                    continue
            print("-"*50 + "\n")

            if not processed_any_file:
                 print("❌ ERROR: No files processed")
                 print(f"No data available for {mercado} between {start_date} and {end_date}")
                 return

            # Data Processing Steps
            print("🔄 PROCESSING DATA")
            print("-"*50)
            print("1. Concatenating dataframes...")
            combined_raw_df = pd.concat(all_raw_dfs, ignore_index=True)
            print(f"   Combined shape: {combined_raw_df.shape}")

            print("\n2. Filtering date range...")
            processed_df = self._filter_data_by_mode(
                combined_raw_df,
                'multiple',
                start_date=start_date,
                end_date=end_date
            )

            if processed_df.empty:
                print("❌ ERROR: No data after filtering")
                print(f"No records found for {mercado} in specified date range")
                return

            print(f"   Filtered shape: {processed_df.shape}")

            print("\n3. Applying market transformation...")
            self._route_to_market_saver(processed_df, mercado)
            print("-"*50)

            print("\n✅ PROCESS COMPLETE")
            print(f"Successfully processed {mercado} data")
            print(f"Period: {start_date} to {end_date}")
            print("="*80 + "\n")

        except ValueError as ve:
             print("\n❌ VALIDATION ERROR")
             print(f"Error in {mercado} ({start_date} to {end_date})")
             print(f"Details: {str(ve)}")
             print("="*80 + "\n")
        except Exception as e:
            print("\n❌ UNEXPECTED ERROR")
            print(f"Error in {mercado} ({start_date} to {end_date})")
            print(f"Details: {str(e)}")
            print(traceback.format_exc())
            print("="*80 + "\n")

    def transform_data_for_all_markets(self, start_date: str = None, end_date: str = None, mercados: list[str] = None, mode: str = 'latest') -> None:
        """
        Transforms data for all specified markets based on the mode.
        
        Args:
            mercados: List of market names to process. If None, processes all markets.
            mode: One of 'latest', 'batch', 'single', or 'multiple'
        """
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
        
        try:
            for mercado in mercados:
                if mode == 'batch':
                    self._process_batch_mode(mercado)
                elif mode == 'single':
                    self._process_single_day(mercado, start_date)
                elif mode == 'latest':
                    self._process_latest_day(mercado)
                elif mode == 'multiple':
                    self._process_date_range(mercado, start_date, end_date)
            
        except Exception as e:
            print(f"Error transforming data for {mercado}: {e}")
            return

    def process_diario_market(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        mercado = 'diario'
        self._save_transformed_data(raw_df, mercado)
        return

    def process_intra_market(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        mercado = 'intra'
        self._save_transformed_data(raw_df, mercado)
        return

    def process_secundaria_market(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        mercado = 'secundaria'
        self._save_transformed_data(raw_df, mercado)
        return

    def process_terciaria_market(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        mercado = 'terciaria'
        self._save_transformed_data(raw_df, mercado)
        return

    def process_rr_market(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        mercado = 'rr'
        self._save_transformed_data(raw_df, mercado)
        return
    