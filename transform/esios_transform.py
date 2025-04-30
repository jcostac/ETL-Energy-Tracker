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
            print("--------------------------------")
            print(f"Raw data loaded ({len(raw_df)} rows). Starting transformation...")
            print("--------------------------------")
            processed_df = self.processor.transform_price_data(raw_df)
            print("--------------------------------")
            print(f"Processed data head: \n{processed_df.head()}")
            print("--------------------------------")

            if processed_df.empty:
                print(f"Transformation resulted in empty DataFrame for {mercado}.")
                return

        except Exception as e:
            print(f"Error transforming data for {mercado}: {e}")
            return

        try:
        
            # 3. Save Processed Data
            print(f"Transformation complete ({len(processed_df)} rows). Saving processed data...")
            self.processed_file_utils.write_processed_parquet(processed_df, mercado, value_col='precio', dataset_type=self.dataset_type)

        except Exception as e:
            print(f"Error saving processed data for {mercado}: {e}")
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
        years = self.raw_file_utils.get_raw_folder_list(mercado)
        
        for year in years:
            months = self.raw_file_utils.get_raw_folder_list(mercado, year)
            
            for month in months:
                try:
                    print(f"Processing {mercado} for {year}-{month:02d}")
                    raw_df = self.raw_file_utils.read_raw_file(year, month, self.dataset_type, mercado)
                    raw_df = self._filter_data_by_mode(raw_df, 'batch')
                    self._route_to_market_saver(raw_df, mercado)
                    print(f"Successfully processed {mercado} for {year}-{month:02d}")


                except Exception as e:
                    print(f"Error processing {mercado} for {year}-{month:02d}: {e}")
                    continue

    def _process_single_day(self, mercado, date: str):
        """Processes data for a single specified day."""
        try:
            # Parse the date string to get year and month
            try:
                target_date = pd.to_datetime(date)
                target_year = target_date.year
                target_month = target_date.month

            except ValueError:
                print(f"Invalid date format: {date}. Please use YYYY-MM-DD.")
                return

            print(f"Processing data for {mercado} on {date}")

            # Check if the required year and month folders exist
            available_years = self.raw_file_utils.get_raw_folder_list(mercado)
            if target_year not in available_years:
                print(f"No data found for market {mercado} in year {target_year}")
                return

            available_months = self.raw_file_utils.get_raw_folder_list(mercado, target_year)
            if target_month not in available_months:
                print(f"No data found for market {mercado} in year {target_year}, month {target_month:02d}")
                return

            # Read the specific raw file for the given year and month
            raw_df = self.raw_file_utils.read_raw_file(target_year, target_month, self.dataset_type, mercado)
            print("--------------------------------")
            print(f"Raw data before filtering by single day:")
            print("\n", raw_df.head())
            print(f"Shape: \n{raw_df.shape}")
            print("--------------------------------")

            # Filter the DataFrame for the specific date and process
            filtered_df = self._filter_data_by_mode(raw_df, mode='single', start_date=date, end_date=date)

            # Check if data exists for the specific date after filtering
            if filtered_df.empty:
                print("--------------------------------")
                print(f"No data found on {date} for mercado {mercado.upper()} on month {target_month} of year {target_year}.")
                print(f"Range of available dates in the raw dataset:")
                print(f"- {raw_df['datetime_utc'].dt.date.min()} to {raw_df['datetime_utc'].dt.date.max()}")
                print("--------------------------------")
                return
            
            #save data
            self._route_to_market_saver(filtered_df, mercado)

        except FileNotFoundError:
             print(f"Raw file not found for {mercado} for {target_year}-{target_month:02d}.")
        except Exception as e:
            print(f"Error processing data for {mercado} on {date}: {e}")

    def _process_latest_day(self, mercado):
        """Processes the most recent day's data."""
        try:
            # Get the latest year
            years = sorted(self.raw_file_utils.get_raw_folder_list(mercado), reverse=True)
            if not years:
                print(f"No data found for market {mercado}")
                return
            
            latest_year = years[0]
            
            # Get the latest month for the latest year
            months = sorted(self.raw_file_utils.get_raw_folder_list(mercado, latest_year), reverse=True)
            if not months:
                print(f"No data found for market {mercado} in year {latest_year}")
                return
            
            latest_month = months[0]
            
            print(f"Processing latest data for {mercado}: {latest_year}-{latest_month}")
            raw_df = self.raw_file_utils.read_raw_file(latest_year, latest_month, self.dataset_type, mercado)

            print("--------------------------------")
            print(f"Raw data before filtering by single day:")
            print("\n", raw_df.head())
            print(f"Shape: \n{raw_df.shape}")
            print("--------------------------------")

            raw_df = self._filter_data_by_mode(raw_df, 'latest')

            #save data
            self._route_to_market_saver(raw_df, mercado)

        except Exception as e:
            print(f"Error processing latest data for {mercado}: {e}")
    
    def _process_date_range(self, mercado: str, start_date: str, end_date: str):
        """Processes data for a specified date range."""
        print(f"Starting multiple transform for {mercado} from {start_date} to {end_date}")
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
            print(f"Identifying and reading relevant files for {mercado} between {start_date} and {end_date}...")
            for year, month in year_months:
                try:
                    print(f"Attempting to read raw file: {year}-{month:02d} for {mercado}")
                    df = self.raw_file_utils.read_raw_file(year, month, self.dataset_type, mercado)

                    #appned the df to existing df list if not empty
                    if not df.empty:
                        all_raw_dfs.append(df)
                        processed_any_file = True
                        print(f"Successfully read raw file: {year}-{month:02d} for {mercado}")
                    else:
                        print(f"Raw file {year}-{month:02d} for {mercado} is empty. Skipping.")

                except FileNotFoundError:
                    print(f"Warning: Raw file not found for {mercado} for {year}-{month:02d}. Skipping.")
                except Exception as e:
                    print(f"Error reading raw file for {mercado} for {year}-{month:02d}: {e}")
                    continue # Continue with next file

            if not processed_any_file:
                 print(f"No raw files found or successfully read for {mercado} within the specified date range {start_date} to {end_date}.")
                 return

            # Concatenate all dataframes
            print("Concatenating raw dataframes...")
            combined_raw_df = pd.concat(all_raw_dfs, ignore_index=True)
            print(f"Combined raw dataframe shape: {combined_raw_df.shape}")

            # Filter the combined DataFrame for the exact date range using the helper method
            print(f"Filtering combined data for range {start_date} to {end_date}...")
            processed_df = self._filter_data_by_mode(
                combined_raw_df,
                'multiple',
                start_date=start_date,
                end_date=end_date
            )

            # Check if data exists after filtering for the date range
            if processed_df.empty:
                print(f"No data found for {mercado} between {start_date} and {end_date} after filtering the combined data.")
                return

            print(f"Filtered data shape for transformation: {processed_df.shape}")

            # Apply the final transformation and save (assuming _route_to_market_saver handles this)
            print(f"Applying market transformation for {mercado}...")
            self._route_to_market_saver(processed_df, mercado)
            print(f"🆗 Successfully processed and saved data for {mercado} from {start_date} to {end_date} 🆗")

        except ValueError as ve:
             # Catch specific errors like date parsing or invalid range
             print(f"Configuration or Value Error during multiple transform for {mercado} ({start_date}-{end_date}): {ve}")
        except Exception as e:
            print(f"An unexpected error occurred during multiple transform for {mercado} ({start_date}-{end_date}): {e}")
            print(traceback.format_exc()) # Print stack trace for debugging

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
    