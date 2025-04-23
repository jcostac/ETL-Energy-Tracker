import pandas as pd
from datetime import datetime
from utilidades.etl_date_utils import DateUtilsETL
import pytz
import pretty_errors
import sys

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
    
    def transform_data_for_all_markets(self, start_date: str = None, end_date: str = None, mercados: list[str] = None, transform_type: str = 'latest') -> None:
        """
        Transforms data for all specified markets based on the transform_type.
        
        Args:
            mercados: List of market names to process. If None, processes all markets.
            transform_type: One of 'latest', 'batch', 'single', or 'multiple'
        """
        if transform_type not in self.transform_types:
            raise ValueError(f"Invalid transform type: {transform_type}. Must be one of: {self.transform_types}")

        # Get list of all markets to process
        if mercados is None:
            mercados = ESIOSConfig().esios_precios_markets
        
        for mercado in mercados:
            if transform_type == 'batch':
                self._transform_batch(mercado)
            elif transform_type == 'single':
                self._transform_single(mercado, start_date)
            elif transform_type == 'latest':
                self._transform_latest(mercado)
            elif transform_type == 'multiple':
                self._transform_multiple(mercado, start_date, end_date)

        return print(f"✅Successfully transformed data for all markets✅")

    def transform_diario(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        mercado = 'diario'
        self._transform_save(raw_df, mercado)
        return

    def transform_intra(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        mercado = 'intra'
        self._transform_save(raw_df, mercado)
        return

    def transform_secundaria(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        mercado = 'secundaria'
        self._transform_save(raw_df, mercado)
        return

    def transform_terciaria(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        mercado = 'terciaria'
        self._transform_save(raw_df, mercado)
        return

    def transform_rr(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        mercado = 'rr'
        self._transform_save(raw_df, mercado)
        return
    
    def _process_df_based_on_transform_type(self, raw_df: pd.DataFrame, transform_type: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
         Transform the dataframe based on the transform type. If single pick the last day in the dataframe.
         If batch process the entire dataframe.

         Args:
            raw_df (pd.DataFrame): The dataframe to transform.
            transform_type (str): The type of transformation to perform.
            start_date (str): The start date to process. Optional for single and multiple modes.
            end_date (str): The end date to process. Optional for multiple modes.

         Returns:
            pd.DataFrame: The transformed dataframe.

        Note:
            - If transform_type is 'single' or 'multiple', start_date and end_date must be provided.
            - For single mode, we only use the start_date to filter the dataframe.
            - For multiple mode, we take all days in the range start_date to end_date (inclusive).
        """

        #check that start_date and end_date are provided for single and multiple modes
        if transform_type == 'multiple' and (start_date is None or end_date is None):
            raise ValueError("start_date and end_date must be provided for multiple mode")
        if transform_type == 'single' and start_date is None:
            raise ValueError("start_date must be provided for single mode")

        if not raw_df.empty:
            if transform_type == 'latest':
                # Process only the last day in the dataframe
                last_day = raw_df['datetime_utc'].dt.date.max()
              
                print(f"Processing in single mode for {last_day}")
                # Filter dataframe to only include the last day
                raw_df = raw_df[raw_df['datetime_utc'].dt.date == last_day]
                return raw_df

            elif transform_type == 'batch':
                # Process the entire dataframe
                print(f"Processing in batch mode with {len(raw_df['datetime_utc'].dt.date.unique())} days")
                return raw_df

            elif transform_type == 'single':  
                #convert start date to series first 
                start_date_series = pd.Series(start_date)    

                #convert from naive to utc
                start_date_utc = self.date_utils.convert_naive_to_utc(start_date_series)

                # Process a single day
                print(f"Processing in single mode for {start_date}")
                raw_df = raw_df[raw_df['datetime_utc'].dt.date == start_date_utc]
                return raw_df
            
            elif transform_type == 'multiple':
                #convert to series first 
                start_date_series = pd.Series(start_date)
                end_date_series = pd.Series(end_date)

                #convert from naive to utc
                start_date_utc = self.date_utils.convert_naive_to_utc(start_date_series)
                end_date_utc = self.date_utils.convert_naive_to_utc(end_date_series)

                # Process a range of days
                print(f"Processing in multiple mode for {start_date} to {end_date}")
                raw_df = raw_df[raw_df['datetime_utc'].dt.date.isin(pd.date_range(start=start_date_utc, end=end_date_utc))]
                return raw_df
            
            else:
                raise ValueError(f"Invalid transform type: {transform_type}")

    def _transform_save(self, raw_df: pd.DataFrame, mercado: str):
        try:
            # 1. Transform Data
            print(f"Raw data loaded ({len(raw_df)} rows). Starting transformation...")
            processed_df = self.processor.transform_price_data(raw_df) # Geo_name defaults to 'España'

            if processed_df.empty:
                print(f"Transformation resulted in empty DataFrame for {mercado}.")
                return

        except Exception as e:
            print(f"Error transforming data for {mercado}: {e}")
            return

        try:
            # 3. Save Processed Data
            print(f"Transformation complete ({len(processed_df)} rows). Saving processed data...")
            self.processed_file_utils.write_processed_parquet(processed_df, mercado)
            date_range = f"{processed_df['datetime_utc'].min().date()} to {processed_df['datetime_utc'].max().date()}"
            print(f"Successfully saved processed {mercado} data for {date_range}.")

        except Exception as e:
            print(f"Error saving processed data for {mercado}: {e}")
            return

    def _transform_market(self, raw_df: pd.DataFrame, mercado: str):
        """
        Calls the appropriate transform method based on market type.
        
        Args:
            raw_df: DataFrame containing the raw data.
            mercado: Market name to determine which transform to apply.
        """
        if mercado == 'diario':
            self.transform_diario(raw_df)
        elif mercado == 'intra':
            self.transform_intra(raw_df)
        elif mercado == 'secundaria':
            self.transform_secundaria(raw_df)
        elif mercado == 'terciaria':
            self.transform_terciaria(raw_df)
        elif mercado == 'rr':
            self.transform_rr(raw_df)

    def _transform_batch(self, mercado):
        """
        Process all years and months for batch mode.
        
        Args:
            mercado: Market name to process.
        """
        years = self.raw_file_utils.get_raw_folder_list(mercado)
        
        for year in years:
            months = self.raw_file_utils.get_raw_folder_list(mercado, year)
            
            for month in months:
                try:
                    print(f"Processing {mercado} for {year}-{month:02d}")
                    raw_df = self.raw_file_utils.read_raw_file(year, month, self.dataset_type, mercado)
                    raw_df = self._process_df_based_on_transform_type(raw_df, 'batch')
                    self._transform_market(raw_df, mercado)
                    print(f"Successfully processed {mercado} for {year}-{month:02d}")

                except Exception as e:
                    print(f"Error processing {mercado} for {year}-{month:02d}: {e}")
                    continue

    def _transform_single(self, mercado, date: str):
        """
        Process only a single day specified by the user.

        Args:
            mercado: Market name to process.
            date: Date to process in the format "YYYY-MM-DD".
        """
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

            # Filter the DataFrame for the specific date and process
            raw_df = self._process_df_based_on_transform_type(raw_df, 'single', start_date=date, end_date=date)

            # Check if data exists for the specific date after filtering
            if raw_df.empty:
                print(f"No data found for {mercado} on the specific date {date} within the file {target_year}-{target_month:02d}.")
                return

            self._transform_market(raw_df, mercado)
            print(f"Successfully processed data for {mercado} on {date}")

        except FileNotFoundError:
             print(f"Raw file not found for {mercado} for {target_year}-{target_month:02d}.")
        except Exception as e:
            print(f"Error processing data for {mercado} on {date}: {e}")

    def _transform_latest(self, mercado):
        """
        Process only the latest data (last day available in the data set).
        
        Args:
            mercado: Market name to process.
        """
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
            raw_df = self._process_df_based_on_transform_type(raw_df, 'latest')
            self._transform_market(raw_df, mercado)
            print(f"Successfully processed latest data for {mercado}: {latest_year}-{latest_month}")

        except Exception as e:
            print(f"Error processing latest data for {mercado}: {e}")
    
    def _transform_multiple(self, mercado: str, start_date: str, end_date: str):
        """
        Process a range of days specified by the user. Reads relevant monthly files,
        concatenates them, filters by the exact date range, and then transforms.

        Args:
            mercado (str): Market name to process.
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.
        """
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
                    # Decide whether to continue or stop on read error
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
            processed_df = self._process_df_based_on_transform_type(
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

            # Apply the final transformation and save (assuming _transform_market handles this)
            print(f"Applying market transformation for {mercado}...")
            self._transform_market(processed_df, mercado)
            print(f"Successfully processed and saved data for {mercado} from {start_date} to {end_date}")

        except ValueError as ve:
             # Catch specific errors like date parsing or invalid range
             print(f"Configuration or Value Error during multiple transform for {mercado} ({start_date}-{end_date}): {ve}")
        except Exception as e:
            # Catch broader exceptions during the process
            import traceback
            print(f"An unexpected error occurred during multiple transform for {mercado} ({start_date}-{end_date}): {e}")
            print(traceback.format_exc()) # Print stack trace for debugging

    