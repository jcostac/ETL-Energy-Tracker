import pandas as pd
from datetime import datetime
import pytz
import pretty_errors
import sys
from pathlib import Path
from typing import Optional, List, Dict
import traceback

# Add necessary imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utilidades.etl_date_utils import DateUtilsETL
from utilidades.raw_file_utils import RawFileUtils
from utilidades.processed_file_utils import ProcessedFileUtils
from transform.procesadores._procesador_omie import OMIEProcessor
from configs.omie_config import OMIEConfig, DiarioConfig, IntraConfig, IntraContinuoConfig

class TransformadorOMIE:
    """
    OMIE data transformer class that handles processing of diario, intra, and continuo market data.
    Supports multiple transformation modes and produces structured results with status tracking.
    """
    
    def __init__(self):
        """
        Initialize the TransformadorOMIE instance with utility classes, supported transformation modes, OMIE market types, and market configuration mappings.
        """
        # Initialize core components
        self.processor = OMIEProcessor()
        self.raw_file_utils = RawFileUtils()
        self.processed_file_utils = ProcessedFileUtils()
        self.date_utils = DateUtilsETL()

        # Define dataset types and transformation modes
        self.transform_types = ['latest', 'single', 'multiple']


        # OMIE market configuration
        self.omie_markets = ['diario', 'intra', 'continuo']
        

        # Map market names to config classes
        self.market_config_map = {
            'diario': DiarioConfig,
            'intra': IntraConfig,
            'continuo': IntraContinuoConfig
        }

    def _filter_data_by_mode(self, raw_df: pd.DataFrame, transform_type: str, fecha_inicio: str = None, fecha_fin: str = None) -> pd.DataFrame:
        """
        Filter a DataFrame of OMIE market data according to the specified transformation mode and date parameters.
        
        Depending on the `transform_type`, this method selects rows for the latest available date, a single specified date, or a date range. It ensures the 'Fecha' column is present and valid, removing rows with missing or invalid dates before filtering. Returns an empty DataFrame if filtering is not possible due to missing data or errors.
        
        Parameters:
            raw_df (pd.DataFrame): Input DataFrame containing OMIE market data with a 'Fecha' column.
            transform_type (str): Transformation mode; one of 'latest', 'single', or 'multiple'.
            fecha_inicio (str, optional): Start date for filtering (required for 'single' and 'multiple' modes).
            fecha_fin (str, optional): End date for filtering (required for 'multiple' mode).
        
        Returns:
            pd.DataFrame: Filtered DataFrame containing only the rows matching the specified date criteria, or an empty DataFrame if filtering fails.
        """
        # Validate inputs
        if transform_type == 'multiple' and (fecha_inicio is None or fecha_fin is None):
            raise ValueError("fecha_inicio and fecha_fin must be provided for multiple transform_type")
        if transform_type == 'single' and fecha_inicio is None:
            raise ValueError("fecha_inicio must be provided for single transform_type")

        if raw_df.empty:
            print("⚠️  Input DataFrame is empty, returning empty DataFrame.")
            return pd.DataFrame()

        # Ensure we have a date column to work with
        if 'Fecha' not in raw_df.columns:
            print("❌ Missing 'Fecha' column in raw data. Cannot apply date filtering.")
            return pd.DataFrame()

        # Ensure Fecha column is properly converted to datetime
        if not pd.api.types.is_datetime64_any_dtype(raw_df['Fecha']):
            # Store original values before conversion
            original_dates = raw_df['Fecha'].copy()

            # Convert with coerce - problematic dates become NaT
            raw_df['Fecha'] = pd.to_datetime(raw_df['Fecha'], errors='coerce')

            # Find which dates failed to convert
            failed_mask = raw_df['Fecha'].isna()
            failed_dates = original_dates[failed_mask]

            # Check for any NaT values that were created
            nat_count = raw_df['Fecha'].isna().sum()
            if nat_count > 0:
                print(f"⚠️ Found {nat_count} invalid dates converted to NaT")
                print(f"\n Failed dates: {failed_dates.head()}")

        # Remove rows with NaT (Not a Time) values in Fecha column that will casue errors in the filtering process by datetime
        initial_count = len(raw_df)
        raw_df = raw_df.dropna(subset=['Fecha'])
        cleaned_count = len(raw_df)
        
        if initial_count != cleaned_count:
            print(f"⚠️  Removed {initial_count - cleaned_count} rows with invalid/missing dates")
        
        if raw_df.empty:
            print("❌ No valid dates found after cleaning. Cannot apply date filtering.")
            return pd.DataFrame()

        print(f"ℹ️  Processing in {transform_type.upper()} transform_type")

        try:
            if transform_type == 'latest':
                # Process only the last day in the dataframe
                last_day = raw_df['Fecha'].dt.date.max()
                print(f"📅 Latest day in data: {last_day}")
                filtered_df = raw_df[raw_df['Fecha'].dt.date == last_day]
                print(f"📊 Records found after filtering: {len(filtered_df)}")
                return filtered_df

            elif transform_type == 'single':
                # Process a single day
                target_date = pd.to_datetime(fecha_inicio).date()
                print(f"📅 Single day selected: {target_date}")
                filtered_df = raw_df[raw_df['Fecha'].dt.date == target_date]
                print(f"📊 Records found: {len(filtered_df)}")
                return filtered_df

            elif transform_type == 'multiple':
                # Process a range of days
                target_fecha_inicio = pd.to_datetime(fecha_inicio).date()
                target_fecha_fin = pd.to_datetime(fecha_fin).date()
                print(f"📅 Processing range: {target_fecha_inicio} to {target_fecha_fin}")
                
                filtered_df = raw_df[
                    (raw_df['Fecha'].dt.date >= target_fecha_inicio) & 
                    (raw_df['Fecha'].dt.date <= target_fecha_fin)
                ]
                print(f"📊 Records found in range: {len(filtered_df)}")
                return filtered_df
            
            else:
                raise ValueError(f"Invalid transform type: {transform_type}")

        except Exception as e:
            print(f"❌ Error during date filtering ({transform_type} transform_type): {e}")
            return pd.DataFrame()

    def _transform_market_data(self, raw_df: pd.DataFrame, mercado: str) -> Optional[pd.DataFrame]:
        """
        Transforms raw OMIE market data for the specified market type and returns the processed DataFrame.
        
        Parameters:
        	raw_df (pd.DataFrame): Raw input data to be transformed.
        	mercado (str): Market type to process ('diario', 'intra', or 'continuo').
        
        Returns:
        	pd.DataFrame or None: The transformed DataFrame if successful; otherwise, None if transformation fails or results in empty data.
        """
        try:
            print("\n" + "="*80)
            print(f"🔄 TRANSFORMING DATA FOR {mercado.upper()}")
            print("="*80)

            print("\n📊 INPUT DATA")
            print("-"*50)
            print(f"Raw records to process: {len(raw_df)}")

            # Route to appropriate transformation method
            if mercado == 'diario':
                processed_df = self.processor.transform_omie_diario(raw_df)
            elif mercado == 'intra':
                processed_df = self.processor.transform_omie_intra(raw_df)
            elif mercado == 'continuo':
                processed_df = self.processor.transform_omie_continuo(raw_df)
            else:
                raise ValueError(f"Unknown market: {mercado}")

            if processed_df is None or processed_df.empty:
                print("\n❌ TRANSFORMATION RESULTED IN EMPTY DATAFRAME")
                print(f"Market: {mercado}")
                print("="*80 + "\n")
                return None

            print("\n📋 PROCESSED DATA PREVIEW")
            print("-"*50)
            print("First 5 records:")
            print(processed_df.head().to_string(
                index=True,
                justify='right',
                float_format=lambda x: f"{x:.6f}" if pd.api.types.is_numeric_dtype(type(x)) else str(x)
            ))
            print("-"*50)
            print("Last 5 records:")
            print(processed_df.tail().to_string(
                index=True,
                justify='right',
                float_format=lambda x: f"{x:.6f}" if pd.api.types.is_numeric_dtype(type(x)) else str(x)
            ))
            print("-"*50)

            print(f"✅ Transformation successful for {mercado}. Records processed: {len(processed_df)}")
            print("="*80 + "\n")
            return processed_df

        except Exception as e:
            print("\n❌ ERROR DURING TRANSFORMATION")
            print("-"*50)
            print(f"Market: {mercado}")
            print(f"Error: {str(e)}")
            print(traceback.format_exc())
            print("="*80 + "\n")
            return None

    def _process_single_day(self, mercado: str, date: str) -> Optional[pd.DataFrame]:
        """
        Process and transform OMIE market data for a single specified date.
        
        Attempts to locate and read the raw data file for the given market and date, filters the data to the target day, and applies the appropriate market transformation. Returns the processed DataFrame, or `None` if the data is unavailable, the date is invalid, or processing fails.
        
        Parameters:
        	mercado (str): The market name ('diario', 'intra', or 'continuo').
        	date (str): The target date in 'YYYY-MM-DD' format.
        
        Returns:
        	Optional[pd.DataFrame]: The processed DataFrame for the specified date, or `None` if processing is unsuccessful.
        """
        print("\n" + "="*80)
        print(f"🔄 STARTING SINGLE DAY TRANSFORM")
        print(f"Market: {mercado.upper()}")
        print(f"Date: {date}")
        print("="*80 + "\n")

        processed_df = None
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
            raw_df = self.raw_file_utils.read_raw_file(target_year, target_month, "volumenes_omie", mercado)
            print(f"   Records found: {len(raw_df)}")

            print("\n2. Filtering for target date...")
            filtered_df = self._filter_data_by_mode(raw_df, transform_type='single', fecha_inicio=date)

            if filtered_df.empty:
                print("\n❌ NO DATA FOUND FOR TARGET DATE")
                print(f"No records for {date} in the raw file.")
                if not raw_df.empty and 'Fecha' in raw_df.columns:
                    print(f"Available date range in file: {raw_df['Fecha'].dt.date.min()} to {raw_df['Fecha'].dt.date.max()}")
                print("="*80 + "\n")
                return None

            print(f"   Filtered records for {date}: {len(filtered_df)}")

            print("\n3. Applying market transformation...")
            processed_df = self._transform_market_data(filtered_df, mercado)
            print("-"*50)

            if processed_df is not None and not processed_df.empty:
                print("\n✅ PROCESS COMPLETE")
                print(f"Successfully transformed {mercado} data for {date}. Records: {len(processed_df)}")
            elif processed_df is None:
                print("\n❌ TRANSFORMATION FAILED")
            else:
                print("\n⚠️  TRANSFORMATION RESULTED IN EMPTY DATAFRAME")

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

        return processed_df

    def _check_raw_file_exists(self, mercado: str, year: int, month: int, dataset_type: str = "volumenes_omie") -> bool:
        """
        Determine whether a raw data file exists and contains data for the specified market, year, month, and dataset type.
        
        Returns:
            True if the file exists and is non-empty; otherwise, False.
        """
        try:
            # Try to read the file - if it succeeds, the file exists and is valid
            raw_df = self.raw_file_utils.read_raw_file(year, month, dataset_type, mercado)
            return not raw_df.empty
        except (FileNotFoundError, Exception):
            return False

    def _find_latest_available_file(self, mercado: str, dataset_type: str = "volumenes_omie") -> Optional[tuple]:
        """
        Searches for the most recent year and month combination that contains a valid raw data file for the specified market and dataset type.
        
        Parameters:
        	mercado (str): The market name to search within.
        
        Returns:
        	A tuple (year, month) of the latest available file if found, or None if no valid file exists.
        """
        print("🔍 SEARCHING FOR LATEST AVAILABLE FILE")
        print("-"*50)
        
        # Get all available years in descending order
        years = sorted(self.raw_file_utils.get_raw_folder_list(mercado), reverse=True)
        if not years:
            print("❌ No year folders found")
            return None
            
        print(f"✓ Available years: {years}")
        
        # Search through years and months to find the latest file
        for year in years:
            months = sorted(self.raw_file_utils.get_raw_folder_list(mercado, year), reverse=True)
            if not months:
                print(f"⚠️  No month folders found for year {year}")
                continue
                
            print(f"📅 Checking year {year}, available months: {months}")
            
            for month in months:
                print(f"   📌 Checking {year}-{month:02d} for {dataset_type} file...")
                raw_file_exists = self._check_raw_file_exists(mercado, year, month, dataset_type)
                if raw_file_exists: #if true
                    print(f"   ✅ Found valid file: {year}-{month:02d}")
                    return (year, month)
                else:
                    print(f"   ❌ No valid {dataset_type} file found")
        
        print("❌ NO VALID DATA FOUND")
        print(f"No {dataset_type} files found for market {mercado}")
        return None

    def _process_latest_day(self, mercado: str) -> Optional[pd.DataFrame]:
        """
        Processes and transforms the most recent available day's data for the specified OMIE market.
        
        Parameters:
        	mercado (str): The market type to process ('diario', 'intra', or 'continuo').
        
        Returns:
        	pd.DataFrame or None: The processed DataFrame for the latest day, or None if no data is available or processing fails.
        """
        print("\n" + "="*80)
        print(f"🔄 STARTING LATEST DAY TRANSFORM")
        print(f"Market: {mercado.upper()}")
        print("="*80 + "\n")

        processed_df = None

        try:
            print("📂 LOCATING LATEST DATA")
            print("-"*50)

            # Find the latest year/month combination that actually has a volumenes_omie file
            latest_file_info = self._find_latest_available_file(mercado, "volumenes_omie")
            
            if latest_file_info is None:
                print("="*80 + "\n")
                return None
                
            latest_year, latest_month = latest_file_info
            print(f"✓ Latest file found: {latest_year}-{latest_month:02d}")
            print("-"*50 + "\n")

            # Process data
            print("🔄 PROCESSING DATA")
            print("-"*50)
            print(f"1. Reading latest raw file ({latest_year}-{latest_month:02d})...")
            raw_df = self.raw_file_utils.read_raw_file(latest_year, latest_month, "volumenes_omie", mercado)
            print(f"   Records found: {len(raw_df)}")

            if raw_df.empty:
                print("❌ Raw file is empty. Cannot process latest day.")
                print("="*80 + "\n")
                return None

            print("\n2. Filtering for latest day in the file...")
            filtered_df = self._filter_data_by_mode(raw_df, 'latest')

            if filtered_df.empty:
                print("❌ NO RECORDS FOUND FOR LATEST DAY")
                print("The latest raw file might not contain data for the expected latest date.")
                if 'Fecha' in raw_df.columns:
                    print(f"Latest date in file was: {raw_df['Fecha'].dt.date.max()}")
                print("="*80 + "\n")
                return None

            latest_date = filtered_df['Fecha'].dt.date.max()
            print(f"   Latest date being processed: {latest_date}")
            print(f"   Filtered records for latest day: {len(filtered_df)}")

            print("\n3. Applying market transformation...")
            processed_df = self._transform_market_data(filtered_df, mercado)
            print("-"*50)

            if processed_df is not None and not processed_df.empty:
                print("\n✅ PROCESS COMPLETE")
                print(f"Successfully transformed latest {mercado} data.")
                print(f"Date processed: {latest_date}. Records: {len(processed_df)}")
            elif processed_df is None:
                print("\n❌ TRANSFORMATION FAILED")
            else:
                print("\n⚠️  TRANSFORMATION RESULTED IN EMPTY DATAFRAME")

            print("="*80 + "\n")

        except FileNotFoundError:
            print("\n❌ FILE ERROR")
            print(f"Could not find raw file for latest period.")
            print("="*80 + "\n")
            processed_df = None
        except Exception as e:
            print("\n❌ UNEXPECTED ERROR")
            print(f"Error processing latest data: {str(e)}")
            print(traceback.format_exc())
            print("="*80 + "\n")
            processed_df = None

        return processed_df

    def _process_date_range(self, mercado: str, fecha_inicio: str, fecha_fin: str) -> Optional[pd.DataFrame]:
        """
        Processes and transforms OMIE market data for a specified date range, returning a combined DataFrame.
        
        Parameters:
        	mercado (str): The market type to process ('diario', 'intra', or 'continuo').
        	fecha_inicio (str): Start date in 'YYYY-MM-DD' format.
        	fecha_fin (str): End date in 'YYYY-MM-DD' format.
        
        Returns:
        	pd.DataFrame or None: The processed DataFrame containing data within the specified date range, or None if processing fails or no data is found.
        """
        print("\n" + "="*80)
        print(f"🔄 STARTING MULTIPLE DAY TRANSFORM (Date Range)")
        print(f"Market: {mercado.upper()}")
        print(f"Period: {fecha_inicio} to {fecha_fin}")
        print("="*80 + "\n")

        processed_df_final = None

        try:
            # Parse dates
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

            year_months = sorted(list(year_months))

            all_raw_dfs = []
            processed_any_file = False

            # Read data for each relevant year-month
            print("\n📂 RAW FILE READING")
            print("-"*50)
            for year, month in year_months:
                try:
                    print(f"📌 Attempting to read {year}-{month:02d}...")
                    df = self.raw_file_utils.read_raw_file(year, month, "volumenes_omie", mercado)

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
                    continue

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
            filtered_df = self._filter_data_by_mode(
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
            processed_df_final = self._transform_market_data(filtered_df, mercado)
            print("-"*50)

            if processed_df_final is not None and not processed_df_final.empty:
                print("\n✅ PROCESS COMPLETE")
                print(f"Successfully transformed {mercado} data for the period.")
                print(f"Period: {fecha_inicio} to {fecha_fin}. Records: {len(processed_df_final)}")
            elif processed_df_final is None:
                print("\n❌ TRANSFORMATION FAILED")
            else:
                print("\n⚠️  TRANSFORMATION RESULTED IN EMPTY DATAFRAME")

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

        return processed_df_final

    def transform_data_for_all_markets(self, fecha_inicio: str = None, fecha_fin: str = None,
                                      mercados_lst: List[str] = None) -> Dict:
        """
        Transforms OMIE market data for specified markets and date parameters, returning processed results and detailed status.
        
        Parameters:
            fecha_inicio (str, optional): Start date in 'YYYY-MM-DD' format. Determines the transformation mode if provided.
            fecha_fin (str, optional): End date in 'YYYY-MM-DD' format. Used for date range transformations.
            mercados_lst (List[str], optional): List of OMIE market names to process. If not provided, all supported markets are processed.
        
        Returns:
            Dict: A dictionary with:
                - 'data': Processed DataFrames for each market (or None if processing failed).
                - 'status': Dictionary containing:
                    - 'success': Boolean indicating if at least one market was processed successfully.
                    - 'details': Dictionary with lists of processed and failed markets, the transformation type used, the date range, and error details if any.
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
            if transform_type not in self.transform_types:
                raise ValueError(f"Invalid transform type: {transform_type}. Must be one of: {self.transform_types}")
                
            # Add validation for required parameters
            if transform_type == 'single' and not fecha_inicio:
                raise ValueError("fecha_inicio must be provided for single transform_type")
            if transform_type == 'multiple' and (not fecha_inicio or not fecha_fin):
                raise ValueError("Both fecha_inicio and fecha_fin must be provided for multiple transform_type")
            
            # Get list of all markets to process
            if mercados_lst is None:
                mercados_lst = self.omie_markets
            elif isinstance(mercados_lst, str):  # Allow single market string
                mercados_lst = [mercados_lst]
                
            # Validate markets
            invalid_markets = [m for m in mercados_lst if m not in self.omie_markets]
            if invalid_markets:
                raise ValueError(f"Invalid markets: {invalid_markets}. Must be one of: {self.omie_markets}")
                
            print("\n" + "="*80)
            print(f"🚀 STARTING OMIE TRANSFORMATION PIPELINE")
            print(f"Mode: {transform_type.upper()}")
            print(f"Markets: {', '.join(mercados_lst)}")
            if fecha_inicio:
                print(f"Date Range: {fecha_inicio}" + (f" to {fecha_fin}" if fecha_fin else ""))
            print("="*80)
                
            # Process each market and track status
            for mercado in mercados_lst:
                market_result = None
                try:
                    print(f"\n🏭 PROCESSING MARKET: {mercado.upper()}")
                    print(f"Dataset Type: volumenes_omie")
                    print("-"*60)
                    
                    # Process each market based on the transform type
                    if transform_type == 'single':
                        market_result = self._process_single_day(mercado, fecha_inicio)
                    elif transform_type == 'latest':
                        market_result = self._process_latest_day(mercado)
                    elif transform_type == 'multiple':
                        market_result = self._process_date_range(mercado, fecha_inicio, fecha_fin)
                    
                    # Check if transformation was successful
                    if market_result is not None and (
                        (isinstance(market_result, pd.DataFrame) and not market_result.empty) or
                        (isinstance(market_result, list) and any(df is not None and not df.empty for df in market_result))
                    ):
                        status_details["markets_processed"].append(mercado)
                        results[mercado] = market_result
                        print(f"✅ Market {mercado} processed successfully")
                    else:
                        status_details["markets_failed"].append({
                            "market": mercado,
                            "error": "Transformation produced no valid data"
                        })
                        overall_success = False
                        results[mercado] = None
                        print(f"⚠️  Market {mercado} produced no valid data")
                        
                except Exception as e:
                    status_details["markets_failed"].append({
                        "market": mercado,
                        "error": str(e)
                    })
                    overall_success = False
                    results[mercado] = None
                    print(f"❌ Market {mercado} failed: {str(e)}")
                    
            # If all markets failed, consider the entire transformation failed
            if not status_details["markets_processed"]:
                overall_success = False
                
            print("\n" + "="*80)
            print(f"🏁 OMIE TRANSFORMATION PIPELINE COMPLETE")
            print(f"✅ Successful: {len(status_details['markets_processed'])}")

            if not overall_success:
                print(f"❌ Failed: {len(status_details['markets_failed'])}")
                print("="*80 + "\n")
                
        except Exception as e:
            overall_success = False
            status_details["error"] = str(e)
            print(f"\n❌ PIPELINE ERROR: {str(e)}")
            print("="*80 + "\n")
            
        # Return both the results and the status
        return {
            "data": results,
            "status": {
                "success": overall_success,
                "details": status_details
            }
        }

    def process_diario_market(self, raw_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Transform raw OMIE data for the 'diario' market.
        
        Parameters:
        	raw_df (pd.DataFrame): Raw input DataFrame containing OMIE market data.
        
        Returns:
        	Optional[pd.DataFrame]: Processed DataFrame for the 'diario' market, or None if transformation fails or results in empty data.
        """
        return self._transform_market_data(raw_df, 'diario')

    def process_intra_market(self, raw_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Transforms raw OMIE data for the 'intra' market and returns the processed DataFrame.
        
        Parameters:
        	raw_df (pd.DataFrame): Raw input data for the 'intra' market.
        
        Returns:
        	Optional[pd.DataFrame]: Processed DataFrame for the 'intra' market, or None if transformation fails or results in empty data.
        """
        return self._transform_market_data(raw_df, 'intra')

    def process_continuo_market(self, raw_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Transforms raw OMIE data for the 'continuo' market and returns the processed DataFrame.
        
        Parameters:
        	raw_df (pd.DataFrame): Raw market data to be transformed.
        
        Returns:
        	Optional[pd.DataFrame]: Processed DataFrame for the 'continuo' market, or None if transformation fails or results in empty data.
        """
        return self._transform_market_data(raw_df, 'continuo')