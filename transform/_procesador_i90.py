import pandas as pd
from typing import Dict, List, Optional, Any
import sys
from pathlib import Path
import pytz
import traceback
from datetime import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Import actual config classes from configs.i90_config
from configs.i90_config import (
        I90Config, # Base might be useful too
        DiarioConfig, SecundariaConfig, TerciariaConfig, RRConfig,
        CurtailmentConfig, P48Config, IndisponibilidadesConfig, RestriccionesConfig,
        IntraConfig
    )
from utilidades.etl_date_utils import DateUtilsETL
from utilidades.data_validation_utils import DataValidationUtils
from utilidades.progress_utils import with_progress
from utilidades.storage_file_utils import RawFileUtils


class I90Processor:
    """
    Processor class for I90 data (Volumenes and Precios).
    Handles data cleaning, validation, filtering, and transformation.
    """
    def __init__(self):
        """
        Initializes the I90Processor with utilities for date handling, data validation, and raw file operations.
        """
        self.date_utils = DateUtilsETL()
        self.data_validation_utils = DataValidationUtils()
        self.raw_file_utils = RawFileUtils()
       
    # === FILTERING ===
    def _apply_market_filters_and_id(self, df: pd.DataFrame, market_config: I90Config) -> pd.DataFrame:
        """
        Filter the input DataFrame according to the provided market configuration and assign the appropriate market ID.
        
        If the DataFrame contains the 'sheet_i90_volumenes' column (for intra market data), maps it directly to 'id_mercado' using the configuration mapping. Otherwise, applies filters for each market ID based on 'Sentido' and 'Redespacho' columns as specified in the market configuration, assigns 'id_mercado', and concatenates the results. Returns the filtered DataFrame with the 'id_mercado' column, or an empty DataFrame if no data matches.
         
        Returns:
            pd.DataFrame: Filtered DataFrame with 'id_mercado' assigned, or empty if no rows match the filters.
        """
        if df.empty:
            return pd.DataFrame() # Return empty if input is empty

        # --- Direct mapping for intra markets ---
        if 'sheet_i90_volumenes' in df.columns:
            # Build the mapping from config (invert volumenes_sheet)
            sheet_to_market_id = {
                str(sheet): str(market_id)
                for market_id, sheet in market_config.volumenes_sheet.items()
                if sheet is not None
            }
            # Map the column (ensure both are strings for matching)
            df['id_mercado'] = df['sheet_i90_volumenes'].astype(str).map(sheet_to_market_id)
            df['id_mercado'] = df['id_mercado'].astype(int)
            df = df.drop(columns=['sheet_i90_volumenes'])
            return df


        all_market_dfs = []
        print(f"Applying market filters and id to DataFrame with shape: {df.shape}")

        required_cols = ['volumenes'] #required cols for volumenes
        if 'Sentido' in df.columns: 
            required_cols.append('Sentido')
        if 'Redespacho' in df.columns: 
            required_cols.append('Redespacho')

        # Ensure required columns exist
        if not all(col in df.columns for col in required_cols if col in ['Sentido', 'Redespacho']):
             print(f"Warning: Input DataFrame for market transformation might be missing 'Sentido' or 'Redespacho' columns if required by config.")
             # Decide if this is fatal or if filtering can proceed partially

        # Ensure market_config has the necessary attributes
        if not hasattr(market_config, 'market_ids') or not hasattr(market_config, 'sentido_map') or not hasattr(market_config, 'get_redespacho_filter'):
             print(f"Error: Provided market_config object ({type(market_config).__name__}) is missing required attributes/methods (market_ids, sentido_map, get_redespacho_filter).")
             return pd.DataFrame()

        for market_id in market_config.market_ids:
            # market_id from config is already a string
            sentido = market_config.sentido_map.get(market_id) # Use market_id directly
            redespacho_filter = market_config.get_redespacho_filter(market_id) # Use market_id directly

            filtered_df = df.copy() # Start with the full data for this specific market_id iteration

            # Apply sentido filter
            if sentido and 'Sentido' in filtered_df.columns:
                 # Ensure consistent comparison (e.g., handle case sensitivity if needed)
                filtered_df = filtered_df[filtered_df['Sentido'] == sentido]

            elif sentido and 'Sentido' not in filtered_df.columns:
                 print(f"Warning: Config requires filtering by Sentido='{sentido}' for market_id {market_id}, but 'Sentido' column is missing.")
                 # If sentido filter is required but column missing, no rows will match
                 filtered_df = pd.DataFrame() # Clear DF

            # Apply redespacho filter
            if redespacho_filter and 'Redespacho' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['Redespacho'].isin(redespacho_filter)]
            elif redespacho_filter and 'Redespacho' not in filtered_df.columns:
                 print(f"Warning: Config requires filtering by Redespacho='{redespacho_filter}' for market_id {market_id}, but 'Redespacho' column is missing.")
                 # If redespacho filter is required but column missing, no rows will match
                 filtered_df = pd.DataFrame() # Clear DF


            if not filtered_df.empty:
                # Add the market_id (which is already a string from the config)
                filtered_df['id_mercado'] = market_id
                all_market_dfs.append(filtered_df)
                print(f"Processed Market ID: {market_id}, Filtered rows: {len(filtered_df)}")
            else:
                 print(f"No data matched filters for Market ID: {market_id}")


        if not all_market_dfs:
            return pd.DataFrame() # Return empty if no market_id yielded data

        # Combine results for all market_ids handled by this config
        final_df = pd.concat(all_market_dfs, ignore_index=True)
        # Ensure id_mercado is string type after concat (should be if added as string)
        if 'id_mercado' in final_df.columns:
             final_df['id_mercado'] = final_df['id_mercado'].astype(int)
        return final_df

    # === DATETIME HANDLING ===
    def _standardize_datetime(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """
        Standardizes and converts input datetimes to a UTC column with 15-minute granularity, handling DST transitions and multiple input formats.
        
        Splits the input DataFrame by granularity and DST transition days, applies appropriate datetime parsing and conversion methods for each subset, and combines the results into a single DataFrame with a standardized `datetime_utc` column. Drops intermediate columns and invalid datetimes before returning the processed DataFrame.
        
        Parameters:
            dataset_type (str): Specifies the type of dataset being processed, affecting downstream datetime conversion logic.
        
        Returns:
            pd.DataFrame: DataFrame with standardized UTC datetimes at 15-minute intervals.
        """
        return self.date_utils.standardize_datetime(df, dataset_type)

    @with_progress(message="Processing hourly data...", interval=2)
    def _process_hourly_data(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """
        Processes hourly market data with possible DST suffixes, generating UTC datetimes at 15-minute intervals.
        
        Converts "HH-HH+1" formatted time strings (including 'a'/'b' suffixes for DST fall-back) into timezone-aware local datetimes, then to UTC. Expands each hourly entry into four 15-minute intervals for downstream analysis.
        
        Returns:
            pd.DataFrame: DataFrame with standardized UTC datetimes at 15-minute granularity, or an empty DataFrame on error.
        """
        return self.date_utils.process_hourly_data(df, dataset_type)

    @with_progress(message="Processing 15-minute data...", interval=2)
    def _process_15min_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process 15-minute data (numeric index "1" to "96/92/100").
        Creates timezone-aware datetime_local series and converts to UTC.
        """
        return self.date_utils.process_15min_data(df)

    @with_progress(message="Processing 15-minute data (vectorized)...", interval=2)
    def _process_15min_data_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process 15-minute data (numeric index "1" to "96/92/100") using vectorized operations.
        Creates timezone-aware UTC datetime series by processing each day individually.
        """
        return self.date_utils.process_15min_data_vectorized(df)
        
     # New vectorized version for hourly data
    
    @with_progress(message="Processing hourly data (vectorized)...", interval=2)
    def _process_hourly_data_vectorized(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """
        Vectorized processing of hourly data with possible DST suffixes, converting to UTC and expanding to 15-minute intervals.
        
        This method parses hourly time strings (e.g., "02-03a", "02-03b"), handles daylight saving time transitions using suffixes, localizes datetimes to Europe/Madrid, converts them to UTC, and expands each hour to four 15-minute intervals. Returns a DataFrame with standardized UTC datetimes and 15-minute granularity.
        """
        return self.date_utils.process_hourly_data_vectorized(df, dataset_type)
    
    def _parse_hourly_datetime_local(self, fecha, hora_str) -> pd.Timestamp:
        """
        Parse hourly format data (e.g., "00-01", "02-03a", "02-03b") into a timezone-aware
        datetime in Europe/Madrid timezone.
        
        Args:
            fecha: Date object or datetime object
            hora_str: Hour string in format "HH-HH+1" potentially with 'a' or 'b' suffix
        
        Returns:
            Timezone-aware pd.Timestamp in Europe/Madrid timezone
        """
        return self.date_utils.parse_hourly_datetime_local(fecha, hora_str)

    def _parse_15min_datetime_local(self, fecha, hora_index_str) -> pd.Timestamp:
        """
        Parse a 15-minute interval index for a given date into a timezone-aware datetime in Europe/Madrid, correctly handling daylight saving time transitions.
        
        Parameters:
            fecha: The date of the interval, as a string, pandas Timestamp, or date object.
            hora_index_str: The 1-based index (as string or integer) of the 15-minute interval within the day.
        
        Returns:
            pd.Timestamp: The corresponding timezone-aware datetime in Europe/Madrid.
        
        Raises:
            ValueError: If the interval index is less than 1 or exceeds the number of intervals for the given date.
        """
        return self.date_utils.parse_15min_datetime_local(fecha, hora_index_str)

    # === COLUMN FINALIZATION ===
    def _select_and_finalize_columns(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """
        Standardizes and filters DataFrame columns to match the required schema for the specified dataset type.
        
        Renames columns for consistency (e.g., "Unidad de Programación" to "up", "precios" to "precio", "Tipo Transacción" to "tipo_transaccion") and selects only the columns required for the given dataset type ("volumenes_i90" or "precios_i90"). If the "Tipo Transacción" column is present, it is included in the output.
        """
     
        if "Unidad de Programación" in df.columns:
            df = df.rename(columns={"Unidad de Programación": "up"})

        if dataset_type == 'volumenes_i90':
            required_cols = self.data_validation_utils.processed_volumenes_i90_required_cols.copy()
        elif dataset_type == 'precios_i90':
            #rename precios to precio
            df = df.rename(columns={'precios': 'precio'})
            required_cols = self.data_validation_utils.processed_price_required_cols.copy()

        # ADD TIPO TRANSACCIÓN HANDLING - only if column exists
        if "Tipo Transacción" in df.columns:
            df = df.rename(columns={"Tipo Transacción": "tipo_transaccion"})
            required_cols.append('tipo_transaccion')
        print(f"Filtering columns: {required_cols}")
        return df

    def _get_value_col(self, dataset_type: str) -> Optional[str]:
        """
        Return the name of the value column corresponding to the specified dataset type.
        
        Parameters:
            dataset_type (str): The type of dataset, either 'volumenes_i90' or 'precios_i90'.
        
        Returns:
            str or None: The column name ('volumenes' or 'precios') for the given dataset type, or None if not recognized.
        """
        if dataset_type == 'volumenes_i90':
            return 'volumenes'
        elif dataset_type == 'precios_i90':
            return 'precios'
        return None

    # === VALIDATION ===
    def _validate_raw_data(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """Validate raw data structure."""
        if not df.empty:
            validation_schema_type = "volumenes_i90" if dataset_type == 'volumenes_i90' else "precios_i90" # Map to validation schema names more specifically
            try:
                self.data_validation_utils.validate_raw_data(df, validation_schema_type)
                print("Raw data validation successful.")
            except Exception as e:
                print(f"Error during raw data validation: {e}")
                # Decide if this should return empty df or raise
                # Raising allows the error to propagate up
                raise # Reraise validation error
        else:
            print("Skipping validation for empty DataFrame.")
        return df
    
    def _validate_final_data(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """Validate final data structure."""
        if not df.empty:
            validation_schema_type = "volumenes_i90" if dataset_type == 'volumenes_i90' else "precios_i90" # Map to validation schema names more specifically
            try:
                 # Assuming DataValidationUtils.validate_data expects specific schema names
                 self.data_validation_utils.validate_processed_data(df, validation_schema_type)
                 print("Final data validation successful.")
            except KeyError as e:
                 print(f"Validation Error: Schema '{validation_schema_type}' not found in DataValidationUtils. Skipping validation. Error: {e}")
            except Exception as e:
                 print(f"Error during final data validation: {e}")
                 # Decide if this should return empty df or raise
                 # Raising allows the error to propagate up
                 raise # Reraise validation error
            
            print("Skipping validation for empty DataFrame.")
        return df

    # === UTILITY ===
    def _empty_output_df(self, dataset_type: str) -> pd.DataFrame:
        """
        Create an empty DataFrame with the appropriate columns for the specified dataset type.
        
        Parameters:
            dataset_type (str): The type of dataset, such as 'volumenes_i90' or 'precios_i90'.
        
        Returns:
            pd.DataFrame: An empty DataFrame with columns matching the expected schema for the given dataset type.
        """
        value_col = self._get_value_col(dataset_type)
        cols = ['id_mercado', 'datetime_utc', value_col]
        if dataset_type == 'volumenes_i90':
            cols.append('up')
        df = pd.DataFrame(columns=cols)
        df.index.name = 'id'
        return df

    # === INTRA DATA PROCESSING ===
    def _process_cumulative_volumenes_intra(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """
        Calculate actual intra-day volumes by computing differences between cumulative intra session data and baseline diario data.
        
        For each intra session, the function subtracts the previous session's cumulative volumes (starting with diario as baseline) to obtain the actual volume for that session. Returns a DataFrame with the calculated differences for all intra sessions. Raises an error if required diario data is missing or if no sessions are processed.
        
        Parameters:
            df (pd.DataFrame): Intra session cumulative volume data with 'datetime_utc' and 'id_mercado' columns.
            dataset_type (str): Dataset type, expected to be 'volumenes_i90' for intra processing.
        
        Returns:
            pd.DataFrame: DataFrame containing actual intra session volumes calculated as differences from previous sessions.
        """
        if df.empty:
            return df
            
        print("\n🔄 PROCESSING INTRA DATA - CUMULATIVE CALCULATIONS")
        print("="*70)
        
        try:        
             # Get the date by taking unique dates from datetime_utc and selecting the latest one
            unique_dates = df['datetime_utc'].dt.date.unique()
            target_date = max(unique_dates)  # Get the latest/last date
            print(f"Target date for cumulative calculations: {target_date}")
            target_year = target_date.year
            target_month = target_date.month
            
            print(f"📅 Processing intra data for: {target_date}")
            print(f"📊 Input data shape: {df.shape}")
            print(f"📈 Sessions found: {sorted(df['id_mercado'].unique())}")
            
            # Load corresponding diario data
            diario_df = self._load_diario_data_for_intra(target_year, target_month, target_date, dataset_type)
            
            if diario_df.empty:
                print("⚠️  No diario data found. Returning original intra data without processing.")
                return df
                
            # Split intra data by sessions (id_mercado) and store in a dictionary
            #intra sessions start at id mercado 2, 3, 4, etc.
            intra_dfs = {} #ie {2: df[df['id_mercado'] == 2], 3: df[df['id_mercado'] == 3], 4: df[df['id_mercado'] == 4]}

            for session_id in sorted(df['id_mercado'].unique()):
                session_data = df[df['id_mercado'] == session_id].copy()
                intra_dfs[session_id] = session_data
                print(f"📋 Session {session_id}: {len(session_data)} records")

            
            # Prepare diario data as baseline (session 0)
            diario_processed = self._prepare_diario_baseline(diario_df, target_date)
            
            if diario_processed.empty:
                print("⚠️  No processed diario data available. Cannot calculate cumulative differences.")
                raise ValueError("No processed diario data available. Cannot calculate cumulative differences.")
                
            # Calculate cumulative differences
            processed_sessions = []
            previous_session_data = diario_processed  #start with diario data as previous session kinda like a session 0
            
            for session_id in sorted(intra_dfs.keys()): #ie session_id = 2, 3, 4
                
                current_session = intra_dfs[session_id]
                
                # Calculate difference: current_session - previous_session
                session_with_differences = self._calculate_session_differences(
                    current_session, previous_session_data, session_id
                )
                
                if not session_with_differences.empty:
                    processed_sessions.append(session_with_differences)
                    # Update previous_session_data for next iteration
                    previous_session_data = current_session
                    print(f"✅ Session {session_id}: {len(session_with_differences)} records processed")
                else:
                    print(f"⚠️  Session {session_id}: No differences calculated")
            
            # Combine all processed sessions
            if processed_sessions:
                final_df = pd.concat(processed_sessions, ignore_index=True)
                print(f"\n✅ INTRA PROCESSING COMPLETE")
                print(f"Final shape: {final_df.shape}")
                return final_df
            else:
                raise ValueError("❌ No sessions were successfully processed")
                
        except Exception as e:
            raise ValueError(f"Error in intra data processing: {e}")

    def _load_diario_data_for_intra(self, year: int, month: int, target_date, dataset_type: str) -> pd.DataFrame:
        """
        Loads and processes diario market data for a specific date to serve as a baseline for intra-day calculations.
        
        Parameters:
        	year (int): The year of the target date.
        	month (int): The month of the target date.
        	target_date: The date for which diario data is required.
        	dataset_type (str): The type of dataset, either 'volumenes_i90' or 'precios_i90'.
        
        Returns:
        	pd.DataFrame: Processed diario data for the target date, filtered and standardized for use in intra-day baseline calculations.
        
        Raises:
        	ValueError: If diario data is missing, the 'fecha' column is absent, or any error occurs during processing.
        """
        print("="*70)
        print(f"\n📂 Loading diario data for {target_date}")
        
        try:
            # Read diario raw data for the same year/month
            diario_raw = self.raw_file_utils.read_raw_file(year, month, dataset_type, 'diario')
            
            if diario_raw.empty:
                print("❌ No diario raw data found")
                return pd.DataFrame()
            
            # Filter for the specific target date
            if 'fecha' in diario_raw.columns:
                diario_raw['fecha'] = pd.to_datetime(diario_raw['fecha'])
                diario_filtered = diario_raw[diario_raw['fecha'].dt.date == target_date].copy()

                if diario_filtered.empty:
                    raise ValueError(f"❌ No diario data found for date {target_date}")
                
            else:
                raise ValueError("❌ 'fecha' column not found in diario data")
            
            
            # Transform diario data using the same pipeline (but without intra processing)
            diario_config = DiarioConfig()
            
            # Apply basic transformations (excluding intra processing)
            diario_processed = self._apply_market_filters_and_id(diario_filtered, diario_config)
            diario_processed = self._standardize_datetime(diario_processed, dataset_type)
            diario_processed = self._select_and_finalize_columns(diario_processed, dataset_type)
            
            print(f"✅ Loaded diario data: {len(diario_processed)} records")
            return diario_processed
            
        except Exception as e:
            raise ValueError(f"Error loading diario data for intra processing: {e}")

    def _prepare_diario_baseline(self, diario_df: pd.DataFrame, target_date) -> pd.DataFrame:
        """
        Prepares diario market data as a baseline for intra-day cumulative volume calculations.
        
        Filters the input DataFrame for 'Mercado' transactions if applicable, ensures required columns are present, fills missing values, and aggregates volumes by unit and UTC datetime. The resulting DataFrame is structured for use as the baseline in intra-session difference calculations.
        
        Parameters:
        	diario_df (pd.DataFrame): The processed diario market data.
        	target_date: The date for which the baseline is prepared.
        
        Returns:
        	pd.DataFrame: Baseline DataFrame with columns ['datetime_utc', 'up', 'volumenes', 'id_mercado'].
        """
        if diario_df.empty:
            return pd.DataFrame()
        print("="*70)
        print(f"\n🔧 Preparing diario baseline for {target_date}")
        
        try:
            baseline_df = diario_df.copy()
            
            # Filter by tipo_transaccion = 'Mercado' if the column exists
            if 'tipo_transaccion' in baseline_df.columns:
                print("📋 Filtering by tipo_transaccion = 'Mercado'")
                baseline_df = baseline_df[baseline_df['tipo_transaccion'] == 'Mercado']
                print(f"   Records after filter: {len(baseline_df)}")
            
            # Ensure required columns exist
            required_cols = ['datetime_utc', 'up', 'volumenes']
            missing_cols = [col for col in required_cols if col not in baseline_df.columns]
            if missing_cols:
                print(f"❌ Missing required columns: {missing_cols}")
                raise ValueError(f"❌ Missing required columns: {missing_cols}")
            
            # Fill null/zero volumes
            baseline_df = baseline_df.fillna(0)
            
            # Group by UP and datetime to aggregate volumes (in case of duplicates)
            baseline_df = baseline_df.groupby(['datetime_utc', 'up']).agg({
                'volumenes': 'sum',
                'id_mercado': 'first'  # Keep the id_mercado (should be 1 for diario)
            }).reset_index()
            
            # Set id_mercado to 1 for diario (baseline)
            baseline_df['id_mercado'] = 1
            
            print(f"✅ Baseline prepared: {len(baseline_df)} records")
            print(f"   UPs in baseline: {baseline_df['up'].nunique()}")
            print(f"   Time range: {baseline_df['datetime_utc'].min()} to {baseline_df['datetime_utc'].max()}")

            
            return baseline_df
            
        except Exception as e:
            raise ValueError(f"Error preparing diario baseline for intra processing: {e}")

    def _calculate_session_differences(self, current_session: pd.DataFrame, 
                                     previous_session: pd.DataFrame, 
                                     session_id: int) -> pd.DataFrame:
        """
                                     Compute the difference in volumes between the current intra session and the previous session for each unit and timestamp.
                                     
                                     Parameters:
                                     	current_session (pd.DataFrame): Data for the current intra session.
                                     	previous_session (pd.DataFrame): Data for the previous session (either diario or prior intra session).
                                     	session_id (int): Identifier for the current session.
                                     
                                     Returns:
                                     	pd.DataFrame: DataFrame containing the volume differences per unit and timestamp, with the current session ID assigned.
                                     """
        try:
            print("="*70)
            print(f"\n🔧 Calculating differences for Session {session_id}")
            print(f"   📊 Current session records: {len(current_session)}")
            print(f"   📊 Previous session records: {len(previous_session)}")
            
            # Merge on UP and datetime_utc to align the data
            merged = pd.merge(
                current_session[['datetime_utc', 'up', 'volumenes']],
                previous_session[['datetime_utc', 'up', 'volumenes']],
                on=['datetime_utc', 'up'],
                how='left',
                suffixes=('_current', '_previous')
            )
            
            # Fill missing previous values with 0 (in case UP exists in current but not in previous)
            merged['volumenes_previous'] = merged['volumenes_previous'].fillna(0)
            
            # Calculate the difference: current - previous
            merged['volumenes_diff'] = merged['volumenes_current'] - merged['volumenes_previous']

            
            # Rename the difference column back to 'volumenes'
            session_result_df = merged.rename(columns={'volumenes_diff': 'volumenes'})

            
            # Keep only required columns
            session_result_df = session_result_df[['datetime_utc', 'up', 'volumenes']] 
            session_result_df['id_mercado'] = session_id
            
            print(f"   ✅ Differences calculated: {len(session_result_df)} non-zero programs")
            
            return session_result_df
            
        except Exception as e:
            raise ValueError(f"Error calculating differences: {e}")

    # === MAIN PIPELINE ===
    def transform_raw_i90_data(self, df: pd.DataFrame, market_config: I90Config, dataset_type: str) -> pd.DataFrame:
        """
        Executes the full I90 data transformation pipeline for volumes or prices, applying validation, filtering, datetime standardization, and optional intra cumulative processing.
        
        Parameters:
            df (pd.DataFrame): The input DataFrame containing raw I90 data.
            market_config (I90Config): Market configuration specifying filtering and processing rules.
            dataset_type (str): The dataset type, either 'volumenes_i90' or 'precios_i90'.
        
        Returns:
            pd.DataFrame: The fully processed DataFrame, or an empty DataFrame with expected columns if input is empty or an error occurs during processing.
        
        Raises:
            ValueError: If the DataFrame becomes empty at any pipeline step (except final validation).
            Exception: For unexpected errors encountered during processing.
        """
      
        print("\n" + "="*80)
        print(f"🔄 STARTING {dataset_type.upper()} TRANSFORMATION")
        print("="*80)

        if df.empty:
            print("\n❌ EMPTY INPUT")
            print("Input DataFrame is empty. Skipping transformation.")
            return self._empty_output_df(dataset_type)

        # Define the pipeline steps as (function, kwargs)
        pipeline = [
            (self._validate_raw_data, {"dataset_type": dataset_type}),
            (self._apply_market_filters_and_id, {"market_config": market_config}),
            (self._standardize_datetime, {"dataset_type": dataset_type}),
            (self._select_and_finalize_columns, {"dataset_type": dataset_type}),
            (self._validate_final_data, {"dataset_type": dataset_type}),
        ]

        #Apply intra data processing if market_config is IntraConfig
        if isinstance(market_config, IntraConfig):
            pipeline.append((self._process_cumulative_volumenes_intra, {"dataset_type": "volumenes_i90"})) #volumnes_i90 always be the dataset type for intra

        try:
            df_processed = df.copy()
            total_steps = len(pipeline)

            for i, (step_func, step_kwargs) in enumerate(pipeline, 1):
                print("\n" + "-"*50)
                print(f"📍 STEP {i}/{total_steps}: {step_func.__name__.replace('_', ' ').title()}")
                print("-"*50)

                # Apply the function with its arguments
                df_processed = step_func(df_processed, **step_kwargs)

                print(f"\n📊 Data Status:")
                print(f"   Rows: {df_processed.shape[0]}")
                print(f"   Columns: {df_processed.shape[1]}")

                if df_processed.empty and step_func != self._validate_final_data:
                    print("\n❌ PIPELINE HALTED")
                    print(f"DataFrame became empty after step: {step_func.__name__}")
                    raise ValueError(f"DataFrame empty after: {step_func.__name__}")

            print("\n✅ TRANSFORMATION COMPLETE")
            print(f"Final shape: {df_processed.shape}")
            print("="*80 + "\n")
            return df_processed

        except ValueError as e:
            print("\n❌ PROCESSING PIPELINE ERROR")
            print(f"Error: {str(e)}")
            print("="*80 + "\n")
            raise

        except Exception as e:
            print("\n❌ UNEXPECTED ERROR")
            print(f"Error: {str(e)}")
            print(traceback.format_exc())
            print("="*80 + "\n")
            raise
