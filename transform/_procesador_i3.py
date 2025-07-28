import pandas as pd
import pytz
from datetime import datetime
import traceback
from typing import Optional

from utilidades.etl_date_utils import DateUtilsETL
from utilidades.data_validation_utils import DataValidationUtils
from utilidades.storage_file_utils import RawFileUtils, ProcessedFileUtils
from configs.i3_config import I3Config, DiarioConfig, IntraConfig
from utilidades.progress_utils import with_progress
from utilidades.db_utils import DatabaseUtils

class I3Processor:
    """
    Processor class for I3 data (Volumenes).
    Handles data cleaning, validation, filtering, and transformation.
    """
    def __init__(self):
        """
        Initialize the I3Processor with utility classes for date handling, data validation, file operations, and database access.
        """
        self.date_utils = DateUtilsETL()
        self.data_validation_utils = DataValidationUtils()
        self.raw_file_utils = RawFileUtils()
        self.processed_file_utils = ProcessedFileUtils()
        self.db_utils = DatabaseUtils()

    # === FILTERING ===
    def _filter_by_technology(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter rows to include only those with allowed technologies as defined in the 'tecnologias_generacion' database table.
        
        Returns:
            pd.DataFrame: DataFrame containing only rows where the 'Concepto' column matches an allowed technology. If the column is missing or an error occurs, returns the original DataFrame.
        """
        if df.empty or 'Concepto' not in df.columns:
            return df
        
        engine = None
        try:
            # Create engine for the correct database (replace 'your_db_name' with the actual DB name)
            engine = self.db_utils.create_engine('energy_tracker')
            # Read allowed technologies from the table
            tech_df = self.db_utils.read_table(engine, 'tecnologias_generacion', columns=['tecnologia'])
            allowed_technologies = tech_df['tecnologia'].tolist()

            filtered_df = df[df['Concepto'].isin(allowed_technologies)].copy()
            return filtered_df
        
        except Exception as e:
            print(f"Error filtering by technology: {e}")
            return df
        
        finally: 
            if engine:
                engine.dispose()
        
    def _apply_market_filters_and_id(self, df: pd.DataFrame, market_config: I3Config) -> pd.DataFrame:
        """
        Filters the input DataFrame according to the specified market configuration and assigns the appropriate market ID to each row.
        
        For intra-day market configurations, maps session identifiers to market names and IDs, filters relevant sessions, and removes intermediate columns. For other market types, applies technology, direction ('Sentido'), and redispatch ('Redespacho') filters as defined in the configuration, assigning the corresponding market ID to each filtered subset. Returns a DataFrame containing only rows matching the market configuration, or an empty DataFrame if no data matches or required columns/configuration attributes are missing.
        if df.empty:
            return pd.DataFrame()
        
        # --- Special handling for Intra markets using 'Programa' column ---
        if isinstance(market_config, IntraConfig):
            if 'Programa' not in df.columns:
                print("Warning: 'Programa' column not found for IntraConfig processing. Cannot map market IDs.")
                return pd.DataFrame()
            
            # Map 'Programa' (e.g., 'PHF-1') to market name (e.g., 'Intra 1')
            df['mercado_name'] = df['Programa'].map(market_config.phf_intra_map)
            
            # Map market name (e.g., 'Intra 1') to market ID (e.g., '2')
            df['id_mercado'] = df['mercado_name'].map(market_config.id_mercado_map)

            # Clean up
            df = df.dropna(subset=['id_mercado'])
            if not df.empty:
                df['id_mercado'] = df['id_mercado'].astype(int)
            
            # Drop intermediate columns
            cols_to_drop = ['Programa', 'mercado_name']
            df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
            
            # Filter for the sessions relevant to the config's date
            df = df[df['id_mercado'].isin([int(x) for x in market_config.market_ids])]

            return df

        # --- General logic for other markets ---
        all_market_dfs = []
        print(f"Applying market filters and id to DataFrame with shape: {df.shape}")

        df = self._filter_by_technology(df)

        required_cols = ['volumenes']  # required cols for volumenes
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
        # Ensure id_mercado is int type after concat (should be if added as string)
        if 'id_mercado' in final_df.columns:
             final_df['id_mercado'] = final_df['id_mercado'].astype(int)
        return final_df

    # === STANDARDIZATION ===
    def _standardize_datetime(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """
        Standardizes datetimes in the input DataFrame to UTC with 15-minute intervals, handling daylight saving time transitions and various input formats.
        
        Returns:
            pd.DataFrame: DataFrame with a standardized `datetime_utc` column and invalid datetimes removed.
        """
        return self.date_utils.standardize_datetime(df, dataset_type)

    # === DATETIME PROCESSING ===
    @with_progress(message="Processing hourly data...", interval=2)
    def _process_hourly_data(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """
        Process hourly market data by converting time strings (including DST suffixes) to UTC datetimes and expanding each hour into four 15-minute intervals.
        
        Returns:
            pd.DataFrame: DataFrame with standardized UTC datetimes at 15-minute intervals, or an empty DataFrame if processing fails.
        """
        return self.date_utils.process_hourly_data(df, dataset_type)
    
    @with_progress(message="Processing 15-minute data...", interval=2)
    def _process_15min_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes 15-minute interval data by generating timezone-aware local datetimes and converting them to UTC.
        
        Returns:
            pd.DataFrame: DataFrame with standardized UTC datetime columns for 15-minute intervals.
        """
        return self.date_utils.process_15min_data(df)

    @with_progress(message="Processing hourly data (vectorized)...", interval=2)
    def _process_hourly_data_vectorized(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """
        Processes hourly data in a vectorized manner, converting local time strings (including DST suffixes) to UTC and expanding each hour into four 15-minute intervals.
        
        Returns:
            pd.DataFrame: DataFrame with standardized UTC datetimes at 15-minute granularity.
        """
        return self.date_utils.process_hourly_data_vectorized(df, dataset_type)

    @with_progress(message="Processing 15-minute data (vectorized)...", interval=2)
    def _process_15min_data_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes 15-minute interval data using vectorized operations for efficient datetime conversion and standardization.
        
        Returns:
            pd.DataFrame: DataFrame with standardized 15-minute interval datetimes and processed columns.
        """
        return self.date_utils.process_15min_data_vectorized(df)

    def _parse_hourly_datetime_local(self, fecha, hora_str) -> pd.Timestamp:
        """
        Convert an hourly time string (with possible DST suffixes) and a date into a timezone-aware datetime in the Europe/Madrid timezone.
        
        Parameters:
            fecha: The date corresponding to the time interval, as a date or datetime object.
            hora_str: Hourly interval string in the format "HH-HH+1", optionally suffixed with 'a' or 'b' for DST transitions.
        
        Returns:
            pd.Timestamp: A timezone-aware timestamp in the Europe/Madrid timezone representing the start of the interval.
        """
        return self.date_utils.parse_hourly_datetime_local(fecha, hora_str)

    def _parse_15min_datetime_local(self, fecha, hora_index_str) -> pd.Timestamp:
        """
        Convert a 1-based 15-minute interval index and date into a timezone-aware Europe/Madrid datetime, accounting for daylight saving time transitions.
        
        Parameters:
            fecha: The date of the interval, as a string, pandas Timestamp, or date object.
            hora_index_str: The 1-based index (string or integer) of the 15-minute interval within the day.
        
        Returns:
            pd.Timestamp: Timezone-aware datetime in Europe/Madrid corresponding to the specified interval.
        
        Raises:
            ValueError: If the interval index is less than 1 or exceeds the valid number of intervals for the date.
        """
        return self.date_utils.parse_15min_datetime_local(fecha, hora_index_str)

    # === COLUMN FINALIZATION ===
    def _select_and_finalize_columns(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """
        Rename the 'Concepto' column to 'tecnologia' and select the required columns for the final processed DataFrame.
        
        Raises:
            ValueError: If the 'Concepto' column is missing from the input DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame containing only the required columns in the standardized format.
        """
        
        if "Concepto" in df.columns:
            df = df.rename(columns={"Concepto": "tecnologia"})
        else:
            print(f"Columns in the input DataFrame: {df.columns}")
            raise ValueError("'Concepto' column not found in the input DataFrame.")
        
        required_cols = self.data_validation_utils.processed_volumenes_i3_required_cols
        return df[[col for col in required_cols if col in df.columns]]

    # === DATA VALIDATION ===
    def _validate_raw_data(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """
        Validates the structure of raw input data against the expected schema for the specified dataset type.
        
        If the DataFrame is empty, validation is skipped. If the schema is missing or an error occurs during validation, an error message is printed and the exception is raised. Returns the original DataFrame.
        """
        if not df.empty:
            try:
                self.data_validation_utils.validate_raw_data(df, dataset_type)
                print("Raw data validation successful.")
            except KeyError as e:
                print(f"Validation Error: Schema '{dataset_type}' not found in DataValidationUtils. Skipping validation. Error: {e}")
            except Exception as e:
                print(f"Error during raw data validation: {e}")
                raise # Reraise validation error
        else:
            print("Skipping validation for empty DataFrame.")
        return df
    
    def _validate_final_data(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """
        Validate the final processed DataFrame against the expected schema for the specified dataset type.
        
        If the DataFrame is empty, validation is skipped. If the schema is missing or another error occurs during validation, an error message is printed and the exception is re-raised.
        
        Returns:
            pd.DataFrame: The input DataFrame, unchanged.
        """
        if not df.empty:
            try:
                self.data_validation_utils.validate_processed_data(df, dataset_type)
                print("Final data validation successful.")
            except KeyError as e:
                print(f"Validation Error: Schema '{dataset_type}' not found in DataValidationUtils. Skipping validation. Error: {e}")
            except Exception as e:
                print(f"Error during final data validation: {e}")
                raise # Reraise validation error
        else:
            print("Skipping validation for empty DataFrame.")
        return df

    # === INTRA DATA PROCESSING ===
    def _process_cumulative_volumenes_intra(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """
        Compute net intra-day volumes by differencing cumulative session data for each technology.
        
        For each intra-day session, calculates the net volume by subtracting the previous session's cumulative values (starting from the daily baseline). Returns a DataFrame with net session volumes for all intra-day sessions on the target date.
        
        Parameters:
            df (pd.DataFrame): DataFrame containing cumulative intra-day session data.
            dataset_type (str): The dataset type identifier used for loading baseline data.
        
        Returns:
            pd.DataFrame: DataFrame with net intra-day session volumes per technology.
        """
        if df.empty:
            return df

        unique_dates = df['datetime_utc'].dt.date.unique()
        target_date = max(unique_dates)
        
        diario_df = self._load_diario_data_for_intra(target_date.year, target_date.month, target_date, dataset_type)
        if diario_df.empty:
            print("No diario data found for baseline. Returning original intra data.")
            return df

        intra_dfs = {sid: df[df['id_mercado'] == sid].copy() for sid in sorted(df['id_mercado'].unique())}
        
        diario_baseline = self._prepare_diario_baseline(diario_df)
        if diario_baseline.empty:
            raise ValueError("Failed to create diario baseline.")

        processed_sessions = []
        previous_session_data = diario_baseline
        
        for session_id in sorted(intra_dfs.keys()):
            current_session = intra_dfs[session_id]
            diff_session = self._calculate_session_differences(current_session, previous_session_data, session_id)
            if not diff_session.empty:
                processed_sessions.append(diff_session)
                previous_session_data = current_session
        
        if not processed_sessions:
            raise ValueError("No intra-day sessions were processed.")
        
        return pd.concat(processed_sessions, ignore_index=True)

    def _load_diario_data_for_intra(self, year: int, month: int, target_date, dataset_type: str) -> pd.DataFrame:
        """
        Load and process diario market data for a specific date to provide a baseline for intra-day calculations.
        
        Parameters:
        	year (int): The year of the diario data to load.
        	month (int): The month of the diario data to load.
        	target_date: The target date (as a date object) for which to extract diario data.
        	dataset_type (str): The dataset type identifier.
        
        Returns:
        	pd.DataFrame: Processed diario data for the specified date, or an empty DataFrame if no data is found.
        
        Raises:
        	ValueError: If an error occurs during loading or processing of diario data.
        """
        try:
            diario_raw = self.raw_file_utils.read_raw_file(year, month, dataset_type, 'diario')
            if diario_raw.empty:
                return pd.DataFrame()

            diario_raw['fecha'] = pd.to_datetime(diario_raw['fecha'])
            diario_filtered = diario_raw[diario_raw['fecha'].dt.date == target_date].copy()
            if diario_filtered.empty:
                return pd.DataFrame()

            diario_config = DiarioConfig()
            diario_processed = self._apply_market_filters_and_id(diario_filtered, diario_config)
            diario_processed = self._standardize_datetime(diario_processed, dataset_type)
            diario_processed = self._select_and_finalize_columns(diario_processed, dataset_type)
            return diario_processed
        except Exception as e:
            raise ValueError(f"Error loading diario data for intra baseline: {e}")

    def _prepare_diario_baseline(self, diario_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate and prepare diario market data as a baseline for cumulative intra-day volume calculations.
        
        Fills missing values with zero, groups by UTC datetime and technology, sums volumes, and assigns the diario market ID. Raises an error if required columns are missing.
        
        Parameters:
        	diario_df (pd.DataFrame): Diario market data to be used as the baseline.
        
        Returns:
        	pd.DataFrame: Aggregated diario baseline data with columns ['datetime_utc', 'tecnologia', 'volumenes', 'id_mercado'].
        """
        if diario_df.empty:
            return pd.DataFrame()
        
        baseline_df = diario_df.copy()
        required_cols = ['datetime_utc', 'tecnologia', 'volumenes']
        if not all(col in baseline_df.columns for col in required_cols):
            raise ValueError(f"Diario baseline is missing required columns: {required_cols}")

        baseline_df = baseline_df.fillna(0)
        baseline_df = baseline_df.groupby(['datetime_utc', 'tecnologia']).agg({
            'volumenes': 'sum',
            'id_mercado': 'first'
        }).reset_index()
        baseline_df['id_mercado'] = 1 # Diario market ID
        return baseline_df

    def _calculate_session_differences(self, current_session: pd.DataFrame, previous_session: pd.DataFrame, session_id: int) -> pd.DataFrame:
        """
        Compute net volume changes for each technology and timestamp by differencing the current session's volumes with those from the previous session.
        
        Parameters:
        	current_session (pd.DataFrame): DataFrame containing 'datetime_utc', 'tecnologia', and 'volumenes' columns for the current session.
        	previous_session (pd.DataFrame): DataFrame containing 'datetime_utc', 'tecnologia', and 'volumenes' columns for the previous session.
        	session_id (int): Identifier for the current session's market.
        
        Returns:
        	pd.DataFrame: DataFrame with net volume changes per technology and timestamp, including the assigned market ID.
        
        Raises:
        	ValueError: If an error occurs during the calculation process.
        """
        try:
            merged = pd.merge(
                current_session[['datetime_utc', 'tecnologia', 'volumenes']],
                previous_session[['datetime_utc', 'tecnologia', 'volumenes']],
                on=['datetime_utc', 'tecnologia'],
                how='left',
                suffixes=('_current', '_previous')
            )
            merged['volumenes_previous'] = merged['volumenes_previous'].fillna(0)
            merged['volumenes_diff'] = merged['volumenes_current'] - merged['volumenes_previous']
            
            session_result = merged.rename(columns={'volumenes_diff': 'volumenes'})
            session_result = session_result[['datetime_utc', 'tecnologia', 'volumenes']]
            session_result['id_mercado'] = session_id
            return session_result
        except Exception as e:
            raise ValueError(f"Error calculating session differences: {e}")

    # === VALIDATION ===
    def _validate_raw_data(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """
        Validates the structure of raw input data for the specified dataset type.
        
        If the DataFrame is not empty, checks its structure using the data validation utility and raises an exception on validation failure. Skips validation for empty DataFrames.
        
        Returns:
            pd.DataFrame: The original DataFrame, regardless of validation outcome.
        """
        if not df.empty:
            try:
                self.data_validation_utils.validate_raw_data(df, dataset_type)
                print("Raw data validation successful.")
            except Exception as e:
                print(f"Error during raw data validation: {e}")
                raise # Reraise validation error
        else:
            print("Skipping validation for empty DataFrame.")
        return df

    # === UTILITY ===
    def _empty_output_df(self, dataset_type: str) -> pd.DataFrame:
        """
        Create and return an empty DataFrame with columns matching the processed volumenes_i3 schema.
        
        Returns:
            pd.DataFrame: An empty DataFrame with the required columns and index named 'id'.
        """
        cols = self.data_validation_utils.processed_volumenes_i3_required_cols
        df = pd.DataFrame(columns=cols)
        df.index.name = 'id'
        return df

    # === MAIN PIPELINE ===
    def transform_raw_i3_data(self, df: pd.DataFrame, market_config: I3Config, dataset_type: str) -> pd.DataFrame:
        """
        Transforms raw I3 market volume data through a configurable processing pipeline, including validation, filtering, datetime standardization, column selection, and optional intra-day cumulative volume calculation.
        
        Parameters:
            df (pd.DataFrame): Raw input DataFrame containing I3 market volume data.
            market_config (I3Config): Market configuration object specifying filtering and processing rules.
            dataset_type (str): Identifier for the dataset type (e.g., 'diario', 'intra').
        
        Returns:
            pd.DataFrame: Fully processed DataFrame ready for downstream use, or an empty DataFrame if input is empty or processing fails.
        """
        print("\n" + "="*80)
        print(f"🔄 STARTING {dataset_type.upper()} TRANSFORMATION")
        print("="*80)

        if df.empty:
            print("\n❌ EMPTY INPUT")
            print("Input DataFrame is empty. Skipping transformation.")
            return self._empty_output_df(dataset_type)

        pipeline = [
            (self._validate_raw_data, {"dataset_type": dataset_type}),
            (self._apply_market_filters_and_id, {"market_config": market_config}),
            (self._standardize_datetime, {"dataset_type": dataset_type}),
            (self._select_and_finalize_columns, {"dataset_type": dataset_type}),
            (self._validate_final_data, {"dataset_type": dataset_type}),
        ]

        if isinstance(market_config, IntraConfig):
            pipeline.append((self._process_cumulative_volumenes_intra, {"dataset_type": dataset_type}))

        try:
            df_processed = df.copy()
            total_steps = len(pipeline)

            for i, (step_func, step_kwargs) in enumerate(pipeline, 1):
                print("\n" + "-"*50)
                print(f"📍 STEP {i}/{total_steps}: {step_func.__name__.replace('_', ' ').title()}")
                print("-"*50)

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