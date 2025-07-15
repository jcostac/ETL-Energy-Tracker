import pandas as pd
import pytz
from datetime import datetime
import traceback
from typing import Optional

from utilidades.etl_date_utils import DateUtilsETL, TimeUtils
from utilidades.data_validation_utils import DataValidationUtils
from utilidades.storage_file_utils import RawFileUtils, ProcessedFileUtils
from configs.i3_config import I3Config, DiarioConfig, IntraConfig
from utilidades.progress_utils import with_progress

class I3Processor:
    """
    Processor class for I3 data (Volumenes).
    Handles data cleaning, validation, filtering, and transformation.
    """
    def __init__(self):
        """
        Initializes the I3Processor with utilities for date handling, data validation, and file operations.
        """
        self.date_utils = DateUtilsETL()
        self.data_validation_utils = DataValidationUtils()
        self.raw_file_utils = RawFileUtils()
        self.processed_file_utils = ProcessedFileUtils()

    def _apply_market_filters_and_id(self, df: pd.DataFrame, market_config: I3Config) -> pd.DataFrame:
        """
        Filter the input DataFrame according to the market configuration and assign the market ID.
        """
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



    def _standardize_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardizes and converts input datetimes to a UTC column.
        """
        if df.empty:
            return df

        required_cols = ['fecha', 'hora', "granularity"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Required columns for datetime standardization not found: {required_cols}")

        df['fecha'] = pd.to_datetime(df['fecha'])
        
        # Vectorized processing for 15-minute data
        is_15_min = df['granularity'] == 'Quince minutos'
        df_15min = df[is_15_min].copy()
        
        if not df_15min.empty:
            df.loc[is_15_min, 'datetime_utc'] = self._process_15min_data_vectorized(df_15min)['datetime_utc']

        # Fallback to older processing for hourly (due to DST 'a'/'b' logic)
        is_hourly = df['granularity'] == 'Hora'
        df_hourly = df[is_hourly].copy()

        if not df_hourly.empty:
             df.loc[is_hourly, 'datetime_utc'] = self._process_hourly_data(df_hourly)['datetime_utc']

        cols_to_drop = ['fecha', 'hora', 'granularity']
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
        
        df = df.dropna(subset=['datetime_utc'])
        if not df.empty:
            df = df.sort_values(by='datetime_utc').reset_index(drop=True)
        
        return df

    def _process_hourly_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Processes hourly data, including DST, and converts to 15-minute intervals."""
        try:
            result_df = df.copy()
            result_df['datetime_local'] = result_df.apply(
                lambda row: self._parse_hourly_datetime_local(row['fecha'], row['hora']),
                axis=1
            )
            utc_df = self.date_utils.convert_local_to_utc(result_df['datetime_local'])
            result_df['datetime_utc'] = utc_df['datetime_utc']
            result_df = self.date_utils.convert_hourly_to_15min(result_df, 'volumenes')
            return result_df
        except Exception as e:
            print(f"Error processing hourly data: {e}\n{traceback.format_exc()}")
            return pd.DataFrame()

    def _parse_hourly_datetime_local(self, fecha, hora_str) -> Optional[pd.Timestamp]:
        """Parses an hourly time string (like '01-02' or '02-03a') into a localized datetime."""
        if not isinstance(hora_str, str):
            return None
        
        base_hour_str = hora_str.split('-')[0]
        base_hour = int(base_hour_str)
        
        is_dst = None
        if base_hour_str.endswith('a'):
            is_dst = True
            base_hour = int(base_hour_str[:-1])
        elif base_hour_str.endswith('b'):
            is_dst = False
            base_hour = int(base_hour_str[:-1])

        naive_dt = pd.Timestamp(year=fecha.year, month=fecha.month, day=fecha.day, hour=base_hour)
        tz = pytz.timezone('Europe/Madrid')

        try:
            return tz.localize(naive_dt, is_dst=is_dst)
        except pytz.exceptions.AmbiguousTimeError:
            return tz.localize(naive_dt, is_dst=True if is_dst is None else is_dst)
        except pytz.exceptions.NonExistentTimeError:
            return tz.localize(naive_dt + pd.Timedelta(hours=1), is_dst=True)

    def _process_15min_data_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """Processes 15-minute data using vectorized operations."""
        try:
            input_df = df.copy()
            tz = pytz.timezone('Europe/Madrid')

            input_df['fecha'] = pd.to_datetime(input_df['fecha'])
            input_df['hora'] = pd.to_numeric(input_df['hora'], errors='coerce').astype('Int64')
            input_df = input_df.dropna(subset=['fecha', 'hora'])

            if input_df.empty:
                return pd.DataFrame()

            def localize_day(group: pd.DataFrame) -> pd.DataFrame:
                group = group.sort_values(by='hora')
                time_offset = pd.to_timedelta((group['hora'] - 1) * 15, unit='m')
                naive_dt = group['fecha'] + time_offset
                try:
                    local_dt = naive_dt.dt.tz_localize(tz, ambiguous='infer', nonexistent='shift_forward')
                    group['datetime_utc'] = local_dt.dt.tz_convert('UTC')
                except Exception:
                    group['datetime_utc'] = pd.NaT
                return group

            result_df = input_df.groupby('fecha', group_keys=False).apply(localize_day)
            result_df = result_df.dropna(subset=['datetime_utc'])
            return result_df

        except Exception as e:
            print(f"Error in vectorized 15-min processing: {e}\n{traceback.format_exc()}")
            return pd.DataFrame()

    def _select_and_finalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Selects and renames columns to the final required format. """
        
        if "Concepto" in df.columns:
            df = df.rename(columns={"Concepto": "tecnologia"})
        else:
            print(f"Columns in the input DataFrame: {df.columns}")
            raise ValueError("'Concepto' column not found in the input DataFrame.")
        
        required_cols = self.data_validation_utils.processed_volumenes_i3_required_cols
        return df[[col for col in required_cols if col in df.columns]]

    def _validate_final_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validates the final processed data."""
        if not df.empty:
            self.data_validation_utils.validate_processed_data(df, "volumenes_i3")
        return df

    def _process_cumulative_volumenes_intra(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates net intra-day volumes by computing differences between cumulative sessions,
        using 'tecnologia' as the grouping key.
        """
        if df.empty:
            return df

        unique_dates = df['datetime_utc'].dt.date.unique()
        target_date = max(unique_dates)
        
        diario_df = self._load_diario_data_for_intra(target_date.year, target_date.month, target_date)
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

    def _load_diario_data_for_intra(self, year: int, month: int, target_date) -> pd.DataFrame:
        """Loads and processes diario market data to serve as a baseline."""
        try:
            diario_raw = self.raw_file_utils.read_raw_file(year, month, 'volumenes_i3', 'diario')
            if diario_raw.empty:
                return pd.DataFrame()

            diario_raw['fecha'] = pd.to_datetime(diario_raw['fecha'])
            diario_filtered = diario_raw[diario_raw['fecha'].dt.date == target_date].copy()
            if diario_filtered.empty:
                return pd.DataFrame()

            diario_config = DiarioConfig()
            diario_processed = self._apply_market_filters_and_id(diario_filtered, diario_config)
            diario_processed = self._standardize_datetime(diario_processed)
            diario_processed = self._select_and_finalize_columns(diario_processed)
            return diario_processed
        except Exception as e:
            raise ValueError(f"Error loading diario data for intra baseline: {e}")

    def _prepare_diario_baseline(self, diario_df: pd.DataFrame) -> pd.DataFrame:
        """Prepares diario data for use as a baseline in cumulative calculations."""
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
        """Calculates the difference in volumes between two consecutive sessions."""
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

    def transform_volumenes_i3(self, df: pd.DataFrame, market_config: I3Config) -> pd.DataFrame:
        """
        Executes the full I3 volume data transformation pipeline.
        """
        if df.empty:
            return pd.DataFrame(columns=self.data_validation_utils.processed_volumenes_i3_required_cols)

        pipeline = [
            (self._apply_market_filters_and_id, {"market_config": market_config}),
            (self._standardize_datetime, {}),
        ]
        
        if isinstance(market_config, IntraConfig):
             pipeline.append((self._process_cumulative_volumenes_intra, {}))
        
        pipeline.extend([
            (self._select_and_finalize_columns, {}),
            (self._validate_final_data, {})
        ])

        try:
            df_processed = df.copy()
            for step_func, kwargs in pipeline:
                df_processed = step_func(df_processed, **kwargs)
                if df_processed.empty and step_func != self._validate_final_data:
                    raise ValueError(f"DataFrame became empty after step: {step_func.__name__}")
            return df_processed
        
        except Exception as e:
            print(f"I3 transformation pipeline failed: {e}\n{traceback.format_exc()}")
            raise 