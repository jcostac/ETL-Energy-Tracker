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
from utilidades.etl_date_utils import TimeUtils
from utilidades.progress_utils import with_progress
from utilidades.storage_file_utils import RawFileUtils


class I90Processor:
    """
    Processor class for I90 data (Volumenes and Precios).
    Handles data cleaning, validation, filtering, and transformation.
    """
    def __init__(self):
        """
        Initialize the processor.
        """
        self.date_utils = DateUtilsETL()
        self.data_validation_utils = DataValidationUtils()
        self.raw_file_utils = RawFileUtils()
       
    # === FILTERING ===
    def _apply_market_filters_and_id(self, df: pd.DataFrame, market_config: I90Config) -> pd.DataFrame:
        """
        Filters the DataFrame based on market config and adds id_mercado.
        
        Args:
            df (pd.DataFrame): Input DataFrame (potentially pre-processed by extractor).
                               Expected columns might include 'Sentido', 'Redespacho', 'UP', 'hora', 'volumen'/'precio', 'fecha'/'datetime_utc'.
            market_config (I90Config): The configuration object instance for the specific market
                                     (e.g., SecundariaConfig(), RestriccionesConfig()).
            
        Returns:
            pd.DataFrame: DataFrame filtered and augmented with 'id_mercado',
                          or an empty DataFrame if no data matches.
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
                 # Assuming Sentido in DataFrame and config map are comparable (e.g., 'Subir', 'Bajar', None)
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
        Ensures a standard UTC datetime column with 15-minute granularity.
        Handles different input formats ('hora' column can be numeric index or HH-HH+1 format)
        and properly processes DST transitions.
        """
        if df.empty: 
            return df

        # Verify required columns exist
        required_cols = ['fecha', 'hora', "granularity"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Required columns: {required_cols} not found in DataFrame.")

        # Ensure fecha is datetime
        df['fecha'] = pd.to_datetime(df['fecha'])
        
        # Get timezone object
        tz = pytz.timezone('Europe/Madrid')
        
        # Get transition dates for relevant year range
        year_min = df['fecha'].dt.year.min() - 1
        year_max = df['fecha'].dt.year.max() + 1
        start_range = pd.Timestamp(year=year_min, month=1, day=1)
        end_range = pd.Timestamp(year=year_max, month=12, day=31)
        transition_dates = TimeUtils.get_transition_dates(start_range, end_range)
        
        # Create mask for DST transition days
        df['is_dst_day'] = df['fecha'].dt.date.apply(lambda x: x in transition_dates)
        
        # Split data by granularity
        df_hourly_dst = df[(df['granularity'] == 'Hora') & (df['is_dst_day'])].copy()
        df_hourly_normal = df[(df['granularity'] == 'Hora') & (~df['is_dst_day'])].copy()
        df_15min_dst = df[(df['granularity'] == 'Quince minutos') & (df['is_dst_day'])].copy()
        df_15min_normal = df[(df['granularity'] == 'Quince minutos') & (~df['is_dst_day'])].copy()
        
        # Process hourly data
        df_hourly_processed_dst = pd.DataFrame()
        if not df_hourly_dst.empty:
            print(f"Processing {len(df_hourly_dst)} rows of hourly DST data with regular method...")
            df_hourly_processed_dst = self._process_hourly_data(df_hourly_dst, dataset_type)
        
        df_hourly_processed_normal = pd.DataFrame()
        if not df_hourly_normal.empty:
            print(f"Processing {len(df_hourly_normal)} rows of hourly non-DST data with vectorized method...")
            df_hourly_processed_normal = self._process_hourly_data_vectorized(df_hourly_normal, dataset_type)
        
        # Process 15-minute data
        df_15min_processed_dst = pd.DataFrame()
        if not df_15min_dst.empty:
            print(f"Processing {len(df_15min_dst)} rows of 15-minute DST data with regular method...")
            df_15min_processed_dst = self._process_15min_data(df_15min_dst)
        
        df_15min_processed_normal = pd.DataFrame()
        if not df_15min_normal.empty:
            print(f"Processing {len(df_15min_normal)} rows of 15-minute non-DST data with vectorized method...")
            df_15min_processed_normal = self._process_15min_data_vectorized(df_15min_normal)
        
        # Combine results
        final_df = pd.concat([
            df_hourly_processed_dst, 
            df_hourly_processed_normal,
            df_15min_processed_dst, 
            df_15min_processed_normal
        ], ignore_index=True)
        
        # Ensure we have datetime_utc column and drop intermediate columns
        if 'datetime_utc' not in final_df.columns:
            print("Error: datetime_utc column not created during processing.")
            return pd.DataFrame()
        
        cols_to_drop = ['fecha', 'hora', 'granularidad', 'datetime_local', 'is_dst_day']
        final_df = final_df.drop(columns=[c for c in cols_to_drop if c in final_df.columns], errors='ignore')
        
        # Drop rows with invalid datetimes and sort
        final_df = final_df.dropna(subset=['datetime_utc'])
        if not final_df.empty:
            final_df = final_df.sort_values(by='datetime_utc').reset_index(drop=True)
        
        return final_df

    @with_progress(message="Processing hourly data...", interval=2)
    def _process_hourly_data(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """
        Process hourly data ("HH-HH+1" format, possibly with 'a'/'b' suffix for fall-back DST).
        Creates timezone-aware datetime_local series and converts to UTC.
        """
        try:
            # Create a copy to avoid modifying the original
            result_df = df.copy()
            
            # Apply the parsing function to create datetime_local
            result_df['datetime_local'] = result_df.apply(
                lambda row: self._parse_hourly_datetime_local(row['fecha'], row['hora']), 
                axis=1
            )
            
            # Convert to UTC
            utc_df = self.date_utils.convert_local_to_utc(result_df['datetime_local'])
            
            # Add the UTC datetime column to our result
            result_df['datetime_utc'] = utc_df['datetime_utc']
            
            # Convert to 15-minute frequency
            result_df = self.date_utils.convert_hourly_to_15min(result_df, dataset_type)
            
            return result_df
        
        except Exception as e:
            print(f"Error processing hourly data: {e}")
            print(traceback.format_exc())
            return pd.DataFrame()

    @with_progress(message="Processing 15-minute data...", interval=2)
    def _process_15min_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process 15-minute data (numeric index "1" to "96/92/100").
        Creates timezone-aware datetime_local series and converts to UTC.
        """
        try:
            # Create a copy to avoid modifying the original
            result_df = df.copy()
            
            # Apply the parsing function to create datetime_local
            result_df['datetime_local'] = result_df.apply(
                lambda row: self._parse_15min_datetime_local(row['fecha'], row['hora']), 
                axis=1
            )
            
            # Convert to UTC
            utc_df = self.date_utils.convert_local_to_utc(result_df['datetime_local'])
            
            # Add the UTC datetime column to our result
            result_df['datetime_utc'] = utc_df['datetime_utc']
            
            return result_df
        
        except Exception as e:
            print(f"Error processing 15-minute data: {e}")
            import traceback
            print(traceback.format_exc())
            return pd.DataFrame()

    @with_progress(message="Processing 15-minute data (vectorized)...", interval=2)
    def _process_15min_data_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process 15-minute data (numeric index "1" to "96/92/100") using vectorized operations.
        Creates timezone-aware UTC datetime series by processing each day individually.
        """
        try:
            input_df = df.copy() # Use a different name to avoid confusion within apply
            tz = pytz.timezone('Europe/Madrid')

            # Ensure 'fecha' is datetime and 'hora' is integer
            input_df['fecha'] = pd.to_datetime(input_df['fecha'])
            input_df['hora'] = pd.to_numeric(input_df['hora'], errors='coerce').astype('Int64')
            input_df = input_df.dropna(subset=['fecha', 'hora'])

            if input_df.empty:
                 print("Warning: 15-min data empty after initial cleaning.")
                 return pd.DataFrame()

            # --- Group by day and apply localization ---
            def localize_day(group: pd.DataFrame) -> pd.DataFrame:
                # Sort within the day by 'hora'
                group = group.sort_values(by='hora')

                # Calculate naive datetime for the day
                time_offset = pd.to_timedelta((group['hora'] - 1) * 15, unit='m')
                # Ensure 'fecha' Series used here has the same index as the group
                naive_dt = group['fecha'] + time_offset

                # Localize, handling DST transitions for this specific day
                try:
                    local_dt = naive_dt.dt.tz_localize(tz, ambiguous='infer', nonexistent='shift_forward')
                    group['datetime_utc'] = local_dt.dt.tz_convert('UTC')
                except Exception as loc_err:
                     # Log error for the specific day/group if needed
                     print(f"Error localizing group for date {group['fecha'].iloc[0].date()}: {loc_err}")
                     group['datetime_utc'] = pd.NaT # Assign NaT for error cases in this group

                return group

            print("Localizing timestamps day by day...")
            # Apply the localization function to each daily group
            # group_keys=False prevents adding 'fecha' as an index level
            result_df = input_df.groupby('fecha', group_keys=False).apply(localize_day)

            # --- End Grouping ---

            # Drop rows where UTC conversion failed (marked as NaT)
            result_df = result_df.dropna(subset=['datetime_utc'])

            print("Finished localization and conversion to UTC.")
            return result_df

        except Exception as e:
            print(f"Error processing 15-minute data (vectorized): {e}")
            import traceback
            print(traceback.format_exc()) # Print full traceback for debugging
            return pd.DataFrame()
        
     # New vectorized version for hourly data
    
    @with_progress(message="Processing hourly data (vectorized)...", interval=2)
    def _process_hourly_data_vectorized(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """
        Process hourly data ("HH-HH+1" format, possibly with 'a'/'b' suffix) using vectorized operations.
        Creates timezone-aware UTC datetime series and converts to 15-min intervals.
        """
        try:
            result_df = df.copy()
            tz = pytz.timezone('Europe/Madrid')

            # Ensure 'fecha' is datetime and 'hora' is string
            result_df['fecha'] = pd.to_datetime(result_df['fecha'])
            result_df['hora_str'] = result_df['hora'].astype(str)

            # 1. Extract Base Hour and Suffix using regex
            # Pattern captures:
            #   Group 1: The starting hour (digits)
            #   Group 2: Optional suffix 'a' or 'b'
            # Examples: "01-02" -> ('01', None), "02-03a" -> ('02', 'a'), "23-00b" -> ('23', 'b')
            # Handles cases without hyphen too: "2" -> ('2', None) - if they occur
            pat = r'^(\d{1,2})(?:-\d{1,2})?([ab]?)$'
            extracted = result_df['hora_str'].str.extract(pat, expand=True)
            extracted.columns = ['base_hour_str', 'suffix']

            # Convert base hour to numeric, coerce errors to NaT/NaN
            base_hour = pd.to_numeric(extracted['base_hour_str'], errors='coerce')

            # Drop rows where base hour couldn't be parsed
            valid_mask = base_hour.notna()
            result_df = result_df[valid_mask].copy()
            base_hour = base_hour[valid_mask]
            extracted = extracted[valid_mask]
            
            if result_df.empty:
                print("Warning: Hourly data empty after parsing base hour.")
                return pd.DataFrame()

            # 2. Create Naive Timestamps
            # Combine date part of 'fecha' with the extracted 'base_hour'
            # Important: Use dt.normalize() to set time to 00:00:00 before adding hours
            naive_dt = result_df['fecha'].dt.normalize() + pd.to_timedelta(base_hour, unit='h')

            # 3. Prepare Localization Parameters
            # Determine 'ambiguous' based on suffix 'a'/'b'
            # Default for ambiguous times without suffix: True (first occurrence, matches old logic)
            ambiguous_param = pd.Series(True, index=result_df.index) # Default
            ambiguous_param[extracted['suffix'] == 'a'] = True
            ambiguous_param[extracted['suffix'] == 'b'] = False

            # 4. Localize Vectorized
            # Use the 'ambiguous_param' Series.
            # Use nonexistent='shift_forward' to handle spring forward hour automatically.
            try:
                # Ensure naive_dt is Series before using .dt accessor
                if not isinstance(naive_dt, pd.Series):
                     naive_dt = pd.Series(naive_dt)

                local_dt = naive_dt.dt.tz_localize(tz, ambiguous=ambiguous_param, nonexistent='shift_forward')

            except Exception as loc_err:
                 # Handle potential errors during vector localization if needed
                 print(f"Warning: Vectorized tz_localize for hourly data failed: {loc_err}. Errors may occur.")
                 # Implement fallback or return empty if critical
                 return pd.DataFrame()

            # 5. Convert to UTC
            result_df['datetime_utc'] = local_dt.dt.tz_convert('UTC')

            # 6. Convert to 15-minute frequency using the utility function
            # Ensure the utility function can handle the input DataFrame structure
            result_df = self.date_utils.convert_hourly_to_15min(result_df, dataset_type)

            # Clean up intermediate columns
            result_df = result_df.drop(columns=['hora_str'], errors='ignore')

            return result_df

        except Exception as e:
            print(f"Error processing hourly data (vectorized): {e}")
            import traceback
            print(traceback.format_exc())
            return pd.DataFrame()
    
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
        # Ensure fecha is a date object
        if isinstance(fecha, pd.Timestamp):
            fecha = fecha.date()
        elif isinstance(fecha, str):
            fecha = pd.to_datetime(fecha).date()
        
        # Handle the hora string format
        hora_str = str(hora_str)  # Ensure string
        
        # Check for fall-back DST suffix ('a' or 'b')
        is_dst = None  # Default - let pytz figure it out
        if hora_str.endswith('a'):
            hora_str = hora_str[:-1]  # Remove suffix
            is_dst = True  # First occurrence during ambiguous hour (still on DST)
        elif hora_str.endswith('b'):
            hora_str = hora_str[:-1]  # Remove suffix
            is_dst = False  # Second occurrence during ambiguous hour (standard time)
        
        # Extract the hour from the format "HH-HH+1"
        if '-' in hora_str:
            base_hour = int(hora_str.split('-')[0])
        else:
            # For cases where we have just the hour as a number
            base_hour = int(hora_str)

        # Create naive datetime from date and hour
        naive_dt = pd.Timestamp(
            year=fecha.year,
            month=fecha.month,
            day=fecha.day,
            hour=base_hour
        )
        
        # Get timezone object
        tz = pytz.timezone('Europe/Madrid')
        
        # Check if this is a DST transition date
        year_range = (naive_dt.year - 1, naive_dt.year + 1)
        
        # Create a much smaller date range for the transition check
        start_range = pd.Timestamp(year=year_range[0], month=1, day=1)
        end_range = pd.Timestamp(year=year_range[1], month=12, day=31)
        transition_dates = TimeUtils.get_transition_dates(start_range, end_range)
        
        is_transition_date = naive_dt.date() in transition_dates
        if is_transition_date:
            transition_type = transition_dates[naive_dt.date()]
            
            # Handle spring forward (2:00 → 3:00)
            if transition_type == 2 and base_hour == 2:
                # The hour 2:00-2:59 doesn't exist - shift to 3:00
                naive_dt = naive_dt.replace(hour=3)
        
        # Localize the datetime with DST handling
        try:
            # Attempt localization with is_dst hint
            local_dt = tz.localize(naive_dt, is_dst=is_dst)
        except pytz.exceptions.AmbiguousTimeError:
            # For ambiguous times (fall-back), default to DST if not specified
            local_dt = tz.localize(naive_dt, is_dst=True if is_dst is None else is_dst)
        except pytz.exceptions.NonExistentTimeError:
            # For non-existent times (spring-forward), shift forward
            local_dt = tz.localize(naive_dt.replace(hour=3), is_dst=True)
        
        return local_dt

    def _parse_15min_datetime_local(self, fecha, hora_index_str) -> pd.Timestamp:
        """
        Parse 15-minute format data ) into a timezone-aware
        datetime in Europe/Madrid timezone, handling DST transitions correctly.
        """
        # Ensure fecha is a date object
        if isinstance(fecha, pd.Timestamp):
            fecha = fecha.date()
        elif isinstance(fecha, str):
            fecha = pd.to_datetime(fecha).date()
        
        # Ensure index is an integer
        index = int(hora_index_str)
        if index < 1:
            raise ValueError(f"Invalid 15-minute index: {hora_index_str}. Must be ≥ 1.")
        
        # Get timezone object
        tz = pytz.timezone('Europe/Madrid')
        
        # Check if this is a DST transition date
        year_range = (fecha.year - 1, fecha.year + 1)
        start_range = pd.Timestamp(year=year_range[0], month=1, day=1)
        end_range = pd.Timestamp(year=year_range[1], month=12, day=31)

        # Get the transition dates for the year range 
        transition_dates = TimeUtils.get_transition_dates(start_range, end_range)
        
        # Generate the sequence of timestamps for this day
        # Default: normal day (96 intervals)
        num_intervals = 96
        skip_hour = None
        transition_type = None
        
        is_transition_date = fecha in transition_dates
        if is_transition_date:
            transition_type = transition_dates[fecha]
            
            if transition_type == 2:  # Spring forward: skip hour 2
                num_intervals = 92  # 96 - 4 (skipped 15-min intervals)
                skip_hour = 2
            elif transition_type == 1:  # Fall back: repeat hour 2
                num_intervals = 100  # 96 + 4 (repeated 15-min intervals)
        
        # When we have an out-of-bounds index on a spring forward day, adjust the index
        if is_transition_date and transition_type == 2 and index > 92:
            # Calculate the equivalent time by mapping to the right hour
            # For spring forward, indices from 9-12 (hour 2) are skipped
            # So index 93 should map to 97, etc.
            adjusted_index = index + 4  # Add the 4 skipped intervals
            
            # Create the time directly based on the adjusted index
            hour = (adjusted_index - 1) // 4
            minute = ((adjusted_index - 1) % 4) * 15
            
            # Check that hour is valid (0-23)
            if hour >= 24:
                hour = 23
                minute = 45  # Set to last interval of the day
            
            # Create timestamp with the correct hour and minute
            ts = pd.Timestamp.combine(fecha, time(hour=hour, minute=minute))
            try:
                # For times after spring forward, is_dst should be True
                aware_ts = tz.localize(ts, is_dst=True)
                return aware_ts
            except (pytz.AmbiguousTimeError, pytz.NonExistentTimeError) as e:
                print(f"Warning: Could not localize adjusted time {ts} on date {fecha}: {e}")
                # Return a reasonable fallback
                ts = ts.replace(hour=max(3, min(hour, 23)))  # Ensure we're after spring forward and within valid range
                return tz.localize(ts, is_dst=True)
        
        # Generate the base sequence of datetimes for the day
        naive_dt = pd.Timestamp(year=fecha.year, month=fecha.month, day=fecha.day, hour=0)
        
        # Generate the complete sequence for the day
        if not is_transition_date or transition_type == 1:
            # For normal days (96 intervals) or fall-back days (100 intervals)
            ts_sequence = pd.date_range(
                start=naive_dt,
                periods=num_intervals,
                freq='15min'
            )
            
            # Handle DST transitions for each timestamp
            aware_ts_sequence = []
            for ts in ts_sequence:
                try:
                    is_dst = None
                    
                    if is_transition_date and transition_type == 1 and ts.hour == 2:
                        first_2am_block = len([t for t in aware_ts_sequence if t.hour == 2]) < 4
                        is_dst = first_2am_block
                    
                    aware_ts = tz.localize(ts, is_dst=is_dst)
                    aware_ts_sequence.append(aware_ts)
                    
                except (pytz.AmbiguousTimeError, pytz.NonExistentTimeError) as e:
                    if 'NonExistentTimeError' in str(e) and ts.hour == 2:
                        shifted_ts = ts.replace(hour=3)
                        aware_ts = tz.localize(shifted_ts, is_dst=True)
                        aware_ts_sequence.append(aware_ts)
                    else:
                        print(f"Warning: Could not localize {ts} on date {fecha}: {e}")
        else:
            # For spring-forward (skip hour 2)
            aware_ts_sequence = []
            
            for h in range(24):
                if h == skip_hour:
                    continue
                
                for m in range(0, 60, 15):
                    # Use datetime.time() instead of pd.Timestamp
        
                    ts = pd.Timestamp.combine(fecha, time(hour=h, minute=m))
                    try:
                        is_dst = h >= 3 if skip_hour == 2 else None
                        aware_ts = tz.localize(ts, is_dst=is_dst)
                        aware_ts_sequence.append(aware_ts)
                    except (pytz.AmbiguousTimeError, pytz.NonExistentTimeError) as e:
                        print(f"Warning: Could not localize {ts} on date {fecha}: {e}")
        
        # Get the requested timestamp (1-based indexing)
        if index <= len(aware_ts_sequence):
            return aware_ts_sequence[index - 1]
        else:
            raise ValueError(f"Index {index} out of bounds for date {fecha} with {len(aware_ts_sequence)} intervals.")

    # === COLUMN FINALIZATION ===
    def _select_and_finalize_columns(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """Selects, orders, and standardizes final columns, filtering by required columns."""
     
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
        Returns an empty DataFrame with the expected columns for the dataset_type.
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
        Process intra data by calculating cumulative differences with diario data.
        
        For intra session data, volumes are aggregated and need to be "unwrapped":
        - Session 1: actual_volume = intra_session_1 - diario_volume
        - Session 2: actual_volume = intra_session_2 - intra_session_1
        - Session 3: actual_volume = intra_session_3 - intra_session_2
        
        Args:
            df (pd.DataFrame): Input DataFrame with intra data
            dataset_type (str): Type of dataset being processed (always 'volumenes_i90' for intra)
            
        Returns:
            pd.DataFrame: Processed DataFrame with calculated differences
        """
        if df.empty:
            return df
            
        print("\n🔄 PROCESSING INTRA DATA - CUMULATIVE CALCULATIONS")
        print("="*70)
        
        try:        
            # Get the date (assuming all data is for the same day)
            target_date = df['datetime_utc'].iloc[0].date()
            target_year = target_date.year
            target_month = target_date.month
            
            print(f"📅 Processing intra data for: {target_date}")
            print(f"📊 Input data shape: {df.shape}")
            print(f"📈 Sessions found: {sorted(df['id_mercado'].unique())}")
            
            # Load corresponding diario data
            diario_df = self._load_diario_data_for_intra(target_year, target_month, target_date, dataset_type)

            breakpoint()
            
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

            breakpoint()
            
            # Prepare diario data as baseline (session 0)
            diario_processed = self._prepare_diario_baseline(diario_df, target_date)

            breakpoint()
            
            if diario_processed.empty:
                print("⚠️  No processed diario data available. Cannot calculate cumulative differences.")
                raise ValueError("No processed diario data available. Cannot calculate cumulative differences.")
                
            # Calculate cumulative differences
            processed_sessions = []
            previous_session_data = diario_processed  #start with diario data as previous session kinda like a session 0
            
            for session_id in sorted(intra_dfs.keys()): #ie session_id = 2, 3, 4
                print(f"\n🔢 Calculating differences for Session {session_id}")
                
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
        Load diario data for the same date as the intra data for baseline calculations.
        
        Args:
            year (int): Year of the target date
            month (int): Month of the target date  
            target_date: Date object for the target date
            dataset_type (str): Type of dataset ('volumenes_i90' or 'precios_i90')
            
        Returns:
            pd.DataFrame: Diario data for the target date
        """
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
        Prepare diario data as baseline for intra calculations.
        
        This method:
        1. Filters diario data by tipo_transaccion = 'Mercado' (if column exists)
        2. Ensures proper datetime and UP column structure
        3. Aggregates by UP and datetime_utc to create baseline volumes
        
        Args:
            diario_df (pd.DataFrame): Processed diario data
            target_date: Target date for filtering
            
        Returns:
            pd.DataFrame: Prepared baseline data with columns [datetime_utc, up, volumenes, id_mercado]
        """
        if diario_df.empty:
            return pd.DataFrame()
            
        print(f"🔧 Preparing diario baseline for {target_date}")
        
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
        Calculate volume differences between current intra session and previous session.
        
        Args:
            current_session (pd.DataFrame): Current intra session data
            previous_session (pd.DataFrame): Previous session data (diario or previous intra)
            session_id (int): Current session ID
            
        Returns:
            pd.DataFrame: DataFrame with calculated differences
        """
        try:
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
    def transform_volumenes_or_precios_i90(self, df: pd.DataFrame, market_config: I90Config, dataset_type: str) -> pd.DataFrame:
        """
        Wrapper method to orchestrate the I90 processing pipeline with formatted debug printouts.
        Args:
            df (pd.DataFrame): Input DataFrame.
            market_config (I90Config): Market configuration object.
            dataset_type (str): 'volumenes_i90' or 'precios_i90'.
        Returns:
            pd.DataFrame: Processed DataFrame or empty DataFrame with correct columns on error.
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
