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
        Initializes the I3Processor with utilities for date handling, data validation, and file operations.
        """
        self.date_utils = DateUtilsETL()
        self.data_validation_utils = DataValidationUtils()
        self.raw_file_utils = RawFileUtils()
        self.processed_file_utils = ProcessedFileUtils()
        self.db_utils = DatabaseUtils()
        self.dataset_type = "volumenes_i3"

    # === FILTERING ===
    def _filter_by_technology(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filters the DataFrame by the 'Concepto' column, keeping only rows where the value is in the allowed technologies from the database table 'tecnologias_generacion'.
        """
        if df.empty or 'Concepto' not in df.columns:
            return df
        
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
        Standardizes and converts input datetimes to a UTC column with 15-minute granularity, handling DST transitions and multiple input formats.
        
        Splits the input DataFrame by granularity and DST transition days, applies appropriate datetime parsing and conversion methods for each subset, and combines the results into a single DataFrame with a standardized `datetime_utc` column. Drops intermediate columns and invalid datetimes before returning the processed DataFrame.
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
        transition_dates = DateUtilsETL.get_transition_dates(start_range, end_range)
        
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
        
        cols_to_drop = ['fecha', 'hora', 'granularity', 'datetime_local', 'is_dst_day']
        final_df = final_df.drop(columns=[c for c in cols_to_drop if c in final_df.columns], errors='ignore')
        
        # Drop rows with invalid datetimes and sort
        final_df = final_df.dropna(subset=['datetime_utc'])
        if not final_df.empty:
            final_df = final_df.sort_values(by='datetime_utc').reset_index(drop=True)
        
        return final_df

    # === DATETIME PROCESSING ===
    @with_progress(message="Processing hourly data...", interval=2)
    def _process_hourly_data(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """
        Processes hourly market data with possible DST suffixes, generating UTC datetimes at 15-minute intervals.
        
        Converts "HH-HH+1" formatted time strings (including 'a'/'b' suffixes for DST fall-back) into timezone-aware local datetimes, then to UTC. Expands each hourly entry into four 15-minute intervals for downstream analysis.
        
        Returns:
            pd.DataFrame: DataFrame with standardized UTC datetimes at 15-minute granularity, or an empty DataFrame on error.
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

    @with_progress(message="Processing hourly data (vectorized)...", interval=2)
    def _process_hourly_data_vectorized(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """
        Vectorized processing of hourly data with possible DST suffixes, converting to UTC and expanding to 15-minute intervals.
        
        This method parses hourly time strings (e.g., "02-03a", "02-03b"), handles daylight saving time transitions using suffixes, localizes datetimes to Europe/Madrid, converts them to UTC, and expands each hour to four 15-minute intervals. Returns a DataFrame with standardized UTC datetimes and 15-minute granularity.
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

    @with_progress(message="Processing 15-minute data (vectorized)...", interval=2)
    def _process_15min_data_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """Processes 15-minute data using vectorized operations."""
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
        transition_dates = DateUtilsETL.get_transition_dates(start_range, end_range)
        
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
        Parse a 15-minute interval index for a given date into a timezone-aware datetime in Europe/Madrid, correctly handling daylight saving time transitions.
        
        Parameters:
            fecha: The date of the interval, as a string, pandas Timestamp, or date object.
            hora_index_str: The 1-based index (as string or integer) of the 15-minute interval within the day.
        
        Returns:
            pd.Timestamp: The corresponding timezone-aware datetime in Europe/Madrid.
        
        Raises:
            ValueError: If the interval index is less than 1 or exceeds the number of intervals for the given date.
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
        transition_dates = DateUtilsETL.get_transition_dates(start_range, end_range)
        
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
            ts = pd.Timestamp.combine(fecha, datetime.time(hour=hour, minute=minute))
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
        
                    ts = pd.Timestamp.combine(fecha, datetime.time(hour=h, minute=m))
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
        """ Selects and renames columns to the final required format. """
        
        if "Concepto" in df.columns:
            df = df.rename(columns={"Concepto": "tecnologia"})
        else:
            print(f"Columns in the input DataFrame: {df.columns}")
            raise ValueError("'Concepto' column not found in the input DataFrame.")
        
        required_cols = self.data_validation_utils.processed_volumenes_i3_required_cols
        return df[[col for col in required_cols if col in df.columns]]

    # === DATA VALIDATION ===
    def _validate_raw_data(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """Validate raw data structure."""
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
        """Validates the final processed data."""
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
        Calculates net intra-day volumes by computing differences between cumulative sessions,
        using 'tecnologia' as the grouping key.
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
        """Loads and processes diario market data to serve as a baseline."""
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

    # === VALIDATION ===
    def _validate_raw_data(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """Validate raw data structure."""
        if not df.empty:
            try:
                self.data_validation_utils.validate_raw_data(df, self.dataset_type)
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
        Create an empty DataFrame with the appropriate columns for volumenes_i3.
        
        Returns:
            pd.DataFrame: An empty DataFrame with columns matching the expected schema.
        """
        cols = self.data_validation_utils.processed_volumenes_i3_required_cols
        df = pd.DataFrame(columns=cols)
        df.index.name = 'id'
        return df

    # === MAIN PIPELINE ===
    def transform_raw_i3_data(self, df: pd.DataFrame, market_config: I3Config, dataset_type: str) -> pd.DataFrame:
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