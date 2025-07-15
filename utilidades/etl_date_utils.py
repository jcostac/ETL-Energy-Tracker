

__all__ = ['DateUtilsETL'] #Export the class

import pandas as pd
import pytz
from datetime import datetime
from typing import Dict, Union, Tuple
import pretty_errors
from datetime import timedelta, timezone, time
from deprecated import deprecated
import numpy as np
import traceback
from utilidades.progress_utils import with_progress

class DateUtilsETL:

    @staticmethod
    def get_transition_dates(fecha_inicio: datetime, fecha_fin: datetime, timezone: str = 'Europe/Madrid') -> Dict[datetime.date, int]:
        """
        Get dictionary of special dates (daylight saving transitions) between two dates
        
        Args:
            fecha_inicio (datetime): Start date
            fecha_fin (datetime): End date
            timezone (str): Timezone to use for the transition dates (default: 'Europe/Madrid')
            
        Returns:
            dict: Dictionary with dates as keys and transition type as values
                 transition type: 1 = 25-hour day (fall back)
                                2 = 23-hour day (spring forward)
        """
        # Get the timezone object for Spain (Europe/Madrid)
        timezone = pytz.timezone(timezone)
        
        # Access the list of UTC transition datetimes from the pytz timezone object
        # The [1:] slice is used to skip the initial transition at the epoch if present
        utc_transition_times = timezone._utc_transition_times[1:]
        
        # Convert the UTC transition datetimes to the local 'Europe/Madrid' timezone
        localized_transition_times = [
            pytz.utc.localize(transition).astimezone(timezone) 
            for transition in utc_transition_times
        ]
        
        # Localize the provided start and end dates to the 'Europe/Madrid' timezone
        # This makes them timezone-aware for comparison
        fecha_inicio_local = timezone.localize(fecha_inicio)
        fecha_fin_local = timezone.localize(fecha_fin)
        
        # Filter the localized transition times to include only those within the specified date range
        filtered_transition_times = [
            transition 
            for transition in localized_transition_times 
            if fecha_inicio_local <= transition <= fecha_fin_local
        ]

        # Create a dictionary mapping the date of each transition to its type
        # The transition type is determined by inspecting the UTC offset change:
        # - Spring forward (e.g., +01:00 to +02:00): The 4th char from the end of ISO format ('2') -> type 2 (23-hour day)
        # - Fall back (e.g., +02:00 to +01:00): The 4th char from the end of ISO format ('1') -> type 1 (25-hour day)
        # Note: This logic relies on the specific offset format and might be fragile for other timezones.
        transition_dates = {dt.date(): int(dt.isoformat()[-4]) for dt in filtered_transition_times}
        
        # Return the dictionary of transition dates and their types
        return transition_dates

    # === DATETIME PROCESSING for i90 and i3 (hourly / 15-minute) ===
    @staticmethod
    def standardize_datetime(df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
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
            df_hourly_processed_dst = DateUtilsETL.process_hourly_data(df_hourly_dst, dataset_type)
        
        df_hourly_processed_normal = pd.DataFrame()
        if not df_hourly_normal.empty:
            print(f"Processing {len(df_hourly_normal)} rows of hourly non-DST data with vectorized method...")
            df_hourly_processed_normal = DateUtilsETL.process_hourly_data_vectorized(df_hourly_normal, dataset_type)
        
        # Process 15-minute data
        df_15min_processed_dst = pd.DataFrame()
        if not df_15min_dst.empty:
            print(f"Processing {len(df_15min_dst)} rows of 15-minute DST data with regular method...")
            df_15min_processed_dst = DateUtilsETL.process_15min_data(df_15min_dst)
        
        df_15min_processed_normal = pd.DataFrame()
        if not df_15min_normal.empty:
            print(f"Processing {len(df_15min_normal)} rows of 15-minute non-DST data with vectorized method...")
            df_15min_processed_normal = DateUtilsETL.process_15min_data_vectorized(df_15min_normal)
        
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
    
    @staticmethod
    @with_progress(message="Processing hourly data...", interval=2)
    def process_hourly_data(df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
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
                lambda row: DateUtilsETL.parse_hourly_datetime_local(row['fecha'], row['hora']), 
                axis=1
            )
            
            # Convert to UTC
            utc_df = DateUtilsETL.convert_local_to_utc(result_df['datetime_local'])
            
            # Add the UTC datetime column to our result
            result_df['datetime_utc'] = utc_df['datetime_utc']
            
            # Convert to 15-minute frequency
            result_df = DateUtilsETL.convert_hourly_to_15min(result_df, dataset_type)
            
            return result_df
        
        except Exception as e:
            print(f"Error processing hourly data: {e}")
            print(traceback.format_exc())
            return pd.DataFrame()
        
    @staticmethod
    @with_progress(message="Processing 15-minute data...", interval=2)
    def process_15min_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Process 15-minute data (numeric index "1" to "96/92/100").
        Creates timezone-aware datetime_local series and converts to UTC.
        """
        try:
            # Create a copy to avoid modifying the original
            result_df = df.copy()
            
            # Apply the parsing function to create datetime_local
            result_df['datetime_local'] = result_df.apply(
                lambda row: DateUtilsETL.parse_15min_datetime_local(row['fecha'], row['hora']), 
                axis=1
            )
            
            # Convert to UTC
            utc_df = DateUtilsETL.convert_local_to_utc(result_df['datetime_local'])
            
            # Add the UTC datetime column to our result
            result_df['datetime_utc'] = utc_df['datetime_utc']
            
            return result_df
        
        except Exception as e:
            print(f"Error processing 15-minute data: {e}")
            import traceback
            print(traceback.format_exc())
            return pd.DataFrame()
        
    @staticmethod
    @with_progress(message="Processing hourly data (vectorized)...", interval=2)
    def process_hourly_data_vectorized(df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
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
            result_df = DateUtilsETL.convert_hourly_to_15min(result_df, dataset_type)

            # Clean up intermediate columns
            result_df = result_df.drop(columns=['hora_str'], errors='ignore')

            return result_df

        except Exception as e:
            print(f"Error processing hourly data (vectorized): {e}")
            import traceback
            print(traceback.format_exc())
            return pd.DataFrame()

    @staticmethod
    @with_progress(message="Processing 15-minute data (vectorized)...", interval=2)
    def process_15min_data_vectorized(df: pd.DataFrame) -> pd.DataFrame:
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
            result_df = input_df.groupby('fecha', group_keys=False).apply(localize_day, include_groups=True)

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

    @staticmethod
    def parse_hourly_datetime_local(fecha, hora_str) -> pd.Timestamp:
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

    @staticmethod
    def parse_15min_datetime_local(fecha, hora_index_str) -> pd.Timestamp:
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

    # === DATETIME TYPE CONVERSION UTC, LOCAL, NAIVE ===
    @staticmethod
    def convert_local_to_utc(dt_local_series: pd.Series) -> pd.DataFrame:
        """
        Converts a local datetime string to a UTC datetime object and adds a corresponding timezone column.

        Args:
            dt_local_series (pd.Series): A Series (single column) containing local datetime strings.

        Returns:
            pd.DataFrame: A DataFrame with datetime objects converted from the UTC date strings
                         and a corresponding timezone column.
        """
        # Ensure the input is a Series
        if not isinstance(dt_local_series, pd.Series):
            raise ValueError("Input must be a pandas Series.")
        
        # Check if the Series is empty
        if dt_local_series.empty:
            raise ValueError("Datetime column is empty.")
        
        # Always convert to datetime with UTC timezone
        dt_utc_converted = pd.to_datetime(dt_local_series, utc=True)

        # Get the local timezone from the datetime object
        converted_timezone = dt_utc_converted.dt.tz

        # Check if the timezone is UTC
        if converted_timezone is not None and str(converted_timezone) == 'UTC':
            print(f"Conversion successful: datetime object timezone is {converted_timezone}")
        else:
            print(f"Conversion failed: datetime object timezone is {converted_timezone}")
            raise ValueError("Unsuccessful conversion: datetime object timezone is not UTC.")
        
        # Create a new DataFrame with the datetime column and timezone column
        result_df = pd.DataFrame({
            'datetime_utc': dt_utc_converted
        })
        
        return result_df
        
    @staticmethod
    def convert_utc_to_local(dt_utc_series: pd.Series, tz_name: str) -> pd.DataFrame:
        """
        Converts a UTC datetime object to a local datetime object and adds a corresponding timezone column.
        Handles potential timezone conversion errors with proper exception handling.

        Args:
            dt_utc_series (pd.Series): A Series (single column) containing UTC datetime strings.
            tz_name (str): The name of the timezone to convert to.

        Returns:
            pd.DataFrame: A DataFrame with datetime objects converted from the UTC date strings
                         to the local timezone.
        """
        # Ensure the input is a Series
        if not isinstance(dt_utc_series, pd.Series):
            raise ValueError("Input must be a pandas Series.")
        
        # Check if the Series is empty
        if dt_utc_series.empty:
            raise ValueError("Datetime column is empty.")
        
        # Always convert to datetime with UTC timezone, then convert to target timezone
        try:
            dt_local_converted = pd.to_datetime(dt_utc_series, utc=True).dt.tz_convert(tz_name)
        except Exception as e:
            raise ValueError(f"Error converting UTC to local timezone: {str(e)}")

        # Create a new DataFrame with the datetime column
        result_df = pd.DataFrame({
            'datetime_local': dt_local_converted,
        })
        
        return result_df
    
    @staticmethod
    def convert_local_to_naive(dt_local_series: pd.Series) -> pd.DataFrame:
        """
        Converts a timezone-aware local datetime object to a naive datetime object.
        Removes timezone information while preserving the local time.
        
        Args:
            dt_local_series (pd.Series): A Series containing timezone-aware datetime objects or strings.
            
        Returns:
            pd.DataFrame: A DataFrame with naive datetime objects (without timezone information).
        """
        # Ensure the input is a Series
        if not isinstance(dt_local_series, pd.Series):
            raise ValueError("Input must be a pandas Series.")
        
        # Check if the Series is empty
        if dt_local_series.empty:
            raise ValueError("Datetime column is empty.")
        
        try:
            # Convert to datetime if not already
            dt_local = pd.to_datetime(dt_local_series)
            
            # Create a Series to store naive datetime objects
            dt_naive = pd.Series(index=dt_local.index, dtype='object')
            
            # Process each datetime
            for idx, dt in dt_local.items():
                # Convert to naive datetime by removing timezone info
                naive_dt = dt.replace(tzinfo=None)
                dt_naive[idx] = naive_dt
                
        except Exception as e:
            raise ValueError(f"Error converting local to naive datetime: {str(e)}")
        
        # Create a new DataFrame with the naive datetime column
        result_df = pd.DataFrame({
            'datetime_naive': dt_naive
        })
        
        return result_df

    @staticmethod
    def convert_naive_to_local(dt_naive: pd.Series, tz_name: str) -> pd.DataFrame:
        """
        Converts naive datetime objects to timezone-aware local datetime objects with proper DST handling.
        Uses transition dates to determine correct UTC offset for each datetime.
        
        Args:
            dt_naive (pd.Series): Series containing naive datetime objects or strings
            tz_name (str): Target timezone name (e.g. 'Europe/Madrid')
            
        Returns:
            pd.DataFrame: DataFrame with timezone-aware datetime objects in the specified timezone
        """
        # Input validation
        if not isinstance(dt_naive, pd.Series):
            raise ValueError("Input must be a pandas Series")
        if dt_naive.empty:
            raise ValueError("Input Series is empty")
        
        # Convert to datetime if not already
        dt_series = pd.to_datetime(dt_naive)
        
        # Get timezone info
        tz = pytz.timezone(tz_name)
        
        # Get all transition dates for the year range in the data
        min_year = dt_series.dt.year.min()
        max_year = dt_series.dt.year.max()
        
        # Create datetime objects for the full year range
        start_date = datetime(min_year - 1, 1, 1)  # Previous year to handle edge cases
        end_date = datetime(max_year + 1, 12, 31)  # Next year to handle edge cases
        
        # Get transition dates using existing method
        transitions = DateUtilsETL.get_transition_dates(start_date, end_date)
        
        # Convert transitions dict to sorted list of (datetime, is_dst_start) tuples
        transition_list = []
        for date, transition_type in transitions.items():
            # transition_type 2 = spring forward (DST start), 1 = fall back (DST end)
            dt = datetime.combine(date, datetime.min.time())
            if transition_type == 2:  # Spring forward
                transition_list.append((dt.replace(hour=2), True))  # DST starts
            else:  # Fall back
                transition_list.append((dt.replace(hour=3), False))  # DST ends
            
        transition_list.sort()
        
        # Create result series
        result = pd.Series(index=dt_series.index, dtype='object')
        
        # Process each datetime
        for idx, dt in dt_series.items():
            # Find the relevant transition period
            is_dst = False
            for transition_time, starts_dst in transition_list:
                if dt < transition_time:
                    break
                is_dst = starts_dst
            
            # Create timezone-aware datetime
            try:
                # Handle spring-forward gap (2:00-3:00)
                if is_dst and any(t[0].date() == dt.date() and t[1] for t in transition_list):
                    if dt.hour == 2:
                        # Shift forward to 3:00 for non-existent hour
                        dt = dt.replace(hour=3)
                    
                # Handle fall-back ambiguity (2:00-3:00 occurs twice)
                if not is_dst and any(t[0].date() == dt.date() and not t[1] for t in transition_list):
                    if dt.hour == 2:
                        # Use first occurrence (still on DST)
                        is_dst = True
                    
                # Create aware datetime with correct UTC offset
                if is_dst:
                    offset = timezone(timedelta(hours=2))  # UTC+2 for DST
                else:
                    offset = timezone(timedelta(hours=1))  # UTC+1 for standard time
                    
                aware_dt = dt.replace(tzinfo=offset)
                
                # Convert to target timezone
                local_dt = aware_dt.astimezone(tz)
                result[idx] = local_dt
                
            except (pytz.NonExistentTimeError, pytz.AmbiguousTimeError) as e:
                # Handle any remaining edge cases
                try:
                    # Try localization with nonexistent='shift_forward' as fallback
                    naive_dt = dt.replace(tzinfo=None)
                    local_dt = tz.localize(naive_dt, nonexistent='shift_forward', is_dst=is_dst)
                    result[idx] = local_dt
                except Exception as e2:
                    raise ValueError(f"Could not convert datetime {dt}: {str(e2)}")
                
        # Create result DataFrame
        result_df = pd.DataFrame({
            'datetime_local': result
        })
        
        return result_df

    @staticmethod
    def convert_naive_to_utc(dt_naive: pd.Series) -> pd.DataFrame:
        """
        Converts naive datetime objects to UTC datetime objects.
        """
        df_local = DateUtilsETL.convert_naive_to_local(dt_naive, 'Europe/Madrid')
        df_utc = DateUtilsETL.convert_local_to_utc(df_local['datetime_local'])
        return df_utc

    @staticmethod
    def convert_utc_to_naive(dt_utc: pd.Series) -> pd.DataFrame:
        """
        Converts a pandas Series of UTC datetime objects to naive datetime objects in the 'Europe/Madrid' local time.
        
        Parameters:
        	dt_utc (pd.Series): Series of UTC datetime objects or strings.
        
        Returns:
        	pd.DataFrame: DataFrame containing the corresponding naive datetime objects in local time.
        """
        df_local = DateUtilsETL.convert_utc_to_local(dt_utc, 'Europe/Madrid')
        df_naive = DateUtilsETL.convert_local_to_naive(df_local['datetime_local'], 'Europe/Madrid')
        return df_naive
    
    # === DATETIME GRANULARITY CONVERSION (hourly / 15-minute) ===
    @staticmethod
    def convert_hourly_to_15min(df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """
        Converts hourly data to 15-minute intervals, adjusting values based on dataset type.
        
        Each hourly row is expanded into four 15-minute rows with appropriate minute offsets. If the dataset type contains "volumenes", the 'volumenes' column is divided by 4 for each new row; otherwise, values are simply replicated.
        
        Parameters:
            df (pd.DataFrame): Hourly data with a 'datetime_utc' column.
            dataset_type (str): String indicating the type of dataset, used to determine value adjustment.
        
        Returns:
            pd.DataFrame: DataFrame with 15-minute granularity.
        """
        # Sort by datetime_utc to ensure proper ordering
        df = df.sort_values(by='datetime_utc').reset_index(drop=True)

        # Ensure 'datetime_utc' is in datetime format
        df['datetime_utc'] = pd.to_datetime(df['datetime_utc'], utc=True)

        # Repeat each row 4 times
        expanded_df = df.loc[df.index.repeat(4)].reset_index(drop=True)
        
        # Create the minute offsets [0, 15, 30, 45] for each original row
        # Calculate the number of original rows before repeating
        num_original_rows = len(df)
        minute_offsets = pd.Series([0, 15, 30, 45] * num_original_rows)
        
        # Convert offsets to Timedelta and add to the datetime
        expanded_df['datetime_utc'] = expanded_df['datetime_utc'] + pd.to_timedelta(minute_offsets, unit='m')

        if dataset_type and "volumenes" in dataset_type:
            print("Converting volumenes to 15-minute data by dividing each replicated row by 4")
            expanded_df['volumenes'] = expanded_df['volumenes'] / 4

        else:
            print("Converting precios to 15-minute data by replicating values 4 times")

        return expanded_df

    @staticmethod
    def convert_15min_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts 15-minute data to hourly data by aggregating within each hour and
        combinations of specified ID columns (id_mercado, up_id, tecnologia_id if present).
        Averages numeric values and keeps the first value for non-numeric columns within each group.
        Assumes input DataFrame has 15-minute frequency and a 'datetime_utc' column.

        Args:
            df (pd.DataFrame): DataFrame containing 15-minute data, including a 'datetime_utc' column
                               and potentially 'id_mercado', 'up_id', 'tecnologia_id'.

        Returns:
            pd.DataFrame: DataFrame containing hourly aggregated data, grouped by time and ID columns.
        """
        # Ensure 'datetime_utc' is in datetime format and UTC
        df['datetime_utc'] = pd.to_datetime(df['datetime_utc'], utc=True)

        # Define potential grouping columns
        potential_id_cols = ['id_mercado', 'up_id', 'tecnologia_id']
        grouping_cols = ['datetime_utc'] # Start with time column

        # Add existing ID columns to the grouping list
        for col in potential_id_cols:
            if col in df.columns:
                grouping_cols.append(col)

        # Sort by all grouping columns (time first) to ensure 'first' aggregation is consistent
        df = df.sort_values(by=grouping_cols).reset_index(drop=True)

        # Create the hourly grouping key based on the datetime
        hourly_group_key = df['datetime_utc'].dt.floor('H')

        # Replace the original datetime_utc with the hourly key for grouping
        df_for_grouping = df.copy()
        df_for_grouping['datetime_utc'] = hourly_group_key

        # Identify columns for aggregation (exclude all grouping columns)
        cols_to_aggregate = df.columns.drop(grouping_cols).tolist()

        # Handle case where there are no columns left to aggregate
        if not cols_to_aggregate:
            print("No columns to aggregate, returning unique hourly timestamps")
            # Group by all grouping columns and keep the unique combinations
            # The 'first' datetime_utc within each group will be the floored hourly timestamp
            df_hourly = df_for_grouping[grouping_cols].drop_duplicates().sort_values(by=grouping_cols).reset_index(drop=True)
            return df_hourly

        # Identify numeric and non-numeric columns among those to be aggregated
        numeric_cols = df[cols_to_aggregate].select_dtypes(include=np.number).columns.tolist()
        non_numeric_cols = df[cols_to_aggregate].select_dtypes(exclude=np.number).columns.tolist()

        # Create aggregation dictionary for the columns to aggregate
        agg_dict = {}
        for col in numeric_cols:
            agg_dict[col] = 'mean'
        for col in non_numeric_cols:
            agg_dict[col] = 'first'

        # Perform the groupby aggregation using all grouping columns
        df_hourly = df_for_grouping.groupby(grouping_cols).agg(agg_dict)

        # The grouping columns become the index, reset it
        df_hourly = df_hourly.reset_index()

        # Reorder columns: grouping columns first, then aggregated columns
        # Ensure the order of grouping columns is preserved
        final_cols_order = grouping_cols + [col for col in df.columns if col in agg_dict]
        # Filter df_hourly columns to only include those that exist
        final_cols = [col for col in final_cols_order if col in df_hourly.columns]
        df_hourly = df_hourly[final_cols]

        return df_hourly
        
def DateUtilsETL_example_usage():
    """Example usage of DateUtilsETL class
    
    Creates a DataFrame with datetime examples including DST transition times:
    - naive: Original datetime string without timezone information
    - local: Datetime string with local timezone (Europe/Madrid)
    - utc: Datetime string converted to UTC timezone
    
    Also demonstrates handling of DST transitions for Europe/Madrid timezone:
    - Spring forward (March): 01:59:59 -> 03:00:00 (2:00 doesn't exist)
    - Fall back (October): 02:59:59 -> 02:00:00 (2:00-2:59 exists twice)
    """

    # Get DST transition dates in 2024
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    transition_dates = DateUtilsETL.get_transition_dates(start_date, end_date)
    print(f"DST transition dates in 2024: {transition_dates}")
    


    print("\n===== Naive DateTime Conversion Tests =====")# Example usage
    naive_series = pd.Series([
    "2025-07-01 12:00:00",  # Regular summer time
    "2025-03-30 01:59:00",  # Just before spring forward
    "2025-03-30 02:30:00",  # During spring forward gap
    "2025-10-26 01:59:00",  # Before fall back
    "2025-10-26 02:30:00",  # During fall back (ambiguous)
    "2025-10-26 03:01:00"   # After fall back
    ])

    utc_series = pd.Series([
    "2025-07-01 12:00:00+00:00",  # Regular summer time
    "2025-03-30 02:00:00+00:00",  # Just before spring forward
    "2025-03-30 03:00:00+00:00",  # Adjusted to exact hour during spring forward gap
    "2025-10-26 02:00:00+00:00",  # Before fall back
    "2025-10-26 03:00:00+00:00",  # Adjusted to exact hour during fall back (ambiguous)
    "2025-10-27 02:00:00+00:00"   # Adjusted to exact hour during fall back (ambiguous)
    ])

    price_series = pd.Series([100, 101, 102, 103, 104, 105])

    df = pd.DataFrame({'datetime_utc': utc_series, 'price': price_series})

    result = DateUtilsETL.convert_naive_to_local(naive_series, 'Europe/Madrid')
    print("\nNaive to Local Conversion:")
    print(result)

    result = DateUtilsETL.convert_local_to_naive(result['datetime_local'])
    print("\nLocal to Naive Conversion:")
    print(result)

    result_local= DateUtilsETL.convert_naive_to_local(naive_series, 'Europe/Madrid')
    result_utc= DateUtilsETL.convert_local_to_utc(result_local['datetime_local'])
    print("\nNaive to Local to UTC Conversion:")
    print(result_utc)

    result_local= DateUtilsETL.convert_utc_to_local(result_utc['datetime_utc'], 'Europe/Madrid')
    result_naive= DateUtilsETL.convert_local_to_naive(result_local['datetime_local'])
    print("\nUTC to Local to Naive Conversion:")
    print(result_naive)

    result_15min = DateUtilsETL.convert_hourly_to_15min(df)
    print("\nOriginal DataFrame:")
    print(df)
    print("\nHourly to 15-minute Conversion:")
    print(result_15min)
    

if __name__ == "__main__":
    #TimeUtils_example_usage() 
    DateUtilsETL_example_usage()