""" Contains deprecated TimeUtils class/methods and
current implementation of uitlity datetime conversion methods 
used in ETL pipeline (DateUtilsETL class)"""

__all__ = ['TimeUtils', "DateUtilsETL"] #Export the class

import pandas as pd
import pytz
from datetime import datetime
from typing import Dict, Union, Tuple
import pretty_errors
from datetime import timedelta, timezone
from deprecated import deprecated
import numpy as np

class TimeUtils:
    """Utility class for handling time-related operations for ESIOS, OMIE and I90 data. 

    Especially for handling daylight saving time transitions and conversions between hourly and 15-minute intervals."""


    @staticmethod
    def ajuste_quinceminutal_a_horario_i90(row, is_special_date=False, tipo_cambio_hora=None):
        """Convert 15-minute data to hourly format
        
        Args:
            row (pd.Series): Row containing hour data
            is_special_date (bool): Whether the date is a daylight saving transition date
            tipo_cambio_hora (int): Type of hour change (1=fall back, 2=spring forward)
            
        Returns:
            int: Hour in hourly format (0-23)
        """
        if is_special_date and tipo_cambio_hora == 2:   # 23-hour day (spring forward)
            if row['hora'] > 8:
                row['hora'] = (row['hora']+3)//4 - 1
            else:
                row['hora'] = (row['hora']+3)//4
        else:              # Normal days and 25-hour days
            row['hora'] = (row['hora']+3)//4
        return row['hora']

    @staticmethod
    def ajuste_horario_i90(row, is_special_date=False, tipo_cambio_hora=None):
        """Adjust hourly data for special dates
        
        Args:
            row (pd.Series): Row containing hour data
            is_special_date (bool): Whether the date is a daylight saving transition date
            tipo_cambio_hora (int): Type of hour change (1=fall back, 2=spring forward)
            
        Returns:
            int: Adjusted hour value
        """
        # Convert hour to string if it's not already
        hora = str(row['hora']) if not isinstance(row['hora'], str) else row['hora']
        
        if is_special_date:
            if tipo_cambio_hora == 2:  # 23-hour day (spring forward)
                hour_value = int(str(hora)[-2:]) if len(str(hora)) >= 2 else int(hora)
                if hour_value < 3:
                    return hour_value
                else:
                    return hour_value - 1
            
            if tipo_cambio_hora == 1:  # 25-hour day (fall back)
                if hora[-1].isdigit():
                    hour_value = int(str(hora)[-2:]) if len(str(hora)) >= 2 else int(hora)
                    if hour_value < 3:
                        return hour_value
                    else:
                        return hour_value + 1
                elif hora[-1] == 'a':
                    return int(hora[-3:-1])
                elif hora[-1] == 'b':
                    return int(hora[-3:-1]) + 1
        else:    # Normal days
            return int(str(hora)[-2:]) if len(str(hora)) >= 2 else int(hora)

    @staticmethod
    def ajuste_horario_ESIOS(row, special_dates=None):
        """Adjust hourly data for special dates
        
        Args:
            row (pd.Series): Row containing hour data
            special_dates (dict): Dictionary of special dates """
    
        if row['fecha'] in special_dates:
            print("Dia especial", row['fecha'],row['hora_real'])
            if special_dates[row['fecha']] == 2 and row['zona_horaria'] == 2:     #Dia 23 horas a partir del cambio horario
                row['hora'] = row['hora_real'] - 1
            elif special_dates[row['fecha']] == 1 and row['zona_horaria'] == 1:   #Dia 25 horas a partir del cambio horario
                row['hora'] = row['hora_real'] + 1
            else:   # Dias de 23 o 25 horas antes del cambio horario
                row['hora'] = row['hora_real']
        
        else:  #Resto de dias
            row['hora'] = row['hora_real']

        return row['hora']
        
    @staticmethod
    def ajuste_quinceminutal_i90(row, is_special_date=False, tipo_cambio_hora=None):
        """Adjust 15-minute data for special dates
        
        Args:
            row (pd.Series): Row containing hour data
            is_special_date (bool): Whether the date is a daylight saving transition date
            tipo_cambio_hora (int): Type of hour change (1=fall back, 2=spring forward)
            
        Returns:
            str: Hour in 15-minute format (HH:MM)
        """
        minutos_dict = {0:":00", 1:":15", 2:":30", 3:":45"}
        
        # Convert hour to int if it's a string
        hora_value = int(row['hora']) if isinstance(row['hora'], str) else row['hora']
        
        if is_special_date and tipo_cambio_hora == 2:   # 23-hour day (spring forward)
            if hora_value > 8:
                hora = str((hora_value+3)//4 - 2).zfill(2)
            else:
                hora = str((hora_value+3)//4 - 1).zfill(2)
        else:              # Normal days and 25-hour days
            hora = str((hora_value+3)//4 - 1).zfill(2)
            
        minutos = minutos_dict[(hora_value+3)%4]
        return hora + minutos
        
    @staticmethod
    def ajuste_quinceminutal_ESIOS(row, special_dates=None):
        """Adjust 15-minute data for special dates
        
        Args:
            row (pd.Series): Row containing hour data
            is_special_date (bool): Whether the date is a daylight saving transition date"""
        
        if row['fecha'] in special_dates:
        #print("Dia especial", row['fecha'],row['hora_real'])
            if special_dates[row['fecha']] == 2 and row['zona_horaria'] == 2:     #Dia 23 horas a partir del cambio horario
                minuto = row['hora_real'][-3:]
                hora = str(int(row['hora_real'][:2])-1).zfill(2)
                row['hora'] = hora + minuto
            elif special_dates[row['fecha']] == 1 and row['zona_horaria'] == 1:   #Dia 25 horas a partir del cambio horario
                minuto = row['hora_real'][-3:]
                hora = str(int(row['hora_real'][:2])+1).zfill(2)
                row['hora'] = hora + minuto
            else:   # Dias de 23 o 25 horas antes del cambio horario
                row['hora'] = row['hora_real']
            
        else:  #Resto de dias
            row['hora'] = row['hora_real']

        return row['hora']
        
    @staticmethod
    def ajuste_horario_a_quinceminutal_i90(row, is_special_date=False, tipo_cambio_hora=None):
        """Convert hourly data to 15-minute format
        
        Args:
            row (pd.Series): Row containing hour data
            is_special_date (bool): Whether the date is a daylight saving transition date
            tipo_cambio_hora (int): Type of hour change (1=fall back, 2=spring forward)
            
        Returns:
            str: Hour in 15-minute format (HH:MM)
        """
        minutos_dict = {0:":00", 1:":15", 2:":30", 3:":45"}
        
        # Handle hour values that might be in the format "00-01"
        hora_value = row['hora']
        if isinstance(hora_value, str) and '-' in hora_value:
            # Extract the second part of the time range (e.g., "01" from "00-01")
            base_hora = int(hora_value.split('-')[1])
        else:
            # For non-range values, try to convert directly
            try:
                base_hora = int(hora_value)
            except ValueError:
                # If it's still a string but not a range, try to get the last 2 chars
                if isinstance(hora_value, str) and len(hora_value) >= 2:
                    base_hora = int(hora_value[-2:])
                else:
                    raise ValueError(f"Cannot convert '{hora_value}' to an integer hour value")
        
        # Apply daylight saving time adjustments
        if is_special_date:
            if tipo_cambio_hora == 2:  # 23-hour day (spring forward)
                if base_hora >= 2: 
                    base_hora = base_hora + 1
            elif tipo_cambio_hora == 1:  # 25-hour day (fall back)
                if base_hora >= 2:
                    base_hora = base_hora - 1
        
        # Create 15-minute intervals
        quarter = int(row.name) % 4  # Use row index to determine which quarter
        hora_str = str(base_hora).zfill(2) + minutos_dict[quarter]  # concatenate hour and minute
        
        return hora_str

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

    @staticmethod
    def convert_granularity_i90(df: pd.DataFrame, current_format: str, target_format: str, is_special_date: bool = False, tipo_cambio_hora: int = None) -> Union[pd.Series, pd.DataFrame]:
        """Convert data between hourly and 15-minute granularity

        Args:
            df (pd.DataFrame): DataFrame containing the data to convert
            current_format (str): Current time granularity format ('hora' or '15min')
            target_format (str): Target time granularity format ('hora' or '15min')
            is_special_date (bool): Whether the date is a daylight saving transition date
            tipo_cambio_hora (int): Type of hour change (1=fall back, 2=spring forward)

        Returns:
            Union[pd.Series, pd.DataFrame]: Series containing the converted hour values or expanded DataFrame for hourly to 15-min

        Raises:
            ValueError: If unsupported granularity conversion is requested
        """
        if current_format == "hora" and target_format == "hora":
            df['hora'] = df.apply(TimeUtils.ajuste_horario_i90, 
                                 axis=1, 
                                 is_special_date=is_special_date, 
                                 tipo_cambio_hora=tipo_cambio_hora)
            
        elif current_format == "15min" and target_format == "15min":
            df['hora'] = df.apply(TimeUtils.ajuste_quinceminutal_i90, 
                                 axis=1, 
                                 is_special_date=is_special_date, 
                                 tipo_cambio_hora=tipo_cambio_hora)
                                 
        elif current_format == "15min" and target_format == "hora":
            df['hora'] = df.apply(TimeUtils.ajuste_quinceminutal_a_horario_i90, 
                                 axis=1, 
                                 is_special_date=is_special_date, 
                                 tipo_cambio_hora=tipo_cambio_hora)
            
        elif current_format == "hora" and target_format == "15min":  # for hourly to 15-min conversion, we need to expand each row into 4 rows
            # First, convert the hour format
            original_df = df.copy()
            expanded_df = pd.DataFrame()
            
            for idx, row in original_df.iterrows():
                # Create 4 copies of the row for each 15-min interval
                for quarter in range(4):
                    new_row = row.copy()
                    # Use the row index and quarter to determine the 15-min interval
                    expanded_df = pd.concat([expanded_df, pd.DataFrame([new_row])], ignore_index=True)
            
            # Apply the conversion to the expanded dataframe
            expanded_df['hora'] = expanded_df.reset_index().apply(
                TimeUtils.ajuste_horario_a_quinceminutal_i90,
                axis=1,
                is_special_date=is_special_date,
                tipo_cambio_hora=tipo_cambio_hora
            )
            
            # If there's a 'valor' column, divide it by 4
            if 'valor' in expanded_df.columns:
                expanded_df['valor'] = expanded_df['valor'] / 4
                
            # Return the entire expanded dataframe
            return expanded_df
        else: 
            raise ValueError(f"Unsupported granularity conversion: {current_format} to {target_format}")
        
        return df['hora']

def TimeUtils_example_usage():
    """Example usage of TimeUtils class"""
    # Get DST transition dates in 2024
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    transition_dates = TimeUtils.get_transition_dates(start_date, end_date)
    print(f"DST transition dates in 2024: {transition_dates}")
    
    # Create sample data
    data = {'fecha': ['2024-03-31', '2024-03-31', '2024-10-27', '2024-10-27'],
            'hora': ['02:00', '03:00', '02:00', '03:00']}
    df = pd.DataFrame(data)
    
    # Convert hourly to 15-minute
    for index, row in df.iterrows():
        date = datetime.strptime(row['fecha'], '%Y-%m-%d').date()
        is_special = date in transition_dates
        tipo_cambio = transition_dates.get(date, None)
        
        print(f"Converting {row['fecha']} {row['hora']} - Special: {is_special}, Type: {tipo_cambio}")
        result = TimeUtils.convert_granularity_i90(
            pd.DataFrame([row]), 
            'hora', 
            '15min',
            is_special,
            tipo_cambio
        )
        print(result)

class DateUtilsETL:

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
        local_timezone = dt_utc_converted.dt.tz.zone
        
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
        transitions = TimeUtils.get_transition_dates(start_date, end_date)
        
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
        Converts UTC datetime objects to naive datetime objects.
        """
        df_local = DateUtilsETL.convert_utc_to_local(dt_utc, 'Europe/Madrid')
        df_naive = DateUtilsETL.convert_local_to_naive(df_local['datetime_local'], 'Europe/Madrid')
        return df_naive
    
    @staticmethod
    def convert_hourly_to_15min(df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts hourly data to 15-minute data using vectorized operations.
        Assumes input DataFrame has hourly frequency and 'datetime_utc' column.
        
        Args:
            df (pd.DataFrame): DataFrame containing hourly data, including a 'datetime_utc' column.

        Returns:
            pd.DataFrame: DataFrame containing 15-minute data.
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