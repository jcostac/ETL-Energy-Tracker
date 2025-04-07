""" Contains deprecated TimeUtils class and methods and
current implementation of uitlity datetime conversion methods 
used in ETL pipeline (DateUtilsETL)"""

__all__ = ['TimeUtils', "DateUtilsETL"] #Export the class

import pandas as pd
import pytz
from datetime import datetime
from typing import Dict, Union, Tuple


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
    def get_transition_dates(fecha_inicio: datetime, fecha_fin: datetime) -> Dict[datetime.date, int]:
        """
        Get dictionary of special dates (daylight saving transitions) between two dates
        
        Args:
            fecha_inicio (datetime): Start date
            fecha_fin (datetime): End date
            
        Returns:
            dict: Dictionary with dates as keys and transition type as values
                 transition type: 1 = 25-hour day (fall back)
                                2 = 23-hour day (spring forward)
        """
        spain_timezone = pytz.timezone('Europe/Madrid')
        utc_transition_times = spain_timezone._utc_transition_times[1:]
        localized_transition_times = [
            pytz.utc.localize(transition).astimezone(spain_timezone) 
            for transition in utc_transition_times
        ]
        
        fecha_inicio_local = spain_timezone.localize(fecha_inicio)
        fecha_fin_local = spain_timezone.localize(fecha_fin)
        
        filtered_transition_times = [
            transition 
            for transition in localized_transition_times 
            if fecha_inicio_local <= transition <= fecha_fin_local
        ]

        transition_dates = {dt.date(): int(dt.isoformat()[-4]) for dt in filtered_transition_times}
        
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
        
        # Convert the series to datetime
        dt_utc_converted = pd.to_datetime(dt_local_series, utc=True)

        # Get the local timezone from the datetime object
        local_timezone = dt_utc_converted.tzname()
    
        
        # Create a new DataFrame with the datetime column and timezone column
        result_df = pd.DataFrame({
            'datetime': dt_utc_converted,
            'local_timezone': local_timezone  # Add timezone column with 'UTC' value
        })
        
        return result_df
        
    @staticmethod
    def convert_utc_to_local(dt_utc_series: pd.Series, tz_name: str) -> pd.DataFrame:
        """
        Converts a UTC datetime object to a local datetime object and adds a corresponding timezone column.

        Args:
            dt_utc_series (pd.Series): A Series (single column) containing UTC datetime strings.
            tz_name (str): The name of the timezone to convert to.

        Returns:
            pd.DataFrame: A DataFrame with datetime objects converted from the UTC date strings
                         and a corresponding timezone column.
        """
        # Ensure the input is a Series
        if not isinstance(dt_utc_series, pd.Series):
            raise ValueError("Input must be a pandas Series.")
        
        # Check if the Series is empty
        if dt_utc_series.empty:
            raise ValueError("Datetime column is empty.")
        
        # Convert the series to datetime
        dt_local_converted = pd.to_datetime(dt_utc_series, tz=tz_name)

        # Create a new DataFrame with the datetime column and timezone column   
        result_df = pd.DataFrame({
            'datetime': dt_local_converted,
        })
        
        return result_df
    
    @staticmethod
    def convet_naive_to_utc(dt_naive: datetime, tz_name: str) -> pd.DataFrame:
        """
        Converts a naive datetime object to a UTC datetime object.

        Args:
            dt_naive (datetime): A naive datetime object.
            tz_name (str): The name of the timezone to convert to.

        Returns:
            pd.DataFrame: A DataFrame with datetime objects converted from the UTC date strings
                         and a corresponding timezone column.
        """
        # Ensure the input is a datetime object
        if not isinstance(dt_naive, datetime):
            naive_dt = pd.to_datetime(dt_naive)

        # Get the timezone object
        local_tz = pytz.timezone(tz_name)

        # Localize the naive datetime to Europe/Madrid timezone
        local_datetime = local_tz.localize(naive_dt)

        # Convert to UTC
        utc_datetime = local_datetime.astimezone(pytz.utc)

        #create a new dataframe with the datetime column
        result_df = pd.DataFrame({
            'datetime': utc_datetime,
            'timezone': tz_name
        })

        return result_df

    @staticmethod
    def convert_utc_to_naive(dt_utc: pd.Series, tz_name: str) -> pd.DataFrame:
        """
        Converts a UTC datetime object to a naive datetime object in the specified timezone.
        
        Args:
            dt_utc (pd.Series): A Series containing UTC datetime objects or strings.
            tz_name (str): The name of the timezone to convert to before removing timezone info.
            
        Returns:
            pd.DataFrame: A DataFrame with naive datetime objects in format 'yyyy-mm-dd hh:mm:ss'
                         converted from the UTC datetime objects.
        """
        # Ensure the input is a Series
        if not isinstance(dt_utc, pd.Series):
            raise ValueError("Input must be a pandas Series.")
        
        # Check if the Series is empty
        if dt_utc.empty:
            raise ValueError("Datetime column is empty.")
        
        # Convert the series to datetime with UTC timezone if not already
        dt_utc_converted = pd.to_datetime(dt_utc, utc=True)
        
        # Convert to the specified timezone
        dt_local = dt_utc_converted.dt.tz_convert(tz_name)
        
        # Remove timezone information to create naive datetime
        dt_naive = dt_local.dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Convert back to datetime objects without timezone
        dt_naive = pd.to_datetime(dt_naive)
        
        # Create a new DataFrame with the naive datetime column
        result_df = pd.DataFrame({
            'datetime': dt_naive
        })
        
        return result_df


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

def DateUtilsETL_example_usage():
    """Example usage of DateUtilsETL class
    
    Creates a DataFrame with three datetime columns in different formats:
    - naive: Original datetime string without timezone information
    - local: Datetime string with local timezone (Europe/Madrid)
    - utc: Datetime string converted to UTC timezone
    """
    # Create a sample datetime string
    dt_naive = "2024-03-31 02:00:00"
    dt_local = "2025-04-07T07:00:00.000+02:00"
    dt_utc = "2025-04-07T05:00:00Z"
    
    # Create local timezone version
    local_tz = pytz.timezone('Europe/Madrid')
    dt_local = local_tz.localize(dt_naive)
    
    # Create UTC version
    dt_utc = dt_local.astimezone(pytz.UTC)
    
    # Create DataFrame with all three versions as strings
    example_df = pd.DataFrame({
        'naive': [dt_naive],
        'local': [dt_local],
        'utc': [dt_utc]
    })

    # Convert naive datetime to UTC
    result_df = DateUtilsETL.convet_naive_to_utc(dt_naive, 'Europe/Madrid')
    print(f"Result of converting NAIVE datetime to UTC: {result_df}")

    # Convert local datetime to UTC
    result_df = DateUtilsETL.convert_local_to_utc(dt_local)
    print(f"Result of converting LOCAL datetime to UTC: {result_df}")

    # Convert UTC datetime to local
    result_df = DateUtilsETL.convert_utc_to_local(dt_utc, 'Europe/Madrid')
    print(f"Result of converting UTC datetime to LOCAL: {result_df}")

    result_df = DateUtilsETL.convert_utc_to_naive(dt_utc, 'Europe/Madrid')
    print(f"Result of converting UTC datetime to NAIVE: {result_df}")




if __name__ == "__main__":
    #TimeUtils_example_usage() 
    DateUtilsETL_example_usage()