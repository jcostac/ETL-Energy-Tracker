from typing import Dict, List, Optional, Union
import pandas as pd
from datetime import datetime
import sys
import os
import numpy as np
import pytz
import time

# Add necessary imports
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Ensure the project root is added to sys.path if necessary
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utilidades.etl_date_utils import DateUtilsETL, TimeUtils
from utilidades.data_validation_utils import DataValidationUtils
from utilidades.progress_utils import with_progress

class OMIEProcessor:
    """
    Processor class for OMIE data (Diario, Intra, and Continuo markets).
    Handles data cleaning, validation, and transformation operations based on carga_omie.py logic.
    """
    def __init__(self):
        """
        Initialize the transformer with OMIE processing utilities.
        """
        self.date_utils = DateUtilsETL()
        self.data_validator = DataValidationUtils()

    # === 1. CLEANING ===
    def _clean_empty_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows that are completely empty or have only NaN values in critical columns.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        print("\n🧹 CLEANING EMPTY ROWS")
        print("-"*30)
        
        initial_rows = len(df)
        
        # Remove completely empty rows
        df = df.dropna(how='all')
        
        # Remove rows where all important columns are NaN
        important_cols = ['Fecha', 'Hora', 'Unidad']
        existing_important_cols = [col for col in important_cols if col in df.columns]
        if existing_important_cols:
            df = df.dropna(subset=existing_important_cols, how='all')
        
        removed_rows = initial_rows - len(df)
        if removed_rows > 0:
            print(f"   Removed {removed_rows} empty/invalid rows")
        else:
            print(f"   No empty rows found")
        
        print(f"   Rows: {initial_rows} → {len(df)}")
        print("-"*30)
        return df

    # === 2. PROCESSING ENERGY, PRICE, UOF, GRANULARITY COLUMNS ===
    def _add_granularity_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add granularity column based on the unique values in the 'Hora' column.
        Now that columns are standardized, we always check 'Hora' column.
        
        Args:
            df (pd.DataFrame): Input DataFrame with 'Hora' column
            
        Returns:
            pd.DataFrame: DataFrame with added 'granularity' column
        """
        print("\n⏱️ ADDING GRANULARITY COLUMN")
        print("-"*30)
        
        if 'Hora' in df.columns:
            # Check if Hora contains H2Q4 format (15-minute intervals)
            sample_hora = str(df['Hora'].iloc[0]) if not df.empty else None
            if 'H' in sample_hora and 'Q' in sample_hora:
                #if H2Q4 format, then 15-minute granularity
                granularity = 'Quince minutos'
            else:
                #if not H2Q4 format, then hourly granularity
                granularity = 'Hora'
        else:
            raise Exception("'Hora' column not found, cannot determine granularity")
        
        # Add granularity column
        df['granularity'] = granularity
        
        print(f"   Assigned granularity: {granularity}")
        print("-"*30)
        return df
    
    def _process_and_filter_energy_column(self, df: pd.DataFrame, mercado: str) -> pd.DataFrame:
        """
        Process energy columns and apply filtering/multipliers based on market type.
        Now that column names are standardized in extract phase, we always expect 'Energía Compra/Venta'.
        Division by 4 only occurs when we detect 15-minute granularity (H2Q4 format).
        
        Args:
            df (pd.DataFrame): Input DataFrame
            mercado (str): Market type ('diario', 'intra', 'continuo')
            
        Returns:
            pd.DataFrame: DataFrame with processed energy column and applied filters
        """
        print("\n⚡ PROCESSING ENERGY AND FILTERS")
        print("-"*30)
        
        initial_rows = len(df)
        
        # Helper function to clean numeric column
        def clean_numeric_column(column_series):
            if pd.api.types.is_numeric_dtype(column_series):
                return column_series
            else:
                return (column_series.str.replace('.', '')
                                  .str.replace(',', '.')
                                  .astype(float))
        
        if mercado in ['diario', 'intra']:
            # Filter only matched units (Casada)
            if 'Ofertada (O)/Casada (C)' in df.columns:
                df = df[df['Ofertada (O)/Casada (C)'] == 'C']
                print(f"   Filtered to matched units: {len(df)} rows")
            
            # Process energy column (now standardized to always be 'Energía Compra/Venta')
            if 'Energía Compra/Venta' in df.columns:
                # Clean and convert to numeric
                df['Energía Compra/Venta'] = clean_numeric_column(df['Energía Compra/Venta'])
                
                # Check if we need to convert from power to energy (15-minute granularity)
                if 'granularity' in df.columns and df['granularity'].iloc[0] == 'Quince minutos':
                    # Convert from power (per hour) to energy (per 15 minutes)
                    df['Energía Compra/Venta'] = df['Energía Compra/Venta'] / 4
                    print(f"   Converted from power to energy (÷4) for 15-minute granularity")
                
                df = df.rename(columns={'Energía Compra/Venta': 'volumenes'})
                print(f"   Processed and renamed 'Energía Compra/Venta' to 'volumenes'")
            else:
                raise ValueError("'Energía Compra/Venta' column not found for diario/intra market")
                
            # Apply buy/sell multiplier logic
            if 'Tipo Oferta' in df.columns and 'volumenes' in df.columns:
                # Create multiplier: Compra (C) = -1, Venta (V) = 1
                df['Extra'] = np.where(df['Tipo Oferta'] == 'C', -1, 1)
                df['volumenes'] = df['volumenes'] * df['Extra']
                print(f"   Applied buy/sell multiplier (C=-1, V=1)")
                # Drop the temporary Extra column
                df = df.drop(columns=['Extra'])
        
        elif mercado == 'continuo':
            # For continuo market, process Cantidad column
            if 'Cantidad' in df.columns:
                df['Cantidad'] = clean_numeric_column(df['Cantidad'])
                df = df.rename(columns={'Cantidad': 'volumenes'})
                print(f"   Processed 'Cantidad' column for continuo market")
            else:
                raise ValueError("'Cantidad' column not found for continuo market")
        
        print(f"   Rows: {initial_rows} → {len(df)}")
        print("-"*30)
        return df

    def _standardize_price_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize the price column for continuo market data.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with standardized price column
        """
        if 'Precio' in df.columns:
            df['Precio'] = df['Precio'].str.replace('.', '')
            df['Precio'] = df['Precio'].str.replace(',', '.')
            df['Precio'] = df['Precio'].astype(float)
            df = df.rename(columns={'Precio': 'precio'})
        
        return df

    def _process_uof_columns(self, df: pd.DataFrame, mercado: str) -> pd.DataFrame:
        """
        Process unit columns based on market type. For diairo and intra markets, rename Unidad to uof. 
        For continuo markets, handle buy and sell units separately, making sell volumes positive and buy volumes negative.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            mercado (str): Market type ('diario', 'intra', 'continuo')
            
        Returns:
            pd.DataFrame: DataFrame with processed unit columns
        """
        print("\n🏭 PROCESSING UNIT COLUMNS")
        print("-"*30)
        
        if mercado in ['diario', 'intra']:
            # For diario and intra markets, rename Unidad to uof
            if 'Unidad' in df.columns:
                df = df.rename(columns={'Unidad': 'uof'})
                print(f"   Renamed 'Unidad' to 'uof'")
        
        elif mercado == 'continuo':
            # For continuo market, handle buy and sell units separately
            df_buy = pd.DataFrame()
            df_sell = pd.DataFrame()
            
            if 'Unidad compra' in df.columns: 
                df_buy = df[df['Unidad compra'].notna()].copy()
                df_buy = df_buy.rename(columns={'Unidad compra': 'uof'})
                # Make buy volumes negative
                if 'volumenes' in df_buy.columns:
                    df_buy['volumenes'] = -df_buy['volumenes']
                print(f"   Processed buy units: {len(df_buy)} rows")
            
            if 'Unidad venta' in df.columns:
                df_sell = df[df['Unidad venta'].notna()].copy()
                df_sell = df_sell.rename(columns={'Unidad venta': 'uof'})
                print(f"Processed sell units: {len(df_sell)} rows")
            
            # Combine buy and sell data
            if not df_buy.empty and not df_sell.empty:
                df = pd.concat([df_buy, df_sell], ignore_index=True)
            elif not df_buy.empty:
                df = df_buy
            elif not df_sell.empty:
                df = df_sell
            
            print(f"   Combined units: {len(df)} rows")
        
        print("-"*30)
        return df
    
    # === 3. PROCESSING DATETIME COLUMN ===
    def _preprocess_datetime_column(self, df: pd.DataFrame, mercado: str) -> pd.DataFrame:
        """
        Standardize datetime columns (Fecha and Hora) for all market types.
        This is the first step before UTC conversion.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            mercado (str): Market type ('diario', 'intra', 'continuo')
            
        Returns:
            pd.DataFrame: DataFrame with standardized Fecha and Hora columns
        """
        print("\n🕐 STANDARDIZING DATETIME COLUMNS")
        print("-"*30)
        
        # Convert Fecha to datetime for all market types
        df['Fecha'] = pd.to_datetime(df['Fecha'], format="mixed")  # some dates have hours ie 00:00:00
        
        if mercado == 'continuo':
            # For continuo market, extract from Contrato column
            if 'Contrato' in df.columns:
                # Extract delivery date from contract string (first 8 characters: YYYYMMDD)
                df['delivery_date_str'] = df['Contrato'].str.strip().str[:8]
                df['Fecha'] = pd.to_datetime(df['delivery_date_str'], format="%Y%m%d")
                
                # Extract delivery hour (keep 1-based for consistency)
                df['delivery_hour'] = df['Contrato'].str.strip()[9:11].astype(int)
                df['Hora'] = df['delivery_hour']
                
                print(f"   Extracted Fecha and Hora from Contrato for continuo market (1-based hour)")
            
            # Store file date for continuo market
            df['fecha_fichero'] = df['Fecha'].dt.strftime('%Y-%m-%d')
        
        elif mercado in ['diario', 'intra']:

            #if granularity is quince minutos, parse the Periodo column to create the Hora column
            if df['granularity'].iloc[0] == 'Quince minutos':

                # Expected format: HxQy where x is hour (1-24) and y is quarter (1-4)
                periodo_regex = r'H(\d{1,2})Q(\d)'
                
                # Extract hour and quarter
                extracted = df['Hora'].str.extract(periodo_regex)
                extracted.columns = ['hour_str', 'quarter_str']
                
                # Convert to numeric
                hour = pd.to_numeric(extracted['hour_str'])
                quarter = pd.to_numeric(extracted['quarter_str'])
                
                # Calculate 15-minute interval index (1-based)
                # (hour - 1) * 4 gives the start of the hour block-
                # + quarter gives the specific interval, ex: H2Q1 becomes period 
                df['Hora'] = (hour - 1) * 4 + quarter
                
                print(f"   Successfully created 'Hora' as 15-min interval index from 'Periodo'")
            
            df['Hora'] = df['Hora'].astype(int)  # Keep 1-based, conver to int
        
        print(f"   Standardized Fecha and Hora columns for {mercado} market")
        print("-"*30)
        return df
    
    def _standardize_datetime(self, df: pd.DataFrame, mercado: str) -> pd.DataFrame:
        """
        Ensures a standard UTC datetime column with proper granularity handling.
        Handles different input formats and properly processes DST transitions.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            mercado (str): Market type ('diario', 'intra', 'continuo')
            
        Returns:
            pd.DataFrame: DataFrame with standardized datetime_utc column
        """
        if df.empty: 
            return df

        # Verify required columns exist
        required_cols = ['Fecha', 'Hora', 'granularity']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Required columns: {required_cols} not found in DataFrame.")

        # Ensure Fecha is datetime
        df['Fecha'] = pd.to_datetime(df['Fecha'])
        
        # Get timezone object
        tz = pytz.timezone('Europe/Madrid')
        
        # Get transition dates for relevant year range
        year_min = df['Fecha'].dt.year.min() - 1
        year_max = df['Fecha'].dt.year.max() + 1
        start_range = pd.Timestamp(year=year_min, month=1, day=1)
        end_range = pd.Timestamp(year=year_max, month=12, day=31)
        transition_dates = TimeUtils.get_transition_dates(start_range, end_range)
        
        # Create mask for DST transition days
        df['is_dst_day'] = df['Fecha'].dt.date.apply(lambda x: x in transition_dates)
        
        # Split data by granularity and DST status
        #hourly data
        df_hourly_dst = df[(df['granularity'] == 'Hora') & (df['is_dst_day'])].copy()
        df_hourly_normal = df[(df['granularity'] == 'Hora') & (~df['is_dst_day'])].copy()

        #15min data *TODO: Implement 15min data processing
        df_15min_dst = df[(df['granularity'] == 'Quince minutos') & (df['is_dst_day'])].copy()
        df_15min_normal = df[(df['granularity'] == 'Quince minutos') & (~df['is_dst_day'])].copy()
        
        # Process hourly data
        df_hourly_processed_dst = pd.DataFrame()
        if not df_hourly_dst.empty:
            print(f"Processing {len(df_hourly_dst)} rows of hourly DST data...")
            df_hourly_processed_dst = self._process_omie_hourly_dst_data(df_hourly_dst, transition_dates)
        
        df_hourly_processed_normal = pd.DataFrame()
        if not df_hourly_normal.empty:
            print(f"Processing {len(df_hourly_normal)} rows of hourly non-DST data...")
            df_hourly_processed_normal = self._process_omie_hourly_normal_data(df_hourly_normal)
        
        # Process 15-minute data (when implemented)
        df_15min_processed_dst = pd.DataFrame()
        if not df_15min_dst.empty:
            print(f"Processing {len(df_15min_dst)} rows of 15-minute DST data...")
            df_15min_processed_dst = self._process_omie_15min_dst_data(df_15min_dst, transition_dates)
        
        df_15min_processed_normal = pd.DataFrame()
        if not df_15min_normal.empty:
            print(f"Processing {len(df_15min_normal)} rows of 15-minute non-DST data...")
            df_15min_processed_normal = self._process_omie_15min_normal_data(df_15min_normal)
        
        # Combine results
        final_df = pd.concat([
            df_hourly_processed_dst, 
            df_hourly_processed_normal,
            df_15min_processed_dst, 
            df_15min_processed_normal
        ], ignore_index=True)
        
        # Clean up intermediate columns
        cols_to_drop = ['Fecha', 'Hora', 'granularity', 'is_dst_day', 'datetime_local', 'datetime_naive']
        final_df = final_df.drop(columns=[c for c in cols_to_drop if c in final_df.columns], errors='ignore')
        
        # Drop rows with invalid datetimes and sort
        final_df = final_df.dropna(subset=['datetime_utc'])
        if not final_df.empty:
            final_df = final_df.sort_values(by='datetime_utc').reset_index(drop=True)
        
        return final_df

    @with_progress(message="Processing OMIE hourly DST data")
    def _process_omie_hourly_dst_data(self, df: pd.DataFrame, transition_dates: Dict) -> pd.DataFrame:
        """
        Process OMIE hourly data for DST transition days.
        Handles spring forward (23 hours: 1-23, skip hour 2) and fall back (25 hours: repeat hour 2).
        
        Args:
            df (pd.DataFrame): Input DataFrame with DST transition days
            transition_dates (Dict): Dictionary of DST transition dates
            
        Returns:
            pd.DataFrame: DataFrame with processed datetime_utc
        """
        try:
            result_df = df.copy()
            
            # Apply the parsing function to create datetime_local
            result_df['datetime_local'] = result_df.apply(
                lambda row: self._parse_omie_hourly_datetime_local(row['Fecha'], row['Hora'], transition_dates), 
                axis=1
            )
            
            # Convert to UTC
            utc_df = self.date_utils.convert_local_to_utc(result_df['datetime_local'])
            result_df['datetime_utc'] = utc_df['datetime_utc']
            
            # Convert to 15-minute frequency
            result_df = self.date_utils.convert_hourly_to_15min(result_df, "volumenes_omie")
            
            return result_df
        
        except Exception as e:
            print(f"Error processing OMIE hourly DST data: {e}")
            return pd.DataFrame()

    @with_progress(message="Processing OMIE hourly normal data (vectorized)")
    def _process_omie_hourly_normal_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process OMIE hourly data for normal (non-DST transition) days using vectorized operations.
        
        Args:
            df (pd.DataFrame): Input DataFrame with normal days
            
        Returns:
            pd.DataFrame: DataFrame with processed datetime_utc
        """
        try:
            result_df = df.copy()
            tz = pytz.timezone('Europe/Madrid')
            
            # Create naive datetime by combining date and hour (convert from 1-based to 0-based here)
            result_df['datetime_naive'] = result_df['Fecha'] + pd.to_timedelta(result_df['Hora'] - 1, unit='h')
            
            # Vectorized localization for normal days (no DST ambiguity)
            local_dt = result_df['datetime_naive'].dt.tz_localize(tz, ambiguous='infer', nonexistent='shift_forward')
            result_df['datetime_utc'] = local_dt.dt.tz_convert('UTC')
            
            # Convert to 15-minute frequency
            result_df = self.date_utils.convert_hourly_to_15min(result_df, "volumenes_omie")
            
            return result_df
        
        except Exception as e:
            print(f"Error processing OMIE hourly normal data: {e}")
            return pd.DataFrame()

    @with_progress(message="Processing OMIE 15-minute DST data")
    def _process_omie_15min_dst_data(self, df: pd.DataFrame, transition_dates: Dict) -> pd.DataFrame:
        """
        Process OMIE 15-minute data for DST transition days.
        Handles spring forward (92 intervals) and fall back (100 intervals).
        
        Args:
            df (pd.DataFrame): Input DataFrame with DST transition days
            transition_dates (Dict): Dictionary of DST transition dates
            
        Returns:
            pd.DataFrame: DataFrame with processed datetime_utc
        """
        try:
            result_df = df.copy()
            
            # Apply the parsing function to create datetime_local
            result_df['datetime_local'] = result_df.apply(
                lambda row: self._parse_omie_15min_datetime_local(row['Fecha'], row['Hora'], transition_dates), 
                axis=1
            )
            
            # Convert to UTC
            utc_df = self.date_utils.convert_local_to_utc(result_df['datetime_local'])
            result_df['datetime_utc'] = utc_df['datetime_utc']
            
            return result_df
        
        except Exception as e:
            print(f"Error processing OMIE 15-minute DST data: {e}")
            return pd.DataFrame()

    @with_progress(message="Processing OMIE 15-minute normal data (vectorized)")
    def _process_omie_15min_normal_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process OMIE 15-minute data for normal (non-DST transition) days using vectorized operations.
        
        Args:
            df (pd.DataFrame): Input DataFrame with normal days
            
        Returns:
            pd.DataFrame: DataFrame with processed datetime_utc
        """
        try:
            result_df = df.copy()
            tz = pytz.timezone('Europe/Madrid')
            
            # Create naive datetime by combining date and hour (convert from 1-based to 0-based here)
            # For 15-minute data, Hora represents the 15-minute interval (1-96), convert to 0-95
            result_df['datetime_naive'] = result_df['Fecha'] + pd.to_timedelta((result_df['Hora'] - 1) * 15, unit='m')
            
            # Vectorized localization for normal days (no DST ambiguity)
            local_dt = result_df['datetime_naive'].dt.tz_localize(tz, ambiguous='infer', nonexistent='shift_forward')
            result_df['datetime_utc'] = local_dt.dt.tz_convert('UTC')
            
            return result_df
        
        except Exception as e:
            print(f"Error processing OMIE 15-minute normal data: {e}")
            return pd.DataFrame()

    def _parse_omie_hourly_datetime_local(self, fecha, hora_int, transition_dates: Dict) -> pd.Timestamp:
        """
        Parse OMIE hourly data into timezone-aware datetime in Europe/Madrid timezone.
        Handles DST transitions:
        - Spring forward: 23 hours (1-23), hour 2 doesn't exist
        - Fall back: 25 hours (1-25), hour 2 is repeated (first occurrence and second occurrence)
        
        Args:
            fecha: Date object or datetime object
            hora_int: Hour integer (1-based indexing: 1=00:00, 2=01:00, etc.)
            transition_dates: Dictionary of DST transition dates
            
        Returns:
            Timezone-aware pd.Timestamp in Europe/Madrid timezone
        """
        # Ensure fecha is a date object
        if isinstance(fecha, pd.Timestamp):
            fecha = fecha.date()
        elif isinstance(fecha, str):
            fecha = pd.to_datetime(fecha).date()
    
        # Get timezone object
        tz = pytz.timezone('Europe/Madrid')
        
        # Convert 1-based hora_int to 0-based hour (1=00:00, 2=01:00, etc.)
        hour_24 = hora_int - 1
        
        # Check if this is a DST transition date
        is_transition_date = fecha in transition_dates
        # Determine the type of DST transition for this date:
        # - 0: Normal day (no transition)
        # - 1: Fall back transition (25-hour day, hour repeated)
        # - 2: Spring forward transition (23-hour day, hour skipped)
        transition_type = transition_dates.get(fecha, 0) if is_transition_date else 0
        
        # Initialize is_dst for DST handling
        is_dst = None
        
        # Handle DST transitions
        if is_transition_date:
            if transition_type == 2:  # Spring forward: 23-hour day (skip hour 2:00)
                if hour_24 == 2:
                    # Hour 2:00 doesn't exist on spring forward day, shifting to hour 3
                    print(f"Warning: Hour 2 found on spring forward day {fecha}, shifting to hour 3")
                    hour_24 = 3
                    is_dst = True
                elif hour_24 >= 3:
                    # Hours after 2:00 are already correct (DST is in effect)
                    is_dst = True
            
            elif transition_type == 1:  # Fall back: 25-hour day (repeat hour 2:00)
                if hour_24 == 1:  # This is 1:00 AM (still DST)
                    is_dst = True
                elif hora_int >= 3 and hora_int <= 25:
                    # This represents the period after the fall back
                    # Map hora_int 3-25 to actual hours 2-24 in standard time
                    if hora_int == 3:
                        # This is the second occurrence of 2:00 AM (now in standard time)
                        hour_24 = 2
                        is_dst = False
                    else:
                        # Map remaining hours: hora_int 4-25 -> hours 3-24
                        hour_24 = hora_int - 2
                        is_dst = False
                else:
                    is_dst = None  # Let pytz decide
        else:
            is_dst = None  # Normal day, let pytz decide
        
        # Create naive datetime
        naive_dt = pd.Timestamp(
            year=fecha.year,
            month=fecha.month,
            day=fecha.day,
            hour=hour_24,
            minute=0,
            second=0
        )
        
        # Localize the datetime with DST handling
        try:
            if is_transition_date and transition_type in [1, 2]:
                local_dt = tz.localize(naive_dt, is_dst=is_dst)
            else:
                local_dt = tz.localize(naive_dt)
        except pytz.exceptions.AmbiguousTimeError:
            # For ambiguous times (fall-back), default to first occurrence if not specified
            local_dt = tz.localize(naive_dt, is_dst=True if is_dst is None else is_dst)
        except pytz.exceptions.NonExistentTimeError:
            # For non-existent times (spring-forward), shift forward
            shifted_dt = naive_dt.replace(hour=3)
            local_dt = tz.localize(shifted_dt, is_dst=True)
        
        return local_dt
    
    def _parse_omie_15min_datetime_local(self, fecha, hora_index, transition_dates: Dict) -> pd.Timestamp:
        """
        Parse OMIE 15-minute format data into a timezone-aware datetime in Europe/Madrid timezone.
        Handles DST transitions correctly.
        
        Args:
            fecha: Date object or datetime object
            hora_index: 15-minute interval index (1-96 for normal days, 1-92 for spring forward, 1-100 for fall back)
            transition_dates: Dictionary of DST transition dates
            
        Returns:
            Timezone-aware pd.Timestamp in Europe/Madrid timezone
        """
        # Ensure fecha is a date object
        if isinstance(fecha, pd.Timestamp):
            fecha = fecha.date()
        elif isinstance(fecha, str):
            fecha = pd.to_datetime(fecha).date()
        
        # Ensure index is an integer
        index = int(hora_index)
        if index < 1:
            raise ValueError(f"Invalid 15-minute index: {hora_index}. Must be ≥ 1.")
        
        # Get timezone object
        tz = pytz.timezone('Europe/Madrid')
        
        # Check if this is a DST transition date
        is_transition_date = fecha in transition_dates
        transition_type = transition_dates.get(fecha, 0) if is_transition_date else 0
        
        # Calculate number of intervals for the day
        if is_transition_date:
            if transition_type == 2:  # Spring forward: skip hour 2
                num_intervals = 92  # 96 - 4 (skipped 15-min intervals)
            elif transition_type == 1:  # Fall back: repeat hour 2
                num_intervals = 100  # 96 + 4 (repeated 15-min intervals)
        else:
            num_intervals = 96  # Normal day
        
        # Handle out-of-bounds index
        if index > num_intervals:
            raise ValueError(f"Index {index} out of bounds for date {fecha} with {num_intervals} intervals.")
        
        # Calculate hour and minute
        hour = (index - 1) // 4
        minute = ((index - 1) % 4) * 15
        
        # Handle DST transitions
        if is_transition_date:
            if transition_type == 2:  # Spring forward
                if hour >= 2:
                    hour += 1  # Skip hour 2
            elif transition_type == 1:  # Fall back
                if hour >= 2:
                    # For fall back, we need to handle the repeated hour 2
                    if index > 8:  # After first occurrence of hour 2
                        hour += 1
        
        # Create timestamp
        ts = pd.Timestamp.combine(fecha, time(hour=hour, minute=minute))
        
        # Localize with appropriate DST handling
        try:
            is_dst = None
            if is_transition_date:
                if transition_type == 1:  # Fall back
                    # For fall back, first occurrence of hour 2 is DST, second is not
                    is_dst = hour < 2 or (hour == 2 and index <= 8)
                elif transition_type == 2:  # Spring forward
                    # For spring forward, all times after the transition are DST
                    is_dst = hour >= 3
            
            local_dt = tz.localize(ts, is_dst=is_dst)
            return local_dt
            
        except (pytz.AmbiguousTimeError, pytz.NonExistentTimeError) as e:
            print(f"Warning: Could not localize {ts} on date {fecha}: {e}")
            # Return a reasonable fallback
            if transition_type == 2:  # Spring forward
                ts = ts.replace(hour=max(3, hour))  # Ensure we're after spring forward
            return tz.localize(ts, is_dst=True)
    

    # === 4. AGGREGATION ===
    def _aggregate_data(self, df: pd.DataFrame, mercado: str) -> pd.DataFrame:
        """
        Aggregate data by grouping columns based on market type.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            mercado (str): Market type ('diario', 'intra', 'continuo')
            
        Returns:
            pd.DataFrame: Aggregated DataFrame
        """
        print("\n📊 AGGREGATING DATA")
        print("-"*30)
        
        initial_rows = len(df)
        
        if mercado in ['diario', 'intra']:
            # Group by uof, date, hour, and id_mercado
            group_cols = ['uof', 'datetime_utc', 'id_mercado']
            df = df.groupby(group_cols).agg({'volumenes': 'sum'}).reset_index()
        
        elif mercado == 'continuo':
            # For continuo, no aggregation needed as each row represents a trade
            pass
        
        print(f"   Rows: {initial_rows} → {len(df)}")
        print("-"*30)
        return df

    # === 5. SELECT AND FINALIZE COLUMNS ===
    def _select_and_finalize_columns(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """
        Select final columns based on dataset type and validation requirements.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            dataset_type (str): Type of dataset ('volumenes_omie' or 'volumenes_mic')
            
        Returns:
            pd.DataFrame: DataFrame with final columns
        """
        print("\n📋 FINALIZING COLUMNS")
        print("-"*30)
        
        if dataset_type == 'volumenes_omie':
            required_cols = self.data_validator.processed_volumenes_omie_required_cols
        elif dataset_type == 'volumenes_mic':
            required_cols = self.data_validator.processed_volumenes_mic_required_cols
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        # Filter to only include required columns that exist
        available_cols = [col for col in required_cols if col in df.columns]
        df = df[available_cols]
        
        print(f"   Selected columns: {available_cols}")
        print("-"*30)
        return df

    # === 6. DATA VALIDATION ===
    def _validate_final_data(self, df: pd.DataFrame, dataset_type: str, validation_type: str = "processed") -> pd.DataFrame:
        """
        Validate data structure and types. Deffault for omie is processed (we dont check for raw data validation)
        
        Args:
            df (pd.DataFrame): Input DataFrame
            validation_type (str): 'raw' or 'processed'
            dataset_type (str): Dataset type for validation
            
        Returns:
            pd.DataFrame: Validated DataFrame
        """
        print(f"\n🔍 DATA VALIDATION ({validation_type.upper()})")
        print("-"*30)
        
        if df.empty:
            print("⚠️  Empty DataFrame - Skipping validation")
            return df
        
        try:
            if validation_type == "processed":
                df = self.data_validator.validate_processed_data(df, dataset_type)
            elif validation_type == "raw":
                df = self.data_validator.validate_raw_data(df, dataset_type)
            
            print("✅ Validation passed")
        except Exception as e:
            print(f"❌ Validation failed: {str(e)}")
            raise
        
        print("-"*30)
        return df

    # === MAIN PIPELINE FUNCTIONS ===
    def transform_omie_diario(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform OMIE diario market data.
        
        Args:
            df (pd.DataFrame): Raw diario market data
            
        Returns:
            pd.DataFrame: Processed diario market data
        """
        print("\n" + "="*80)
        print("🔄 STARTING OMIE DIARIO TRANSFORMATION")
        print("="*80)

        if df.empty:
            print("\n❌ EMPTY INPUT")
            empty_df = pd.DataFrame(columns=self.data_validator.processed_volumenes_omie_required_cols)
            return empty_df
        
        mercado = "diario"
        dataset_type = "volumenes_omie"

        # Define processing pipeline
        pipeline = [
            (self._clean_empty_rows, {}),
            (self._add_granularity_column, {}),
            (self._process_and_filter_energy_column, {"mercado": mercado}),
            (self._process_uof_columns, {"mercado": mercado}),
            (self._preprocess_datetime_column, {"mercado": mercado}),
            (self._standardize_datetime, {"mercado": mercado}),
            (self._aggregate_data, {"mercado": mercado}),
            (self._select_and_finalize_columns, {"dataset_type": dataset_type}),
            (self._validate_final_data, {"dataset_type": dataset_type}),
        ]

        return self._execute_pipeline(df, pipeline, "DIARIO")   

    def transform_omie_intra(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform OMIE intra market data.
        
        Args:
            df (pd.DataFrame): Raw intra market data
            
        Returns:
            pd.DataFrame: Processed intra market data
        """
        print("\n" + "="*80)
        print("🔄 STARTING OMIE INTRA TRANSFORMATION")
        print("="*80)

        if df.empty:
            print("\n❌ EMPTY INPUT")
            empty_df = pd.DataFrame(columns=self.data_validator.processed_volumenes_omie_required_cols)
            return empty_df

        mercado = "intra"
        dataset_type = "volumenes_omie"

        # Define processing pipeline
        pipeline = [
            (self._clean_empty_rows, {}),
            (self._add_granularity_column, {}),
            (self._process_and_filter_energy_column, {"mercado": mercado}),
            (self._process_uof_columns, {"mercado": mercado}),
            (self._preprocess_datetime_column, {"mercado": mercado}),
            (self._standardize_datetime, {"mercado": mercado}),
            (self._aggregate_data, {"mercado": mercado}),
            (self._select_and_finalize_columns, {"dataset_type": dataset_type}),
            (self._validate_final_data, {"validation_type": "processed", "dataset_type": dataset_type}),
        ]

        return self._execute_pipeline(df, pipeline, "INTRA")

    def transform_omie_continuo(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform OMIE continuo market data.
        
        Args:
            df (pd.DataFrame): Raw continuo market data
            
        Returns:
            pd.DataFrame: Processed continuo market data
        """
        print("\n" + "="*80)
        print("🔄 STARTING OMIE CONTINUO TRANSFORMATION")
        print("="*80)

        mercado = "continuo"
        dataset_type = "volumenes_mic" #this is not a real dataset type (a filename), this identifier is just used to validate the data

        if df.empty:
            print("\n❌ EMPTY INPUT")
            empty_df = pd.DataFrame(columns=self.data_validator.processed_volumenes_mic_required_cols)
            return empty_df

        # Define processing pipeline
        pipeline = [
            (self._clean_empty_rows, {}),
            (self._process_and_filter_energy_column, {"mercado": mercado}),
            (self._process_uof_columns, {"mercado": mercado}),
            (self._preprocess_datetime_column, {"mercado": mercado}),
            (self._standardize_datetime, {"mercado": mercado}),
            (self._aggregate_data, {"mercado": mercado}),
            (self._select_and_finalize_columns, {"dataset_type": dataset_type}),
            (self._validate_final_data, {"validation_type": "processed", "dataset_type": dataset_type}),
        ]

        return self._execute_pipeline(df, pipeline, "CONTINUO")

    def _execute_pipeline(self, df: pd.DataFrame, pipeline: List, market_name: str) -> pd.DataFrame:
        """
        Execute a processing pipeline with error handling and progress tracking printouts.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            pipeline (List): List of (function, kwargs) tuples
            market_name (str): Market name for logging
            
        Returns:
            pd.DataFrame: Processed DataFrame
        """
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

                if df_processed.empty and step_func.__name__ != '_validate_final_data':
                    print("\n❌ PIPELINE HALTED")
                    print(f"DataFrame became empty after step: {step_func.__name__}")
                    raise ValueError(f"DataFrame empty after: {step_func.__name__}")

            print("\n✅ TRANSFORMATION COMPLETE")
            print(f"Final shape: {df_processed.shape}")
            print("="*80 + "\n")
            return df_processed

        except Exception as e:
            print("\n❌ TRANSFORMATION FAILED")
            print(f"Error in {market_name} processing: {str(e)}")
            print("="*80 + "\n")
            raise


def example_usage():    
    processor = OMIEProcessor()
    df_intra = pd.read_csv("C:/Users/Usuario/OneDrive - OPTIMIZE ENERGY/Escritorio/Optimize Energy/timescale_v_duckdb_testing/data/raw/intra/2024/03/volumenes_omie.csv")
    df_diario = pd.read_csv("C:/Users/Usuario/OneDrive - OPTIMIZE ENERGY/Escritorio/Optimize Energy/timescale_v_duckdb_testing/data/raw/diario/2024/10/volumenes_omie.csv")
    df_continuo = pd.read_csv("C:/Users/Usuario/OneDrive - OPTIMIZE ENERGY/Escritorio/Optimize Energy/timescale_v_duckdb_testing/data/raw/continuo/2025/02/volumenes_omie.csv")
    
    df_intra = processor.transform_omie_intra(df_intra)
    print(df_intra)
    breakpoint()

    df_diario = processor.transform_omie_diario(df_diario)
    print(df_diario)
    breakpoint()

    df_continuo = processor.transform_omie_continuo(df_continuo)
    print(df_continuo)

if __name__ == "__main__":
    example_usage()
