from typing import Dict, List, Optional, Union
import pandas as pd
from datetime import datetime
import sys
import os
import numpy as np

# Add necessary imports
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Ensure the project root is added to sys.path if necessary
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utilidades.etl_date_utils import DateUtilsETL, TimeUtils
from utilidades.data_validation_utils import DataValidationUtils

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

    def _filter_matched_units(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter data to only include matched units (Casada) and apply buy/sell logic.
        Based on the logic from carga_omie.py for intra and diario markets.
        
        Args:
            df (pd.DataFrame): Input DataFrame with OMIE data
            
        Returns:
            pd.DataFrame: Filtered DataFrame with only matched units
        """
        print("\n🔍 FILTERING MATCHED UNITS")
        print("-"*30)
        
        initial_rows = len(df)
        
        # Filter only matched units (Casada)
        if 'Ofertada (O)/Casada (C)' in df.columns:
            df = df[df['Ofertada (O)/Casada (C)'] == 'C']
            print(f"   Filtered to matched units: {len(df)} rows")
        
        # Apply buy/sell multiplier logic
        if 'Tipo Oferta' in df.columns:
            df['Extra'] = np.where(df['Tipo Oferta'] == 'C', -1, 1)
            print(f"   Applied buy/sell multiplier")
        
        print(f"   Rows: {initial_rows} → {len(df)}")
        print("-"*30)
        
        return df

    def _standardize_energy_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize the energy column by cleaning numeric formatting.
        Based on the logic from carga_omie.py.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with standardized energy column
        """
        print("\n⚡ STANDARDIZING ENERGY COLUMN")
        print("-"*30)
        
        if 'Energía Compra/Venta' in df.columns:
            # Clean numeric formatting (remove thousands separator and fix decimal)
            df['Energía Compra/Venta'] = df['Energía Compra/Venta'].str.replace('.', '')
            df['Energía Compra/Venta'] = df['Energía Compra/Venta'].str.replace(',', '.')
            df['Energía Compra/Venta'] = df['Energía Compra/Venta'].astype(float)
            
            # Rename to standard column name
            df = df.rename(columns={'Energía Compra/Venta': 'volumenes'})
            print(f"   Renamed to 'volumenes'")
        
        elif 'Cantidad' in df.columns:
            # For continuo market data
            df['Cantidad'] = df['Cantidad'].str.replace('.', '')
            df['Cantidad'] = df['Cantidad'].str.replace(',', '.')
            df['Cantidad'] = df['Cantidad'].astype(float)
            df = df.rename(columns={'Cantidad': 'volumenes'})
            print(f"   Processed 'Cantidad' column for continuo market")
        
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

    def _process_unit_columns(self, df: pd.DataFrame, mercado: str) -> pd.DataFrame:
        """
        Process unit columns based on market type.
        
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

    def _process_datetime_intra_diario(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process datetime for intra and diario markets.
        Combines 'Fecha' and 'Hora' columns to create datetime_utc.
        
        Args:
            df (pd.DataFrame): Input DataFrame with 'Fecha' and 'Hora' columns
            
        Returns:
            pd.DataFrame: DataFrame with datetime_utc column
        """
        print("\n🕐 PROCESSING DATETIME (INTRA/DIARIO)")
        print("-"*30)
        
        if 'Fecha' in df.columns and 'Hora' in df.columns:
            # Convert Fecha to datetime
            df['Fecha'] = pd.to_datetime(df['Fecha'], format="%d/%m/%Y")
            
            # Convert Hora to integer
            df['Hora'] = df['Hora'].astype(int)
            
            # Create naive datetime by combining date and hour
            df['datetime_naive'] = df['Fecha'] + pd.to_timedelta(df['Hora'] - 1, unit='h')
            
            # Convert to UTC using DateUtilsETL
            utc_df = self.date_utils.convert_naive_to_utc(df['datetime_naive'])
            df['datetime_utc'] = utc_df['datetime_utc']
            
            # Clean up intermediate columns
            df = df.drop(columns=['datetime_naive'], errors='ignore')
            
            print(f"   Created datetime_utc from Fecha and Hora")
        
        print("-"*30)
        return df

    def _process_datetime_continuo(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process datetime for continuo market.
        Uses 'Contrato' column to extract delivery period and applies DST adjustments.
        
        Args:
            df (pd.DataFrame): Input DataFrame with 'Contrato' column
            
        Returns:
            pd.DataFrame: DataFrame with delivery_period column in UTC
        """
        print("\n🕐 PROCESSING DATETIME (CONTINUO)")
        print("-"*30)
        
        if 'Fecha' in df.columns:
            # Process file date (fecha_fichero)
            df['Fecha'] = pd.to_datetime(df['Fecha'], format="%d/%m/%Y")
            df['fecha_fichero'] = df['Fecha'].dt.strftime('%Y-%m-%d')
        
        if 'Contrato' in df.columns:
            # Extract delivery date from contract string (first 8 characters: YYYYMMDD)
            df['delivery_date_str'] = df['Contrato'].str.strip().str[:8]
            df['delivery_date'] = pd.to_datetime(df['delivery_date_str'], format="%Y%m%d")
            
            # Get transition dates for DST handling
            year_min = df['delivery_date'].dt.year.min()
            year_max = df['delivery_date'].dt.year.max()
            start_range = datetime(year_min - 1, 1, 1)
            end_range = datetime(year_max + 1, 12, 31)
            transition_dates = TimeUtils.get_transition_dates(start_range, end_range)
            
            # Apply hour adjustment logic from ajuste_horario function
            df['delivery_hour'] = df.apply(
                lambda row: self._adjust_continuo_hour(row, transition_dates), 
                axis=1
            )
            
            # Create naive delivery period datetime (local time)
            df['delivery_period_naive'] = df['delivery_date'] + pd.to_timedelta(df['delivery_hour'] - 1, unit='h')
            
            # Convert naive datetime to UTC using DateUtilsETL
            utc_df = self.date_utils.convert_naive_to_utc(df['delivery_period_naive'])
            df['delivery_period'] = utc_df['datetime_utc']
            
            # Clean up intermediate columns
            df = df.drop(columns=['delivery_date_str', 'delivery_date', 'delivery_hour', 'delivery_period_naive'], errors='ignore')
            
            print(f"   Created delivery_period in UTC from Contrato")
        
        print("-"*30)
        return df

    def _adjust_continuo_hour(self, row, transition_dates: Dict) -> int:
        """
        Adjust hour for continuo market based on DST transitions.
        Implements the ajuste_horario logic from carga_omie.py.
        
        Args:
            row: DataFrame row with delivery date and contract info
            transition_dates: Dictionary of DST transition dates
            
        Returns:
            int: Adjusted hour
        """
        date_ref = row['delivery_date'].date()
        is_special_date = date_ref in transition_dates
        tipo_cambio_hora = transition_dates.get(date_ref, 0)
        
        # Extract hour from contract string (positions 9-11)
        contract_hour = int(row['Contrato'].strip()[9:11])
        
        if is_special_date:
            if tipo_cambio_hora == 2:  # 23-hour day (spring forward)
                if (contract_hour + 1) < 3:
                    return contract_hour + 1
                else:
                    return contract_hour
            
            elif tipo_cambio_hora == 1:  # 25-hour day (fall back)
                if row['Contrato'].strip()[-1].isdigit():
                    if (contract_hour + 1) < 3:
                        return contract_hour + 1
                    else:
                        return contract_hour + 2
                elif row['Contrato'].strip()[-1] == 'A':
                    return contract_hour + 1
                elif row['Contrato'].strip()[-1] == 'B':
                    return contract_hour + 2
        
        # Normal days
        return contract_hour + 1

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
            group_cols = ['uof', 'Fecha', 'Hora', 'id_mercado']
            df = df.groupby(group_cols).agg({'volumenes': 'sum'}).reset_index()
        
        elif mercado == 'continuo':
            # For continuo, no aggregation needed as each row represents a trade
            pass
        
        print(f"   Rows: {initial_rows} → {len(df)}")
        print("-"*30)
        return df

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

    def _validate_data(self, df: pd.DataFrame, validation_type: str, dataset_type: str) -> pd.DataFrame:
        """
        Validate data structure and types.
        
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

        # Define processing pipeline
        pipeline = [
            (self._standardize_energy_column, {}),
            (self._process_unit_columns, {"mercado": "diario"}),
            (self._process_datetime_intra_diario, {}),
            (self._aggregate_data, {"mercado": "diario"}),
            (self._select_and_finalize_columns, {"dataset_type": "volumenes_omie"}),
            (self._validate_data, {"validation_type": "processed", "dataset_type": "volumenes_omie"})
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

        # Define processing pipeline
        pipeline = [
            (self._standardize_energy_column, {}),
            (self._process_unit_columns, {"mercado": "intra"}),
            (self._process_datetime_intra_diario, {}),
            (self._aggregate_data, {"mercado": "intra"}),
            (self._select_and_finalize_columns, {"dataset_type": "volumenes_omie"}),
            (self._validate_data, {"validation_type": "processed", "dataset_type": "volumenes_omie"})
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

        if df.empty:
            print("\n❌ EMPTY INPUT")
            empty_df = pd.DataFrame(columns=self.data_validator.processed_volumenes_mic_required_cols)
            return empty_df

        # Define processing pipeline
        pipeline = [
            (self._standardize_energy_column, {}),
            (self._standardize_price_column, {}),
            (self._process_unit_columns, {"mercado": "continuo"}),
            (self._process_datetime_continuo, {}),
            (self._select_and_finalize_columns, {"dataset_type": "volumenes_mic"}),
            (self._validate_data, {"validation_type": "processed", "dataset_type": "volumenes_mic"})
        ]

        return self._execute_pipeline(df, pipeline, "CONTINUO")

    def _execute_pipeline(self, df: pd.DataFrame, pipeline: List, market_name: str) -> pd.DataFrame:
        """
        Execute a processing pipeline with error handling and progress tracking.
        
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

                if df_processed.empty and step_func.__name__ != '_validate_data':
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
