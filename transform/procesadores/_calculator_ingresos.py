import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import os
from utilidades.processed_file_utils import ProcessedFileUtils
from utilidades.data_validation_utils import DataValidationUtils
from configs.ingresos_config import IngresosConfig
from configs.storage_config import VALID_DATASET_TYPES
from read._parquet_reader import ParquetReader

class IngresosCalculator:
    def __init__(self):
        self.file_utils = ProcessedFileUtils()
        self.config = IngresosConfig()
        self.parquet_reader = ParquetReader()

    def _find_latest_partition_path(self, mercado, id_mercado, dataset_type):
        base_path = self.file_utils.processed_path / f"mercado={mercado}" / f"id_mercado={id_mercado}"
        all_files = list(base_path.glob("**/year=*/month=*/{}.parquet".format(dataset_type)))
        if not all_files:
            return None
        latest_file = None
        latest_year = -1
        latest_month = -1
        for f in all_files:
            parts = f.parts
            try:
                year = int([p for p in parts if p.startswith('year=')][0].split('=')[1])
                month = int([p for p in parts if p.startswith('month=')][0].split('=')[1])
                if year > latest_year or (year == latest_year and month > latest_month):
                    latest_year = year
                    latest_month = month
                    latest_file = f
            except (ValueError, IndexError):
                continue
        return latest_file

    def _get_latest_df(self, mercado, id_mercado, dataset_type):
        path = self._find_latest_partition_path(mercado, id_mercado, dataset_type)
        if path:
            df = pd.read_parquet(path)
            print("Processed parquet:")
            print(df.head())
            print(df.tail())
            print(f"\n")
            return df
        return pd.DataFrame()

    def _find_latest_common_date(self, volumes_df, prices_df):
        if volumes_df.empty or prices_df.empty:
            return None
        volumes_dates = volumes_df['datetime_utc'].dt.date.unique()
        prices_dates = prices_df['datetime_utc'].dt.date.unique()
        common_dates = set(volumes_dates) & set(prices_dates)
        if not common_dates:
            print(f"Warning: No common dates found between volumes and prices.")
            return None
        return max(common_dates)

    def _get_df_for_date_range(self, mercado, id_mercado, dataset_type, fecha_inicio, fecha_fin):
        """
        Get DataFrame for a specific date range using ParquetReader.
        
        Args:
            mercado (str): Market type (e.g., "intra", "secundaria")
            id_mercado (int): Market ID
            dataset_type (str): Type of dataset (e.g., "precios", "volumenes")
            fecha_inicio (str): Start date in YYYY-MM-DD format
            fecha_fin (str): End date in YYYY-MM-DD format
            
        Returns:
            pd.DataFrame: Filtered DataFrame for the specified date range
        """
        try:
            # Use ParquetReader to get the data (now accepts single values)
            df = self.parquet_reader.read_parquet_data(
                fecha_inicio_lectura=fecha_inicio,
                fecha_fin_lectura=fecha_fin,
                mercado_lst=mercado,  # Single string value
                dataset_type=dataset_type,
                mercado_id_lst=id_mercado  
            )
            
            # Apply additional date filtering if needed (ParquetReader might return broader range)
            if not df.empty and 'datetime_utc' in df.columns:
                start_dt = pd.to_datetime(fecha_inicio).date()
                end_dt = pd.to_datetime(fecha_fin).date()
                df = df[(df['datetime_utc'].dt.date >= start_dt) & (df['datetime_utc'].dt.date <= end_dt)].copy()
            
            return df
            
        except Exception as e:
            print(f"Warning: Could not read data for {mercado} (id: {id_mercado}), {dataset_type}: {e}")
            return pd.DataFrame()

    def _calculate_ingresos(self, volumes_df, prices_df, market_key, id_mercado):
        if volumes_df.empty and prices_df.empty:
            return pd.DataFrame()

        if volumes_df.empty:
            raise ValueError(f"No volume data for market '{market_key}' (id: {id_mercado}) for the given period. No ingresos calculated.")
        
        if prices_df.empty:
            raise ValueError(f"No price data for market '{market_key}' (id: {id_mercado}) for the given period. No ingresos calculated.")

        if market_key == 'restricciones':
            merged = pd.merge(volumes_df, prices_df, on=['datetime_utc', 'id_mercado', 'up'], how='inner')
        else:
            merged = pd.merge(volumes_df, prices_df, on=['datetime_utc', 'id_mercado'], how='inner')
        
        if merged.empty:
            raise ValueError(f"For market '{market_key}' (id: {id_mercado}), data for volumes and prices could not be merged for the given period (no common timestamps). No ingresos calculated.")
            
        merged['ingresos'] = merged['volumenes'] * merged['precio']
        result = merged[['datetime_utc', 'up', 'ingresos', 'id_mercado']]
        return result

    def calculate_latest(self, market_key, date):
        ids = self.config.mercado_name_id_map.get(market_key, [])
        all_results = []
        for id_mercado in ids:
            volumes_df = self._get_latest_df(market_key, id_mercado, 'volumenes_i90')
            precio_id = self.config.get_precios_from_id_mercado(id_mercado)
            
            # Use different price dataset for restricciones
            price_dataset = 'precios_i90' if market_key == 'restricciones' else 'precios_esios'
            prices_df = self._get_latest_df(market_key, precio_id, price_dataset)
            
            latest_date = self._find_latest_common_date(volumes_df, prices_df)
            if latest_date is None:
                continue
            
            volumes_day = volumes_df[volumes_df['datetime_utc'].dt.date == latest_date]
            prices_day = prices_df[prices_df['datetime_utc'].dt.date == latest_date]
            
            ingresos_df = self._calculate_ingresos(volumes_day, prices_day, market_key, id_mercado)
            all_results.append(ingresos_df)
        if not all_results:
            return pd.DataFrame()
        combined = pd.concat(all_results, ignore_index=True)
        return combined

    def calculate_single(self, market_key, fecha):
        fecha_dt = pd.to_datetime(fecha)
        ids = self.config.mercado_name_id_map.get(market_key, [])
        all_results = []
        for id_mercado in ids:
            volumes_df = self._get_df_for_date_range(market_key, id_mercado, 'volumenes_i90', fecha, fecha)
            precio_id = self.config.get_precios_from_id_mercado(id_mercado, fecha_dt)
            print(f"Precio id: {precio_id}")
            breakpoint()
            
            # Use different price dataset for restricciones
            price_dataset = 'precios_i90' if market_key == 'restricciones' else 'precios_esios'
            prices_df = self._get_df_for_date_range(market_key, precio_id, price_dataset, fecha, fecha)
            
            ingresos_df = self._calculate_ingresos(volumes_df, prices_df, market_key, id_mercado)
            all_results.append(ingresos_df)
        if not all_results:
            return pd.DataFrame()
        return pd.concat(all_results, ignore_index=True)

    def calculate_multiple(self, market_key, fecha_inicio, fecha_fin):
        ids = self.config.mercado_name_id_map.get(market_key, [])
        all_results = []
        for id_mercado in ids:
            volumes_df = self._get_df_for_date_range(market_key, id_mercado, 'volumenes_i90', fecha_inicio, fecha_fin)

            # Handle prices by segmenting the date range if the precio_id can change
            prices_dfs = []
            start_dt = pd.to_datetime(fecha_inicio)
            end_dt = pd.to_datetime(fecha_fin)
            srs_date = self.config.dia_inicio_SRS

            # Define segments based on the SRS change date for relevant markets
            segments = []
            if id_mercado in [14, 15, 18, 19] and start_dt < srs_date <= end_dt:
                # Split into two segments around the SRS date
                segments.append((start_dt, srs_date - timedelta(days=1)))
                segments.append((srs_date, end_dt))
            else:
                # Process as a single segment
                segments.append((start_dt, end_dt))

            for seg_start, seg_end in segments:
                if seg_start > seg_end:
                    continue
                
                # Get the correct precio_id for the start of the current segment
                precio_id = self.config.get_precios_from_id_mercado(id_mercado, seg_start)
                price_dataset = 'precios_i90' if market_key == 'restricciones' else 'precios_esios'
                
                segment_prices_df = self._get_df_for_date_range(
                    market_key, 
                    precio_id, 
                    price_dataset, 
                    seg_start.strftime('%Y-%m-%d'), 
                    seg_end.strftime('%Y-%m-%d')
                )
                
                if not segment_prices_df.empty:
                    prices_dfs.append(segment_prices_df)

            prices_df = pd.concat(prices_dfs, ignore_index=True) if prices_dfs else pd.DataFrame()
            
            ingresos_df = self._calculate_ingresos(volumes_df, prices_df, market_key, id_mercado)
            all_results.append(ingresos_df)

        if not all_results:
            return pd.DataFrame()
        return pd.concat(all_results, ignore_index=True)

class ContinuoIngresosCalculator(IngresosCalculator):
    def calculate_latest(self, market_key):
        ids = self.config.mercado_name_id_map.get(market_key, [])
        all_results = []
        for id_mercado in ids:
            df = self._get_latest_df(market_key, id_mercado, 'volumenes_omie')  # Assuming volumenes_omie for continuo
            if df.empty:
                continue
            latest_date = df['datetime_utc'].dt.date.max()
            day_df = df[df['datetime_utc'].dt.date == latest_date]
            day_df['ingresos'] = day_df['volumenes'] * day_df['precio']
            result = day_df[['datetime_utc', 'uof', 'ingresos', 'id_mercado']].rename(columns={'uof': 'up'})
            all_results.append(result)
        if not all_results:
            return pd.DataFrame()
        combined = pd.concat(all_results, ignore_index=True)
        return combined

    def calculate_single(self, market_key, fecha):
        ids = self.config.mercado_name_id_map.get(market_key, [])
        all_results = []
        for id_mercado in ids:
            df = self._get_df_for_date_range(market_key, id_mercado, 'volumenes_omie', fecha, fecha)
            if df.empty:
                continue
            df['ingresos'] = df['volumenes'] * df['precio']
            result = df[['datetime_utc', 'uof', 'ingresos', 'id_mercado']].rename(columns={'uof': 'up'})
            all_results.append(result)
        if not all_results:
            return pd.DataFrame()
        return pd.concat(all_results, ignore_index=True)

    def calculate_multiple(self, market_key, fecha_inicio, fecha_fin):
        ids = self.config.mercado_name_id_map.get(market_key, [])
        all_results = []
        for id_mercado in ids:
            df = self._get_df_for_date_range(market_key, id_mercado, 'volumenes_omie', fecha_inicio, fecha_fin)
            if df.empty:
                continue
            df['ingresos'] = df['volumenes'] * df['precio']
            result = df[['datetime_utc', 'uof', 'ingresos', 'id_mercado']].rename(columns={'uof': 'up'})
            all_results.append(result)
        if not all_results:
            return pd.DataFrame()
        return pd.concat(all_results, ignore_index=True)

