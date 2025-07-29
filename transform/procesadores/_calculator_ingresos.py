import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import os
from utilidades.processed_file_utils import ProcessedFileUtils
from utilidades.data_validation_utils import DataValidationUtils
from configs.ingresos_config import IngresosConfig
from configs.storage_config import VALID_DATASET_TYPES

class IngresosCalculator:
    def __init__(self):
        self.file_utils = ProcessedFileUtils()
        self.config = IngresosConfig()

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
            return pd.read_parquet(path)
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
        start_dt = pd.to_datetime(fecha_inicio).date()
        end_dt = pd.to_datetime(fecha_fin).date()
        
        all_days_in_range = pd.date_range(start_dt, end_dt, freq='D')
        if all_days_in_range.empty:
            return pd.DataFrame()
            
        year_months = sorted(list(set([(d.year, d.month) for d in all_days_in_range])))
        
        all_dfs = []
        for year, month in year_months:
            path = self.file_utils.processed_path / f"mercado={mercado}" / f"id_mercado={id_mercado}" / f"year={year}" / f"month={month}" / f"{dataset_type}.parquet"
            if path.exists():
                try:
                    df = pd.read_parquet(path)
                    all_dfs.append(df)
                except Exception as e:
                    print(f"Warning: Could not read parquet file {path}: {e}")

        if not all_dfs:
            return pd.DataFrame()
        
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        if 'datetime_utc' not in combined_df.columns:
            return pd.DataFrame()
            
        return combined_df[(combined_df['datetime_utc'].dt.date >= start_dt) & (combined_df['datetime_utc'].dt.date <= end_dt)].copy()

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

    def calculate_latest(self, market_key):
        ids = self.config.mercado_name_id_map.get(market_key, [])
        all_results = []
        for id_mercado in ids:
            volumes_df = self._get_latest_df(market_key, id_mercado, 'volumenes_i90')
            precio_id = self.config.get_precios_from_id_mercado(id_mercado, datetime.now())
            
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

