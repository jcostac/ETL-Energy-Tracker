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
            return None
        return max(common_dates)

    def _calculate_ingresos(self, volumes_df, prices_df, market_key, id_mercado):
        latest_date = self._find_latest_common_date(volumes_df, prices_df)
        if latest_date is None:
            return pd.DataFrame()

        volumes_day = volumes_df[volumes_df['datetime_utc'].dt.date == latest_date]
        prices_day = prices_df[prices_df['datetime_utc'].dt.date == latest_date]

        merged = pd.merge(volumes_day, prices_day, on=['datetime_utc', 'id_mercado'], how='inner')
        merged['ingresos'] = merged['volumenes'] * merged['precio']
        result = merged[['datetime_utc', 'up', 'ingresos', 'id_mercado']]
        return result

    def _save_ingresos(self, df):
        if df.empty:
            return
        df['mercado'] = 'ingresos'
        self.file_utils.write_processed_parquet(df, 'ingresos', value_cols=['ingresos'], dataset_type='ingresos')

    def calculate_latest(self, market_key):
        ids = self.config.mercado_name_id_map.get(market_key, [])
        all_results = []
        for id_mercado in ids:
            volumes_df = self._get_latest_df(market_key, id_mercado, 'volumenes_i90')
            precio_id = self.config.get_precios_from_id_mercado(id_mercado, datetime.now())
            prices_df = self._get_latest_df(market_key, precio_id, 'precios_esios')
            ingresos_df = self._calculate_ingresos(volumes_df, prices_df, market_key, id_mercado)
            all_results.append(ingresos_df)
        combined = pd.concat(all_results, ignore_index=True)
        return combined

    def calculate_single(self, market_key, fecha):
        # Stub: Implement similar to latest but filter to specific date/month partition
        pass

    def calculate_multiple(self, market_key, fecha_inicio, fecha_fin):
        # Stub: Implement range reading and calculation
        pass

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
        combined = pd.concat(all_results, ignore_index=True)
        return combined

class RestriccionesIngresosCalculator(IngresosCalculator):
    def _calculate_ingresos(self, volumes_df, prices_df, market_key, id_mercado):
        latest_date = self._find_latest_common_date(volumes_df, prices_df)
        if latest_date is None:
            return pd.DataFrame()
        volumes_day = volumes_df[volumes_df['datetime_utc'].dt.date == latest_date]
        prices_day = prices_df[prices_df['datetime_utc'].dt.date == latest_date]
        merged = pd.merge(volumes_day, prices_day, on=['datetime_utc', 'id_mercado', 'up'], how='inner')
        merged['ingresos'] = merged['volumenes'] * merged['precio']
        result = merged[['datetime_utc', 'up', 'ingresos', 'id_mercado']]
        return result

    def calculate_latest(self, market_key):
        ids = self.config.mercado_name_id_map.get(market_key, [])
        all_results = []
        for id_mercado in ids:
            volumes_df = self._get_latest_df(market_key, id_mercado, 'volumenes_i90')
            precio_id = self.config.get_precios_from_id_mercado(id_mercado, datetime.now())
            prices_df = self._get_latest_df(market_key, precio_id, 'precios_i90')
            ingresos_df = self._calculate_ingresos(volumes_df, prices_df, market_key, id_mercado)
            all_results.append(ingresos_df)
        combined = pd.concat(all_results, ignore_index=True)
        return combined