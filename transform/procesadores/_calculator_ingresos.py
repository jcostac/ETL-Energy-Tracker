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
        print(f"    📂 Loading latest {dataset_type} data for mercado={mercado}, id_mercado={id_mercado}")
        path = self._find_latest_partition_path(mercado, id_mercado, dataset_type)
        if path:
            df = pd.read_parquet(path)
            print(f"    ✅ Successfully loaded {len(df):,} rows from {path.name}")
            print("Latest processed parquet:")
            print(df.head())
            print(df.tail())
            print(f"\n")
            return df
        print(f"    ❌ No data found for {dataset_type}")
        return pd.DataFrame()

    def _find_latest_common_date(self, volumes_df, prices_df):
        if volumes_df.empty or prices_df.empty:
            print("    ⚠️  One or both dataframes are empty, no common date found")
            return None
        volumes_dates = volumes_df['datetime_utc'].dt.date.unique()
        prices_dates = prices_df['datetime_utc'].dt.date.unique()
        common_dates = set(volumes_dates) & set(prices_dates)
        print(f"    📅 Found {len(common_dates)} common dates: {sorted(common_dates)}")
        if not common_dates:
            print(f"    ⚠️  Warning: No common dates found between volumes and prices.")
            return None
        latest_date = max(common_dates)
        print(f"    🎯 Using latest common date: {latest_date}")
        return latest_date

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
        print(f"    📊 Loading {dataset_type} data for mercado={mercado}, id_mercado={id_mercado}")
        print(f"        📅 Date range: {fecha_inicio} to {fecha_fin}")
        
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
            
            if not df.empty:
                print(f"    ✅ Successfully loaded {len(df):,} rows of {dataset_type} data")
                date_range = f"{df['datetime_utc'].dt.date.min()} to {df['datetime_utc'].dt.date.max()}"
                print(f"        📅 Actual data range: {date_range}")
            else:
                print(f"    ❌ No {dataset_type} data found for specified date range")
            
            return df
            
        except Exception as e:
            print(f"    ❌ Warning: Could not read data for {mercado} (id: {id_mercado}), {dataset_type}: {e}")
            return pd.DataFrame()

    def _analyze_total_ingresos(self, df: pd.DataFrame, context: str = "calculation") -> pd.DataFrame:
        """
        Group ingresos by UP/UOF and display top 3 highest earners.
        
        Args:
            df (pd.DataFrame): DataFrame with 'up' and 'ingresos' columns
            context (str): Context description for the analysis
            
        Returns:
            pd.DataFrame: Grouped data sorted by ingresos descending
        """
        if df.empty or 'up' not in df.columns or 'ingresos' not in df.columns:
            print(f"    ⚠️  Cannot analyze ingresos by UP - missing data or columns")
            return pd.DataFrame()
        
        # Group by UP and sum ingresos
        grouped = df.groupby('up')['ingresos'].agg(['sum', 'count']).reset_index()
        grouped.columns = ['up', 'total_ingresos', 'records_count']
        grouped = grouped.sort_values('total_ingresos', ascending=False)
        
        total_ups = len(grouped)
        total_ingresos = grouped['total_ingresos'].sum()
        
        print(f"    📈 INGRESOS ANALYSIS ({context}):")
        print(f"        💰 Total ingresos: {total_ingresos:,.2f} €")
        print(f"        🏭 Total UPs: {total_ups}")
        
        if total_ups > 0:
            print(f"    🏆 TOP 3 HIGHEST EARNING UPs:")
            top_3 = grouped.head(3)
            for i, (_, row) in enumerate(top_3.iterrows(), 1):
                percentage = (row['total_ingresos'] / total_ingresos) * 100
                print(f"        {i}. UP {row['up']}: {row['total_ingresos']:,.2f} € ({percentage:.1f}%) - {row['records_count']:,} records")
            
            if total_ups > 3:
                remaining_ingresos = grouped.iloc[3:]['total_ingresos'].sum()
                remaining_percentage = (remaining_ingresos / total_ingresos) * 100
                print(f"        ... {total_ups - 3} other UPs: {remaining_ingresos:,.2f} € ({remaining_percentage:.1f}%)")
        
        return grouped

    def _calculate_ingresos(self, volumes_df, prices_df, market_key, id_mercado):
        print(f"    🧮 Starting ingresos calculation for market '{market_key}' (id: {id_mercado})")
        
        if volumes_df.empty and prices_df.empty:
            print(f"    ⚠️  Both volumes and prices dataframes are empty")
            return pd.DataFrame()

        if volumes_df.empty:
            print(f"    ❌ No volume data available")
            raise ValueError(f"No volume data for market '{market_key}' (id: {id_mercado}) for the given period. No ingresos calculated.")
        
        if prices_df.empty:
            print(f"    ❌ No price data available")
            raise ValueError(f"No price data for market '{market_key}' (id: {id_mercado}) for the given period. No ingresos calculated.")

        print(f"    📊 Input data: {len(volumes_df):,} volume rows, {len(prices_df):,} price rows")
        
        # 🔧 FIX: Check if volume and price data have different id_mercado values
        vol_ids = set(volumes_df['id_mercado'].unique())
        price_ids = set(prices_df['id_mercado'].unique())
        ids_match = bool(vol_ids & price_ids)  # Check if there's any overlap
        
        if not ids_match:
            print(f"    🔄 Different id_mercado values detected - merging on datetime_utc only")
            print(f"        Volume IDs: {sorted(vol_ids)}")
            print(f"        Price IDs: {sorted(price_ids)}")
        
        if market_key == 'restricciones':
            # For restricciones, always merge on datetime_utc, id_mercado, up
            merged = pd.merge(volumes_df, prices_df, on=['datetime_utc', 'id_mercado', 'up'], how='inner')
            print(f"    🔗 Merged on: datetime_utc, id_mercado, up")
        else:
            if ids_match:
                # Standard merge when IDs match
                merged = pd.merge(volumes_df, prices_df, on=['datetime_utc', 'id_mercado'], how='inner')
                print(f"    🔗 Merged on: datetime_utc, id_mercado")
            else:
                # Merge only on datetime_utc when IDs don't match (e.g., secundaria before SRS)
                merged = pd.merge(volumes_df, prices_df, on=['datetime_utc'], how='inner')
                print(f"    🔗 Merged on: datetime_utc only (different id_mercado values)")
                # Keep the volume data's id_mercado for the result
                merged['id_mercado'] = merged['id_mercado_x']
                merged = merged.drop(columns=['id_mercado_x', 'id_mercado_y'])
        
        if merged.empty:
            print(f"    ❌ Merge resulted in empty dataframe (no common timestamps)")
            raise ValueError(f"For market '{market_key}' (id: {id_mercado}), data for volumes and prices could not be merged for the given period (no common timestamps). No ingresos calculated.")
            
        print(f"    ✅ Merge successful: {len(merged):,} rows")
        merged['ingresos'] = (merged['volumenes'] * merged['precio']).round(2)
        
        result = merged[['datetime_utc', 'up', 'ingresos', 'id_mercado']]
        
        # Analyze and print top ingresos by UP
        self._analyze_total_ingresos(result, f"market {market_key} (id: {id_mercado})")
        
        print(f"    ✅ Calculation complete: {len(result):,} ingresos records generated")
        return result

    def calculate_latest(self, market_key, date):
        print(f"\n🚀 STARTING LATEST INGRESOS CALCULATION")
        print(f"    Market: {market_key}")
        print(f"    Date: {date}")
        
        ids = self.config.get_market_ids(market_key, date)
        print(f"    Processing {len(ids)} market IDs: {ids}")
        
        all_results = []
        for i, id_mercado in enumerate(ids, 1):
            print(f"\n📍 Processing market ID {id_mercado} ({i}/{len(ids)})")
            
            volumes_df = self._get_latest_df(market_key, id_mercado, 'volumenes_i90')
            precio_id = self.config.get_precios_from_id_mercado(id_mercado)
            print(f"    💰 Using precio_id: {precio_id}")
            
            # Use different price dataset for restricciones
            price_dataset = 'precios_i90' if market_key == 'restricciones' else 'precios_esios'
            print(f"    📊 Using price dataset: {price_dataset}")
            prices_df = self._get_latest_df(market_key, precio_id, price_dataset)
            
            latest_date = self._find_latest_common_date(volumes_df, prices_df)
            if latest_date is None:
                print(f"    ⏭️  Skipping market ID {id_mercado} - no common dates")
                continue
            
            volumes_day = volumes_df[volumes_df['datetime_utc'].dt.date == latest_date]
            prices_day = prices_df[prices_df['datetime_utc'].dt.date == latest_date]
            print(f"    📅 Filtered to {latest_date}: {len(volumes_day):,} volume rows, {len(prices_day):,} price rows")
            
            try:
                ingresos_df = self._calculate_ingresos(volumes_day, prices_day, market_key, id_mercado)
                all_results.append(ingresos_df)
                print(f"    ✅ Market ID {id_mercado} completed successfully")
            except Exception as e:
                print(f"    ❌ Market ID {id_mercado} failed: {e}")
                
        if not all_results:
            print(f"\n❌ CALCULATION FAILED: No results generated for any market ID")
            return pd.DataFrame()
            
        combined = pd.concat(all_results, ignore_index=True)
        total_records = len(combined)
        
        # Final analysis of all combined results
        print(f"\n🎯 FINAL COMBINED RESULTS:")
        print(f"    📊 Total records: {total_records:,}")
        print(f"    🏢 Markets processed: {len(all_results)}/{len(ids)}")
        self._analyze_total_ingresos(combined, "FINAL COMBINED")
        
        print(f"\n🎉 CALCULATION COMPLETED SUCCESSFULLY!")
        return combined

    def calculate_single(self, market_key, fecha):
        print(f"\n🚀 STARTING SINGLE DATE INGRESOS CALCULATION")
        print(f"    Market: {market_key}")
        print(f"    Date: {fecha}")
        
        fecha_dt = pd.to_datetime(fecha)
        ids = self.config.get_market_ids(market_key, fecha_dt)
        print(f"    Processing {len(ids)} market IDs: {ids}")
        
        all_results = []
        for i, id_mercado in enumerate(ids, 1):
            print(f"\n📍 Processing market ID {id_mercado} ({i}/{len(ids)})")
            
            volumes_df = self._get_df_for_date_range(market_key, id_mercado, 'volumenes_i90', fecha, fecha)
            precio_id = self.config.get_precios_from_id_mercado(id_mercado, fecha_dt)
            print(f"    💰 Using precio_id: {precio_id}")
            
            # Use different price dataset for restricciones
            price_dataset = 'precios_i90' if market_key == 'restricciones' else 'precios_esios'
            print(f"    📊 Using price dataset: {price_dataset}")
            prices_df = self._get_df_for_date_range(market_key, precio_id, price_dataset, fecha, fecha)
            
            try:
                ingresos_df = self._calculate_ingresos(volumes_df, prices_df, market_key, id_mercado)
                all_results.append(ingresos_df)
                print(f"    ✅ Market ID {id_mercado} completed successfully")
            except Exception as e:
                print(f"    ❌ Market ID {id_mercado} failed: {e}")
                
        if not all_results:
            print(f"\n❌ CALCULATION FAILED: No results generated for any market ID")
            return pd.DataFrame()
            
        combined = pd.concat(all_results, ignore_index=True)
        total_records = len(combined)
        
        # Final analysis of all combined results
        print(f"\n🎯 FINAL COMBINED RESULTS:")
        print(f"    📊 Total records: {total_records:,}")
        print(f"    🏢 Markets processed: {len(all_results)}/{len(ids)}")
        self._analyze_total_ingresos(combined, "FINAL COMBINED")
        
        print(f"\n🎉 CALCULATION COMPLETED SUCCESSFULLY!")
        return combined

    def calculate_multiple(self, market_key, fecha_inicio, fecha_fin):
        print(f"\n🚀 STARTING MULTIPLE DATE INGRESOS CALCULATION")
        print(f"    Market: {market_key}")
        print(f"    Date range: {fecha_inicio} to {fecha_fin}")
        
        start_dt = pd.to_datetime(fecha_inicio)
        end_dt = pd.to_datetime(fecha_fin)
        
        # Handle intra market reduction date splitting
        if market_key == 'intra' and start_dt < self.config.intra_reduction_date <= end_dt:
            print(f"    📅 Intra reduction date ({self.config.intra_reduction_date.date()}) falls within range - splitting processing")
            
            all_results = []
            
            # Period 1: Before intra reduction (use all 7 markets)
            period1_end = self.config.intra_reduction_date - timedelta(days=1)
            print(f"    📊 Period 1: {start_dt.date()} to {period1_end.date()} (7 intra markets)")
            ids_period1 = [2, 3, 4, 5, 6, 7, 8]
            
            results_period1 = self._process_market_ids_for_period(
                market_key, ids_period1, start_dt.strftime('%Y-%m-%d'), period1_end.strftime('%Y-%m-%d')
            )
            all_results.extend(results_period1)
            
            # Period 2: After intra reduction (use only 3 markets)
            print(f"    📊 Period 2: {self.config.intra_reduction_date.date()} to {end_dt.date()} (3 intra markets)")
            ids_period2 = [2, 3, 4]
            
            results_period2 = self._process_market_ids_for_period(
                market_key, ids_period2, self.config.intra_reduction_date.strftime('%Y-%m-%d'), end_dt.strftime('%Y-%m-%d')
            )
            all_results.extend(results_period2)
            
        else:
            # Standard processing (no date splitting needed)
            ids = self.config.get_market_ids(market_key, start_dt)
            print(f"    Processing {len(ids)} market IDs: {ids}")
            
            all_results = self._process_market_ids_for_period(market_key, ids, fecha_inicio, fecha_fin)
        
        # Combine all results
        combined = pd.concat(all_results, ignore_index=True)
        total_records = len(combined)
        date_range = f"{combined['datetime_utc'].dt.date.min()} to {combined['datetime_utc'].dt.date.max()}"
        
        # Final analysis of all combined results
        print(f"\n🎯 FINAL COMBINED RESULTS:")
        print(f"    📊 Total records: {total_records:,}")
        print(f"    📅 Date range covered: {date_range}")
        print(f"    🏢 Markets processed: {len(all_results)}/{len(ids)}") # This line needs to be updated based on how ids is determined
        self._analyze_total_ingresos(combined, "FINAL COMBINED")
        
        print(f"\n🎉 CALCULATION COMPLETED SUCCESSFULLY!")
        return combined

    def _process_market_ids_for_period(self, market_key, ids, fecha_inicio, fecha_fin):
        """Process a specific set of market IDs for a given date range"""
        results = []
        
        for i, id_mercado in enumerate(ids, 1):
            print(f"\n📍 Processing market ID {id_mercado} ({i}/{len(ids)})")
            
            volumes_df = self._get_df_for_date_range(market_key, id_mercado, 'volumenes_i90', fecha_inicio, fecha_fin)

            # Handle prices by segmenting the date range if the precio_id can change
            print(f"    🔄 Setting up price data segments...")
            prices_dfs = []
            start_dt = pd.to_datetime(fecha_inicio)
            end_dt = pd.to_datetime(fecha_fin)
            srs_date = self.config.dia_inicio_SRS

            # Define segments based on the SRS change date for relevant markets
            segments = []
            if id_mercado in [14, 15, 18, 19] and start_dt < srs_date <= end_dt:
                # Split into two segments around the SRS date
                print(f"    📅 SRS date ({srs_date}) falls within range - splitting into segments")
                segments.append((start_dt, srs_date - timedelta(days=1)))
                segments.append((srs_date, end_dt))
                print(f"        Segment 1: {segments[0][0].date()} to {segments[0][1].date()}")
                print(f"        Segment 2: {segments[1][0].date()} to {segments[1][1].date()}")
            else:
                # Process as a single segment
                print(f"    📅 Processing as single segment")
                segments.append((start_dt, end_dt))

            for j, (seg_start, seg_end) in enumerate(segments, 1):
                if seg_start > seg_end:
                    print(f"    ⏭️  Skipping invalid segment {j}: start > end")
                    continue
                
                print(f"    📊 Processing price segment {j}/{len(segments)}: {seg_start.date()} to {seg_end.date()}")
                
                # Get the correct precio_id for the start of the current segment
                precio_id = self.config.get_precios_from_id_mercado(id_mercado, seg_start)
                price_dataset = 'precios_i90' if market_key == 'restricciones' else 'precios_esios'
                print(f"        💰 Using precio_id: {precio_id}, dataset: {price_dataset}")
                
                segment_prices_df = self._get_df_for_date_range(
                    market_key, 
                    precio_id, 
                    price_dataset, 
                    seg_start.strftime('%Y-%m-%d'), 
                    seg_end.strftime('%Y-%m-%d')
                )
                
                if not segment_prices_df.empty:
                    prices_dfs.append(segment_prices_df)
                    print(f"        ✅ Segment {j} added: {len(segment_prices_df):,} price rows")
                else:
                    print(f"        ❌ Segment {j} empty")

            if prices_dfs:
                prices_df = pd.concat(prices_dfs, ignore_index=True)
                print(f"    🔗 Combined price segments: {len(prices_df):,} total rows")
            else:
                prices_df = pd.DataFrame()
                print(f"    ❌ No price data from any segment")
            
            try:
                ingresos_df = self._calculate_ingresos(volumes_df, prices_df, market_key, id_mercado)
                results.append(ingresos_df)
                print(f"    ✅ Market ID {id_mercado} completed successfully")
            except Exception as e:
                print(f"    ❌ Market ID {id_mercado} failed: {e}")
        
        return results

class ContinuoIngresosCalculator(IngresosCalculator):
    def calculate_latest(self, market_key):
        print(f"\n🚀 STARTING CONTINUO LATEST INGRESOS CALCULATION")
        print(f"    Market: {market_key}")
        
        ids = self.config.mercado_name_id_map.get(market_key, [])
        print(f"    Processing {len(ids)} market IDs: {ids}")
        
        all_results = []
        for i, id_mercado in enumerate(ids, 1):
            print(f"\n📍 Processing market ID {id_mercado} ({i}/{len(ids)})")
            
            df = self._get_latest_df(market_key, id_mercado, 'volumenes_omie')  # Assuming volumenes_omie for continuo
            if df.empty:
                print(f"    ⏭️  Skipping market ID {id_mercado} - no data")
                continue
                
            latest_date = df['datetime_utc'].dt.date.max()
            print(f"    📅 Using latest date: {latest_date}")
            day_df = df[df['datetime_utc'].dt.date == latest_date]
            print(f"    📊 Filtered to {len(day_df):,} rows for {latest_date}")
            
            day_df['ingresos'] = day_df['volumenes'] * day_df['precio']
            
            result = day_df[['datetime_utc', 'uof', 'ingresos', 'id_mercado']].rename(columns={'uof': 'up'})
            
            # Analyze ingresos for this market
            self._analyze_total_ingresos(result, f"market {market_key} (id: {id_mercado})")
            
            all_results.append(result)
            print(f"    ✅ Market ID {id_mercado} completed: {len(result):,} records")
            
        if not all_results:
            print(f"\n❌ CALCULATION FAILED: No results generated for any market ID")
            return pd.DataFrame()
            
        combined = pd.concat(all_results, ignore_index=True)
        total_records = len(combined)
        
        # Final analysis of all combined results
        print(f"\n🎯 FINAL COMBINED RESULTS:")
        print(f"    📊 Total records: {total_records:,}")
        print(f"    🏢 Markets processed: {len(all_results)}/{len(ids)}")
        self._analyze_total_ingresos(combined, "FINAL COMBINED")
        
        print(f"\n🎉 CALCULATION COMPLETED SUCCESSFULLY!")
        return combined

    def calculate_single(self, market_key, fecha):
        print(f"\n🚀 STARTING CONTINUO SINGLE DATE INGRESOS CALCULATION")
        print(f"    Market: {market_key}")
        print(f"    Date: {fecha}")
        
        ids = self.config.mercado_name_id_map.get(market_key, [])
        print(f"    Processing {len(ids)} market IDs: {ids}")
        
        all_results = []
        for i, id_mercado in enumerate(ids, 1):
            print(f"\n📍 Processing market ID {id_mercado} ({i}/{len(ids)})")
            
            df = self._get_df_for_date_range(market_key, id_mercado, 'volumenes_omie', fecha, fecha)
            if df.empty:
                print(f"    ⏭️  Skipping market ID {id_mercado} - no data")
                continue
                
            print(f"    📊 Processing {len(df):,} rows for {fecha}")
            df['ingresos'] = df['volumenes'] * df['precio']
            
            result = df[['datetime_utc', 'uof', 'ingresos', 'id_mercado']].rename(columns={'uof': 'up'})
            
            # Analyze ingresos for this market
            self._analyze_total_ingresos(result, f"market {market_key} (id: {id_mercado})")
            
            all_results.append(result)
            print(f"    ✅ Market ID {id_mercado} completed: {len(result):,} records")
            
        if not all_results:
            print(f"\n❌ CALCULATION FAILED: No results generated for any market ID")
            return pd.DataFrame()
            
        combined = pd.concat(all_results, ignore_index=True)
        total_records = len(combined)
        
        # Final analysis of all combined results
        print(f"\n🎯 FINAL COMBINED RESULTS:")
        print(f"    📊 Total records: {total_records:,}")
        print(f"    🏢 Markets processed: {len(all_results)}/{len(ids)}")
        self._analyze_total_ingresos(combined, "FINAL COMBINED")
        
        print(f"\n🎉 CALCULATION COMPLETED SUCCESSFULLY!")
        return combined

