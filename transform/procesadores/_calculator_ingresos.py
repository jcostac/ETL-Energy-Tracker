import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utilidades.processed_file_utils import ProcessedFileUtils
from configs.ingresos_config import IngresosConfig
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
        #print(f"    📂 Loading latest {dataset_type} data for mercado={mercado}, id_mercado={id_mercado}")
        path = self._find_latest_partition_path(mercado, id_mercado, dataset_type)
        if path:
            df = pd.read_parquet(path)
            #print(f"    ✅ Successfully loaded {len(df):,} rows from {path.name}")
            #print("Latest processed parquet:")
            #print(df.head())
            #print(df.tail())
            #print(f"\n")
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
        #print(f"    📅 Found {len(common_dates)} common dates: {sorted(common_dates)}")
        if not common_dates:
            print(f"    ⚠️  Warning: No common dates found between volumes and prices.")
            return None
        latest_date = max(common_dates)
        #print(f"    🎯 Using latest common date: {latest_date}")
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
        print("================================================")
        print(f"📊 Loading {dataset_type} data for mercado={mercado}, id_mercado={id_mercado}")
        #print(f"        📅 Date range: {fecha_inicio} to {fecha_fin}")
        
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
                print(f"✅ Successfully loaded {len(df):,} rows of {dataset_type} data")
                print("================================================")
                date_range = f"{df['datetime_utc'].dt.date.min()} to {df['datetime_utc'].dt.date.max()}"
                #print(f"📅 Actual data range: {date_range}")
            else:
                print(f"    ❌ No {dataset_type} data found for specified date range")
            
            return df
            
        except Exception as e:
            print(f"    ❌ Warning: Could not read data for {mercado} (id: {id_mercado}), {dataset_type}: {e}")
            return pd.DataFrame()

    def _analyze_total_ingresos(self, df: pd.DataFrame, title_context: str = "", plot=False) -> pd.DataFrame:
        """
        Group ingresos by UP/UOF and display top 3 highest earners.
        
        Args:
            df (pd.DataFrame): DataFrame with 'up' and 'ingresos' columns
            title_context (str): Optional context for the plot title.
            plot (bool): Whether to generate a plot.
            
        Returns:
            pd.DataFrame: Grouped data sorted by ingresos descending
        """
        if df.empty or 'up' not in df.columns or 'ingresos' not in df.columns:
            print(f"    ⚠️  Cannot analyze ingresos by UP - missing data or columns")
            return pd.DataFrame()
        
        # Group by UP and sum ingresos
        grouped = df.groupby(['up', 'id_mercado'])['ingresos'].agg(['sum', 'count']).reset_index()
        grouped.columns = ['up', 'id_mercado', 'total_ingresos', 'records_count']
        total_ingresos_abs = grouped['total_ingresos'].abs().sum()
        grouped["% of volume"] = grouped['total_ingresos'].abs() / total_ingresos_abs * 100
        grouped['total_ingresos'] = grouped['total_ingresos'].round(2)
        grouped['% of volume'] = grouped['% of volume'].round(2)
        grouped = grouped.sort_values('total_ingresos', ascending=False).reset_index(drop=True)
        
        total_ups = len(grouped)

        if plot == True:
            # Plot overall distribution
            plot_title = f"Distribución de Ingresos Totales por UP"
            if title_context:
                plot_title += f" ({title_context})"
            self._plot_histogram(grouped['total_ingresos'], title=plot_title, xlabel="Ingresos Totales por UP (€)")
            
            # Plot per id_mercado
            for id_mercado in grouped['id_mercado'].unique():
                mercado_data = grouped[grouped['id_mercado'] == id_mercado]['total_ingresos']
                if not mercado_data.empty:
                    plot_title = f"Distribución de Ingresos por UP - Mercado ID {id_mercado}"
                    if title_context:
                        plot_title += f" ({title_context})"
                    self._plot_histogram(mercado_data, title=plot_title, xlabel=f"Ingresos Totales por UP - Mercado {id_mercado} (€)")
        
        print(f"--------------------------------")
        print(f"📈 TOTAL INGRESOS ANALYSIS:")
        if title_context:
            print(f"   Context: {title_context}")
        print(f"💰 Total volume: {total_ingresos_abs:,.2f} €")
        print(f"🏭 Total UPs: {total_ups}")
        print(f"Top Earners:")
        print(grouped.head())
        print(f"Bottom Earners:")
        print(grouped.tail())
        print(f"--------------------------------")
         
        return grouped

    def _plot_histogram(self, data_series: pd.Series, title: str, xlabel: str):
        """
        Plots a histogram for a given data series, trimming outliers for better visualization.

        Args:
            data_series (pd.Series): Series of data to plot.
            title (str): Title for the plot.
            xlabel (str): Label for the x-axis.
        """
        if data_series.empty:
            print(f"    ⚠️  Cannot plot histogram for '{title}' - empty data series")
            return

        data = data_series.dropna()
        if len(data) < 2:
            print(f"    ⚠️  Not enough data to plot histogram for '{title}'")
            return

        # Trim outliers for better visualization
        lower_percentile = data.quantile(0.01)
        upper_percentile = data.quantile(0.99)
        
        # Only filter if the percentiles are different to avoid removing all data
        if lower_percentile < upper_percentile:
            filtered_data = data[(data >= lower_percentile) & (data <= upper_percentile)]
        else:
            filtered_data = data # Keep original data if percentiles are the same

        if len(filtered_data) < 2:
            print(f"    ⚠️  Not enough data to plot histogram for '{title}' after filtering outliers")
            return
            
        # Recalculate bins based on the filtered data
        q1 = filtered_data.quantile(0.25)
        q3 = filtered_data.quantile(0.75)
        iqr = q3 - q1
        
        if iqr > 0:
            bin_width = 2 * iqr / (len(filtered_data)**(1/3))
            if bin_width > 0:
                num_bins = int(np.ceil((filtered_data.max() - filtered_data.min()) / bin_width))
                num_bins = max(10, min(num_bins, 100))  # Cap bins to a reasonable range
            else:
                num_bins = 30
        else:
            num_bins = 30

        # Calculate stats for legend
        min_val = filtered_data.min()
        max_val = filtered_data.max()
        avg_val = filtered_data.mean()
        
        stats_label = f'Min: {min_val:.2f}\nMax: {max_val:.2f}\nAvg: {avg_val:.2f}'

        plt.figure(figsize=(12, 7))
        plt.hist(filtered_data, bins=num_bins, color='skyblue', edgecolor='black', label=stats_label)
        plt.title(f"{title}")
        plt.xlabel(xlabel)
        plt.ylabel("Frecuencia")
        plt.grid(axis='y', alpha=0.75)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def _calculate_ingresos(self, volumes_df, prices_df, market_key, id_mercado, plot=False):
        print(f"🧮 Starting ingresos calculation for market '{market_key}' - id: {id_mercado}")

        if market_key == "diario":
            if "tipo_transaccion" in volumes_df.columns:
                #print(f"🔄 Filtering volumes_df for diari (tipo_transaccion == Mercado)")
                volumes_df = volumes_df[volumes_df["tipo_transaccion"] == "Mercado"]
            else:
                #print(f"    ⚠️  No tipo_transaccion column found in volumes_df for diario (expected)")
                raise ValueError(f"No tipo_transaccion column found in volumes_df for diario (expected)")
        
        if volumes_df.empty and prices_df.empty:
            print(f"    ⚠️  Both volumes and prices dataframes are empty")
            return pd.DataFrame()

        if volumes_df.empty:
            print(f"    ❌ No volume data available")
            raise ValueError(f"No volume data for market '{market_key}' (id: {id_mercado}) for the given period. No ingresos calculated.")
        
        if prices_df.empty:
            print(f"    ❌ No price data available")
            raise ValueError(f"No price data for market '{market_key}' (id: {id_mercado}) for the given period. No ingresos calculated.")

        print(f"📊 Input data: {len(volumes_df):,} volume rows, {len(prices_df):,} price rows")
        
        # 🔧 FIX: Check if volume and price data have different id_mercado values
        vol_ids = set(volumes_df['id_mercado'].unique())
        price_ids = set(prices_df['id_mercado'].unique())
        ids_match = bool(vol_ids & price_ids)  # Check if there's any overlap
        
        if not ids_match:
            print(f"    🔄 Different id_mercado values detected - merging on datetime_utc only")
            print(f"        Volume IDs: {sorted(vol_ids)}")
            print(f"        Price IDs: {sorted(price_ids)}")
        
        if market_key == 'restricciones_md' or market_key == 'restricciones_tr' or market_key == 'desvios':
            # For restricciones, always merge on datetime_utc, id_mercado, up
            merged = pd.merge(volumes_df, prices_df, on=['datetime_utc', 'id_mercado', 'up', 'redespacho'], how='inner')
            #print(f"    🔗 Merged on: datetime_utc, id_mercado, up, redespacho")
        else:
            if ids_match:
                # Standard merge when IDs match
                merged = pd.merge(volumes_df, prices_df, on=['datetime_utc', 'id_mercado'], how='inner')
                #print(f"    🔗 Merged on: datetime_utc, id_mercado")
            else:
                # Merge only on datetime_utc when IDs don't match (e.g., secundaria before SRS ie single price for two diff markets)
                merged = pd.merge(volumes_df, prices_df, on=['datetime_utc'], how='inner')
                #print(f"    🔗 Merged on: datetime_utc only (different id_mercado values)")
                # Keep the volume data's id_mercado for the result
                merged['id_mercado'] = merged['id_mercado_x']
                merged = merged.drop(columns=['id_mercado_x', 'id_mercado_y'])
        
        if merged.empty:
            print(f"    ❌ Merge resulted in empty dataframe (no common timestamps)")
            raise ValueError(f"For market '{market_key}' (id: {id_mercado}), data for volumes and prices could not be merged for the given period (no common timestamps). No ingresos calculated.")
            
        merged['ingresos'] = (merged['volumenes'] * merged['precio']).round(2)
        
        result = merged[['datetime_utc', 'up', 'ingresos', 'id_mercado']]

        result_grouped = result.groupby(['datetime_utc', 'id_mercado', "up"]).agg({'ingresos': 'sum'}).reset_index()
        

        market_context = f"para {market_key} (id: {id_mercado})"

        if plot == True:
            self._plot_histogram(merged['ingresos'], title=f"Distribución de Ingresos Horarios {market_context}", xlabel="Ingresos (€)")
            self._plot_histogram(merged['volumenes'], title=f"Distribución de Volúmenes Horarios {market_context}", xlabel="Volumen (MWh)")
            self._plot_histogram(merged['precio'], title=f"Distribución de Precios Horarios {market_context}", xlabel="Precio (€/MWh)")
        
        #print(f"    ✅ Calculation complete: {len(result):,} ingresos records generated")
        return result_grouped

    # ============================================================
    # INGRESOS POR DIFERENCIAS
    # ============================================================
    def _calculate_ingresos_diferencias(self, volumes_df, prices_df, spot_df, market_key, id_mercado, plot=False):
        """Calculate ingresos = volumen * (precio – precio_spot)."""
        if volumes_df.empty or prices_df.empty or spot_df.empty:
            print("    ⚠️  Missing data – cannot compute ingresos_diferencias")
            return pd.DataFrame()

        # Merge volume with price
        merged = pd.merge(volumes_df, prices_df, on=['datetime_utc'], how='inner', suffixes=('', '_precio'))
        # Attach spot price
        spot_df = spot_df[['datetime_utc', 'precio']].rename(columns={'precio': 'precio_spot'})
        merged = pd.merge(merged, spot_df, on='datetime_utc', how='inner')

        if merged.empty:
            print("    ⚠️  Merge produced no rows for diferencias")
            return pd.DataFrame()

        merged['ingresos'] = (merged['volumenes'] * (merged['precio'] - merged['precio_spot'])).round(2)

        id_label = 'up' if 'up' in merged.columns else ('uof' if 'uof' in merged.columns else None)
        cols = ['datetime_utc', 'id_mercado', 'ingresos'] + ([id_label] if id_label else [])
        result = merged[cols]
        return result

    def calculate_diferencias_single(self, market_key, fecha, plot=False):
        """Public entry: single-day ingresos por diferencias for standard markets."""
        spot_df = self._get_df_for_date_range('diario', 1, 'precios_esios', fecha, fecha)
        fecha_dt = pd.to_datetime(fecha)
        ids = self.config.get_market_ids(market_key, fecha_dt)

        all_results = []
        for id_mercado in ids:
            volumes_df = self._get_df_for_date_range(market_key, id_mercado, 'volumenes_i90', fecha, fecha)
            precio_id = self.config.get_precios_from_id_mercado(id_mercado, fecha_dt)
            price_dataset = 'precios_i90' if market_key in ['restricciones_md', 'restricciones_tr'] else 'precios_esios'
            prices_df = self._get_df_for_date_range(market_key, precio_id, price_dataset, fecha, fecha)

            diff_df = self._calculate_ingresos_diferencias(volumes_df, prices_df, spot_df, market_key, id_mercado, plot)
            if not diff_df.empty:
                all_results.append(diff_df)

        return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()

    def _process_market_ids_for_period(self, market_key, ids, fecha_inicio, fecha_fin, plot=False):
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
                price_dataset = 'precios_i90' if market_key in ['restricciones_md', 'restricciones_tr'] else 'precios_esios'
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
                ingresos_df = self._calculate_ingresos(volumes_df, prices_df, market_key, id_mercado, plot)
                results.append(ingresos_df)
                print(f"    ✅ Market ID {id_mercado} completed successfully")
            except Exception as e:
                print(f"    ❌ Market ID {id_mercado} failed: {e}")
        
        return results

    def calculate_single(self, market_key, fecha, plot=False):
        print(f"\n🚀 STARTING SINGLE DATE INGRESOS CALCULATION")
        print(f"    Market: {market_key}")
        print(f"    Date: {fecha}")
        
        fecha_dt = pd.to_datetime(fecha)
        ids = self.config.get_market_ids(market_key, fecha_dt)

        all_results = []
        for i, id_mercado in enumerate(ids, 1):
            print(f"\n📍 Processing market ID {id_mercado} ({i}/{len(ids)})")
            
            volumes_df = self._get_df_for_date_range(market_key, id_mercado, 'volumenes_i90', fecha, fecha)
            precio_id = self.config.get_precios_from_id_mercado(id_mercado, fecha_dt)
            print(f"mercado: {market_key} id: {id_mercado} precio_id: {precio_id}")
            
            # Use different price dataset for restricciones
            price_dataset = 'precios_i90' if market_key in ['restricciones_md', 'restricciones_tr'] else 'precios_esios'
            print(f"    📊 Using price dataset: {price_dataset}")
            prices_df = self._get_df_for_date_range(market_key, precio_id, price_dataset, fecha, fecha)
            
            try:
                ingresos_df = self._calculate_ingresos(volumes_df, prices_df, market_key, id_mercado, plot)
                all_results.append(ingresos_df)
                print(f"    ✅ Market ID {id_mercado} completed successfully")
            except Exception as e:
                print(f"    ❌ Market ID {id_mercado} failed: {e}")
                
        if not all_results:
            print(f"\n❌ CALCULATION FAILED: No results generated for any market ID")
            return pd.DataFrame()
            
        combined = pd.concat(all_results, ignore_index=True)
        
        self._analyze_total_ingresos(combined, plot)
        
        print(f"\n🎉 CALCULATION COMPLETED SUCCESSFULLY!")
        return combined

    def calculate_multiple(self, market_key, fecha_inicio, fecha_fin, plot=False):
        print(f"\n🚀 STARTING MULTIPLE DATE INGRESOS CALCULATION")
        print(f"    Market: {market_key}")
        print(f"    Date range: {fecha_inicio} to {fecha_fin}")
        
        #convert to datetime for boolean comparison
        start_dt = pd.to_datetime(fecha_inicio)
        end_dt = pd.to_datetime(fecha_fin)
        
        # Handle intra market reduction date splitting
        if market_key == 'intra' and start_dt < self.config.intra_reduction_date <= end_dt:
            print(f"    📅 Intra reduction date ({self.config.intra_reduction_date.date()}) falls within range - splitting processing")
            
            all_results = []
            
            # Period 1: Before intra reduction (use all 7 markets)
            period1_end = self.config.intra_reduction_date - timedelta(days=1)
            print(f"    📊 Period 1: {start_dt.date()} to {period1_end.date()} (7 intra markets)")
            ids_period1 = self.config.get_market_ids(market_key, start_dt) #get intra keys beofre reduction
            
            results_period1 = self._process_market_ids_for_period(
                market_key, ids_period1,fecha_inicio, period1_end.strftime('%Y-%m-%d'), plot
            )
            all_results.extend(results_period1)
            
            # Period 2: After intra reduction (use only 3 markets)
            print(f"    📊 Period 2: {self.config.intra_reduction_date.date()} to {end_dt.date()} (3 intra markets)")
            ids_period2 = self.config.get_market_ids(market_key, self.config.intra_reduction_date) #get intra keys after reduction
            
            results_period2 = self._process_market_ids_for_period(
                market_key, ids_period2, self.config.intra_reduction_date.strftime('%Y-%m-%d'), fecha_fin, plot
            )
            all_results.extend(results_period2)
            
        else:
            # Standard processing (no date splitting needed)
            ids = self.config.get_market_ids(market_key, start_dt)
            print(f"    Processing {len(ids)} market IDs: {ids}")
            
            all_results = self._process_market_ids_for_period(market_key, ids, fecha_inicio, fecha_fin, plot)
        
        # Combine all results
        combined = pd.concat(all_results, ignore_index=True)

        self._analyze_total_ingresos(combined, title_context=f"Combined for {market_key}", plot=plot)
        
        print(f"\n🎉 CALCULATION COMPLETED SUCCESSFULLY!")
        return combined


class ContinuoIngresosCalculator(IngresosCalculator):
    def calculate_single(self, market_key, fecha, plot=False):
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
            df['ingresos'] = (df['volumenes'] * df['precio']).round(2)
            
            result = df[['datetime_utc', 'uof', 'ingresos', 'id_mercado']].rename(columns={'uof': 'up'})

            # Analyze ingresos for this market
            self._analyze_total_ingresos(result, title_context=f"market {market_key} (id: {id_mercado})", plot=plot)
            
            all_results.append(result)
            print(f"    ✅ Market ID {id_mercado} completed: {len(result):,} records")
            
        if not all_results:
            print(f"\n❌ CALCULATION FAILED: No results generated for any market ID")
            return pd.DataFrame()
            
        combined = pd.concat(all_results, ignore_index=True)
        
        self._analyze_total_ingresos(combined, title_context=f"Combined for {market_key}", plot=plot)
        
        print(f"\n🎉 CALCULATION COMPLETED SUCCESSFULLY!")
        return combined

    def calculate_diferencias_single(self, market_key, fecha, plot=False):
        """Calculate ingresos por diferencias for MIC (id_mercado=21)."""
        spot_df = self._get_df_for_date_range('diario', 1, 'precios_esios', fecha, fecha)

        id_mercado = 21  # MIC market id
        df = self._get_df_for_date_range(market_key, id_mercado, 'volumenes_omie', fecha, fecha)
        if df.empty or spot_df.empty:
            return pd.DataFrame()

        spot_df = spot_df[['datetime_utc', 'precio']].rename(columns={'precio': 'precio_spot'})
        merged = pd.merge(df, spot_df, on='datetime_utc', how='inner')

        if merged.empty:
            return pd.DataFrame()

        merged['ingresos'] = (merged['volumenes'] * (merged['precio'] - merged['precio_spot'])).round(2)
        result = merged[['datetime_utc', 'uof', 'ingresos']].rename(columns={'uof': 'up'})
        result['id_mercado'] = id_mercado
        return result

class DesviosIngresosCalculator(IngresosCalculator):
    def _check_energias_balance(self, fecha):
        """
        Check all i90 volumes for that particular hour for mercados de balance a subir and bajar
        to determine if the price is unique or dual.
        """
        mercados_subir = self.config.mercado_sentido_desvios_map['subir']
        mercados_bajar = self.config.mercado_sentido_desvios_map['bajar']
        
        volumenes_subir = 0
        volumenes_bajar = 0
        
        # Get all volumes for 'subir' markets
        for mercado, ids in mercados_subir.items():
            for id_mercado in ids:
                df = self._get_df_for_date_range(mercado, id_mercado, 'volumenes_i90', fecha, fecha)
                if not df.empty:
                    volumenes_subir += df['volumenes'].sum()
                    
        # Get all volumes for 'bajar' markets
        for mercado, ids in mercados_bajar.items():
            for id_mercado in ids:
                df = self._get_df_for_date_range(mercado, id_mercado, 'volumenes_i90', fecha, fecha)
                if not df.empty:
                    volumenes_bajar += df['volumenes'].sum()

        print(f"Volumenes subir: {volumenes_subir} volumenes bajar: {volumenes_bajar}")

        if volumenes_subir == 0 and volumenes_bajar == 0:
            return None, None  # Desvio nulo (excepcional)

        if volumenes_subir > volumenes_bajar:
            direction = "subir"
            if volumenes_bajar < 0.02 * volumenes_subir:
                return "unico", direction
            else:
                return "dual", None
        else:  # volumenes_bajar >= volumenes_subir
            direction = "bajar"
            if volumenes_subir < 0.02 * volumenes_bajar:
                return "unico", direction
            else:
                return "dual", None

    def _calculate_precios_desvios(self, fecha, sentidos: list) -> dict:
        """Calculate weighted average desvío prices and per-mercado energy fractions for each sentido."""

        precios_desvios = {}
        fractions_by_sentido = {} # useful for debugging

        for sentido in sentidos:
            total_volumen = 0.0
            total_ingresos = 0.0
            total_volumen_abs = 0.0 # For fraction calculation

            mercados = self.config.mercado_sentido_desvios_map[sentido]
            per_market_volume = {mercado: 0.0 for mercado in mercados.keys()}
            per_market_volume_abs = {mercado: 0.0 for mercado in mercados.keys()} # For fraction calculation

            for mercado, ids in mercados.items():
                for id_mercado in ids:

                    #get volumes and prices for the market
                    volumes_df = self._get_df_for_date_range(mercado, id_mercado, 'volumenes_i90', fecha, fecha)
                    if volumes_df.empty:
                        continue

                    precio_id = self.config.get_precios_from_id_mercado(id_mercado, pd.to_datetime(fecha))
                    prices_df = self._get_df_for_date_range(mercado, precio_id, 'precios_esios', fecha, fecha)
                    if prices_df.empty:
                        continue

                    merged_df = pd.merge(volumes_df, prices_df, on='datetime_utc', how='inner') #replicate price for each up entry

                    if not merged_df.empty:
                        ingresos = (merged_df['volumenes'] * merged_df['precio']).sum()
                        volumen = merged_df['volumenes'].sum()

                        total_ingresos += float(ingresos)
                        total_volumen += float(volumen)
                        per_market_volume[mercado] += float(volumen)
                        
                        # Use absolute values for fraction calculation
                        total_volumen_abs += abs(float(volumen))
                        per_market_volume_abs[mercado] += abs(float(volumen))

            if total_volumen != 0:
                precio_ponderado = total_ingresos / total_volumen
                precios_desvios[sentido] = precio_ponderado

                if total_volumen_abs > 0:
                    fractions_by_sentido[sentido] = {m: (per_market_volume_abs[m] / total_volumen_abs) for m in per_market_volume_abs}
                else:
                     fractions_by_sentido[sentido] = {m: 0.0 for m in per_market_volume}
            else:
                precios_desvios[sentido] = 0.0
                fractions_by_sentido[sentido] = {m: 0.0 for m in per_market_volume}

        # Debug print
        print("    ⚖️ Fraction of total FRR/RR energy by mercado per sentido:")
        for sentido, fractions in fractions_by_sentido.items():
            details = ", ".join([f"{m}: {f*100:.2f}%" for m, f in fractions.items()])
            print(f"        {sentido}: {details}")

        return precios_desvios
    
    def _get_precios_desvios(self, fecha):
        """Calculate the precios for the desvios"""

        #extract net direction of all rr and frr energias balance
        price_type, direction = self._check_energias_balance(fecha)

        if price_type is None and direction is None:
            return None
        
        #if the price is unique, we can use the same price for both subir and bajar
        if price_type == "unico":
            precios = self._calculate_precios_desvios(fecha, [direction])
            precio_unico = precios[direction]
            return {"subir": precio_unico, "bajar": precio_unico}

        else: #dual, we have to calculate the price for each sentido
            precios = self._calculate_precios_desvios(fecha, ["subir", "bajar"])
            precio_subir_desvio = precios['bajar']
            precio_bajar_desvio = precios['subir']
            return {"subir": precio_subir_desvio, "bajar": precio_bajar_desvio}
    
    def calculate_single(self, market_key, fecha, plot=False):
        print(f"\n🚀 STARTING DESVIOS SINGLE DATE INGRESOS CALCULATION")
        print(f"    Market: {market_key}")
        print(f"    Date: {fecha}")
        
        # 1. Get desvios prices by checking the relative amount of energias de balance subir/bajar for the day
        precios_desvio = self._get_precios_desvios(fecha)

        if precios_desvio is None:
            print(f"    ℹ️ No FRR activated (or negligible) on {fecha} → no desvío price applicable")
            return pd.DataFrame()
        else:
            print(f"    💸 Precios desvios calculados: Subir={precios_desvio['subir']:.2f} €/MWh, Bajar={precios_desvio['bajar']:.2f} €/MWh")

        # 2. Get desvios volumes
        ids = self.config.get_market_ids(market_key, pd.to_datetime(fecha))
        
        all_volumes_df = []
        for id_mercado in ids:
            df = self._get_df_for_date_range(market_key, id_mercado, 'volumenes_i90', fecha, fecha)
            if not df.empty:
                all_volumes_df.append(df)
        
        if not all_volumes_df:
            print(f"    ❌ No volume data found for desvios on {fecha}")
            return pd.DataFrame()

        volumes_df = pd.concat(all_volumes_df, ignore_index=True)
        
        # 3. Calculate ingresos
        def calculate_row_ingresos(row):
            volumen = row['volumenes']
            if volumen > 0: # Desvío a subir
                return volumen * precios_desvio['subir']
            elif volumen < 0: # Desvío a bajar
                return volumen * precios_desvio['bajar']
            else:
                return 0

        volumes_df['ingresos'] = volumes_df.apply(calculate_row_ingresos, axis=1).round(2)

        result = volumes_df[['datetime_utc', 'up', 'ingresos', 'id_mercado']]

        self._analyze_total_ingresos(result, title_context=f"Desvios for {fecha}", plot=plot)
        
        print(f"\n🎉 CALCULATION COMPLETED SUCCESSFULLY!")
        return result
        