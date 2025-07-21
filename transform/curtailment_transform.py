import pandas as pd
from typing import Optional
from datetime import datetime
from typing import List
import traceback

from utilidades.raw_file_utils import RawFileUtils
from transform._procesador_curtailments import CurtailmentProcessor

class CurtailmentTransformer:
    """
    Transformer class for curtailment data from i90 and i3 sources.
    Handles reading the latest raw data and processing it using CurtailmentProcessor.
    """

    def __init__(self):
        self.raw_utils = RawFileUtils()
        self.processor = CurtailmentProcessor()

    def transform_curtailment_data(self, fecha_inicio: Optional[str] = None, fecha_fin: Optional[str] = None, 
                                   sources_lst: Optional[List[str]] = None) -> dict:
        """
        Transforms curtailment data for specified sources over a given date range or mode.
        
        Automatically determines the transformation mode based on the provided date parameters:
        - If no dates are given, processes the latest available data ('latest' mode).
        - If only `fecha_inicio` is provided or both dates are equal, processes a single day ('single' mode).
        - If both dates are provided and different, processes the full date range ('multiple' mode).

        Parameters:
            fecha_inicio (str, optional): Start date in 'YYYY-MM-DD' format, or the single date to process.
            fecha_fin (str, optional): End date in 'YYYY-MM-DD' format. If omitted or equal to `fecha_inicio`, processes a single day.
            sources_lst (List[str], optional): List of sources to process (e.g., ['curtailment_i90', 'curtailment_i3']). If omitted, processes both.
            
        Returns:
            dict: A dictionary with:
                - 'data': Mapping of source names to processed DataFrames.
                - 'status': Dictionary containing success status and details.
        """
        if sources_lst is None:
            sources_lst = ['curtailment_i90', 'curtailment_i3']

        if fecha_inicio is None and fecha_fin is None:
            transform_type = 'latest'
        elif fecha_inicio is not None and (fecha_fin is None or fecha_inicio == fecha_fin):
            transform_type = 'single'
        elif fecha_inicio is not None and fecha_fin is not None and fecha_inicio != fecha_fin:
            transform_type = 'multiple'

        status_details = {
            "markets_processed": [],
            "markets_failed": [],
            "mode": transform_type,
            "date_range": f"{fecha_inicio} to {fecha_fin}" if fecha_fin else fecha_inicio
        }

        overall_success = True
        results = {}

        print(f"\n===== STARTING CURTAILMENT TRANSFORMATION RUN (Mode: {transform_type.upper()}) =====")
        if fecha_inicio: print(f"Start Date: {fecha_inicio}")
        if fecha_fin: print(f"End Date: {fecha_fin}")
        print("="*60)

        for market in sources_lst:
            print(f"\n-- Source: {market} --")
            source = market.split('_')[1]  # 'i90' or 'i3'
            dataset_type = f'volumenes_{source}'
            try:
                if transform_type == 'single':
                    market_result = self._process_single_day(market, dataset_type, fecha_inicio)
                elif transform_type == 'latest':
                    market_result = self._process_latest_day(market, dataset_type)
                elif transform_type == 'multiple':
                    market_result = self._process_date_range(market, dataset_type, fecha_inicio, fecha_fin)

                if market_result is not None and isinstance(market_result, pd.DataFrame) and not market_result.empty:
                    status_details["markets_processed"].append(market)
                    results[market] = market_result
                else:
                    status_details["markets_failed"].append(market)
                    overall_success = False

            except Exception as e:
                status_details["markets_failed"].append({
                    "market": market,
                    "error": str(e)
                })
                overall_success = False
                results[market] = None
                print(f"❌ Failed to transform for {market} ({transform_type}): {e}")
                print(traceback.format_exc())

        print(f"\n===== CURTAILMENT TRANSFORMATION RUN FINISHED (Mode: {transform_type.upper()}) =====")
        print("Summary:")
        for market, result in results.items():
            status = "Failed/No Data"
            if isinstance(result, pd.DataFrame) and not result.empty:
                status = f"Success ({len(result)} records)"
            print(f"- {market}: {status}")
        print("="*60)

        return {
            "data": results,
            "status": {
                "success": overall_success,
                "details": status_details
            }
        }

    def _transform_data(self, raw_df: pd.DataFrame, market: str) -> pd.DataFrame:
        if raw_df.empty:
            print(f"Skipping transformation for {market}: Input DataFrame is empty.")
            return pd.DataFrame()

        source = market.split('_')[1]
        table_name = f'curtailments_{source}'
        processed_df = self.processor.transform_raw_curtailment_data(raw_df, table_name)
        if processed_df is None or processed_df.empty:
            print(f"Transformation resulted in empty or None DataFrame for {market}.")
            return pd.DataFrame()

        print(f"Processed data for {market}:")
        print(processed_df.head())
        return processed_df

    def _process_df_based_on_transform_type(self, raw_df: pd.DataFrame, transform_type: str, fecha_inicio: Optional[str] = None, fecha_fin: Optional[str] = None) -> pd.DataFrame:
        if raw_df.empty:
            return raw_df

        if "fecha" not in raw_df.columns:
            print("Error: 'fecha' column missing in DataFrame. Cannot apply date filters.")
            raise ValueError("'fecha' column missing. Cannot apply date filters.")
        else:
            if not pd.api.types.is_datetime64_any_dtype(raw_df['fecha']):
                try:
                    raw_df['fecha'] = pd.to_datetime(raw_df['fecha'])
                except Exception as e:
                    print(f"Error converting 'fecha': {e}.")
                    raise e

        try:
            if transform_type == 'latest':
                last_day = raw_df['fecha'].dropna().dt.date.max()
                if pd.isna(last_day):
                    print("Warning: Could not determine the latest day (max date is NaT). Returning empty DataFrame for 'latest' mode.")
                    return pd.DataFrame()
                print(f"Filtering for latest mode: {last_day}")
                return raw_df[raw_df['fecha'].dt.date == last_day].copy()

            elif transform_type == 'single':
                target_date = pd.to_datetime(fecha_inicio).date()
                print(f"Filtering for single mode: {target_date}")
                return raw_df[raw_df['fecha'].dt.date == target_date].copy()

            elif transform_type == 'multiple':
                start_dt = pd.to_datetime(fecha_inicio).date()
                end_dt = pd.to_datetime(fecha_fin).date()
                if start_dt > end_dt: raise ValueError("Start date cannot be after end date.")
                print(f"Filtering for multiple mode: {start_dt} to {end_dt}")
                return raw_df[(raw_df['fecha'].dt.date >= start_dt) & (raw_df['fecha'].dt.date <= end_dt)].copy()

            else:
                raise ValueError(f"Invalid transform type for filtering: {transform_type}")

        except Exception as e:
            print(f"Error during date filtering ({transform_type} mode): {e}")
            return pd.DataFrame()

    def _process_single_day(self, market: str, dataset_type: str, date: str):
        print(f"Starting SINGLE transformation for {market} on {date}")
        try:
            target_date = pd.to_datetime(date)
            target_year = target_date.year
            target_month = target_date.month

            raw_df = self.raw_utils.read_raw_file(target_year, target_month, dataset_type, 'restricciones')

            filtered_df = self._process_df_based_on_transform_type(raw_df, 'single', fecha_inicio=date)

            if filtered_df.empty:
                print(f"No data found for {market} on {date}.")
                return pd.DataFrame()

            processed_df = self._transform_data(filtered_df, market)
            return processed_df

        except Exception as e:
            print(f"Error during single day processing for {market} on {date}: {e}")
            return pd.DataFrame()

    def _process_latest_day(self, market: str, dataset_type: str):
        print(f"Starting LATEST transformation for {market}")
        try:
            years = sorted(self.raw_utils.get_raw_folder_list(mercado='restricciones'), reverse=True)
            if not years:
                print(f"No data years found for restricciones. Skipping latest for {market}.")
                return pd.DataFrame()

            found = False
            for year in years:
                months = sorted(self.raw_utils.get_raw_folder_list(mercado='restricciones', year=year), reverse=True)
                for month in months:
                    files = self.raw_utils.get_raw_file_list('restricciones', year, month)
                    if not files:
                        continue
                    matching_files = [f for f in files if dataset_type in str(f)]
                    if matching_files:
                        print(f"Identified latest available file for {market}: {year}-{month:02d}")
                        raw_df = self.raw_utils.read_raw_file(year, month, dataset_type, 'restricciones')
                        filtered_df = self._process_df_based_on_transform_type(raw_df, 'latest')
                        if filtered_df.empty:
                            print(f"No data found for the latest day within file {year}-{month:02d} for {market}.")
                            return pd.DataFrame()
                        processed_df = self._transform_data(filtered_df, market)
                        found = True
                        return processed_df
                if found:
                    break
            if not found:
                print(f"No raw file found for {market} in any available year/month.")
                return pd.DataFrame()

        except Exception as e:
            print(f"Error during latest day processing for {market}: {e}")
            return pd.DataFrame()

    def _process_date_range(self, market: str, dataset_type: str, fecha_inicio: str, fecha_fin: str):
        print(f"Starting MULTIPLE transformation for {market} from {fecha_inicio} to {fecha_fin}")
        try:
            start_dt = pd.to_datetime(fecha_inicio)
            end_dt = pd.to_datetime(fecha_fin)

            if start_dt > end_dt: raise ValueError("Start date cannot be after end date.")

            all_days_in_range = pd.date_range(start_dt, end_dt, freq='D')
            if all_days_in_range.empty:
                print("Warning: Date range resulted in zero days. Nothing to process for {market}.")
                return pd.DataFrame()

            year_months = sorted(list(set([(d.year, d.month) for d in all_days_in_range])))

            all_raw_dfs = []
            print(f"Reading files for year-months: {year_months} for {market}")
            for year, month in year_months:
                try:
                    raw_df = self.raw_utils.read_raw_file(year, month, dataset_type, 'restricciones')
                    if raw_df is not None and not raw_df.empty:
                        all_raw_dfs.append(raw_df)
                    else:
                        print(f"Warning: No data returned from reading for {year}-{month:02d} for {market}.")

                except FileNotFoundError:
                    print(f"Warning: Raw file not found for {year}-{month:02d} for {market}. Skipping.")
                except Exception as e:
                    print(f"Error reading or processing raw file for {year}-{month:02d} for {market}: {e}")
                    continue

            if not all_raw_dfs:
                print(f"No raw data found for the specified date range {fecha_inicio} to {fecha_fin} for {market}.")
                return pd.DataFrame()

            combined_raw_df = pd.concat(all_raw_dfs, ignore_index=True)

            print(f"Combined raw data ({len(combined_raw_df)} rows) for {market}. Applying date range filtering.")
            filtered_df = self._process_df_based_on_transform_type(combined_raw_df, 'multiple', fecha_inicio=fecha_inicio, fecha_fin=fecha_fin)

            if filtered_df.empty:
                print(f"No data found for {market} between {fecha_inicio} and {fecha_fin} after final filtering.")
                return pd.DataFrame()

            print(f"Filtered data has {len(filtered_df)} rows for {market}. Proceeding with transformation...")
            processed_df = self._transform_data(filtered_df, market)
            return processed_df

        except Exception as e:
            print(f"An unexpected error occurred during multiple transform for {market}: {e}")
            print(traceback.format_exc())
            return pd.DataFrame()

    def transform_curtailment_i90(self) -> dict:
        """
        Wrapper for i90 latest transformation using the standardized method.
        """
        return self.transform_curtailment_data(sources_lst=['curtailment_i90'])

    def transform_curtailment_i3(self) -> dict:
        """
        Wrapper for i3 latest transformation using the standardized method.
        """
        return self.transform_curtailment_data(sources_lst=['curtailment_i3'])
