import os
import pandas as pd
import pytz
from datetime import datetime, timedelta, date
from typing import Optional, List, Dict
import numpy as np
import sys
import os
from tqdm import tqdm

# Add the project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from configs.omie_config import OMIEConfig
from extract._descargador_omie import IntraOMIEDownloader, ContinuoOMIEDownloader, DiarioOMIEDownloader
from utilidades.raw_file_utils import RawFileUtils
from utilidades.db_utils import DatabaseUtils
from utilidades.env_utils import EnvUtils


class OMIEExtractor:
    """Extractor for OMIE market data"""
    
    def __init__(self):
        """
        Initialize the OMIEExtractor with configuration, downloader instances for each market type, utility classes, and a default maximum download window of 93 days.
        """
        # Initialize configuration
        self.config = OMIEConfig()
        
        # Initialize downloaders
        self.intra_downloader = IntraOMIEDownloader()
        self.continuo_downloader = ContinuoOMIEDownloader()
        self.diario_downloader = DiarioOMIEDownloader()
        
        # Initialize utils
        self.raw_file_utils = RawFileUtils()
        self.env_utils = EnvUtils()
        
        # Set the maximum download window (in days)
        self.download_window = 93  # OMIE typically has data available with 90 day delay as well 


    def fecha_input_validation(self, fecha_inicio_carga: str, fecha_fin_carga: str) -> tuple[str, str]:
        """
        Validate and normalize the input date range for OMIE data extraction.
        
        If both start and end dates are provided, ensures the start date is not after the end date. If neither date is provided, defaults to a one-day range starting 93 days ago. Raises a ValueError if only one date is provided.
        
        Returns:
            tuple[str, str]: Validated start and end dates in 'YYYY-MM-DD' format.
        """
        # Check if dates are provided and valid
        if fecha_inicio_carga and fecha_fin_carga:
            fecha_inicio_carga_dt = datetime.strptime(fecha_inicio_carga, '%Y-%m-%d')
            fecha_fin_carga_dt = datetime.strptime(fecha_fin_carga, '%Y-%m-%d')

            # If start date > end date, raise error
            if fecha_inicio_carga_dt > fecha_fin_carga_dt:
                print(f"Wrong input dates: {fecha_inicio_carga} > {fecha_fin_carga}")
                raise ValueError("La fecha de inicio de carga no puede ser mayor que la fecha de fin de carga")
            
            print(f"Descargando datos entre {fecha_inicio_carga} y {fecha_fin_carga}")

        # If no dates provided, set default values
        elif fecha_inicio_carga is None and fecha_fin_carga is None:
            fecha_inicio_carga_dt = datetime.now() - timedelta(days=self.download_window)
            fecha_fin_carga_dt = datetime.now() - timedelta(days=self.download_window) + timedelta(days=1)
            
            fecha_inicio_carga = fecha_inicio_carga_dt.strftime('%Y-%m-%d')
            fecha_fin_carga = fecha_fin_carga_dt.strftime('%Y-%m-%d')
            print(f"No se han proporcionado fechas de carga, se descargarán datos entre {fecha_inicio_carga} y {fecha_fin_carga}")

        else:
            raise ValueError("No se han proporcionado fechas de carga completas")

        return fecha_inicio_carga, fecha_fin_carga

    def extract_omie_diario(self, fecha_inicio_carga: Optional[str] = None, fecha_fin_carga: Optional[str] = None) -> None:
        """
        Extracts daily OMIE market data for a specified date range and saves it as standardized raw CSV files.
        
        Validates the input dates, iterates over each day in the range, downloads daily market data, standardizes column names for consistency, adds a market identifier column, and saves the data using the configured file utility. Skips saving if no data is available for a given day. Raises an exception if an error occurs during extraction.
        """
        # Validate input dates
        self.fecha_input_validation(fecha_inicio_carga, fecha_fin_carga)

        # Convert to datetime objects
        fecha_inicio_carga_dt = datetime.strptime(fecha_inicio_carga, '%Y-%m-%d')
        fecha_fin_carga_dt = datetime.strptime(fecha_fin_carga, '%Y-%m-%d')

        print(f"Processing market: diario")
        # Download data for each day in the range
        for day in tqdm(pd.date_range(start=fecha_inicio_carga_dt, end=fecha_fin_carga_dt), desc="Extracting diario data"):
            # Extract year and month from date
            year = day.year
            month = day.month
            day_str = day.strftime('%Y-%m-%d')

            try:
                # Get data for the day using the diario downloader
                diario_data = self.diario_downloader.descarga_omie_datos(
                    fecha_inicio_carga=day_str,
                    fecha_fin_carga=day_str
                )

                #diario data is a dict with keys as months and values as the dataframes for a market
                for month_data in diario_data.values():
                    if isinstance(month_data, list):
                        for df in month_data:
                            if df is not None and not df.empty:
                                # Standardize column names before processing
                                df = self._standardize_column_names(df)
                                
                                # Add ID column for raw storage
                                df['id_mercado'] = 1  # ID for daily market
                                
                                self.raw_file_utils.write_raw_csv(
                                    year=year, month=month, df=df,
                                    dataset_type='volumenes_omie',
                                    mercado='diario'
                                )

                                print(f"✅ Successfully saved raw diario data for {day_str}")
                            else:
                                print(f" ⚠️ No diario data found for {day_str}. Nothing was saved to raw folder.")
                    else:
                        # Handle case where it's directly a DataFrame (fallback)
                        df = month_data
                        if df is not None and not df.empty:
                            df = self._standardize_column_names(df)
                            df['id_mercado'] = 1

                            self.raw_file_utils.write_raw_csv(
                                year=year, month=month, df=df,
                                dataset_type='volumenes_omie',
                                mercado='diario'
                            )
                            print(f"✅ Successfully saved raw diario data for {day_str}")

            except Exception as e:
                error_msg = f"Error downloading diario data for {day_str}: {e}"
                print(f"  ❌ {error_msg}")
                raise e
            
    def extract_omie_intra(self, fecha_inicio_carga: Optional[str] = None, fecha_fin_carga: Optional[str] = None, 
                          intra_lst: Optional[List[int]] = None) -> None:
        """
                          Extracts intraday (intra) market data from OMIE for a specified date range and set of intraday market sessions.
                          
                          Validates input dates, iterates over each day in the range, downloads intraday data for the specified sessions, standardizes column names, assigns the appropriate market ID (`id_mercado`) based on session and delivery date, and saves the resulting data as raw CSV files. Handles regulatory changes after 2024-06-13, when only intraday markets 1-3 are available.
                          
                          Parameters:
                              fecha_inicio_carga (Optional[str]): Start date in 'YYYY-MM-DD' format. If None, defaults to 93 days ago.
                              fecha_fin_carga (Optional[str]): End date in 'YYYY-MM-DD' format. If None, uses logic in fecha_input_validation.
                              intra_lst (Optional[List[int]]): List of intraday market session IDs to extract. If None, extracts all available sessions.
                          """
    
        # Validate input dates
        self.fecha_input_validation(fecha_inicio_carga, fecha_fin_carga)
        
        # Convert to datetime objects
        fecha_inicio_carga_dt = datetime.strptime(fecha_inicio_carga, '%Y-%m-%d')
        fecha_fin_carga_dt = datetime.strptime(fecha_fin_carga, '%Y-%m-%d')

        print(f"Processing market: intra")
        # Download data for each day in the range
        for day in tqdm(pd.date_range(start=fecha_inicio_carga_dt, end=fecha_fin_carga_dt), desc="Extracting intra data"):
            # Extract year and month from date
            year = day.year
            month = day.month
            day_str = day.strftime('%Y-%m-%d')

            try:
                # Get data for the day using the intra downloader
                intra_data = self.intra_downloader.descarga_omie_datos(
                    fecha_inicio_carga=day_str,
                    fecha_fin_carga=day_str,
                    intras =intra_lst
                )

                for month_data in intra_data.values():
                    if isinstance(month_data, list):
                        for df in month_data:
                            if df is not None and not df.empty:
                                # Standardize column names before processing
                                df = self._standardize_column_names(df)
                                
                                # Add ID column for raw storage - conditional assignment
                                df['id_mercado'] = df['sesion'] + 1
                                
                                # Override for session 2 id mercado (rewrite based on intra sesion logic): check if delivery date matches
                                session_2_mask = df['sesion'] == 2

                               # For rows where sesion == 2 (id_mercado = 3) if the date matches the current day, otherwise (d-1) (id_mercado = 8)
                                df.loc[session_2_mask, 'id_mercado'] = np.where(
                                    df.loc[session_2_mask, 'Fecha'] == day.date(), 3, 8
                                )

                                
                                self.raw_file_utils.write_raw_csv(
                                        year=year, month=month, df=df,
                                        dataset_type='volumenes_omie',
                                        mercado='intra'
                                    )
                                                        
                                print(f"✅ Successfully saved raw intra data for {day_str}")
                            else:
                                print(f" ⚠️ No intra data found for {day_str}. Nothing was saved to raw folder.")

            except Exception as e:
                error_msg = f"Error downloading intra data for {day_str}: {e}"
                print(f"  ❌ {error_msg}")
                raise e 

    def extract_omie_continuo(self, fecha_inicio_carga: Optional[str] = None, fecha_fin_carga: Optional[str] = None) -> None:
        """
        Extracts and saves OMIE continuous market data for each day in the specified date range.
        
        Validates input dates, downloads daily continuous market data, assigns a market identifier, and saves the data as raw CSV files. Skips saving if no data is available for a given day. Raises an exception if data download fails.
        """
        # Validate input dates
        self.fecha_input_validation(fecha_inicio_carga, fecha_fin_carga)
        

        # Convert to datetime objects
        fecha_inicio_carga_dt = datetime.strptime(fecha_inicio_carga, '%Y-%m-%d')
        fecha_fin_carga_dt = datetime.strptime(fecha_fin_carga, '%Y-%m-%d')

        print(f"Processing market: continuo")
        # Download data for each day in the range
        for day in tqdm(pd.date_range(start=fecha_inicio_carga_dt, end=fecha_fin_carga_dt), desc="Extracting continuo data"):
            # Extract year and month from date
            year = day.year
            month = day.month
            day_str = day.strftime('%Y-%m-%d')

            try:
                # Get data for the day using the continuo downloader
                continuo_data = self.continuo_downloader.descarga_omie_datos(
                    fecha_inicio_carga=day_str,
                    fecha_fin_carga=day_str
                )


                for month_data in continuo_data.values():
                    if isinstance(month_data, list):
                        for df in month_data:
                            if df is not None and not df.empty:
                                # Add ID column for raw storage
                                df['id_mercado'] = 21  # ID for continuous market
                                
                                self.raw_file_utils.write_raw_csv(
                                    year=year, month=month, df=df,
                                    dataset_type='volumenes_omie',
                                    mercado='continuo'
                                )
                                print(f"✅ Successfully saved raw continuo data for {day_str}")
                            else:
                                print(f" ⚠️ No continuo data found for {day_str}. Nothing was saved to raw folder.")

            except Exception as e:
                print(f"  ❌ Error downloading continuo data for {day_str}: {e}")
                raise e

    def extract_data_for_all_markets(self, fecha_inicio_carga: Optional[str] = None, fecha_fin_carga: Optional[str] = None , mercados_lst: Optional[List[str]] = None) -> Dict:
        """
        Extracts OMIE market data for all specified markets within a given date range and returns a summary of the extraction status.
        
        Parameters:
        	fecha_inicio_carga (Optional[str]): Start date for extraction in 'YYYY-MM-DD' format. If None, defaults to the latest available window.
        	fecha_fin_carga (Optional[str]): End date for extraction in 'YYYY-MM-DD' format. If None, defaults to the latest available window.
        	mercados_lst (Optional[List[str]]): List of market names to extract ('diario', 'intra', 'continuo'). If None, all markets are extracted.
        
        Returns:
        	Dict: Dictionary containing a boolean 'success' flag and a 'details' dictionary with lists of successfully downloaded and failed markets, as well as the date range.
        """
        # Validate input dates BEFORE starting any extraction
        self.fecha_input_validation(fecha_inicio_carga, fecha_fin_carga)
        
        if (fecha_fin_carga is None and fecha_inicio_carga is None) or fecha_fin_carga == fecha_inicio_carga:
            date_range_str = f"Single day download for latest available day"
        else:
            date_range_str = f"{fecha_inicio_carga} to {fecha_fin_carga}"

        # Initialize status tracking
        status_details = {
            "markets_downloaded": [],
            "markets_failed": [],
            "date_range": date_range_str
        }

        if mercados_lst is None:
            mercados_lst = ['diario', 'intra', 'continuo']
        
        market_successes = []
        try:
            # Track success for each market
            if 'diario' in mercados_lst:
                print("\n--------- Diario ---------")
                success_diario = self._extract_with_status("diario", self.extract_omie_diario, 
                                                        fecha_inicio_carga, fecha_fin_carga, status_details)
                market_successes.append(success_diario)
            if 'intra' in mercados_lst:
                print("\n--------- Intra ---------")
                success_intra = self._extract_with_status("intra", self.extract_omie_intra, 
                                                        fecha_inicio_carga, fecha_fin_carga, status_details)
                market_successes.append(success_intra)
            if 'continuo' in mercados_lst:
                print("\n--------- Continuo ---------")
                success_continuo = self._extract_with_status("continuo", self.extract_omie_continuo, 
                                                       fecha_inicio_carga, fecha_fin_carga, status_details)
                market_successes.append(success_continuo)
            print("\n--------------------------------")
            
            
            # Overall success only if all markets succeeded
            overall_success = all(market_successes)
            
        except Exception as e:
            overall_success = False
            status_details["error"] = str(e)
        
        print("ℹ️ Data extraction pipeline finished.")
        
        # Return status for Airflow task
        return {"success": overall_success, "details": status_details}

    def _extract_with_status(self, market_name: str, extract_function, fecha_inicio_carga: Optional[str], 
                            fecha_fin_carga: Optional[str], status_details: Dict) -> bool:
        """
                            Attempts to extract data for a specific market and updates the extraction status.
                            
                            Calls the provided extraction function for the given market and date range, appending the market name to the appropriate status list in `status_details` based on success or failure.
                            
                            Parameters:
                                market_name (str): The name of the market being extracted.
                                extract_function: The extraction function to invoke for the market.
                                fecha_inicio_carga (Optional[str]): Start date for extraction in 'YYYY-MM-DD' format.
                                fecha_fin_carga (Optional[str]): End date for extraction in 'YYYY-MM-DD' format.
                                status_details (Dict): Dictionary tracking downloaded and failed markets.
                            
                            Returns:
                                bool: True if extraction succeeded, False if an error occurred.
                            """
        try:
            if extract_function == self.extract_omie_diario or extract_function == self.extract_omie_intra:
                error_msg = extract_function(fecha_inicio_carga, fecha_fin_carga)

                if error_msg: #for diario and intra market bc we dont raise an error for these markets but rather return a df and an error msg
                    raise Exception(error_msg)
            
                status_details["markets_downloaded"].append(market_name)
                return True
        

            else: #for continuo market, since we dont return a df or an error msg, we just raise an error
                extract_function(fecha_inicio_carga, fecha_fin_carga)
            
                status_details["markets_downloaded"].append(market_name)
                return True
            
        except Exception as e:
            status_details["markets_failed"].append({
                "market": market_name,
                "error": str(e)
            })
            return False

    def _standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize OMIE DataFrame column names to ensure consistency across schema changes.
        
        Renames columns introduced by OMIE after March 2025 back to their original names, specifically converting 'Periodo' to 'Hora' and 'Potencia Compra/Venta' to 'Energía Compra/Venta'.
        
        Parameters:
            df (pd.DataFrame): DataFrame with OMIE market data that may contain updated column names.
        
        Returns:
            pd.DataFrame: DataFrame with standardized column names for downstream processing.
        """
        if df is None or df.empty:
            return df
            
        # Create a copy to avoid modifying the original
        df_standardized = df.copy()
        
        # Standardize column names
        column_mapping = {
            'Periodo': 'Hora',  # Rename Periodo back to Hora
            'Potencia Compra/Venta': 'Energía Compra/Venta'  # Rename Potencia back to Energía
        }
        
        # Apply renaming only if columns exist
        for old_name, new_name in column_mapping.items():
            if old_name in df_standardized.columns:
                df_standardized = df_standardized.rename(columns={old_name: new_name})
                print(f"   Standardized column: '{old_name}' -> '{new_name}'")
        
        return df_standardized

