import os
import pandas as pd
import pytz
from datetime import datetime, timedelta, date
from typing import Optional, List, Dict
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from configs.omie_config import OMIEConfig
from extract.omie_downloaders import IntraOMIEDownloader, ContinuoOMIEDownloader
from utilidades.raw_file_utils import RawFileUtils
from utilidades.db_utils import DatabaseUtils
from utilidades.env_utils import EnvUtils


class OMIEExtractor:
    """Extractor for OMIE market data"""
    
    def __init__(self):
        """Initialize the OMIE extractor"""
        # Initialize configuration
        self.config = OMIEConfig()
        
        # Initialize downloaders
        self.intra_downloader = IntraOMIEDownloader()
        self.continuo_downloader = ContinuoOMIEDownloader()
        
        # Initialize utils
        self.raw_file_utils = RawFileUtils()
        self.bbdd_engine = DatabaseUtils.create_engine('pruebas_BT')
        self.env_utils = EnvUtils()
        
        # Set the maximum download window (in days)
        self.download_window = 93  # OMIE typically has data available with 90 day delay

    def fecha_input_validation(self, fecha_inicio_carga: str, fecha_fin_carga: str) -> tuple[str, str]:
        """
        Validates the input date range for OMIE data requests.
        
        Args:
            fecha_inicio_carga (str): Start date in 'YYYY-MM-DD' format
            fecha_fin_carga (str): End date in 'YYYY-MM-DD' format
            
        Returns:
            tuple[str, str]: Validated start and end dates in 'YYYY-MM-DD' format
        """
        # Check if dates are provided and valid
        if fecha_inicio_carga and fecha_fin_carga:
            fecha_inicio_carga_dt = datetime.strptime(fecha_inicio_carga, '%Y-%m-%d')
            fecha_fin_carga_dt = datetime.strptime(fecha_fin_carga, '%Y-%m-%d')

            # If start date > end date, raise error
            if fecha_inicio_carga_dt > fecha_fin_carga_dt:
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

    def extract_data(self, fecha_inicio_carga: Optional[str] = None, fecha_fin_carga: Optional[str] = None, 
                     UP_ids: Optional[List[int]] = None, cargar_intras: Optional[List[int]] = None, 
                     cargar_continuo: bool = True) -> dict:
        """
        Extract OMIE data for each day in the specified date range.
        
        Args:
            fecha_inicio_carga (Optional[str]): Start date in 'YYYY-MM-DD' format
            fecha_fin_carga (Optional[str]): End date in 'YYYY-MM-DD' format
            UP_ids (Optional[List[int]]): List of programming unit IDs to filter
            cargar_intras (Optional[List[int]]): List of intraday sessions to download (1-7)
            cargar_continuo (bool): Whether to download continuous market data
            
        Returns:
            dict: Status information about the extraction process
        """
        # Initialize status tracking
        status_details = {
            "markets_downloaded": [],
            "markets_failed": [],
            "date_range": f"{fecha_inicio_carga} to {fecha_fin_carga}" if fecha_inicio_carga and fecha_fin_carga else "Default range"
        }
        
        try:
            # Validate date inputs
            fecha_inicio_carga, fecha_fin_carga = self.fecha_input_validation(fecha_inicio_carga, fecha_fin_carga)
            fecha_inicio_carga_dt = datetime.strptime(fecha_inicio_carga, '%Y-%m-%d')
            fecha_fin_carga_dt = datetime.strptime(fecha_fin_carga, '%Y-%m-%d')
            
            # Get list of programming units
            unidades, dict_unidades = self.config.get_lista_UPs(UP_ids)
            
            # Get error data
            df_errores = self.config.get_error_data()
            
            # Process each day in the date range
            overall_success = True
            mes_ref = None
            año_ref = None
            
            # Get timezone info for special day handling
            spain_timezone = pytz.timezone('Europe/Madrid')
            utc_transition_times = spain_timezone._utc_transition_times[1:]
            localized_transition_times = [pytz.utc.localize(transition).astimezone(spain_timezone) for transition in utc_transition_times]
            fecha_inicio_local = spain_timezone.localize(fecha_inicio_carga_dt)
            fecha_fin_local = spain_timezone.localize(fecha_fin_carga_dt)
            filtered_transition_times = [transition for transition in localized_transition_times if fecha_inicio_local <= transition <= fecha_fin_local]
            filtered_transition_dates = {dt.date():int(dt.isoformat()[-4]) for dt in filtered_transition_times}
            
            # Track zip files for intra and continuo
            intra_zip_file = None
            continuo_zip_file = None
            
            for day in pd.date_range(start=fecha_inicio_carga_dt, end=fecha_fin_carga_dt):
                day_str = day.strftime('%Y-%m-%d')
                
                # Filter errors for the current day
                df_errores_dia = df_errores[df_errores['fecha'] == day.date()]
                sesiones_canceladas = df_errores_dia['tipo_error'].values.tolist()
                
                try:
                    # Process intraday sessions
                    if cargar_intras is not False:
                        # Check if we need to download a new zip file for this month
                        if año_ref != day.year or mes_ref != day.month:
                            intra_zip_file = self.intra_downloader.download_intra_file(day.year, day.month)
                        
                        # Determine which sessions to process
                        if cargar_intras is None or cargar_intras is True:
                            intradiarios = [1, 2, 3, 4, 5, 6]
                        else:
                            intradiarios = cargar_intras
                        
                        # Filter out cancelled sessions
                        intradiarios = [item for item in intradiarios if item not in sesiones_canceladas]
                        
                        # Process each session
                        for intra in intradiarios:
                            # Skip sessions 4, 5, 6 after June 13, 2024
                            if intra not in [4, 5, 6] or day.date() <= date(2024, 6, 13):
                                status_details = self._extract_intra_session(
                                    intra_zip_file, day, intra, unidades, dict_unidades, status_details
                                )
                    
                    # Process continuous market
                    if cargar_continuo:
                        # Check if we need to download a new zip file for this month
                        if año_ref != day.year or mes_ref != day.month:
                            continuo_zip_file = self.continuo_downloader.download_continuo_file(day.year, day.month)
                        
                        status_details = self._extract_continuo_data(
                            continuo_zip_file, day, unidades, dict_unidades, 
                            filtered_transition_dates, status_details
                        )
                    
                    # Update reference month and year
                    mes_ref = day.month
                    año_ref = day.year
                    
                except Exception as e:
                    status_details["markets_failed"].append({
                        "market": "all",
                        "error": str(e),
                        "day": day_str
                    })
                    overall_success = False
                    print(f"Error during extraction for {day_str}: {e}")
            
            # Cleanup temporary files if needed
            if intra_zip_file and os.path.exists(intra_zip_file):
                os.remove(intra_zip_file)
            
            if continuo_zip_file and os.path.exists(continuo_zip_file):
                os.remove(continuo_zip_file)
            
        except Exception as e:
            overall_success = False
            status_details["markets_failed"].append({
                "market": "all",
                "error": str(e),
                "day": "all"
            })
        
        return {"success": overall_success, "details": status_details}

    def _extract_intra_session(self, zip_file: str, day: datetime, session: int, 
                               unidades: List[str], dict_unidades: Dict[str, int], 
                               status_details: Dict) -> Dict:
        """
        Extract data for a specific intraday session
        
        Args:
            zip_file (str): Path to the zip file
            day (datetime): The day to extract data for
            session (int): Session number
            unidades (List[str]): List of programming units
            dict_unidades (Dict[str, int]): Mapping of programming unit names to IDs
            status_details (Dict): Status tracking dictionary
            
        Returns:
            Dict: Updated status details
        """
        try:
            # Extract session file
            session_path = self.intra_downloader.extract_session_file(
                zip_file, day.year, day.month, day.day, session
            )
            
            if not session_path:
                print(f"No session file found for day {day.date()} session {session}")
                return status_details
            
            # Get data from session file
            df = self.intra_downloader.get_intra_data(session_path)
            
            if not df.empty:
                # Minimal processing to add id_mercado
                # For intra session 2, can be either 3 or 8 based on date
                if session == 2:
                    id_mercado = 3 if day.date() <= date(2024, 6, 13) else 8
                else:
                    id_mercado = session + 1
                
                # Add ID column for raw storage
                df['id_mercado'] = id_mercado
                
                # Save raw file
                year = day.year
                month = day.month
                
                # Use appropriate method based on environment
                dev, prod = self.env_utils.check_dev_env()
                if dev and not prod:
                    self.raw_file_utils.write_raw_csv(
                        year=year, month=month, df=df,
                        dataset_type='volumenes_omie',
                        mercado='intra'
                    )
                else:
                    self.raw_file_utils.write_raw_parquet(
                        year=year, month=month, df=df,
                        dataset_type='volumenes_omie',
                        mercado='intra'
                    )
                
                # Update status
                status_details["markets_downloaded"].append({
                    "market": f"intra_{session}",
                    "day": day.strftime('%Y-%m-%d')
                })
            
            # Clean up session file
            if os.path.exists(session_path):
                os.remove(session_path)
            
            return status_details
        
        except Exception as e:
            status_details["markets_failed"].append({
                "market": f"intra_{session}",
                "error": str(e),
                "day": day.strftime('%Y-%m-%d')
            })
            
            # Clean up session file even if there was an error
            if 'session_path' in locals() and session_path and os.path.exists(session_path):
                os.remove(session_path)
            
            raise

    def _extract_continuo_data(self, zip_file: str, day: datetime, 
                               unidades: List[str], dict_unidades: Dict[str, int], 
                               filtered_transition_dates: Dict, status_details: Dict) -> Dict:
        """
        Extract continuous market data for a specific day.
        
        Args:
            zip_file (str): Path to the zip file
            day (datetime): The day to extract data for
            unidades (List[str]): List of programming units
            dict_unidades (Dict[str, int]): Mapping of programming unit names to IDs
            filtered_transition_dates: Dictionary of transition dates for hour adjustments
            status_details (Dict): Status tracking dictionary
            
        Returns:
            Dict: Updated status details
        """
        try:
            # Extract day file
            day_path = self.continuo_downloader.extract_day_file(
                zip_file, day.year, day.month, day.day
            )
            
            if not day_path:
                print(f"No continuous market data found for day {day.date()}")
                return status_details
            
            # Get data from day file
            df = self.continuo_downloader.get_continuo_data(day_path)
            
            if not df.empty:
                # Add ID column for raw storage
                df['id_mercado'] = 21  # ID for continuous market
                
                # Save raw file
                year = day.year
                month = day.month
                
                # Use appropriate method based on environment
                dev, prod = self.env_utils.check_dev_env()
                if dev and not prod:
                    self.raw_file_utils.write_raw_csv(
                        year=year, month=month, df=df,
                        dataset_type='precios_omie',
                        mercado='continuo'
                    )
                else:
                    self.raw_file_utils.write_raw_parquet(
                        year=year, month=month, df=df,
                        dataset_type='precios_omie',
                        mercado='continuo'
                    )
                
                # Update status
                status_details["markets_downloaded"].append({
                    "market": "continuo",
                    "day": day.strftime('%Y-%m-%d')
                })
            
            # Clean up day file
            if os.path.exists(day_path):
                os.remove(day_path)
            
            return status_details
        
        except Exception as e:
            status_details["markets_failed"].append({
                "market": "continuo",
                "error": str(e),
                "day": day.strftime('%Y-%m-%d')
            })
            
            # Clean up day file even if there was an error
            if 'day_path' in locals() and day_path and os.path.exists(day_path):
                os.remove(day_path)
            
            raise