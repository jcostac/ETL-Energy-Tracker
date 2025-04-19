from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
from datetime import datetime, timedelta
import sys
import os
import zipfile
import time
import requests

# Add the project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from extract.descargador_i90 import I90DownloaderDL, DiarioDL, TerciariaDL, SecundariaDL, RRDL, CurtailmentDL, P48DL, IndisponibilidadesDL, RestriccionesDL
from utilidades.storage_file_utils import RawFileUtils
from utilidades.db_utils import DatabaseUtils

class I90VolumenesExtractor:
    """
    Wrapper class for extracting volume data from I90 files via ESIOS API.
    Provides a unified interface for downloading and processing I90 Excel files.
    """
    
    def __init__(self):
        """Initialize the I90 downloader and raw file utils"""

        #downloaders
        self.i90_downloader = I90DownloaderDL()
        self.diario_downloader = DiarioDL()
        self.terciaria_downloader = TerciariaDL()
        self.secundaria_downloader = SecundariaDL()
        self.rr_downloader = RRDL()
        self.p48_downloader = P48DL()
        self.curtailment_downloader = CurtailmentDL()
        self.indisponibilidades_downloader = IndisponibilidadesDL()
        self.restricciones_downloader = RestriccionesDL()

        #utils 
        self.raw_file_utils = RawFileUtils()
        self.bbdd_engine = DatabaseUtils.create_engine('pruebas_BT')
        
        # Set the maximum download window (in days)
        self.download_window = 93  # ESIOS API typically limits requests to 3 months

        #latest i90 data for the day
        self.latest_i90_excel_file_name = None
        self.latest_i90_zip_file_name = None
        self.latest_i90_pestañas_con_error = None


    def fecha_input_validation(self, fecha_inicio_carga: str, fecha_fin_carga: str) -> tuple[str, str]:
        """
        Validates the input date range for ESIOS API requests.
        
        This method checks if the provided date range is valid according to ESIOS API limitations.
        If no dates are provided, it sets default values. The method ensures that:
        1. Start date is not greater than end date
        2. Date range does not exceed the maximum allowed window (typically 93 days)
        
        Args:
            fecha_inicio_carga (str): Start date in 'YYYY-MM-DD' format
            fecha_fin_carga (str): End date in 'YYYY-MM-DD' format
            
        Returns:
            tuple[str, str]: Validated start and end dates in 'YYYY-MM-DD' format
            
        Raises:
            ValueError: If date range is invalid or incomplete
        """
        # Check if fecha inicio < fecha fin, and if time range is valid
        if fecha_inicio_carga and fecha_fin_carga:
            fecha_inicio_carga_dt = datetime.strptime(fecha_inicio_carga, '%Y-%m-%d')
            fecha_fin_carga_dt = datetime.strptime(fecha_fin_carga, '%Y-%m-%d')

            # If fecha inicio > fecha fin, raise error
            if fecha_inicio_carga_dt > fecha_fin_carga_dt:
                raise ValueError("La fecha de inicio de carga no puede ser mayor que la fecha de fin de carga")

            # If fecha inicio y fecha fin are valid, print message
            else:
                print(f"Descargando datos entre {fecha_inicio_carga} y {fecha_fin_carga}")

        # If no fecha inicio y fecha fin, set default values
        elif fecha_inicio_carga is None and fecha_fin_carga is None:
            # Get datetime range for 93 days ago to 92 days from now
            fecha_inicio_carga_dt = datetime.now() - timedelta(days=self.download_window) 
            fecha_fin_carga_dt = datetime.now() - timedelta(days=self.download_window) + timedelta(days=1)
            
            # Convert to string format
            fecha_inicio_carga = fecha_inicio_carga_dt.strftime('%Y-%m-%d') 
            fecha_fin_carga = fecha_fin_carga_dt.strftime('%Y-%m-%d')
            print(f"No se han proporcionado fechas de carga, se descargarán datos entre {fecha_inicio_carga} y {fecha_fin_carga}")

        else:
            raise ValueError("No se han proporcionado fechas de carga completas")

        return fecha_inicio_carga, fecha_fin_carga

    def extract_i90_data(self, day: datetime) -> None:
        """
        Downloads the I90 file for a specific day using the downloader.

        Args:
            day (datetime): The specific day to download the I90 file for.

        Returns:
            Tuple[str, str, List[str]]: A tuple containing:
                - zip_file_name (str): The name of the downloaded zip file.
                - excel_file_name (str): The name of the extracted Excel file.
                - pestañas_con_error (List[str]): A list of sheet IDs that have errors for the given day.
        """
        # Assuming self.downloader is an instance of I90DownloaderDL or a subclass
        # initialized in the class constructor (__init__)
        if not hasattr(self, 'i90_downloader'):
             # Initialize a generic downloader if not present, or raise an error
             # For now, let's assume it exists. If not, this needs adjustment based on class structure.
             raise AttributeError("Downloader instance 'self.i90_downloader' not found.")

        zip_file_name, excel_file_name, pestañas_con_error = self.i90_downloader.get_i90_data(day)
        print(f"Downloaded I90 data for {day.date()}: Zip='{zip_file_name}', Excel='{excel_file_name}', Errors='{pestañas_con_error}'")

        #update latest i90 data
        self.latest_i90_zip_file_name = zip_file_name
        self.latest_i90_excel_file_name = excel_file_name
        self.latest_i90_pestañas_con_error = pestañas_con_error

        return

    def extract_volumenes_terciaria(self, fecha_inicio_carga: Optional[str] = None, fecha_fin_carga: Optional[str] = None, 
                                  UP_ids: Optional[List[int]] = None, sentidos: Optional[List[str]] = None, 
                                  dev: bool = False) -> None:
        """
        Extract tertiary regulation volume data from I90 files.
        
        Args:
            fecha_inicio_carga (Optional[str]): Start date in YYYY-MM-DD format
            fecha_fin_carga (Optional[str]): End date in YYYY-MM-DD format
            UP_ids (Optional[List[int]]): List of Programming Unit IDs to filter data for
            sentidos (Optional[List[str]]): Direction filters - ['Subir', 'Bajar', 'Directa'] 
            dev (bool): If True, save to development files/tables instead of production
        """
        # Validate input dates
        fecha_inicio_carga, fecha_fin_carga = self.fecha_input_validation(fecha_inicio_carga, fecha_fin_carga)

        # Convert to datetime objects
        fecha_inicio_carga_dt = datetime.strptime(fecha_inicio_carga, '%Y-%m-%d')
        fecha_fin_carga_dt = datetime.strptime(fecha_fin_carga, '%Y-%m-%d')
        
        # Set default sentidos if None
        if sentidos is None:
            sentidos = ['Subir', 'Bajar', 'Directa']
        
        # Download data for each day in the range
        for day in pd.date_range(start=fecha_inicio_carga_dt, end=fecha_fin_carga_dt):
            # Extract year and month from date
            year = day.year
            month = day.month
            
            # Get data for the day
            dfs = self.terciaria_downloader.get_volumenes(
                day=day,
                bbdd_engine=self.bbdd_engine,
                UP_ids=UP_ids,
                sentidos=sentidos
            )
            
            # Save data to files
            for df, sentido in dfs:
                if not df.empty:
                    if dev:
                        self.raw_file_utils.write_raw_csv(
                            year=year, month=month, df=df, 
                            dataset_type='volumenes', 
                            mercado=f'terciaria_{sentido.lower()}'
                        )
                    else:
                        self.raw_file_utils.write_raw_parquet(
                            year=year, month=month, df=df, 
                            dataset_type='volumenes', 
                            mercado=f'terciaria_{sentido.lower()}'
                        )
        
        return

    def extract_volumenes_secundaria(self, fecha_inicio_carga: Optional[str] = None, fecha_fin_carga: Optional[str] = None, 
                                   UP_ids: Optional[List[int]] = None, sentidos: Optional[List[str]] = None, 
                                   dev: bool = False) -> None:
        """
        Extract secondary regulation volume data from I90 files.
        
        Args:
            fecha_inicio_carga (Optional[str]): Start date in YYYY-MM-DD format
            fecha_fin_carga (Optional[str]): End date in YYYY-MM-DD format
            UP_ids (Optional[List[int]]): List of Programming Unit IDs to filter data for
            sentidos (Optional[List[str]]): Direction filters - ['Subir', 'Bajar']
            dev (bool): If True, save to development files/tables instead of production
        """
        # Validate input dates
        fecha_inicio_carga, fecha_fin_carga = self.fecha_input_validation(fecha_inicio_carga, fecha_fin_carga)

        # Convert to datetime objects
        fecha_inicio_carga_dt = datetime.strptime(fecha_inicio_carga, '%Y-%m-%d')
        fecha_fin_carga_dt = datetime.strptime(fecha_fin_carga, '%Y-%m-%d')
        
        # Set default sentidos if None
        if sentidos is None:
            sentidos = ['Subir', 'Bajar']
        
        # Download data for each day in the range
        for day in pd.date_range(start=fecha_inicio_carga_dt, end=fecha_fin_carga_dt):
            # Extract year and month from date
            year = day.year
            month = day.month
            
            # Get data for the day
            dfs = self.secundaria_downloader.get_volumenes(
                day=day,
                bbdd_engine=self.bbdd_engine,
                UP_ids=UP_ids,
                sentidos=sentidos
            )
            
            # Save data to files
            for df, sentido in dfs:
                if not df.empty:
                    if dev:
                        self.raw_file_utils.write_raw_csv(
                            year=year, month=month, df=df, 
                            dataset_type='volumenes', 
                            mercado=f'secundaria_{sentido.lower()}'
                        )
                    else:
                        self.raw_file_utils.write_raw_parquet(
                            year=year, month=month, df=df, 
                            dataset_type='volumenes', 
                            mercado=f'secundaria_{sentido.lower()}'
                        )
        
        return

    def extract_volumenes_rr(self, fecha_inicio_carga: Optional[str] = None, fecha_fin_carga: Optional[str] = None, 
                          UP_ids: Optional[List[int]] = None, dev: bool = False) -> None:
        """
        Extract Replacement Reserve (RR) volume data from I90 files.
        
        Args:
            fecha_inicio_carga (Optional[str]): Start date in YYYY-MM-DD format
            fecha_fin_carga (Optional[str]): End date in YYYY-MM-DD format
            UP_ids (Optional[List[int]]): List of Programming Unit IDs to filter data for
            dev (bool): If True, save to development files/tables instead of production
        """
        # Validate input dates
        fecha_inicio_carga, fecha_fin_carga = self.fecha_input_validation(fecha_inicio_carga, fecha_fin_carga)

        # Convert to datetime objects
        fecha_inicio_carga_dt = datetime.strptime(fecha_inicio_carga, '%Y-%m-%d')
        fecha_fin_carga_dt = datetime.strptime(fecha_fin_carga, '%Y-%m-%d')
        
        # Download data for each day in the range
        for day in pd.date_range(start=fecha_inicio_carga_dt, end=fecha_fin_carga_dt):
            # Extract year and month from date
            year = day.year
            month = day.month
            
            # Get data for the day
            df = self.rr_downloader.get_volumenes(
                day=day,
                bbdd_engine=self.bbdd_engine,
                UP_ids=UP_ids
            )
            
            # Save data to files
            if not df.empty:
                if dev:
                    self.raw_file_utils.write_raw_csv(
                        year=year, month=month, df=df, 
                        dataset_type='volumenes', 
                        mercado='rr'
                    )
                else:
                    self.raw_file_utils.write_raw_parquet(
                        year=year, month=month, df=df, 
                        dataset_type='volumenes', 
                        mercado='rr'
                    )
        
        return

    def extract_volumenes_curtailment(self, fecha_inicio_carga: Optional[str] = None, fecha_fin_carga: Optional[str] = None, 
                                    UP_ids: Optional[List[int]] = None, tipos: Optional[List[str]] = None, 
                                    dev: bool = False) -> None:
        """
        Extract curtailment volume data from I90 files.
        
        Args:
            fecha_inicio_carga (Optional[str]): Start date in YYYY-MM-DD format
            fecha_fin_carga (Optional[str]): End date in YYYY-MM-DD format
            UP_ids (Optional[List[int]]): List of Programming Unit IDs to filter data for
            tipos (Optional[List[str]]): Curtailment types - ['Curtailment', 'Curtailment demanda']
            dev (bool): If True, save to development files/tables instead of production
        """
        # Validate input dates
        fecha_inicio_carga, fecha_fin_carga = self.fecha_input_validation(fecha_inicio_carga, fecha_fin_carga)

        # Convert to datetime objects
        fecha_inicio_carga_dt = datetime.strptime(fecha_inicio_carga, '%Y-%m-%d')
        fecha_fin_carga_dt = datetime.strptime(fecha_fin_carga, '%Y-%m-%d')
        
        # Set default tipos if None
        if tipos is None:
            tipos = ['Curtailment', 'Curtailment demanda']
        
        # Download data for each day in the range
        for day in pd.date_range(start=fecha_inicio_carga_dt, end=fecha_fin_carga_dt):
            # Extract year and month from date
            year = day.year
            month = day.month
            
            # Get data for the day
            dfs = self.curtailment_downloader.get_volumenes(
                day=day,
                bbdd_engine=self.bbdd_engine,
                UP_ids=UP_ids,
                tipos=tipos
            )
            
            # Save data to files
            for df, tipo in dfs:
                if not df.empty:
                    if dev:
                        self.raw_file_utils.write_raw_csv(
                            year=year, month=month, df=df, 
                            dataset_type='volumenes', 
                            mercado=tipo.lower().replace(' ', '_')
                        )
                    else:
                        self.raw_file_utils.write_raw_parquet(
                            year=year, month=month, df=df, 
                            dataset_type='volumenes', 
                            mercado=tipo.lower().replace(' ', '_')
                        )
        
        return
