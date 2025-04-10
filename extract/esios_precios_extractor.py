from typing import Dict, List, Optional, Union
import pandas as pd
from datetime import datetime
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from extract.descargador_esios import Diario, Intra, Secundaria, Terciaria, RR
from utilidades.parquet_utils import RawFileUtils

class ESIOSPreciosExtractor:
    """
    Wrapper class for extracting price data from ESIOS API.
    Provides a unified interface for extracting data from different markets.
    """
    
    def __init__(self):
        """Initialize market extractors"""
        self.diario = Diario()
        self.intra = Intra()
        self.secundaria = Secundaria()
        self.terciaria = Terciaria()
        self.rr = RR()
        self.raw_file_utils = RawFileUtils()

        # Set the maximum download window (in days)
        self.download_window = 93  # ESIOS API typically limits requests to 3 months

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

        #check if fecha inicio < fecha fin, and if time range is valid
        if fecha_inicio_carga and fecha_fin_carga:
            fecha_inicio_carga_dt = datetime.strptime(fecha_inicio_carga, '%Y-%m-%d')
            fecha_fin_carga_dt = datetime.strptime(fecha_fin_carga, '%Y-%m-%d')

            #if fecha inicio > fecha fin, raise error
            if fecha_inicio_carga_dt > fecha_fin_carga_dt:
                raise ValueError("La fecha de inicio de carga no puede ser mayor que la fecha de fin de carga")
            
            #if there are more than 93 days between fecha inicio y fecha fin, raise error
            elif (fecha_fin_carga_dt - fecha_inicio_carga_dt).days > self.download_window: #93 days is the max allowed or ESIOS can return errors
                raise ValueError("El rango de fechas no puede ser mayor que tres meses")

            #if fecha inicio y fecha fin are valid, print message
            else:
                print(f"Descargando datos entre {fecha_inicio_carga} y {fecha_fin_carga}")

        #if no fecha inicio y fecha fin, set default values
        elif fecha_inicio_carga is None and fecha_fin_carga is None:

            #get datetitme range for 93 days ago to 92 days from now
            fecha_inicio_carga_dt = datetime.now() - timedelta(days=self.download_window) # 93 days ago
            fecha_fin_carga_dt = datetime.now() - timedelta(days=self.download_window) + timedelta(days=1) # 92 days from now
            
            #convert to string format
            fecha_inicio_carga = fecha_inicio_carga_dt.strftime('%Y-%m-%d') 
            fecha_fin_carga = fecha_fin_carga_dt.strftime('%Y-%m-%d')
            print(f"No se han proporcionado fechas de carga, se descargarán datos entre {fecha_inicio_carga} y {fecha_fin_carga}")

        else:
            raise ValueError("No se han proporcionado fechas de carga completas")

        return fecha_inicio_carga, fecha_fin_carga

    def extract_diario(self, fecha_inicio_carga: Optional[str] = None, fecha_fin_carga: Optional[str] = None) -> pd.DataFrame:
        """
        Extract daily market prices from ESIOS.
        
        Args:
            fecha_inicio (Optional[str]): Start date in YYYY-MM-DD format, default None is 93 days ago
            fecha_fin (Optional[str]): End date in YYYY-MM-DD format, default None is 92 days from now
            
        Returns:
            pd.DataFrame: DataFrame with daily market prices containing the following cols of interest:
                - datetime_utc: datetime in utc
                - value: price
                - granularidad: "Hora" or "Quinceminutos"
        """
        
        #validate input dates
        fecha_inicio_carga, fecha_fin_carga = self.fecha_input_validation(fecha_inicio_carga, fecha_fin_carga)

        #convert to datetime objects
        fecha_inicio_carga_dt = datetime.strptime(fecha_inicio_carga, '%Y-%m-%d')
        fecha_fin_carga_dt = datetime.strptime(fecha_fin_carga, '%Y-%m-%d')

        #download data for each day in the range
        for day in pd.date_range(start=fecha_inicio_carga_dt, end=fecha_fin_carga_dt, ):

            #extract year and month from date
            year = day.year
            month = day.month

            df = self.diario.get_prices(fecha_inicio_carga=day.strftime('%Y-%m-%d'), 
                                    fecha_fin_carga=day.strftime('%Y-%m-%d'))
            
            self.raw_file_utils.write_raw_csv(year=year, month=month, df=df, dataset_type='precios', mercado='diario')

        return 

    def extract_intra(self, 
                     fecha_inicio: str, 
                     fecha_fin: str, 
                     intra_lst: List[int]) -> pd.DataFrame:
        """
        Extract intraday market prices from ESIOS.
        
        Args:
            fecha_inicio (str): Start date in YYYY-MM-DD format
            fecha_fin (str): End date in YYYY-MM-DD format
            intra_lst (List[int]): List of intraday market IDs (1-7)
            
        Returns:
            pd.DataFrame: DataFrame with intraday market prices
        
        Note:
            After 2024-06-13, only Intra 1-3 are available due to regulatory changes
        """
        return self.intra.get_prices(fecha_inicio_carga=fecha_inicio,
                                   fecha_fin_carga=fecha_fin,
                                   intra_lst=intra_lst)

    def extract_secundaria(self, 
                         fecha_inicio: str, 
                         fecha_fin: str, 
                         secundaria_lst: List[int]) -> pd.DataFrame:
        """
        Extract secondary regulation prices from ESIOS.
        
        Args:
            fecha_inicio (str): Start date in YYYY-MM-DD format
            fecha_fin (str): End date in YYYY-MM-DD format
            secundaria_lst (List[int]): List of secondary regulation types [1: up, 2: down]
            
        Returns:
            pd.DataFrame: DataFrame with secondary regulation prices
            
        Note:
            After 2024-11-20, prices are split into up/down regulation
        """
        return self.secundaria.get_prices(fecha_inicio_carga=fecha_inicio,
                                        fecha_fin_carga=fecha_fin,
                                        secundaria_lst=secundaria_lst)

    def extract_terciaria(self, 
                         fecha_inicio: str, 
                         fecha_fin: str, 
                         terciaria_lst: List[int]) -> pd.DataFrame:
        """
        Extract tertiary regulation prices from ESIOS.
        
        Args:
            fecha_inicio (str): Start date in YYYY-MM-DD format
            fecha_fin (str): End date in 
            YYYY-MM-DD format
            terciaria_lst (List[int]): List of tertiary types 
                [1: up, 2: down, 3: direct up, 4: direct down, 5: programmed single]
            
        Returns:
            pd.DataFrame: DataFrame with tertiary regulation prices
            
        Note:
            After 2024-12-10, programmed tertiary uses single price (type 5)
        """
        return self.terciaria.get_prices(fecha_inicio_carga=fecha_inicio,
                                       fecha_fin_carga=fecha_fin,
                                       terciaria_lst=terciaria_lst)

    def extract_rr(self, fecha_inicio: str, fecha_fin: str) -> pd.DataFrame:
        """
        Extract Replacement Reserve (RR) prices from ESIOS.
        
        Args:
            fecha_inicio (str): Start date in YYYY-MM-DD format
            fecha_fin (str): End date in YYYY-MM-DD format
            
        Returns:
            pd.DataFrame: DataFrame with RR prices
        """
        return self.rr.get_rr_data(fecha_inicio_carga=fecha_inicio,
                                  fecha_fin_carga=fecha_fin)

