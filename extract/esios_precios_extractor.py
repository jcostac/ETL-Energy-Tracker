from typing import Dict, List, Optional, Union
import pandas as pd
from datetime import datetime
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from extract.descargador_esios import DiarioPreciosDL, IntraPreciosDL, SecundariaPreciosDL, TerciariaPreciosDL, RRPreciosDL
from utilidades.storage_file_utils import RawFileUtils

class ESIOSPreciosExtractor:
    """
    Wrapper class for extracting price data from ESIOS API.
    Provides a unified interface for extracting data from different markets.
    """
    
    def __init__(self):
        """Initialize market extractors and raw file utils"""
        self.diario = DiarioPreciosDL()
        self.intra = IntraPreciosDL()
        self.secundaria = SecundariaPreciosDL()
        self.terciaria = TerciariaPreciosDL()
        self.rr = RRPreciosDL()
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

    def extract_diario(self, fecha_inicio_carga: Optional[str] = None, fecha_fin_carga: Optional[str] = None, 
                       dev: bool = False) -> pd.DataFrame:
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
            
            if dev == True:
                self.raw_file_utils.write_raw_csv(year=year, month=month, df=df, dataset_type='precios', mercado='diario')
            else:
                self.raw_file_utils.write_raw_parquet(year=year, month=month, df=df, dataset_type='precios', mercado='diario')

        return 

    def extract_intra(self, fecha_inicio_carga: Optional[str] = None, fecha_fin_carga: Optional[str] = None, 
                      intra_lst: Optional[List[int]] = None, dev: bool = False) -> pd.DataFrame:
        """
        Extract intraday market prices from ESIOS.
        
        Args:
            fecha_inicio_carga (Optional[str]): Start date in YYYY-MM-DD format, default None is 93 days ago
            fecha_fin_carga (Optional[str]): End date in YYYY-MM-DD format, default None is 92 days from now
            intra_lst (Optional[List[int]]): List of intraday market IDs (1-7), default None is all available markets
            
        Returns:
            pd.DataFrame: DataFrame with intraday market prices
            
        Note:
            After 2024-06-13, only Intra 1-3 are available due to regulatory changes
        """
        
        # Validate input dates
        fecha_inicio_carga, fecha_fin_carga = self.fecha_input_validation(fecha_inicio_carga, fecha_fin_carga)

        # Set default intra_lst if None
        if intra_lst is None:
            intra_lst = list(range(1, 8))  # Default to all markets 1-7

        # Convert to datetime objects
        fecha_inicio_carga_dt = datetime.strptime(fecha_inicio_carga, '%Y-%m-%d')
        fecha_fin_carga_dt = datetime.strptime(fecha_fin_carga, '%Y-%m-%d')

        # Download data for each day in the range
        for day in pd.date_range(start=fecha_inicio_carga_dt, end=fecha_fin_carga_dt):
            # Extract year and month from date
            year = day.year
            month = day.month

            # Get data for the day
            df = self.intra.get_prices(
                fecha_inicio_carga=day.strftime('%Y-%m-%d'),
                fecha_fin_carga=day.strftime('%Y-%m-%d'),
                intra_lst=intra_lst
            )

            if not df.empty:
                if dev == True:
                    self.raw_file_utils.write_raw_csv(year=year, month=month, df=df, dataset_type='precios', mercado='intra')
                else:
                    self.raw_file_utils.write_raw_parquet(year=year, month=month, df=df, dataset_type='precios', mercado='intra')
        return

    def extract_secundaria(self, fecha_inicio_carga: Optional[str] = None, fecha_fin_carga: Optional[str] = None, 
                           secundaria_lst: Optional[List[int]] = None, dev: bool = False) -> pd.DataFrame:
        """
        Extract secondary regulation prices from ESIOS.
        
        Args:
            fecha_inicio_carga (Optional[str]): Start date in YYYY-MM-DD format, default None is 93 days ago
            fecha_fin_carga (Optional[str]): End date in YYYY-MM-DD format, default None is 92 days from now
            secundaria_lst (Optional[List[int]]): List of secondary regulation types [1: up, 2: down], default None is both
            
        Returns:
            pd.DataFrame: DataFrame with secondary regulation prices
            
        Note:
            After 2024-11-20, prices are split into up/down regulation
        """
        
        # Validate input dates
        fecha_inicio_carga, fecha_fin_carga = self.fecha_input_validation(fecha_inicio_carga, fecha_fin_carga)

        # Set default secundaria_lst if None
        if secundaria_lst is None:
            secundaria_lst = [1, 2]  # Default to both up and down regulation

        # Convert to datetime objects
        fecha_inicio_carga_dt = datetime.strptime(fecha_inicio_carga, '%Y-%m-%d')
        fecha_fin_carga_dt = datetime.strptime(fecha_fin_carga, '%Y-%m-%d')

        # Download data for each day in the range
        for day in pd.date_range(start=fecha_inicio_carga_dt, end=fecha_fin_carga_dt):
            # Extract year and month from date
            year = day.year
            month = day.month

            # Get data for the day
            df = self.secundaria.get_prices(
                fecha_inicio_carga=day.strftime('%Y-%m-%d'),
                fecha_fin_carga=day.strftime('%Y-%m-%d'),
                secundaria_lst=secundaria_lst
            )

            if not df.empty:
                if dev == True:
                    self.raw_file_utils.write_raw_csv(year=year, month=month, df=df, dataset_type='precios', mercado='secundaria')
                else:
                    self.raw_file_utils.write_raw_parquet(year=year, month=month, df=df, dataset_type='precios', mercado='secundaria')

        return

    def extract_terciaria(self, fecha_inicio_carga: Optional[str] = None, fecha_fin_carga: Optional[str] = None, 
                          terciaria_lst: Optional[List[int]] = None, dev: bool = False) -> pd.DataFrame:
        """
        Extract tertiary regulation prices from ESIOS.
        
        Args:
            fecha_inicio_carga (Optional[str]): Start date in YYYY-MM-DD format, default None is 93 days ago
            fecha_fin_carga (Optional[str]): End date in YYYY-MM-DD format, default None is 92 days from now
            terciaria_lst (Optional[List[int]]): List of tertiary types, default None is all types
                [1: up, 2: down, 3: direct up, 4: direct down, 5: programmed single]
            
        Returns:
            pd.DataFrame: DataFrame with tertiary regulation prices
            
        Note:
            After 2024-12-10, programmed tertiary uses single price (type 5)
        """
        
        # Validate input dates
        fecha_inicio_carga, fecha_fin_carga = self.fecha_input_validation(fecha_inicio_carga, fecha_fin_carga)

        # Set default terciaria_lst if None
        if terciaria_lst is None:
            terciaria_lst = list(range(1, 6))  # Default to all types 1-5

        # Convert to datetime objects
        fecha_inicio_carga_dt = datetime.strptime(fecha_inicio_carga, '%Y-%m-%d')
        fecha_fin_carga_dt = datetime.strptime(fecha_fin_carga, '%Y-%m-%d')

        # Download data for each day in the range
        for day in pd.date_range(start=fecha_inicio_carga_dt, end=fecha_fin_carga_dt):
            # Extract year and month from date
            year = day.year
            month = day.month

            # Get data for the day
            df = self.terciaria.get_prices(
                fecha_inicio_carga=day.strftime('%Y-%m-%d'),
                fecha_fin_carga=day.strftime('%Y-%m-%d'),
                terciaria_lst=terciaria_lst
            )

            if not df.empty:
                if dev == True:
                    self.raw_file_utils.write_raw_csv(year=year, month=month, df=df, dataset_type='precios', mercado='terciaria')
                else:
                    self.raw_file_utils.write_raw_parquet(year=year, month=month, df=df, dataset_type='precios', mercado='terciaria')

        return

    def extract_rr(self, fecha_inicio_carga: Optional[str] = None, fecha_fin_carga: Optional[str] = None, dev: bool = False) -> pd.DataFrame:
        """
        Extract Replacement Reserve (RR) prices from ESIOS.
        
        Args:
            fecha_inicio_carga (Optional[str]): Start date in YYYY-MM-DD format, default None is 93 days ago
            fecha_fin_carga (Optional[str]): End date in YYYY-MM-DD format, default None is 92 days from now
            
        Returns:
            pd.DataFrame: DataFrame with RR prices
            
        Note:
            RR uses a single price for both up and down regulation
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
            df = self.rr.get_prices(
                fecha_inicio_carga=day.strftime('%Y-%m-%d'),
                fecha_fin_carga=day.strftime('%Y-%m-%d')
            )

            if not df.empty:
                if dev == True:
                    self.raw_file_utils.write_raw_csv(year=year, month=month, df=df, dataset_type='precios', mercado='rr')
                else:
                    self.raw_file_utils.write_raw_parquet(year=year, month=month, df=df, dataset_type='precios', mercado='rr')

        return


   