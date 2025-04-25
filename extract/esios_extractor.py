from typing import Dict, List, Optional, Union
import pandas as pd
from datetime import datetime, timedelta
import sys
import os
import time
# Add the project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from extract._descargador_esios import DiarioPreciosDL, IntraPreciosDL, SecundariaPreciosDL, TerciariaPreciosDL, RRPreciosDL
from utilidades.storage_file_utils import RawFileUtils
from utilidades.env_utils import EnvUtils
class ESIOSPreciosExtractor:
    """
    Wrapper class for extracting price data from ESIOS API.
    Provides a unified interface for extracting data from different markets.
    """
    
    def __init__(self):
        """Initialize market extractors and raw file utils"""

        #initialize market extractors
        self.diario = DiarioPreciosDL()
        self.intra = IntraPreciosDL()
        self.secundaria = SecundariaPreciosDL()
        self.terciaria = TerciariaPreciosDL()
        self.rr = RRPreciosDL()

        #initialize raw file utils
        self.raw_file_utils = RawFileUtils()

        #initialize env utils
        env_utils = EnvUtils()
        self.dev, self.prod = env_utils.check_dev_env()

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
            fecha_fin_carga_dt = datetime.now() - timedelta(days=self.download_window) + timedelta(days=1) # yesterday to avoid issues with today's data
            
            #convert to string format
            fecha_inicio_carga = fecha_inicio_carga_dt.strftime('%Y-%m-%d') 
            fecha_fin_carga = fecha_fin_carga_dt.strftime('%Y-%m-%d')
            print(f"No se han proporcionado fechas de carga, se descargarán datos entre {fecha_inicio_carga} y {fecha_fin_carga}")

        else:
            raise ValueError("No se han proporcionado fechas de carga completas")

        return fecha_inicio_carga, fecha_fin_carga

    def _extract_and_save_prices(self, fecha_inicio_carga: Optional[str], fecha_fin_carga: Optional[str],
                                 mercado: str, downloader, **kwargs) -> None:
        """
        Helper method to extract and save price data for a given market.

        Args:
            fecha_inicio_carga (Optional[str]): Start date in YYYY-MM-DD format.
            fecha_fin_carga (Optional[str]): End date in YYYY-MM-DD format.
            mercado (str): Name of the market (e.g., 'diario', 'intra').
            downloader: The downloader instance for the specific market.
            **kwargs: Additional arguments specific to the market downloader's get_prices method.
        """
        # Validate input dates
        fecha_inicio_carga, fecha_fin_carga = self.fecha_input_validation(fecha_inicio_carga, fecha_fin_carga)

        # Convert to datetime objects
        fecha_inicio_carga_dt = datetime.strptime(fecha_inicio_carga, '%Y-%m-%d')
        fecha_fin_carga_dt = datetime.strptime(fecha_fin_carga, '%Y-%m-%d')

        print(f"Processing market: {mercado}")
        # Download data for each day in the range
        for day in pd.date_range(start=fecha_inicio_carga_dt, end=fecha_fin_carga_dt):
            # Extract year and month from date
            year = day.year
            month = day.month
            day_str = day.strftime('%Y-%m-%d')

            try:
                # Get data for the day using the specific downloader
                df = downloader.get_prices(
                    fecha_inicio_carga=day_str,
                    fecha_fin_carga=day_str,
                    **kwargs # Pass any additional market-specific arguments (for terciaria, secundaria, intras)
                )


                #sleep for 1 second to avoid rate limit
                time.sleep(1)

                if df is not None and not df.empty:
                    # Determine file format based on self.dev flag
                    if self.dev and not self.prod:
                        status = "DEVELOPMEMT "
                        self.raw_file_utils.write_raw_csv(
                            year=year, month=month, df=df,
                            dataset_type='precios', mercado=mercado
                        )
                    else:
                        status = "PRODUCTION "
                        self.raw_file_utils.write_raw_parquet(
                            year=year, month=month, df=df,
                            dataset_type='precios', mercado=mercado
                        )
                    print(f"  - Successfully processed and saved {status}{mercado} prices for {day_str}")
                else:
                    print(f"  - No {mercado} price data found or extracted for {day_str}")

            except Exception as e:
                print(f"  - Error downloading {mercado} prices for {day_str}: {e}")

    def extract_data_for_all_markets(self, fecha_inicio_carga: Optional[str] = None, fecha_fin_carga: Optional[str] = None) -> None:
        """
        Extract data for all relevant markets from ESIOS API for a given date range.
        Uses the environment setting (DEV/PROD) for file saving format.
        """
        try:
            self.extract_diario(fecha_inicio_carga, fecha_fin_carga)
            self.extract_intra(fecha_inicio_carga, fecha_fin_carga)
            self.extract_secundaria(fecha_inicio_carga, fecha_fin_carga)
            self.extract_terciaria(fecha_inicio_carga, fecha_fin_carga)
            self.extract_rr(fecha_inicio_carga, fecha_fin_carga)

        except Exception as e:
            print(f"Error extracting data: {e}")

    def extract_diario(self, fecha_inicio_carga: Optional[str] = None, fecha_fin_carga: Optional[str] = None) -> None:
        """
        Extract daily market prices from ESIOS.
        Uses the environment setting (DEV/PROD) for file saving format.
        
        Args:
            fecha_inicio_carga (Optional[str]): Start date in YYYY-MM-DD format, default None is 93 days ago
            fecha_fin_carga (Optional[str]): End date in YYYY-MM-DD format, default None uses logic in fecha_input_validation
            
        Returns:
            None
        """
        self._extract_and_save_prices(
            fecha_inicio_carga=fecha_inicio_carga,
            fecha_fin_carga=fecha_fin_carga,
            mercado='diario',
            downloader=self.diario
        )
        return

    def extract_intra(self, fecha_inicio_carga: Optional[str] = None, fecha_fin_carga: Optional[str] = None, 
                      intra_lst: Optional[List[int]] = None) -> None:
        """
        Extract intraday market prices from ESIOS.
        Uses the environment setting (DEV/PROD) for file saving format.
        
        Args:
            fecha_inicio_carga (Optional[str]): Start date in YYYY-MM-DD format, default None is 93 days ago
            fecha_fin_carga (Optional[str]): End date in YYYY-MM-DD format, default None uses logic in fecha_input_validation
            intra_lst (Optional[List[int]]): List of intraday market IDs (1-7), default None is all available markets
            
        Returns:
            None
            
        Note:
            After 2024-06-13, only Intra 1-3 are available due to regulatory changes
        """
        # Set default intra_lst if None
        if intra_lst is None:
            intra_lst = list(range(1, 8))  # Default to all markets 1-7

        self._extract_and_save_prices(
            fecha_inicio_carga=fecha_inicio_carga,
            fecha_fin_carga=fecha_fin_carga,
            mercado='intra',
            downloader=self.intra,
            intra_lst=intra_lst
        )
        return

    def extract_secundaria(self, fecha_inicio_carga: Optional[str] = None, fecha_fin_carga: Optional[str] = None, 
                           secundaria_lst: Optional[List[int]] = None) -> None:
        """
        Extract secondary regulation prices from ESIOS.
        Uses the environment setting (DEV/PROD) for file saving format.
        
        Args:
            fecha_inicio_carga (Optional[str]): Start date in YYYY-MM-DD format, default None is 93 days ago
            fecha_fin_carga (Optional[str]): End date in YYYY-MM-DD format, default None uses logic in fecha_input_validation
            secundaria_lst (Optional[List[int]]): List of secondary regulation types [1: up, 2: down], default None is both
            
        Returns:
            None
            
        Note:
            After 2024-11-20, prices are split into up/down regulation
        """
        # Set default secundaria_lst if None
        if secundaria_lst is None:
            secundaria_lst = [1, 2]  # Default to both up and down regulation

        self._extract_and_save_prices(
            fecha_inicio_carga=fecha_inicio_carga,
            fecha_fin_carga=fecha_fin_carga,
            mercado='secundaria',
            downloader=self.secundaria,
            secundaria_lst=secundaria_lst
        )
        return

    def extract_terciaria(self, fecha_inicio_carga: Optional[str] = None, fecha_fin_carga: Optional[str] = None, 
                          terciaria_lst: Optional[List[int]] = None) -> None:
        """
        Extract tertiary regulation prices from ESIOS.
        Uses the environment setting (DEV/PROD) for file saving format.
        
        Args:
            fecha_inicio_carga (Optional[str]): Start date in YYYY-MM-DD format, default None is 93 days ago
            fecha_fin_carga (Optional[str]): End date in YYYY-MM-DD format, default None uses logic in fecha_input_validation
            terciaria_lst (Optional[List[int]]): List of tertiary types, default None is all types
                [1: up, 2: down, 3: direct up, 4: direct down, 5: programmed single]
            
        Returns:
            None
            
        Note:
            After 2024-12-10, programmed tertiary uses single price (type 5)
        """
        # Set default terciaria_lst if None
        if terciaria_lst is None:
            terciaria_lst = list(range(1, 6))  # Default to all types 1-5

        self._extract_and_save_prices(
            fecha_inicio_carga=fecha_inicio_carga,
            fecha_fin_carga=fecha_fin_carga,
            mercado='terciaria',
            downloader=self.terciaria,
            terciaria_lst=terciaria_lst
        )
        return

    def extract_rr(self, fecha_inicio_carga: Optional[str] = None, fecha_fin_carga: Optional[str] = None) -> None:
        """
        Extract Replacement Reserve (RR) prices from ESIOS.
        Uses the environment setting (DEV/PROD) for file saving format.
        
        Args:
            fecha_inicio_carga (Optional[str]): Start date in YYYY-MM-DD format, default None is 93 days ago
            fecha_fin_carga (Optional[str]): End date in YYYY-MM-DD format, default None uses logic in fecha_input_validation
            
        Returns:
            None
            
        Note:
            RR uses a single price for both up and down regulation
        """
        self._extract_and_save_prices(
            fecha_inicio_carga=fecha_inicio_carga,
            fecha_fin_carga=fecha_fin_carga,
            mercado='rr',
            downloader=self.rr
        )
        return


   