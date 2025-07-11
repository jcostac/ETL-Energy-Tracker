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
        """
        Initialize the ESIOSPreciosExtractor with market-specific downloaders, file utilities, and configuration.
        
        Instantiates downloader objects for each electricity market segment, sets up utilities for raw file handling, and defines the maximum allowed download window for API requests.
        """

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

        # Set the maximum download window (in days)
        self.download_window = 93  # ESIOS API typically limits requests to 3 months

    def fecha_input_validation(self, fecha_inicio_carga: str, fecha_fin_carga: str) -> tuple[str, str]:
        """
        Validate and normalize the input date range for ESIOS API data extraction.
        
        Ensures that both start and end dates are provided and that the start date is not after the end date. If no dates are given, defaults to a window ending yesterday. Raises a ValueError if the input is incomplete or invalid.
        
        Returns:
            tuple[str, str]: Validated start and end dates in 'YYYY-MM-DD' format.
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
            fecha_fin_carga_dt = fecha_inicio_carga_dt  # yesterday to avoid issues with today's data
            
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
                                 Extracts and saves price data for a specified market segment over a given date range.
                                 
                                 For each day in the validated date range, retrieves price data using the provided downloader and saves it in CSV or Parquet format depending on the environment. Handles missing data and logs errors per day without interrupting the overall extraction process.
                                 
                                 Parameters:
                                     fecha_inicio_carga (Optional[str]): Start date in 'YYYY-MM-DD' format.
                                     fecha_fin_carga (Optional[str]): End date in 'YYYY-MM-DD' format.
                                     mercado (str): Market segment name (e.g., 'diario', 'intra').
                                     **kwargs: Additional arguments passed to the downloader's get_prices method.
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
                        environment = "DEVELOPMENT "
                        self.raw_file_utils.write_raw_csv(
                            year=year, month=month, df=df,
                            dataset_type="precios_esios", mercado=mercado
                        )
                    else:
                        environment = "PRODUCTION "
                        self.raw_file_utils.write_raw_parquet(
                            year=year, month=month, df=df,
                            dataset_type="precios_esios", mercado=mercado
                        )
                    print(f"✅ Successfully  saved raw {mercado} prices for {day_str} in {environment} environment")
                else:
                    print(f" ⚠️ No {mercado} price data found for {day_str}. Nothing was saved to raw folder.")

            except Exception as e:
                print(f"  ❌ Error downloading {mercado} prices for {day_str}: {e}")

    def extract_data_for_all_markets(self, fecha_inicio_carga: Optional[str] = None, fecha_fin_carga: Optional[str] = None):
        """
        Extracts and saves price data for all ESIOS electricity markets over a specified date range.
        
        Coordinates the extraction process for daily, intraday, secondary, tertiary, and replacement reserve markets, tracking the success or failure of each. Returns a summary dictionary with overall status and detailed results for integration with orchestration tools such as Airflow.
        
        Parameters:
            fecha_inicio_carga (Optional[str]): Start date for extraction in 'YYYY-MM-DD' format. If not provided, defaults to the maximum allowed window.
            fecha_fin_carga (Optional[str]): End date for extraction in 'YYYY-MM-DD' format. If not provided, defaults to the maximum allowed window.
        
        Returns:
            dict: Dictionary containing overall success status and detailed extraction results for each market.
        """
        if (fecha_fin_carga is None and fecha_inicio_carga is None) or fecha_fin_carga == fecha_inicio_carga:
            date_range_str = f"Single day download for {(datetime.now() - timedelta(days=self.download_window)).strftime('%Y-%m-%d')}"
        else:
            date_range_str = f"{fecha_inicio_carga} to {fecha_fin_carga}"

        # Initialize status trackingS
        status_details = {
            "markets_downloaded": [],
            "markets_failed": [],
            "date_range": date_range_str
        }

        
        try:
            # Track success for each market
            print ("\n--------- Diario ---------")
            success_diario = self._extract_with_status("diario", self.extract_diario, 
                                                     fecha_inicio_carga, fecha_fin_carga, status_details)
        
            print ("\n--------- Intra ---------")
            success_intra = self._extract_with_status("intra", self.extract_intra, 
                                                    fecha_inicio_carga, fecha_fin_carga, status_details)
            print ("\n--------- Secundaria ---------")
            success_secundaria = self._extract_with_status("secundaria", self.extract_secundaria, 
                                                         fecha_inicio_carga, fecha_fin_carga, status_details)
            print ("\n--------- Terciaria ---------")
            success_terciaria = self._extract_with_status("terciaria", self.extract_terciaria, 
                                                        fecha_inicio_carga, fecha_fin_carga, status_details)
            print ("\n--------- RR ---------")
            success_rr = self._extract_with_status("rr", self.extract_rr, 
                                                 fecha_inicio_carga, fecha_fin_carga, status_details)
            print ("\n--------------------------------")
            
            # Overall success only if all markets succeeded
            overall_success = (success_diario and success_intra and success_secundaria 
                              and success_terciaria and success_rr)
            
        except Exception as e:
            overall_success = False
            status_details["error"] = str(e)
        
        print("ℹ️ Data extraction pipeline finished.")
        
        # Return status for Airflow task
        return {"success": overall_success, "details": status_details}
        
    def _extract_with_status(self, market_name, extract_function, fecha_inicio_carga, fecha_fin_carga, status_details):
        """
        Runs a market extraction function and records its success or failure status.
        
        Parameters:
            market_name: Name of the market being extracted.
            extract_function: Function to call for extracting market data.
            fecha_inicio_carga: Start date for extraction.
            fecha_fin_carga: End date for extraction.
            status_details: Dictionary to update with extraction results.
        
        Returns:
            True if extraction succeeds; False if an exception occurs.
        """
        try:
            extract_function(fecha_inicio_carga, fecha_fin_carga)
            status_details["markets_downloaded"].append(market_name)
            return True
        except Exception as e:
            status_details["markets_failed"].append({
                "market": market_name,
                "error": str(e)
            })
            return False

    def extract_diario(self, fecha_inicio_carga: Optional[str] = None, fecha_fin_carga: Optional[str] = None) -> None:
        """
        Extracts daily electricity market prices from the ESIOS API for a specified date range and saves the results to file.
        
        Parameters:
            fecha_inicio_carga (Optional[str]): Start date in 'YYYY-MM-DD' format. If None, defaults to 93 days before yesterday.
            fecha_fin_carga (Optional[str]): End date in 'YYYY-MM-DD' format. If None, defaults to yesterday.
        
        This method validates the date range and delegates extraction and saving to the appropriate downloader.
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
        Extracts Replacement Reserve (RR) market prices from the ESIOS API for a specified date range.
        
        If no dates are provided, defaults are determined by internal validation logic. RR market uses a single price for both up and down regulation.
        """
        self._extract_and_save_prices(
            fecha_inicio_carga=fecha_inicio_carga,
            fecha_fin_carga=fecha_fin_carga,
            mercado='rr',
            downloader=self.rr
        )
        return


if __name__ == "__main__":
    #--example usage
    esios_extractor = ESIOSPreciosExtractor()
    esios_extractor.extract_data_for_all_markets(fecha_inicio_carga="2024-12-01", fecha_fin_carga="2025-02-01")




