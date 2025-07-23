from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
from datetime import datetime, timedelta
import sys
import os
import re
import time 

# Add the project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from extract._descargador_i90 import I90Downloader, DiarioDL, TerciariaDL, SecundariaDL, RRDL, CurtailmentDL, P48DL, IndisponibilidadesDL, RestriccionesDL, IntradiarioDL
from utilidades.raw_file_utils import RawFileUtils
from utilidades.db_utils import DatabaseUtils
from utilidades.env_utils import EnvUtils

class I90Extractor:
    """
    Wrapper class for extracting volume data from I90 files via ESIOS API.
    Provides a unified interface for downloading and processing I90 Excel files.
    """
    
    def __init__(self):
        """Initialize the I90 downloader and raw file utils"""

        #downloaders
        self.i90_downloader = I90Downloader()
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
        self.env_utils = EnvUtils()
        
        # Set the maximum download window (in days)
        self.download_window = 93  #i90 is published with 90 day delay (3 day buffer extra)

        #latest i90 data for the day
        self.latest_i90_excel_file_name = None
        self.latest_i90_zip_file_name = None
        self.latest_i90_pestañas_con_error = None

    def fecha_input_validation(self, fecha_inicio: str, fecha_fin: str) -> tuple[str, str]:
        """
        Validate and normalize the input date range for ESIOS API data extraction.
        
        Ensures that both start and end dates are provided and in the correct order, or defaults to a specific day if none are given. Raises a ValueError if the input is incomplete or invalid.
        
        Parameters:
            fecha_inicio (str): Start date in 'YYYY-MM-DD' format.
            fecha_fin (str): End date in 'YYYY-MM-DD' format.
        
        Returns:
            tuple[str, str]: Validated start and end dates in 'YYYY-MM-DD' format.
        
        Raises:
            ValueError: If the date range is incomplete or the start date is after the end date.
        """
        # Check if fecha inicio < fecha fin, and if time range is valid
        if fecha_inicio and fecha_fin:
            fecha_inicio_carga_dt = datetime.strptime(fecha_inicio, '%Y-%m-%d')
            fecha_fin_carga_dt = datetime.strptime(fecha_fin, '%Y-%m-%d')

            # If fecha inicio > fecha fin, raise error
            if fecha_inicio_carga_dt > fecha_fin_carga_dt:
                raise ValueError("La fecha de inicio de carga no puede ser mayor que la fecha de fin de carga")

            # If fecha inicio y fecha fin are valid, print message
            else:
                print(f"Descargando datos entre {fecha_inicio} y {fecha_fin}")

        # If no fecha inicio y fecha fin, set default values
        elif fecha_inicio is None and fecha_fin is None:
            # Get datetime range for 93 days ago 
            fecha_inicio_carga_dt = datetime.now() - timedelta(days=self.download_window) 
            fecha_fin_carga_dt = datetime.now() - timedelta(days=self.download_window)
            
            # Convert to string format
            fecha_inicio = fecha_inicio_carga_dt.strftime('%Y-%m-%d') 
            fecha_fin = fecha_fin_carga_dt.strftime('%Y-%m-%d')
            print(f"No se han proporcionado fechas de carga, se descargarán datos para el día {fecha_inicio}")

        else:
            raise ValueError("No se han proporcionado fechas de carga completas")

        return fecha_inicio, fecha_fin
    
    def download_i90_data(self, day: datetime) -> None:
        """
        Download the I90 zip and Excel files for a specific day and update the latest file attributes.
        
        Parameters:
            day (datetime): The date for which to download the I90 data.
        
        The method updates the object's attributes with the names of the downloaded zip and Excel files, as well as any sheet IDs with errors. It pauses briefly to avoid rate limiting.
        """
        # Assuming self.downloader is an instance of I90DownloaderDL or a subclass
        # initialized in the class constructor (__init__)
        if not hasattr(self, 'i90_downloader'):
             # Initialize a generic downloader if not present, or raise an error
             # For now, let's assume it exists. If not, this needs adjustment based on class structure.
             raise AttributeError("Downloader instance 'self.i90_downloader' not found.")

        zip_file_name, excel_file_name, pestañas_con_error = self.i90_downloader.download_i90_file(day)
        
        #sleep for 1 second to avoid rate limit
        time.sleep(1)
        print(f"Downloaded I90 data for {day.date()}: Zip='{zip_file_name}', Excel='{excel_file_name}', Errors='{pestañas_con_error}'")

        #update latest i90 data
        self.latest_i90_zip_file_name = zip_file_name
        self.latest_i90_excel_file_name = excel_file_name
        self.latest_i90_pestañas_con_error = pestañas_con_error

        return

    def validate_i90_correct_download(self):
        """
        Checks whether the latest I90 zip and Excel file attributes are present and correctly formatted.
        
        Returns:
            bool: True if all required attributes exist and have valid formats; False otherwise.
        """

        # Check for None values
        if self.latest_i90_zip_file_name is None:
            print("Validation failed: latest_i90_zip_file_name is None.")
            return False
        if self.latest_i90_excel_file_name is None:
            print("Validation failed: latest_i90_excel_file_name is None.")
            return False
        if self.latest_i90_pestañas_con_error is None:
            print("Validation failed: latest_i90_pestañas_con_error is None.")
            return False

        # Check types
        if not isinstance(self.latest_i90_zip_file_name, str):
            print(f"Validation failed: latest_i90_zip_file_name is not a string ({type(self.latest_i90_zip_file_name)}).")
            return False
        if not isinstance(self.latest_i90_excel_file_name, str):
            print(f"Validation failed: latest_i90_excel_file_name is not a string ({type(self.latest_i90_excel_file_name)}).")
            return False
        if not isinstance(self.latest_i90_pestañas_con_error, list):
            print(f"Validation failed: latest_i90_pestañas_con_error is not a list ({type(self.latest_i90_pestañas_con_error)}).")
            return False

        # Validate zip file name format (contains YYYY-MM-DD)
        # Zip filename format is 'I90DIA_YYYY-MM-DD.zip'
        zip_match = re.search(r'(\d{4}-\d{2}-\d{2})', self.latest_i90_zip_file_name)
        if not zip_match:
            print(f"Validation failed: Could not find YYYYMMDD date pattern in zip file name: {self.latest_i90_zip_file_name}")
            return False
        zip_date_str = zip_match.group(1)
        try:
            # Attempt to parse the extracted date string to confirm format
            datetime.strptime(zip_date_str, '%Y-%m-%d')
        except ValueError:
            print(f"Validation failed: Invalid date format '{zip_date_str}' found in zip file name: {self.latest_i90_zip_file_name}")
            return False

        # Validate excel file name format (contains YYYYMMDD)
        # Assumes filename format like 'I90DIA_YYYYMMDD.xlsx'
        # The instruction "yyyy-mm-dd str for excel" might be inaccurate for the filename itself.
        # We validate the presence of a YYYYMMDD string instead.
        excel_match = re.search(r'(\d{8})', self.latest_i90_excel_file_name)
        if not excel_match:
            print(f"Validation failed: Could not find YYYYMMDD date pattern in excel file name: {self.latest_i90_excel_file_name}")
            return False
        excel_date_str = excel_match.group(1)
        try:
            # Attempt to parse the extracted date string to confirm format
            datetime.strptime(excel_date_str, '%Y%m%d')
        except ValueError:
            print(f"Validation failed: Invalid date format '{excel_date_str}' found in excel file name: {self.latest_i90_excel_file_name}")
            return False

        # All checks passed
        return True
    
    def extract_data_for_all_markets(self, fecha_inicio: Optional[str] = None, fecha_fin: Optional[str] = None, mercados_lst: Optional[list[str]] = None) -> dict:
        """
        Extracts data for all specified markets over a given date range, coordinating download, validation, and extraction workflows.
        
        Validates the input date range, iterates through each day, downloads I90 data, checks file integrity, and invokes per-market extraction logic. Tracks and reports extraction success or failure for each market and day, cleans up temporary files, and returns a summary of the extraction process.
        
        Parameters:
            fecha_inicio (Optional[str]): Start date for extraction in 'YYYY-MM-DD' format. If not provided, defaults to 93 days ago.
            fecha_fin (Optional[str]): End date for extraction in 'YYYY-MM-DD' format. If not provided, defaults to 93 days ago.
            mercados_lst (Optional[list[str]]): List of market names to extract. If None, extracts all supported markets.
        
        Returns:
            dict: Dictionary containing overall success status and detailed results for each market and day.
        """
        # Initialize status tracking
        if (fecha_fin is None and fecha_inicio is None) or fecha_fin == fecha_inicio:
            date_range_str = f"Single day download for {(datetime.now() - timedelta(days=self.download_window)).strftime('%Y-%m-%d')}"
        else:
            date_range_str = f"{fecha_inicio} to {fecha_fin}"

        status_details = {
            "markets_downloaded": [],  
            "markets_failed": [],
            "date_range": date_range_str
        }
        
        try:
            fecha_inicio, fecha_fin = self.fecha_input_validation(fecha_inicio, fecha_fin)
            fecha_inicio_carga_dt = datetime.strptime(fecha_inicio, '%Y-%m-%d')
            fecha_fin_carga_dt = datetime.strptime(fecha_fin, '%Y-%m-%d')
            
            overall_success = True #overall success is true if no failed markets
            for day in pd.date_range(start=fecha_inicio_carga_dt, end=fecha_fin_carga_dt):
                day_str = day.strftime('%Y-%m-%d')

                try:
                    # Download the i90 file for the given day
                    self.download_i90_data(day)
                    
                    # Check if the i90 file attributes are set
                    if not self.validate_i90_correct_download():
                        print(f"Skipping day {day_str}: I90 file attributes invalid.")
                        status_details["markets_failed"].append({
                            "market": mercados_lst if mercados_lst else "all",
                            "error": "Invalid I90 file attributes",
                            "day": day_str
                        })
                        overall_success = False
                        return {"success": overall_success, "details": status_details}
                    
                    # Extract data for the given day in range, returns overall success and status details
                    overall_success, status_details = self._extract_data_per_day_all_markets(day, status_details, mercados_lst)

                    #data extraction completed 
                    print(f"ℹ️ Data extraction pipeline finished for {day_str}")

                    if overall_success:
                        print(f"✅ Data extraction fully successful for {day_str}")
                    else:
                        print(f"❌ There were failures in the data extraction pipeline for {day_str}")
                    

                except Exception as e:
                    status_details["markets_failed"].append({
                        "market": mercados_lst if mercados_lst else "all",
                        "error": str(e),
                        "day": day_str
                    })
                    overall_success = False
                    print(f"Error during extraction for {day_str}: {e}")
                
                finally:
                    # Clean up files in the temporary download path
                    print(f"ℹ️ Cleaning up files in the temporary download path for {day_str}")
                    if self.latest_i90_zip_file_name and self.latest_i90_excel_file_name:
                        self.i90_downloader.cleanup_files(self.latest_i90_zip_file_name, self.latest_i90_excel_file_name)
                    
                    # Reset latest i90 data
                    self.latest_i90_zip_file_name = None
                    self.latest_i90_excel_file_name = None
                    self.latest_i90_pestañas_con_error = None
            
        except Exception as e:
            overall_success = False
            status_details["markets_failed"].append({
                "market": mercados_lst if mercados_lst else "all",
                "error": str(e),
                "day": day_str
            })
        
        return {"success": overall_success, "details": status_details}

    def _extract_data_per_day_all_markets(self, day: datetime):
        """
        Abstract method for extracting data for all markets on a given day.
        
        This method must be implemented by subclasses to define the extraction logic for each market segment for the specified date.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def _extract_with_status(self, market_name, extract_function, day, status_details):
        """
        Executes a market extraction function for a given day, updating the status details with success or failure information.
        
        Parameters:
            market_name (str): The name of the market being extracted.
            extract_function (Callable): The extraction function to execute for the market.
            day (datetime): The date for which data is being extracted.
            status_details (dict): Dictionary tracking extraction results.
        
        Returns:
            dict: Updated status details reflecting the outcome of the extraction attempt.
        """
        try:
            extract_function(day)
            status_details["markets_downloaded"].append({
                "market": market_name,
                "day": day.strftime('%Y-%m-%d')
            })
            return status_details
        except Exception as e:
            status_details["markets_failed"].append({
                "market": market_name,
                "error": str(e),
                "day": day.strftime('%Y-%m-%d')
            })
            print(f"❌ Error extracting {market_name} data for {day.strftime('%Y-%m-%d')}: {e}")
            return status_details

class I90VolumenesExtractor(I90Extractor): 
    """
    Extracts volume data from I90 files.
    """
    def __init__(self):
        """
        Initializes the I90VolumenesExtractor, setting up base class attributes and preparing for intradiario downloader initialization.
        """
        super().__init__()
        self.intradiario_downloader = None #initialized in intra method with relevant day parameter passed to get correct number of intrasheets

    def _extract_and_save_volumenes(self, day: datetime, mercado: str, downloader) -> None:
        """
        Extracts and saves volume data for a specified market and day using the provided downloader.
        
        Attempts to extract volume data from the latest I90 Excel file for the given market and date. If data is found, it is saved as a CSV file. Handles missing files and extraction errors gracefully.
        """
        try:
    

            # 1. Call the specific downloader's get_i90_volumenes
            df_volumenes = downloader.get_i90_volumenes(
                excel_file_name=self.latest_i90_excel_file_name,
                pestañas_con_error=self.latest_i90_pestañas_con_error
            )

            # 2. Process/Save the extracted data
            if df_volumenes is not None and not df_volumenes.empty:
                year = day.year
                month = day.month
                dataset_type = 'volumenes_i90' # Consistent dataset type

                # Use appropriate saving method based on dev flag
                # Sticking to write_raw_csv for consistency with recent changes
                self.raw_file_utils.write_raw_csv(
                    year=year, month=month, df=df_volumenes,
                    dataset_type=dataset_type,
                    mercado=mercado
                )
            
                print(f"Successfully processed and saved {mercado} volumenes for {day.date()}")

            else:
                print(f"No {mercado} volumenes data found or extracted for {day.date()} from {self.latest_i90_excel_file_name}")

        except FileNotFoundError:
            print(f"Skipping day {day.date()} for {mercado}: Excel file '{self.latest_i90_excel_file_name}' not found.")
        except Exception as e:
            print(f"Error processing {mercado} volumenes for {day.date()}: {e}")

    def extract_volumenes_diario(self, day: datetime) -> None:
        """
        Extracts and saves daily volume data for the specified date using the Diario market downloader.
        """
        self._extract_and_save_volumenes(day, 'diario', self.diario_downloader)

    def extract_volumenes_intradiario(self, day: datetime) -> None:
        """
        Extracts and saves intradiario (intra-day) volume data for the specified day.
        
        Initializes the intradiario downloader for the given date and processes the extraction and saving of volume data for the "intra" market segment.
        """
        self.intradiario_downloader = IntradiarioDL(fecha=day)
        self._extract_and_save_volumenes(day, 'intra', self.intradiario_downloader)

    def extract_volumenes_terciaria(self, day: datetime) -> None:
        """
        Extracts and saves volume data for the 'terciaria' market segment for the specified day.
        """
        self._extract_and_save_volumenes(day, 'terciaria', self.terciaria_downloader)

    def extract_volumenes_secundaria(self, day: datetime) -> None:
        self._extract_and_save_volumenes(day, 'secundaria', self.secundaria_downloader)

    def extract_volumenes_rr(self, day: datetime) -> None:
        self._extract_and_save_volumenes(day, 'rr', self.rr_downloader)

    def extract_volumenes_curtailment(self, day: datetime) -> None:
        self._extract_and_save_volumenes(day, 'curtailment', self.curtailment_downloader)

    def extract_volumenes_p48(self, day: datetime) -> None:
        self._extract_and_save_volumenes(day, 'p48', self.p48_downloader)

    def extract_volumenes_indisponibilidades(self, day: datetime) -> None:
        self._extract_and_save_volumenes(day, 'indisponibilidades', self.indisponibilidades_downloader)

    def extract_volumenes_restricciones(self, day: datetime) -> None:
        """
        Extracts and saves volume data for the 'restricciones' market segment for a given day.
        """
        self._extract_and_save_volumenes(day, 'restricciones', self.restricciones_downloader)

    def _extract_data_per_day_all_markets(self, day: datetime, status_details: dict, mercados_lst: list[str] = None):
        """
        Extracts volume data for all or selected markets from I90 files for a specific day, updating extraction status for each market.
        
        Parameters:
            day (datetime): The date for which to extract data.
            status_details (dict): Dictionary tracking extraction success and failure per market.
            mercados_lst (list[str], optional): List of market names to extract. If None, extracts all markets.
        
        Returns:
            overall_success (bool): True if all requested markets were extracted successfully; False otherwise.
            status_details (dict): Updated dictionary with per-market extraction results.
        """
        # Track success for each market
        print("\n--------- I90 Volumenes Extraction ---------")
        
        # Define markets and their extraction functions
        mercados = [
            ("diario", self.extract_volumenes_diario),
            ("intra", self.extract_volumenes_intradiario),
            ("terciaria", self.extract_volumenes_terciaria),
            ("secundaria", self.extract_volumenes_secundaria),
            ("rr", self.extract_volumenes_rr),
            ("curtailment", self.extract_volumenes_curtailment),
            ("p48", self.extract_volumenes_p48),
            ("indisponibilidades", self.extract_volumenes_indisponibilidades),
            ("restricciones", self.extract_volumenes_restricciones)
        ]
        
        # Process each market and track individual success
        for mercado, extract_func in mercados:
            print(f"\n--------- {mercado.capitalize()} ---------")
            # If mercados_lst is None, extract all markets; otherwise only extract if mercado is in the list
            if mercados_lst is None or mercado in mercados_lst:
                status_details = self._extract_with_status(mercado, extract_func, day, status_details)
            else:
                print(f"Skipping {mercado} - not in requested markets list")
            print("\n--------------------------------")

        #overall success is true if failed markets is empty, else false
        overall_success = len(status_details["markets_failed"]) == 0

        return overall_success, status_details

class I90PreciosExtractor(I90Extractor):
    def __init__(self):
        """
        Initialize the I90VolumenesExtractor by calling the base class constructor.
        """
        super().__init__()

    def _extract_and_save_precios(self, day: datetime, mercado: str, downloader) -> None:
        """
        Extracts price data for a specified market and day using the provided downloader and saves it as a CSV file.
        
        If no data is found or extracted, a message is printed. Handles missing Excel files and other exceptions gracefully.
        """
        try:
            # 1. Call the specific downloader's get_i90_precios
            df_precios = downloader.get_i90_precios(
                excel_file_name=self.latest_i90_excel_file_name,
                pestañas_con_error=self.latest_i90_pestañas_con_error
            )

            # 2. Process/Save the extracted data
            if df_precios is not None and not df_precios.empty:
                year = day.year
                month = day.month
                dataset_type = 'precios_i90' # Consistent dataset type for prices

                # Use appropriate saving method based on dev flag
                # Assuming write_raw_csv for consistency, adjust if needed (e.g., parquet)
               
                self.raw_file_utils.write_raw_csv(
                    year=year, month=month, df=df_precios,
                    dataset_type=dataset_type,
                    mercado=mercado
                )
                    
                print(f"Successfully processed and saved {mercado} precios for {day.date()}")

            else:
                print(f"No {mercado} precios data found or extracted for {day.date()} from {self.latest_i90_excel_file_name}")

        except FileNotFoundError:
            print(f"Skipping day {day.date()} for {mercado} precios: Excel file '{self.latest_i90_excel_file_name}' not found.")
        except Exception as e:
            print(f"Error processing {mercado} precios for {day.date()}: {e}")

    # --- Public methods for markets with price data ---
    # Note: Diario, Intra, Curtailment, P48 do not have get_i90_precios in their downloaders

    def extract_precios_secundaria(self, day: datetime) -> None:
        """
        Extracts secondary market price data for the specified day and saves it as a CSV file.
        
        Note:
            This method currently delegates to the downloader, but price extraction for the secondary market is not yet implemented and prices are typically retrieved directly from the ESIOS API.
        """
        self._extract_and_save_precios(day, 'secundaria', self.secundaria_downloader)

    def extract_precios_terciaria(self, day: datetime) -> None:
        """
        Not yet implemented in downlaoder since prices are retrived from ESIOS API directly
        """
        self._extract_and_save_precios(day, 'terciaria', self.terciaria_downloader)

    def extract_precios_rr(self, day: datetime) -> None:
        """
        Not yet implemented in downlaoder since prices are retrived from ESIOS API directly
        """
        self._extract_and_save_precios(day, 'rr', self.rr_downloader)

    def extract_precios_indisponibilidades(self, day: datetime) -> None:
        """
        Not yet implemented in downlaoder since prices are retrived from ESIOS API directly
        """
        self._extract_and_save_precios(day, 'indisponibilidades', self.indisponibilidades_downloader)

    def extract_precios_restricciones(self, day: datetime) -> None:
        """
        Extracts and saves price data for the 'restricciones' market segment for a given day.
        """
        self._extract_and_save_precios(day, 'restricciones', self.restricciones_downloader)

    def _extract_data_per_day_all_markets(self, day: datetime, status_details: dict, mercados_lst: list[str] = None):
        """
        Extracts price data for all specified markets from I90 files for a given day, updating the extraction status for each market.
        
        Parameters:
            day (datetime): The date for which to extract price data.
            status_details (dict): Dictionary tracking extraction success and failure per market.
            mercados_lst (list[str], optional): List of market names to extract; if None, extracts all available markets.
        
        Returns:
            tuple: (overall_success (bool), status_details (dict)) indicating whether all extractions succeeded and detailed per-market results.
        """
        # Track success for each market
        print("\n--------- I90 Precios Extraction ---------")
        
        # Define markets and their extraction functions
        markets = [
            # ("secundaria", self.extract_precios_secundaria),
            # ("terciaria", self.extract_precios_terciaria),
            # ("rr", self.extract_precios_rr),
            # ("indisponibilidades", self.extract_precios_indisponibilidades),
            ("restricciones", self.extract_precios_restricciones)
        ]
        
        # Process each market and track individual success
        for market_name, extract_func in markets:
            print(f"\n--------- {market_name.capitalize()} ---------")
            # If mercados_lst is None, extract all markets; otherwise only extract if market is in the list
            if mercados_lst is None or market_name in mercados_lst:
                status_details = self._extract_with_status(market_name, extract_func, day, status_details)
            else:
                print(f"Skipping {market_name} - not in requested markets list")
            print("\n--------------------------------")

        # Overall success is true if failed markets is empty, else false
        overall_success = len(status_details["markets_failed"]) == 0

        return overall_success, status_details

def example_usage():
    """
    Demonstrates how to use the I90VolumenesExtractor and I90PreciosExtractor classes to extract volume and price data for a specified date range.
    """
    i90_volumenes_extractor = I90VolumenesExtractor()
    i90_precios_extractor = I90PreciosExtractor()

    i90_volumenes_extractor.extract_data_for_all_markets(fecha_inicio="2024-12-01", fecha_fin="2025-02-01")
    i90_precios_extractor.extract_data_for_all_markets(fecha_inicio="2024-12-01", fecha_fin="2025-02-01")

if __name__ == "__main__":
    example_usage()

    