from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
from datetime import datetime, timedelta
import sys
import os
import re
import time 

# Add the project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from extract._descargador_i90 import I90Downloader, DiarioDL, TerciariaDL, SecundariaDL, RRDL, CurtailmentDL, P48DL, IndisponibilidadesDL, RestriccionesDL
from utilidades.storage_file_utils import RawFileUtils
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
    
    def download_i90_data(self, day: datetime) -> None:
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

        zip_file_name, excel_file_name, pestañas_con_error = self.i90_downloader.download_i90_file(day)
        
        #sleep for 1 second to avoid rate limit
        time.sleep(1)
        print(f"Downloaded I90 data for {day.date()}: Zip='{zip_file_name}', Excel='{excel_file_name}', Errors='{pestañas_con_error}'")

        #update latest i90 data
        self.latest_i90_zip_file_name = zip_file_name
        self.latest_i90_excel_file_name = excel_file_name
        self.latest_i90_pestañas_con_error = pestañas_con_error

        return

    def validate_i90_attributes(self):
        """Validates the presence and format of the latest I90 download attributes. Used to check if i90 data was downloaded correctly"""

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
    
    def extract_data_for_all_markets(self, fecha_inicio_carga: Optional[str] = None, fecha_fin_carga: Optional[str] = None) -> dict:
        """
        Generic workflow to extract all data for each day in the specified date range.

        This method orchestrates the extraction process by validating the provided date range,
        iterating through each day within that range, and calling the `extract_i90_data` method
        for each day. It also handles potential errors during the extraction process and ensures
        that the latest I90 data attributes are reset after each day's extraction attempt.

        Args:
            fecha_inicio_carga (Optional[str]): The start date for the data extraction in 'YYYY-MM-DD' format.
            fecha_fin_carga (Optional[str]): The end date for the data extraction in 'YYYY-MM-DD' format.

        Raises:
            ValueError: If the date range is invalid or incomplete.
            Exception: If an error occurs during the extraction for a specific day.
        """
        # Initialize status tracking
        status_details = {
            "markets_processed": [],  # Changed to dict to track by day and market
            "markets_failed": [],
            "date_range": f"{fecha_inicio_carga} to {fecha_fin_carga}"
        }
        
        try:
            fecha_inicio_carga, fecha_fin_carga = self.fecha_input_validation(fecha_inicio_carga, fecha_fin_carga)
            fecha_inicio_carga_dt = datetime.strptime(fecha_inicio_carga, '%Y-%m-%d')
            fecha_fin_carga_dt = datetime.strptime(fecha_fin_carga, '%Y-%m-%d')
            
            overall_success = True #overall success is true if no failed markets
            for day in pd.date_range(start=fecha_inicio_carga_dt, end=fecha_fin_carga_dt):
                day_str = day.strftime('%Y-%m-%d')
                
                try:
                    # Download the i90 file for the given day
                    self.download_i90_data(day)
                    
                    # Check if the i90 file attributes are set
                    if not self.validate_i90_attributes():
                        print(f"Skipping day {day.date()}: I90 file attributes invalid.")
                        status_details["markets_failed"].append({
                            "day": day_str,
                            "market": "all",
                            "error": "Invalid I90 file attributes"
                        })
                        overall_success = False
                        return {"success": overall_success, "details": status_details}
                    
                    # Extract data for the given day in range, returns overall success and status details
                    overall_success, status_details = self._extract_data_per_day_all_markets(day, status_details)

                    #data transformation completed 
                    print(f"ℹ️ Data transformation piepline finished for {day.date()}")

                    if overall_success:
                        print(f"✅ Data transformation fully successful for {day.date()}")
                    else:
                        print(f"❌ There were failures in the data transformation pipeline for {day.date()}")
                    

                except Exception as e:
                    status_details["markets_failed"].append({
                        "day": day_str,
                        "market": "all",
                        "error": str(e)
                    })
                    overall_success = False
                    print(f"Error during extraction for {day.date()}: {e}")
                
                finally:
                    # Clean up files in the temporary download path
                    print(f"ℹ️ Cleaning up files in the temporary download path for {day.date()}")
                    if self.latest_i90_zip_file_name and self.latest_i90_excel_file_name:
                        self.i90_downloader.cleanup_files(self.latest_i90_zip_file_name, self.latest_i90_excel_file_name)
                    
                    # Reset latest i90 data
                    self.latest_i90_zip_file_name = None
                    self.latest_i90_excel_file_name = None
                    self.latest_i90_pestañas_con_error = None
            
        except Exception as e:
            overall_success = False
            status_details["error"] = str(e)
        
        return {"success": overall_success, "details": status_details}

    def _extract_data_per_day_all_markets(self, day: datetime):
        """
        To be implemented by child classes.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def _extract_with_status(self, market_name, extract_function, day, status_details):
        """Helper method to track success status for each market extraction"""
        try:
            extract_function(day)
            status_details["markets_processed"].append(market_name)
            return status_details
        except Exception as e:
            status_details["markets_failed"].append({
                "market": market_name,
                "error": str(e)
            })
            print(f"❌ Error extracting {market_name} data for {day.date()}: {e}")
            return status_details

class I90VolumenesExtractor(I90Extractor): 
    """
    Extracts volume data from I90 files.
    """
    def __init__(self):
        super().__init__()

    def _extract_and_save_volumenes(self, day: datetime, mercado: str, downloader) -> None:
        """Helper method to extract and save volumenes data for a given market."""
        try:
            dev, prod = self.env_utils.check_dev_env()

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

                if dev and not prod:
                    status = "DEVELOPMENT "
                    self.raw_file_utils.write_raw_csv(
                        year=year, month=month, df=df_volumenes,
                        dataset_type=dataset_type,
                        mercado=mercado
                    )
                else:
                    status = "PRODUCTION "
                    self.raw_file_utils.write_raw_parquet(
                        year=year, month=month, df=df_volumenes,
                        dataset_type=dataset_type,
                        mercado=mercado
                    )

                print(f"Successfully processed and saved {status}{mercado} volumenes for {day.date()}")

            else:
                print(f"No {mercado} volumenes data found or extracted for {day.date()} from {self.latest_i90_excel_file_name}")

        except FileNotFoundError:
            print(f"Skipping day {day.date()} for {mercado}: Excel file '{self.latest_i90_excel_file_name}' not found.")
        except Exception as e:
            print(f"Error processing {mercado} volumenes for {day.date()}: {e}")

    def extract_volumenes_diario(self, day: datetime) -> None:
        self._extract_and_save_volumenes(day, 'diario', self.diario_downloader)

    def extract_volumenes_terciaria(self, day: datetime) -> None:
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
        self._extract_and_save_volumenes(day, 'restricciones', self.restricciones_downloader)

    def _extract_data_per_day_all_markets(self, day: datetime, status_details: dict):
        """
        Extracts all volumenes data from I90 files for a given day.
        Returns market-specific status information.
        """
        # Track success for each market
        print("\n--------- I90 Volumenes Extraction ---------")
        
        # Define markets and their extraction functions
        markets = [
            ("diario", self.extract_volumenes_diario),
            ("terciaria", self.extract_volumenes_terciaria),
            ("secundaria", self.extract_volumenes_secundaria),
            ("rr", self.extract_volumenes_rr),
            ("curtailment", self.extract_volumenes_curtailment),
            ("p48", self.extract_volumenes_p48),
            ("indisponibilidades", self.extract_volumenes_indisponibilidades),
            ("restricciones", self.extract_volumenes_restricciones)
        ]
        
        # Process each market and track individual success
        
        for market_name, extract_func in markets:
            print(f"\n--------- {market_name.capitalize()} ---------")
            status_details = self._extract_with_status(market_name, extract_func, day, status_details)
            print("\n--------------------------------")

        #overall success is true if failed markets is empty, else false
        overall_success = len(status_details["markets_failed"]) == 0

        return overall_success, status_details
    
class I90PreciosExtractor(I90Extractor):
    def __init__(self):
        super().__init__()

    def _extract_and_save_precios(self, day: datetime, mercado: str, downloader) -> None:
        """Helper method to extract and save precios data for a given market."""
        try:
            dev, prod = self.env_utils.check_dev_env()

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
                if dev and not prod:
                    status = "DEVELOPMENT "
                    self.raw_file_utils.write_raw_csv(
                        year=year, month=month, df=df_precios,
                        dataset_type=dataset_type,
                        mercado=mercado
                    )
                else:
                    status = "PRODUCTION "
                    self.raw_file_utils.write_raw_parquet(
                        year=year, month=month, df=df_precios,
                        dataset_type=dataset_type,
                        mercado=mercado
                    )
                    
                print(f"Successfully processed and saved {status}{mercado} precios for {day.date()}")

            else:
                print(f"No {mercado} precios data found or extracted for {day.date()} from {self.latest_i90_excel_file_name}")

        except FileNotFoundError:
            print(f"Skipping day {day.date()} for {mercado} precios: Excel file '{self.latest_i90_excel_file_name}' not found.")
        except Exception as e:
            print(f"Error processing {mercado} precios for {day.date()}: {e}")

    # --- Public methods for markets with price data ---
    # Note: Diario, Curtailment, P48 do not have get_i90_precios in their downloaders

    def extract_precios_secundaria(self, day: datetime) -> None:
        """
        Not yet implemented in downlaoder since prices are retrived from ESIOS API directly
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
        self._extract_and_save_precios(day, 'restricciones', self.restricciones_downloader)

    def _extract_data_per_day_all_markets(self, day: datetime, status_details: dict):
        """
        Extracts all precios data from I90 files for a given day.
        Returns market-specific status information.
        
        Note: All prices come from API, not I90 file typically except for restricciones
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
            status_details = self._extract_with_status(market_name, extract_func, day, status_details)
            print("\n--------------------------------")

        # Overall success is true if failed markets is empty, else false
        overall_success = len(status_details["markets_failed"]) == 0

        return overall_success, status_details

def example_usage():
    i90_volumenes_extractor = I90VolumenesExtractor()
    i90_precios_extractor = I90PreciosExtractor()

    i90_volumenes_extractor.extract_data_for_all_markets(fecha_inicio_carga="2024-12-01", fecha_fin_carga="2025-02-01")
    i90_precios_extractor.extract_data_for_all_markets(fecha_inicio_carga="2024-12-01", fecha_fin_carga="2025-02-01")

