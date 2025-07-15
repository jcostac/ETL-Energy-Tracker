from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
from datetime import datetime, timedelta
import sys
import os
import re
import time 

# Add the project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from extract._descargador_i3 import I3Downloader, DiarioDL, TerciariaDL, SecundariaDL, RRDL, CurtailmentDL, P48DL, IndisponibilidadesDL, RestriccionesDL, IntradiarioDL
from utilidades.storage_file_utils import RawFileUtils
from utilidades.db_utils import DatabaseUtils
from utilidades.env_utils import EnvUtils

class I3Extractor:
    """
    Wrapper class for extracting volume data from I3 files via ESIOS API.
    Provides a unified interface for downloading and processing I3 Excel files.
    """
    
    def __init__(self):
        """
        Initializes the I3Extractor with downloader instances for each market segment, utility classes, and default configuration.
        
        Sets up downloaders for all supported market segments, file and environment utilities, a database engine, and initializes attributes for tracking the latest downloaded I3 files and error sheets. The default download window is set to 4 days.
        """

        #downloaders
        self.i3_downloader = I3Downloader()
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
        self.download_window = 4 # I3 is typically uploaded with a 4 day lag

        #latest i3 data for the day
        self.latest_i3_excel_file_name = None
        self.latest_i3_zip_file_name = None
        self.latest_i3_pestañas_con_error = None

    def fecha_input_validation(self, fecha_inicio_carga: str, fecha_fin_carga: str) -> tuple[str, str]:
        """
        Validates and normalizes the input date range for data extraction.
        
        Ensures both start and end dates are provided and that the start date is not after the end date. If no dates are provided, defaults to a specific day based on the configured download window. Raises a ValueError if the input is incomplete or invalid.
        
        Returns:
            tuple[str, str]: Validated start and end dates in 'YYYY-MM-DD' format.
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
            # Get datetime range for 1 day ago (assuming I3 is for recent data)
            fecha_inicio_carga_dt = datetime.now() - timedelta(days=self.download_window) 
            fecha_fin_carga_dt = datetime.now() - timedelta(days=self.download_window)
            
            # Convert to string format
            fecha_inicio_carga = fecha_inicio_carga_dt.strftime('%Y-%m-%d') 
            fecha_fin_carga = fecha_fin_carga_dt.strftime('%Y-%m-%d')
            print(f"No se han proporcionado fechas de carga, se descargarán datos para el día {fecha_inicio_carga}")

        else:
            raise ValueError("No se han proporcionado fechas de carga completas")

        return fecha_inicio_carga, fecha_fin_carga
    
    def download_i3_data(self, day: datetime) -> None:
        """
        Download the I3 zip and Excel files for the specified day and update the object's attributes with the latest file names and any sheet errors.
        
        Parameters:
            day (datetime): The date for which to download the I3 data.
        
        Raises:
            AttributeError: If the downloader instance is not available.
        """
        if not hasattr(self, 'i3_downloader'):
             raise AttributeError("Downloader instance 'self.i3_downloader' not found.")

        zip_file_name, excel_file_name, pestañas_con_error = self.i3_downloader.download_i3_file(day)
        
        #sleep for 1 second to avoid rate limit
        time.sleep(1)
        print(f"Downloaded I3 data for {day.date()}: Zip='{zip_file_name}', Excel='{excel_file_name}', Errors='{pestañas_con_error}'")

        #update latest i3 data
        self.latest_i3_zip_file_name = zip_file_name
        self.latest_i3_excel_file_name = excel_file_name
        self.latest_i3_pestañas_con_error = pestañas_con_error

        return

    def validate_i3_correct_download(self):
        """
        Validate that the latest downloaded I3 zip and Excel file attributes are present and correctly formatted.
        
        Returns:
            bool: True if all required attributes exist and have valid date patterns; False otherwise.
        """

        # Check for None values
        if self.latest_i3_zip_file_name is None:
            print("Validation failed: latest_i3_zip_file_name is None.")
            return False
        if self.latest_i3_excel_file_name is None:
            print("Validation failed: latest_i3_excel_file_name is None.")
            return False
        if self.latest_i3_pestañas_con_error is None:
            print("Validation failed: latest_i3_pestañas_con_error is None.")
            return False

        # Check types
        if not isinstance(self.latest_i3_zip_file_name, str):
            print(f"Validation failed: latest_i3_zip_file_name is not a string ({type(self.latest_i3_zip_file_name)}).")
            return False
        if not isinstance(self.latest_i3_excel_file_name, str):
            print(f"Validation failed: latest_i3_excel_file_name is not a string ({type(self.latest_i3_excel_file_name)}).")
            return False
        if not isinstance(self.latest_i3_pestañas_con_error, list):
            print(f"Validation failed: latest_i3_pestañas_con_error is not a list ({type(self.latest_i3_pestañas_con_error)}).")
            return False

        # Validate zip file name format (contains YYYY-MM-DD)
        # Zip filename format is 'I3_YYYY-MM-DD.zip'
        zip_match = re.search(r'(\d{4}-\d{2}-\d{2})', self.latest_i3_zip_file_name)
        if not zip_match:
            print(f"Validation failed: Could not find YYYY-MM-DD date pattern in zip file name: {self.latest_i3_zip_file_name}")
            return False
        zip_date_str = zip_match.group(1)
        try:
            datetime.strptime(zip_date_str, '%Y-%m-%d')
        except ValueError:
            print(f"Validation failed: Invalid date format '{zip_date_str}' found in zip file name: {self.latest_i3_zip_file_name}")
            return False

        # Validate excel file name format (contains YYYYMMDD)
        # Excel filename part is 'YYYYMMDD'
        excel_match = re.search(r'(\d{8})', self.latest_i3_excel_file_name)
        if not excel_match:
            print(f"Validation failed: Could not find YYYYMMDD date pattern in excel file name: {self.latest_i3_excel_file_name}")
            return False
        excel_date_str = excel_match.group(1)
        try:
            datetime.strptime(excel_date_str, '%Y%m%d')
        except ValueError:
            print(f"Validation failed: Invalid date format '{excel_date_str}' found in excel file name: {self.latest_i3_excel_file_name}")
            return False

        # All checks passed
        return True
    
    def extract_data_for_all_markets(self, fecha_inicio_carga: Optional[str] = None, fecha_fin_carga: Optional[str] = None, mercados_lst: Optional[list[str]] = None) -> dict:
        """
        Coordinates the extraction of data for multiple markets over a specified date range.
        
        Validates input dates, downloads I3 files for each day in the range, checks file integrity, and invokes per-market extraction logic. Tracks extraction success or failure for each market and day, cleans up temporary files, and returns a summary of the extraction process.
        
        Parameters:
            fecha_inicio_carga (Optional[str]): Start date for extraction in 'YYYY-MM-DD' format. If not provided, defaults to a date based on the configured download window.
            fecha_fin_carga (Optional[str]): End date for extraction in 'YYYY-MM-DD' format. If not provided, defaults to a date based on the configured download window.
            mercados_lst (Optional[list[str]]): List of market names to extract. If None, extracts all supported markets.
        
        Returns:
            dict: Summary of extraction success and detailed results for each market and day.
        """
        # Initialize status tracking
        if (fecha_fin_carga is None and fecha_inicio_carga is None) or fecha_fin_carga == fecha_inicio_carga:
            date_range_str = f"Single day download for {(datetime.now() - timedelta(days=self.download_window)).strftime('%Y-%m-%d')}"
        else:
            date_range_str = f"{fecha_inicio_carga} to {fecha_fin_carga}"

        status_details = {
            "markets_downloaded": [],  
            "markets_failed": [],
            "date_range": date_range_str
        }
        
        try:
            fecha_inicio_carga, fecha_fin_carga = self.fecha_input_validation(fecha_inicio_carga, fecha_fin_carga)
            fecha_inicio_carga_dt = datetime.strptime(fecha_inicio_carga, '%Y-%m-%d')
            fecha_fin_carga_dt = datetime.strptime(fecha_fin_carga, '%Y-%m-%d')
            
            overall_success = True #overall success is true if no failed markets
            for day in pd.date_range(start=fecha_inicio_carga_dt, end=fecha_fin_carga_dt):
                day_str = day.strftime('%Y-%m-%d')

                try:
                    # Download the i3 file for the given day
                    self.download_i3_data(day)
                    
                    # Check if the i3 file attributes are set
                    if not self.validate_i3_correct_download():
                        print(f"Skipping day {day_str}: I3 file attributes invalid.")
                        status_details["markets_failed"].append({
                            "market": mercados_lst if mercados_lst else "all",
                            "error": "Invalid I3 file attributes",
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
                    if self.latest_i3_zip_file_name and self.latest_i3_excel_file_name:
                        self.i3_downloader.cleanup_files(self.latest_i3_zip_file_name, self.latest_i3_excel_file_name)

                    # Reset latest i3 data
                    self.latest_i3_zip_file_name = None
                    self.latest_i3_excel_file_name = None
                    self.latest_i3_pestañas_con_error = None
            
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
        Defines the interface for extracting data for all market segments on a specific day.
        
        Subclasses must implement this method to perform extraction logic for each relevant market on the given date.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def _extract_with_status(self, market_name, extract_function, day, status_details):
        """
        Run a market extraction function for a specific day and update the status details with the result.
        
        Parameters:
            market_name (str): Name of the market being processed.
            extract_function (Callable): Function to extract data for the market.
            day (datetime): Date for which extraction is performed.
            status_details (dict): Dictionary to record extraction outcomes.
        
        Returns:
            dict: The updated status details reflecting extraction success or failure.
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

class I3VolumenesExtractor(I3Extractor): 
    """
    Extracts volume data from I3 files.
    """
    def __init__(self):
        """
        Initialize the I3VolumenesExtractor, preparing for extraction of volume data across multiple market segments.
        
        Sets up base class attributes and defers initialization of the intradiario downloader until extraction time.
        """
        super().__init__()
        self.intradiario_downloader = None #initialized in intra method with relevant day parameter passed to get correct number of intrasheets

    def _extract_and_save_volumenes(self, day: datetime, mercado: str, downloader) -> None:
        """
        Extracts volume data for a specific market and day from the latest I3 Excel file and saves it as a CSV file.
        
        Attempts to extract volume data using the provided downloader. If data is found, it is saved to disk; otherwise, a message is printed. Handles missing Excel files and other extraction errors gracefully.
        """
        try:
            # 1. Call the specific downloader's get_i3_volumenes
            df_volumenes = downloader.get_i3_volumenes(
                excel_file_name=self.latest_i3_excel_file_name,
                pestañas_con_error=self.latest_i3_pestañas_con_error
            )

            # 2. Process/Save the extracted data
            if df_volumenes is not None and not df_volumenes.empty:
                year = day.year
                month = day.month
                dataset_type = 'volumenes_i3' 

                # Use appropriate saving method based on dev flag
                # Sticking to write_raw_csv for consistency with recent changes
                self.raw_file_utils.write_raw_csv(
                    year=year, month=month, df=df_volumenes,
                    dataset_type=dataset_type,
                    mercado=mercado
                )
            
                print(f"Successfxully processed and saved {mercado} volumenes for {day.date()}")

            else:
                print(f"No {mercado} volumenes data found or extracted for {day.date()} from {self.latest_i3_excel_file_name}")

        except FileNotFoundError:
            print(f"Skipping day {day.date()} for {mercado}: Excel file '{self.latest_i3_excel_file_name}' not found.")
        except Exception as e:
            print(f"Error processing {mercado} volumenes for {day.date()}: {e}")

    def extract_volumenes_diario(self, day: datetime) -> None:
        """
        Extracts and saves daily market volume data for the specified day using the diario downloader.
        """
        self._extract_and_save_volumenes(day, 'diario', self.diario_downloader)

    def extract_volumenes_intradiario(self, day: datetime) -> None:
        """
        Extracts and saves intradiario (intra-day) volume data for the specified day.
        
        Initializes the intradiario downloader for the given date and processes extraction and saving of volume data for the "intra" market segment.
        """
        self.intradiario_downloader = IntradiarioDL(fecha=day)
        self._extract_and_save_volumenes(day, 'intra', self.intradiario_downloader)

    def extract_volumenes_terciaria(self, day: datetime) -> None:
        """
        Extracts and saves volume data for the 'terciaria' market segment for the specified day.
        """
        self._extract_and_save_volumenes(day, 'terciaria', self.terciaria_downloader)

    def extract_volumenes_secundaria(self, day: datetime) -> None:
        """
        Extracts and saves volume data for the 'secundaria' market segment for the specified day.
        """
        self._extract_and_save_volumenes(day, 'secundaria', self.secundaria_downloader)

    def extract_volumenes_rr(self, day: datetime) -> None:
        """
        Extracts and saves volume data for the 'rr' market segment for the specified day.
        """
        self._extract_and_save_volumenes(day, 'rr', self.rr_downloader)

    def extract_volumenes_curtailment(self, day: datetime) -> None:
        """
        Extracts and saves curtailment volume data for the specified day using the curtailment downloader.
        """
        self._extract_and_save_volumenes(day, 'curtailment', self.curtailment_downloader)

    def extract_volumenes_p48(self, day: datetime) -> None:
        """
        Extracts and saves P48 market volume data for the specified day.
        
        Calls the internal extraction and saving routine using the P48 market downloader.
        """
        self._extract_and_save_volumenes(day, 'p48', self.p48_downloader)

    def extract_volumenes_indisponibilidades(self, day: datetime) -> None:
        """
        Extracts and saves volume data for the 'indisponibilidades' market segment for the specified day.
        """
        self._extract_and_save_volumenes(day, 'indisponibilidades', self.indisponibilidades_downloader)

    def extract_volumenes_restricciones(self, day: datetime) -> None:
        """
        Extracts and saves volume data for the 'restricciones' market segment for the specified day.
        """
        self._extract_and_save_volumenes(day, 'restricciones', self.restricciones_downloader)

    def _extract_data_per_day_all_markets(self, day: datetime, status_details: dict, mercados_lst: list[str] = None):
        """
        Extracts volume data for all or specified markets from I3 files for a given day, updating the extraction status for each market.
        
        Parameters:
            day (datetime): The date for which to perform extraction.
            status_details (dict): Dictionary to record extraction results per market.
            mercados_lst (list[str], optional): List of market names to extract. If None, all supported markets are processed.
        
        Returns:
            overall_success (bool): True if extraction succeeded for all requested markets; False if any market extraction failed.
            status_details (dict): Updated dictionary containing per-market extraction outcomes.
        """
        # Track success for each market
        print("\n--------- I3 Volumenes Extraction ---------")
        
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