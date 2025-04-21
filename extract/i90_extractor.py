from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
from datetime import datetime, timedelta
import sys
import os
import re


# Add the project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from extract.descargador_i90 import I90Downloader, DiarioDL, TerciariaDL, SecundariaDL, RRDL, CurtailmentDL, P48DL, IndisponibilidadesDL, RestriccionesDL
from utilidades.storage_file_utils import RawFileUtils
from utilidades.db_utils import DatabaseUtils

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
        print(f"Downloaded I90 data for {day.date()}: Zip='{zip_file_name}', Excel='{excel_file_name}', Errors='{pestañas_con_error}'")

        #update latest i90 data
        self.latest_i90_zip_file_name = zip_file_name
        self.latest_i90_excel_file_name = excel_file_name
        self.latest_i90_pestañas_con_error = pestañas_con_error

        return

    def validate_i90_attributes(self):
        """Validates the presence and format of the latest I90 download attributes."""

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

        # Validate zip file name format (contains YYYYMMDD)
        # Assumes filename format like 'I90DIA_YYYYMMDD.zip'
        zip_match = re.search(r'(\d{8})', self.latest_i90_zip_file_name)
        if not zip_match:
            print(f"Validation failed: Could not find YYYYMMDD date pattern in zip file name: {self.latest_i90_zip_file_name}")
            return False
        zip_date_str = zip_match.group(1)
        try:
            # Attempt to parse the extracted date string to confirm format
            datetime.strptime(zip_date_str, '%Y%m%d')
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
    
    def extract_data_for_all_markets(self, fecha_inicio_carga: Optional[str] = None, fecha_fin_carga: Optional[str] = None, dev: bool = False) -> None:
        """
        Generic workflow to extract all data for each day in the specified date range.

        This method orchestrates the extraction process by validating the provided date range,
        iterating through each day within that range, and calling the `extract_i90_data` method
        for each day. It also handles potential errors during the extraction process and ensures
        that the latest I90 data attributes are reset after each day's extraction attempt.

        Args:
            fecha_inicio_carga (Optional[str]): The start date for the data extraction in 'YYYY-MM-DD' format.
            fecha_fin_carga (Optional[str]): The end date for the data extraction in 'YYYY-MM-DD' format.
            dev (bool): A flag indicating whether to run in development mode, which may alter the behavior
                        of the extraction process (e.g., logging, data handling).

        Raises:
            ValueError: If the date range is invalid or incomplete.
            Exception: If an error occurs during the extraction for a specific day.
        """
        fecha_inicio_carga, fecha_fin_carga = self.fecha_input_validation(fecha_inicio_carga, fecha_fin_carga)
        fecha_inicio_carga_dt = datetime.strptime(fecha_inicio_carga, '%Y-%m-%d')
        fecha_fin_carga_dt = datetime.strptime(fecha_fin_carga, '%Y-%m-%d')

        for day in pd.date_range(start=fecha_inicio_carga_dt, end=fecha_fin_carga_dt):
            try:
                #downlaod the i90 file for the given day
                self.download_i90_data(day)

                #check if the i90 file attributes are set
                if (
                    self.latest_i90_excel_file_name is None or
                    self.latest_i90_zip_file_name is None or
                    self.latest_i90_pestañas_con_error is None
                ):
                    print(f"Skipping day {day.date()}: I90 file attributes not set. Extraction will not proceed for this day.")
                    continue

                #extract data for the given day in range
                self._extract_data_per_day_all_markets(day, dev=dev)

            except Exception as e:
                print(f"Error during extraction for {day.date()}: {e}")

            finally:

                #clean up files in the temporary download path
                self.i90_downloader.cleanup_files(self.latest_i90_zip_file_name, self.latest_i90_excel_file_name)

                #reset latest i90 data
                self.latest_i90_zip_file_name = None
                self.latest_i90_excel_file_name = None
                self.latest_i90_pestañas_con_error = None

    def _extract_data_per_day_all_markets(self, day: datetime, dev: bool = False):
        """
        To be implemented by child classes.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

class I90VolumenesExtractor(I90Extractor): 
    """
    Extracts volume data from I90 files.
    """
    def __init__(self):
        super().__init__()

    def _extract_and_save_volumenes(self, day: datetime, dev: bool, mercado: str, downloader) -> None:
        """Helper method to extract and save volumenes data for a given market."""
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

                if dev == True:
                    status = "DEV "
                    self.raw_file_utils.write_raw_csv(
                        year=year, month=month, df=df_volumenes,
                        dataset_type=dataset_type,
                        mercado=mercado
                    )
                else:
                    self.raw_file_utils.write_raw_parquet(
                        year=year, month=month, df=df_volumenes,
                        dataset_type=dataset_type,
                        mercado=mercado
                    )

                status = ""
                print(f"Successfully processed and saved {status}{mercado} volumenes for {day.date()}")

            else:
                print(f"No {mercado} volumenes data found or extracted for {day.date()} from {self.latest_i90_excel_file_name}")

        except FileNotFoundError:
            print(f"Skipping day {day.date()} for {mercado}: Excel file '{self.latest_i90_excel_file_name}' not found.")
        except Exception as e:
            print(f"Error processing {mercado} volumenes for {day.date()}: {e}")

    def extract_volumenes_diario(self, day: datetime, dev: bool = False) -> None:
        self._extract_and_save_volumenes(day, dev, 'diario', self.diario_downloader)

    def extract_volumenes_terciaria(self, day: datetime, dev: bool = False) -> None:
        self._extract_and_save_volumenes(day, dev, 'terciaria', self.terciaria_downloader)

    def extract_volumenes_secundaria(self, day: datetime, dev: bool = False) -> None:
        self._extract_and_save_volumenes(day, dev, 'secundaria', self.secundaria_downloader)

    def extract_volumenes_rr(self, day: datetime, dev: bool = False) -> None:
        self._extract_and_save_volumenes(day, dev, 'rr', self.rr_downloader)

    def extract_volumenes_curtailment(self, day: datetime, dev: bool = False) -> None:
        self._extract_and_save_volumenes(day, dev, 'curtailment', self.curtailment_downloader)

    def extract_volumenes_p48(self, day: datetime, dev: bool = False) -> None:
        self._extract_and_save_volumenes(day, dev, 'p48', self.p48_downloader)

    def extract_volumenes_indisponibilidades(self, day: datetime, dev: bool = False) -> None:
        self._extract_and_save_volumenes(day, dev, 'indisponibilidades', self.indisponibilidades_downloader)

    def extract_volumenes_restricciones(self, day: datetime, dev: bool = False) -> None:
        self._extract_and_save_volumenes(day, dev, 'restricciones', self.restricciones_downloader)

    def _extract_data_per_day_all_markets(self, day: datetime, dev: bool = False):
        """
        Extracts all volumenes data from I90 files for a given day.
        """
        try:

            self.extract_volumenes_diario(day, dev=dev)
            self.extract_volumenes_terciaria(day, dev=dev)
            self.extract_volumenes_secundaria(day, dev=dev)
            self.extract_volumenes_rr(day, dev=dev)
            self.extract_volumenes_curtailment(day, dev=dev)
            self.extract_volumenes_p48(day, dev=dev)
            self.extract_volumenes_indisponibilidades(day, dev=dev)
            self.extract_volumenes_restricciones(day, dev=dev)

        except Exception as e:
            print(f"Error extracting data: {e}")

class I90PreciosExtractor(I90Extractor):
    def __init__(self):
        super().__init__()

    def _extract_and_save_precios(self, day: datetime, dev: bool, mercado: str, downloader) -> None:
        """Helper method to extract and save precios data for a given market."""
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
                dataset_type = 'precios' # Consistent dataset type for prices

                # Use appropriate saving method based on dev flag
                # Assuming write_raw_csv for consistency, adjust if needed (e.g., parquet)
                if dev == True:
                    status = "DEV "
                    self.raw_file_utils.write_raw_csv(
                        year=year, month=month, df=df_precios,
                        dataset_type=dataset_type,
                        mercado=mercado
                    )
                else:
                    status = ""
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

    def extract_precios_secundaria(self, day: datetime, dev: bool = False) -> None:
        """
        Not yet implemented in downlaoder since prices are retrived from ESIOS API directly
        """
        self._extract_and_save_precios(day, dev, 'secundaria', self.secundaria_downloader)

    def extract_precios_terciaria(self, day: datetime, dev: bool = False) -> None:
        """
        Not yet implemented in downlaoder since prices are retrived from ESIOS API directly
        """
        self._extract_and_save_precios(day, dev, 'terciaria', self.terciaria_downloader)

    def extract_precios_rr(self, day: datetime, dev: bool = False) -> None:
        """
        Not yet implemented in downlaoder since prices are retrived from ESIOS API directly
        """
        self._extract_and_save_precios(day, dev, 'rr', self.rr_downloader)

    def extract_precios_indisponibilidades(self, day: datetime, dev: bool = False) -> None:
        """
        Not yet implemented in downlaoder since prices are retrived from ESIOS API directly
        """
        self._extract_and_save_precios(day, dev, 'indisponibilidades', self.indisponibilidades_downloader)

    def extract_precios_restricciones(self, day: datetime, dev: bool = False) -> None:
        self._extract_and_save_precios(day, dev, 'restricciones', self.restricciones_downloader)


    def _extract_data_per_day_all_markets(self, day: datetime, dev: bool = False):

        """
        Extracts all precios data from I90 files for a given day

        Note: All prices come from API, not I90 file typically except for restricciones
        """

        try:
            # Call extraction methods only for markets with I90 price data

            #self.extract_precios_secundaria(day, dev=dev)
            #self.extract_precios_terciaria(day, dev=dev)
            #self.extract_precios_rr(day, dev=dev)
            #self.extract_precios_indisponibilidades(day, dev=dev)
            self.extract_precios_restricciones(day, dev=dev)

        except Exception as e:
            print(f"Error extracting data: {e}")

        # Note: Diario prices come from API, not I90 file typically.
        # Curtailment and P48 downloaders don't define get_i90_precios.
