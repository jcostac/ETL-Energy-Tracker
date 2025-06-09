import requests
import pandas as pd
import zipfile
import os
import math
import pytz
import time
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from utilidades.db_utils import DatabaseUtils
from configs.i90_config import I90Config, DiarioConfig,TerciariaConfig, SecundariaConfig, RRConfig, CurtailmentConfig, P48Config, RestriccionesConfig, IndisponibilidadesConfig, IntraConfig

class I90Downloader:
    """
    Class for downloading and processing I90 files from ESIOS API.
    
    This class handles:
    - Downloading I90 files from ESIOS API
    - Extracting and processing the data from the Excel files
    - Applying time adjustments for special days (23/25 hours)
    - Filtering data by programming units and markets
    """
    
    def __init__(self):
        """Initialize the I90 downloader with ESIOS API token"""
        self.esios_token = os.getenv('ESIOS_API_KEY')
        self.config = I90Config()
        self.lista_errores = self.config.get_error_data()
        self.temporary_download_path = self.config.temporary_download_path

    @staticmethod
    def extract_date_from_file_name(excel_file_name: str) -> datetime:
        """
        Extract the date from the Excel file name.
        """
        #extarct name from filename ie 20241212.xls --> 2024-12-12 00:00:00
        date_str = excel_file_name.split('.')[0]
        return datetime.strptime(date_str, '%Y%m%d')
    
    def download_i90_file(self, day: datetime) -> Tuple[str, str, List[str]]:
        """
        Get I90 data for a specific day.
        
        Args:
            day (datetime): The specific day to retrieve data for
            volumenes_sheets (List[str]): List of sheet IDs to process for volumenes can be None
            precios_sheets (List[str]): List of sheet IDs to process for precios can be None
            
        Returns:
            pd.DataFrame: Processed data from the I90 file

        Notes:
            - If volumenes_sheets is None, only precios_sheets will be processed
            - If precios_sheets is None, only volumenes_sheets will be processed
        """
        
        # Download the file
        print(f"Downloading I90 file for {day.date()}...")
        print(f"--------------------------------")
        zip_file_name, excel_file_name = self._make_i90_request(day)
        print(f"Successfully downloaded I90 file for {day.date()}...")
        print(f"--------------------------------")

        # Get error data for the day
        df_errores_dia = self.lista_errores[self.lista_errores['fecha'] == day.date()]

        #if there are errors for that particular day
        if not df_errores_dia.empty:
            pestañas_con_error = df_errores_dia['tipo_error'].values.tolist()
        else: 
            pestañas_con_error = []

        return zip_file_name, excel_file_name, pestañas_con_error
        
    def extract_sheets_of_interest(self, excel_file_name: str, pestañas_con_error: List[str], volumenes_sheets: List[str] = None, precios_sheets: List[str] = None) -> pd.DataFrame:
        """
        Process the Excel file by filtering, extracting valid sheets, and converting to a DataFrame.
        
        Args:
            excel_file_name (str): Name of the Excel file to process
            pestañas_con_error (List[str]): List of sheet IDs with errors to skip
            volumenes_sheets (List[str]): List of sheet IDs for volume data, can be None
            precios_sheets (List[str]): List of sheet IDs for price data, can be None
        
        Returns:
            pd.DataFrame: The final melted DataFrame containing the requested data.
        """
        #extract the date from the file name
        fecha = self.extract_date_from_file_name(excel_file_name)

        #REMEMBER: one of these will be none (either precios or volumenes), hence the one that is none will be ignored
        #get valid sheets by filtering out the invalid sheets found in the error list
        volumenes_sheets, precios_sheets = self._get_valid_sheets(volumenes_sheets, precios_sheets, pestañas_con_error)

        #initialize the excel files to ensure finally block is executed (finally blcok will always execute regardless of whether an error is raised or not)
        volumenes_excel_file = None
        precios_excel_file = None

        try:
            #filter the excel file by the valid sheets per market (each market has its own valid sheets for prices and volumenes)
            print(f"Filtering relevant sheets...")
            # _filter_sheets creates the temporary files and returns ExcelFile objects
            volumenes_excel_file, precios_excel_file = self._filter_sheets(excel_file_name, volumenes_sheets, precios_sheets)
            print(f"--------------------------------")

            #excel files to dataframes 
            print(f"Converting excel files to dataframes...")
            # _excel_file_to_df uses the ExcelFile objects (needs them open)
            df = self._excel_file_to_df(fecha, volumenes_excel_file, precios_excel_file) #one of these will be none, hence the one that is none will be ignored
            print(f"--------------------------------")

            if volumenes_excel_file is not None:
                print(f"Successfully processed volumenes for {fecha.date()}:")

            elif precios_excel_file is not None:
                print(f"Successfully processed precios for {fecha.date()}:")
            
            if not df.empty:
                 print(df.head()) # Print head for brevity
            else:
                 print(f"No data processed into DataFrame for {fecha.date()}")


        finally:
            # --- Explicitly close the ExcelFile objects here ---
            # This ensures they are closed even if _excel_file_to_df raises an error
            if volumenes_excel_file is not None:
                try:
                    volumenes_excel_file.close()
                    #print(f"Closed temporary volumenes file object for {fecha.date()}")
                except Exception as e:
                    # Log error if closing fails, but don't stop execution
                    print(f"Warning: Error closing volumenes ExcelFile object: {e}")
            
            if precios_excel_file is not None:
                try:
                    precios_excel_file.close()
                    #print(f"Closed temporary precios file object for {fecha.date()}")
                except Exception as e:
                    print(f"Warning: Error closing precios ExcelFile object: {e}")
            # --- End of closing block ---

        return df # Return the final DataFrame
    
    def _make_i90_request(self, day: datetime) -> Tuple[str, str]:
        """
        Download I90 file for a specific day from ESIOS API.
        
        Args:
            day (datetime): Day to download data for
            
        Returns:
            Tuple[str, str]: File names for the downloaded and extracted files
        """
        # Construct API URL for the I90 file
        address = f"https://api.esios.ree.es/archives/34/download?date_type\u003ddatos\u0026end_date\u003d{day.date()}T23%3A59%3A59%2B00%3A00\u0026locale\u003des\u0026start_date\u003d{day.date()}T00%3A00%3A00%2B00%3A00"
        
        # Download the file
        resp = requests.get(
            address, 
            headers={
                'x-api-key': self.esios_token,
                'Authorization': f'Token token={self.esios_token}'
            }
        )
        
        # Save the downloaded file
        with open(f"{self.temporary_download_path}/{day.date()}.zip", "wb") as fd:
            fd.write(resp.content)
        
        # Create file names for the downloaded and extracted files
        zip_file_name = f"{day.year}-{str(day.month).zfill(2)}-{str(day.day).zfill(2)}" 
        excel_file_name = f"{day.year}{str(day.month).zfill(2)}{str(day.day).zfill(2)}" 
        
        # Extract the zip file
        with zipfile.ZipFile(f"{self.temporary_download_path}/{zip_file_name}.zip", 'r') as zip_ref:
            zip_ref.extractall(self.temporary_download_path)
        
        return zip_file_name, excel_file_name
    
    def _excel_file_to_df(self, fecha: datetime, volumenes_excel_file: pd.ExcelFile, precios_excel_file: pd.ExcelFile) -> pd.DataFrame:
      
        all_dfs = []

        if volumenes_excel_file is not None:
            excel_file = volumenes_excel_file
            value_col_name = "volumenes"
    
        elif precios_excel_file is not None:
            excel_file = precios_excel_file
            value_col_name = "precios"
        else:
            raise ValueError("No valid Excel file provided")
        
        #Loop through each sheet (each sheet represents a particular inidcator for a particualr market)
        for sheet_name in excel_file.sheet_names:

            # Read the sheet without a header to inspect the rows
            df_temp = pd.read_excel(excel_file, sheet_name=sheet_name, header=None)

            # Find the header row by looking for "Total" column
            header_row = 0
            for i, row in df_temp.iterrows():
                if "Total" in row.values:
                    header_row = i
                    break

            if header_row == 0:
                print(f"Could not find header row in sheet {sheet_name}. Skipping this sheet.")
                continue

            # Now read the sheet again, skipping the rows before the header
            df = pd.read_excel(excel_file, sheet_name=sheet_name, skiprows=header_row)

            # Identify the position of the 'Total' column
            total_col_idx = df.columns.get_loc('Total')
            #hora col name == column straight after total col
            hora_col_name = df.columns[total_col_idx + 1]
            
            # Split columns into identifier columns (before Total) and hour columns (after Total)
            id_cols = df.columns[:total_col_idx + 1].tolist()  # Include 'Total'
            hour_cols = df.columns[total_col_idx + 1:].tolist()  # Everything after 'Total'

            
            # Melt the DataFrame using the dynamic id_cols
            df_melted = df.melt(
                id_vars=id_cols,
                value_vars=hour_cols,
                var_name='hora',
                value_name= value_col_name
            )

            #add granularity column to the dataframe based on the hour col name
            if hora_col_name == 1:
                df_melted['granularity'] = "Quince minutos"
            else:
                df_melted['granularity'] = "Hora"
            
            #add date column to the dataframe
            df_melted['fecha'] = fecha
           

            all_dfs.append(df_melted)


        #concat and turn NaNs into 0s
        df_concat = pd.concat(all_dfs)
        #drop all columns that are fully NaNs
        df_concat = df_concat.dropna(axis=1, how='all')
        df_concat = df_concat.fillna(0).infer_objects()

        # Explicitly format the 'fecha' column to ensure consistency in the output
        if not df_concat.empty and 'fecha' in df_concat.columns:
            df_concat['fecha'] = pd.to_datetime(df_concat['fecha']).dt.strftime('%Y-%m-%d')

        return df_concat

    def _get_valid_sheets(self, volumenes_sheets: List[str], precios_sheets: List[str], pestañas_con_error: List[str]) -> Tuple[List[str], List[str]]:
        """
        Get valid sheets from the I90 Excel file.
        """
        # Validate sheets against the list of valid sheets
        # This will filter out any invalid sheet IDs from the input lists
        
        # Check for invalid sheets in volumenes_sheets and remove the
        if volumenes_sheets is not None:
            invalid_volumenes = [sheet for sheet in volumenes_sheets if sheet in pestañas_con_error]
            if invalid_volumenes:
                # Log the invalid sheets being removed
                print(f"Warning: Removing invalid volume sheets: {invalid_volumenes}")
                # Filter out invalid sheets
                volumenes_sheets = [sheet for sheet in volumenes_sheets if sheet not in pestañas_con_error]
                
    
        
        # Check for invalid sheets in precios_sheets and remove the
        if precios_sheets is not None:
            invalid_precios = [sheet for sheet in precios_sheets if sheet in pestañas_con_error]
            if invalid_precios:
                # Log the invalid sheets being removed
                print(f"Warning: Removing invalid price sheets: {invalid_precios}")
            # Filter out invalid sheets
            precios_sheets = [sheet for sheet in precios_sheets if sheet not in pestañas_con_error]
        
        # Check if we have any valid sheets left after filtering
        if not volumenes_sheets and not precios_sheets:
            raise ValueError("No valid sheets found after filtering. Please provide valid sheet IDs.")
        
        return volumenes_sheets, precios_sheets
    
    def _filter_sheets(self, excel_file_name: str, volumenes_sheets: List[str], precios_sheets: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process the I90 Excel file by filtering and keeping only specified sheets.
        
        Args:
            excel_file_name (str): Name of the Excel file to process
            volumenes_sheets (List[str]): List of sheet IDs for volume data
            precios_sheets (List[str]): List of sheet IDs for price data
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Tuple containing:
                - DataFrame with filtered volume data (or None if no volume sheets)
                - DataFrame with filtered price data (or None if no price sheets)
        """
        # Initialize result variables
        result_volumenes = None
        result_precios = None

        excel_file = pd.ExcelFile(f"{self.temporary_download_path}/I90DIA_{excel_file_name}.xls")
        
        # Process volumenes sheets if provided
        if volumenes_sheets:
            print(f"Filtering volumenes sheets: {volumenes_sheets}")
            all_sheets = excel_file.sheet_names
            
            # Match exact sheet numbers 
            filtered_sheets = [sheet for sheet in all_sheets if any(vs in sheet for vs in volumenes_sheets)]
            
            if not filtered_sheets:
                raise ValueError("At least one of the volume sheets must be found in the I90 file")
            
            #create a temporary filefor volumenes
            volumenes_path = f"{self.temporary_download_path}/I90DIA_{excel_file_name}_filtered_volumenes.xls"
            with pd.ExcelWriter(volumenes_path, engine='openpyxl') as writer:
                for sheet in filtered_sheets:
                    df = pd.read_excel(excel_file, sheet_name=sheet)
                    df.to_excel(writer, sheet_name=sheet, index=False)
                    
            result_volumenes = pd.ExcelFile(volumenes_path)
        # Process precios sheets if provided
        if precios_sheets:
            print(f"Filtering precios sheets: {precios_sheets}")
            all_sheets = excel_file.sheet_names
            
            filtered_sheets = [sheet for sheet in all_sheets if any(ps in sheet for ps in precios_sheets)]
            
            if not filtered_sheets:
                raise ValueError("At least one of the price sheets must be found in the I90 file")
            
            #create a temporary file for precios
            precios_path = f"{self.temporary_download_path}/I90DIA_{excel_file_name}_filtered_precios.xls"
            with pd.ExcelWriter(precios_path, engine='openpyxl') as writer:
                for sheet in filtered_sheets:
                    df = pd.read_excel(excel_file, sheet_name=sheet)
                    df.to_excel(writer, sheet_name=sheet, index=False)
                    
            result_precios = pd.ExcelFile(precios_path)

        return result_volumenes, result_precios
    
    def cleanup_files(self, zip_file_name: str, excel_file_name: str) -> None:
        """
        Clean up temporary files after processing, only if they exist.
        
        Args:
            zip_file_name (str): Base file name for the zip file
            excel_file_name (str): Base file name for the Excel file
        """
        # List of files to potentially remove
        files_to_cleanup = [
            f"{self.temporary_download_path}/{zip_file_name}.zip",
            f"{self.temporary_download_path}/I90DIA_{excel_file_name}.xls",
            f"{self.temporary_download_path}/I90DIA_{excel_file_name}_filtered_volumenes.xls",
            f"{self.temporary_download_path}/I90DIA_{excel_file_name}_filtered_precios.xls"
        ]
        
            # Remove each file only if it exists
        for file_path in files_to_cleanup:
            for attempt in range(5):  # Try up to 5 times
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        print(f"Successfully removed temporary file: {file_path}")
                    break  # Exit the retry loop if successful
                except PermissionError:
                    print(f"File {file_path} is in use. Attempt {attempt + 1} of 5.")
                    time.sleep(1)  # Wait for 1 second before retrying
                except Exception as e:
                    # Log the error but continue with other files
                    print(f"Error removing file {file_path}: {str(e)}")
                    break  # Exit the retry loop on other exceptions
        
class DiarioDL(I90Downloader):
    """
    Specialized class for downloading and processing diario data from I90 files.
    """
    
    def __init__(self):
        """Initialize the diario downloader"""
        super().__init__()

        self.config = DiarioConfig()
        self.precios_sheets = self.config.precios_sheets
        self.volumenes_sheets = self.config.volumenes_sheets

    def get_i90_volumenes(self, excel_file_name: str, pestañas_con_error: List[str]) -> pd.DataFrame:
        """
        Get diario (daily market) volume data for a specific day.
        
        Args:
            day (datetime): The specific day to retrieve data for
           
        Returns:
            pd.ExcelFile: Processed volume data from the I90 file containing:
                        - Programming unit data
                        - Hourly volumes
                        - Market specific information
                        
        Notes:
            - Uses sheet 26 from I90 file as specified in DiarioConfig
            - Processes daily market volumes only
            - Returns data in a standardized format for raw data lake insertion
        """
        df_volumenes = super().extract_sheets_of_interest(excel_file_name, pestañas_con_error, volumenes_sheets=self.volumenes_sheets, precios_sheets=None)
    
        return df_volumenes

    def get_i90_precios(self, excel_file_name: str, pestañas_con_error: List[str]) -> pd.DataFrame:
        """
        Get diario (daily market) price data for a specific day.
        
        Args:
            day (datetime): The specific day to retrieve data for
            precios_sheets (List[str]): List of sheet IDs to process for prices
        
        Returns:
            pd.DataFrame: Empty DataFrame as prices are obtained from ESIOS API
            
        Notes:
            - This method is not implemented as price data is retrieved from ESIOS API instead
            - The method exists for interface consistency with other market types
        """
        df_precios = super().extract_sheets_of_interest(excel_file_name, volumenes_sheets=None, precios_sheets=self.precios_sheets)
        return df_precios

class IntradiarioDL(I90Downloader):
    """
    Specialized class for downloading and processing intradiario volume data from I90 files.
    """
    def __init__(self, fecha: datetime = None):
        """
        Initialize the intradiario downloader
        
        Args:
            fecha (datetime, optional): Date for market configuration. If None, uses current date.
        """
        super().__init__()
        #fecha is passed to the IntraConfig to determine which sheets to use for the corresponding intra markets  (all sheets before intra reduciton or sheets for the corresponding intra markets after intra reduction)
        self.config = IntraConfig(fecha=fecha) 
        self.precios_sheets = self.config.precios_sheets #not used for intra i90
        self.volumenes_sheets = self.config.volumenes_sheets  

    def get_i90_volumenes(self, excel_file_name: str, pestañas_con_error: List[str]) -> pd.DataFrame:
        """
        Get intradiario volume data for a specific day.
        """
        df_volumenes = pd.DataFrame()
        for sheet in self.volumenes_sheets:
            df_volumenes
            df_volumenes = pd.concat([df_volumenes, super().extract_sheets_of_interest(excel_file_name, pestañas_con_error, volumenes_sheets=sheet, precios_sheets=None)])

        return df_volumenes
    
    
class SecundariaDL(I90Downloader):
    """
    Specialized class for downloading and processing secundaria volume data from I90 files.
    """
    def __init__(self):
        """Initialize the secundaria downloader"""
        super().__init__()
        self.config = SecundariaConfig()
        self.precios_sheets = self.config.precios_sheets
        self.volumenes_sheets = self.config.volumenes_sheets
              
    def get_i90_volumenes(self, excel_file_name: str, pestañas_con_error: List[str]) -> pd.DataFrame:
        """
        Get secundaria volume data for a specific day.
        
        Args:
            excel_file_name (str): Name of the Excel file to process
            pestañas_con_error (List[str]): List of sheet IDs with errors to skip
            
        Returns:
            pd.DataFrame: Processed volume data from the I90 file
        """
        df_volumenes = super().extract_sheets_of_interest(excel_file_name, pestañas_con_error, volumenes_sheets=self.volumenes_sheets, precios_sheets=None)
        
        #rename column that changed from the date of the SRS 
        if 'Participante del Mercado' in df_volumenes.columns:
            df_volumenes = df_volumenes.rename(columns={'Participante del Mercado': 'Unidad de Programación'})
            
        return df_volumenes

    def get_i90_precios(self, excel_file_name: str, pestañas_con_error: List[str]) -> pd.DataFrame:
        """
        Get secundaria price data for a specific day.
        
        Args:
            excel_file_name (str): Name of the Excel file to process
            pestañas_con_error (List[str]): List of sheet IDs with errors to skip
            
        Returns:
            pd.DataFrame: Processed price data from the I90 file
        """
        df_precios = super().extract_sheets_of_interest(excel_file_name, pestañas_con_error, volumenes_sheets=None, precios_sheets=self.precios_sheets)
        return df_precios

class TerciariaDL(I90Downloader):
    """
    Specialized class for downloading and processing tertiary regulation volume data from I90 files.
    """
    
    def __init__(self):
        """Initialize the tertiary regulation downloader"""
        super().__init__()

        #initialize config
        self.config = TerciariaConfig()
        self.precios_sheets = self.config.precios_sheets 
        self.volumenes_sheets = self.config.volumenes_sheets 

    def get_i90_volumenes(self, excel_file_name: str, pestañas_con_error: List[str]) -> pd.DataFrame:
        """
        Get tertiary regulation volume data.
        
        Args:
            excel_file_name (str): Name of the Excel file to process
            pestañas_con_error (List[str]): List of sheet IDs with errors to skip
            
        Returns:
            pd.DataFrame: Processed volume data containing upward and downward regulation
        """
        df_volumenes = super().extract_sheets_of_interest(excel_file_name, pestañas_con_error, volumenes_sheets=self.volumenes_sheets, precios_sheets=None)
        return df_volumenes
    
    def get_i90_precios(self, excel_file_name: str, pestañas_con_error: List[str]) -> pd.DataFrame:
        """
        Get tertiary regulation price data.
        
        Args:
            excel_file_name (str): Name of the Excel file to process
            pestañas_con_error (List[str]): List of sheet IDs with errors to skip
            
        Returns:
            pd.DataFrame: Processed price data
        """
        df_precios = super().extract_sheets_of_interest(excel_file_name, pestañas_con_error, volumenes_sheets=None, precios_sheets=self.precios_sheets)
        return df_precios

class RRDL(I90Downloader):
    """
    Specialized class for downloading and processing Replacement Reserve (RR) volume data from I90 files.
    """
    
    def __init__(self):
        """Initialize the RR downloader"""
        super().__init__()
        self.config = RRConfig()
        self.precios_sheets = self.config.precios_sheets
        self.volumenes_sheets = self.config.volumenes_sheets

    def get_i90_volumenes(self, excel_file_name: str, pestañas_con_error: List[str]) -> pd.DataFrame:
        """
        Get Replacement Reserve (RR) volume data.
        
        Args:
            excel_file_name (str): Name of the Excel file to process
            pestañas_con_error (List[str]): List of sheet IDs with errors to skip
            
        Returns:
            pd.DataFrame: Processed volume data for upward and downward RR
        """
        df_volumenes = super().extract_sheets_of_interest(excel_file_name, pestañas_con_error, volumenes_sheets=self.volumenes_sheets, precios_sheets=None)
        return df_volumenes
        
    def get_i90_precios(self, excel_file_name: str, pestañas_con_error: List[str]) -> pd.DataFrame:
        """
        Get Replacement Reserve (RR) price data.
        
        Args:
            excel_file_name (str): Name of the Excel file to process
            pestañas_con_error (List[str]): List of sheet IDs with errors to skip
            
        Returns:
            pd.DataFrame: Processed price data
        """
        df_precios = super().extract_sheets_of_interest(excel_file_name, pestañas_con_error, volumenes_sheets=None, precios_sheets=self.precios_sheets)
        return df_precios

class CurtailmentDL(I90Downloader):
    """
    Specialized class for downloading and processing curtailment volume data from I90 files.
    """
    
    def __init__(self):
        """Initialize the curtailment downloader"""
        super().__init__()
        self.config = CurtailmentConfig()
        self.precios_sheets = self.config.precios_sheets
        self.volumenes_sheets = self.config.volumenes_sheets

    def get_i90_volumenes(self, excel_file_name: str, pestañas_con_error: List[str]) -> pd.DataFrame:
        """
        Get curtailment volume data.
        
        Args:
            excel_file_name (str): Name of the Excel file to process
            pestañas_con_error (List[str]): List of sheet IDs with errors to skip
            
        Returns:
            pd.DataFrame: Processed curtailment volume data
        """
        df_volumenes = super().extract_sheets_of_interest(excel_file_name, pestañas_con_error, volumenes_sheets=self.volumenes_sheets, precios_sheets=None)
        return df_volumenes

class RestriccionesDL(I90Downloader):
    """
    Specialized class for downloading and processing restricciones de precios data from I90 files.
    """
    
    def __init__(self):
        """Initialize the restricciones de precios downloader"""
        super().__init__()

        self.config = RestriccionesConfig()
        self.precios_sheets = self.config.precios_sheets
        self.volumenes_sheets = self.config.volumenes_sheets
    
    def get_i90_volumenes(self, excel_file_name: str, pestañas_con_error: List[str]) -> pd.DataFrame:
        """
        Get technical restrictions volume data.
        
        Args:
            excel_file_name (str): Name of the Excel file to process
            pestañas_con_error (List[str]): List of sheet IDs with errors to skip
            
        Returns:
            pd.DataFrame: Processed volume data for all types of restrictions
        """
        df_volumenes = super().extract_sheets_of_interest(excel_file_name, pestañas_con_error, volumenes_sheets=self.volumenes_sheets, precios_sheets=None)
        return df_volumenes

    def get_i90_precios(self, excel_file_name: str, pestañas_con_error: List[str]) -> pd.DataFrame:
        """
        Get technical restrictions price data.
        
        Args:
            excel_file_name (str): Name of the Excel file to process
            pestañas_con_error (List[str]): List of sheet IDs with errors to skip
            
        Returns:
            pd.DataFrame: Processed price data for all types of restrictions
        """
        df_precios = super().extract_sheets_of_interest(excel_file_name, pestañas_con_error, volumenes_sheets=None, precios_sheets=self.precios_sheets)
        return df_precios

class P48DL(I90Downloader):
    """
    Specialized class for downloading and processing P48 data from I90 files.
    """
    
    def __init__(self):
        """Initialize the P48 downloader"""
        super().__init__()

        self.config = P48Config()
        self.volumenes_sheets = self.config.volumenes_sheets      

    def get_i90_volumenes(self, excel_file_name: str, pestañas_con_error: List[str]) -> pd.DataFrame:
        """
        Get P48 (Final Program) volume data.
        
        Args:
            excel_file_name (str): Name of the Excel file to process
            pestañas_con_error (List[str]): List of sheet IDs with errors to skip
            
        Returns:
            pd.DataFrame: Processed P48 volume data
        """
        df_volumenes = super().extract_sheets_of_interest(excel_file_name, pestañas_con_error, volumenes_sheets=self.volumenes_sheets, precios_sheets=None)
        return df_volumenes

class IndisponibilidadesDL(I90Downloader):
    """
    Specialized class for downloading and processing indisponibilidades data from I90 files.
    """
    
    def __init__(self):
        """Initialize the indisponibilidades downloader"""
        super().__init__()

        self.config = IndisponibilidadesConfig()
        self.volumenes_sheets = self.config.volumenes_sheets

    def get_i90_volumenes(self, excel_file_name: str, pestañas_con_error: List[str]) -> pd.DataFrame:
        """
        Get unavailability (indisponibilidades) volume data.
        
        Args:
            excel_file_name (str): Name of the Excel file to process
            pestañas_con_error (List[str]): List of sheet IDs with errors to skip
            
        Returns:
            pd.DataFrame: Processed unavailability volume data
        """
        df_volumenes = super().extract_sheets_of_interest(excel_file_name, pestañas_con_error, volumenes_sheets=self.volumenes_sheets, precios_sheets=None)
        return df_volumenes

    def get_i90_precios(self, excel_file_name: str, pestañas_con_error: List[str]) -> pd.DataFrame:
        """
        Get unavailability (indisponibilidades) price data.
        
        Args:
            excel_file_name (str): Name of the Excel file to process
            pestañas_con_error (List[str]): List of sheet IDs with errors to skip
            
        Returns:
            pd.DataFrame: Processed price data
        """
        df_precios = super().extract_sheets_of_interest(excel_file_name, pestañas_con_error, volumenes_sheets=None, precios_sheets=self.precios_sheets)
        return df_precios


