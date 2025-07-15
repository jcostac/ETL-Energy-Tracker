import requests
import pandas as pd
import zipfile
import os
import time
from datetime import datetime
from typing import List, Tuple
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from configs.i3_config import I3Config, DiarioConfig, SecundariaConfig, TerciariaConfig, RRConfig, CurtailmentConfig, P48Config, IndisponibilidadesConfig, RestriccionesConfig, IntraConfig

class I3Downloader:
    """
    Class for downloading and processing I3 files from ESIOS API.
    
    This class handles:
    - Downloading I3 files from ESIOS API
    - Extracting and processing the data from the Excel files
    - Filtering data by markets
    """
    
    def __init__(self):
        """
        Initialize the I3Downloader with the ESIOS API token and configuration settings.
        
        Loads the API token from the environment and sets the temporary download path for file operations.
        """
        self.esios_token = os.getenv('ESIOS_TOKEN')
        self.config = I3Config()
        # Note: Error data not implemented in i3_config; setting to empty for now
        self.lista_errores = pd.DataFrame()  # Empty DataFrame; adapt if error handling is added later
        self.temporary_download_path = self.config.temporary_download_path

    @staticmethod
    def extract_date_from_file_name(excel_file_name: str) -> datetime:
        """
        Extract the date from the Excel file name.
        """
        date_str = excel_file_name.split('.')[0]
        return datetime.strptime(date_str, '%Y%m%d')
    
    def download_i3_file(self, day: datetime) -> Tuple[str, str, List[str]]:
        """
        Get I3 data for a specific day.
        
        Args:
            day (datetime): The specific day to retrieve data for
            
        Returns:
            zip_file_name, excel_file_name, pestañas_con_error
        """
        
        # Download the file
        print(f"Downloading I3 file for {day.date()}...")
        print(f"--------------------------------")
        zip_file_name, excel_file_name = self._make_i3_request(day)
        print(f"Successfully downloaded I3 file for {day.date()}...")
        print(f"--------------------------------")

        # Get error data for the day (empty for now)
        df_errores_dia = self.lista_errores[self.lista_errores['fecha'] == day.date()] if not self.lista_errores.empty else pd.DataFrame()

        if not df_errores_dia.empty:
            pestañas_con_error = df_errores_dia['tipo_error'].values.tolist()
        else: 
            pestañas_con_error = []

        return zip_file_name, excel_file_name, pestañas_con_error
        
    def extract_sheets_of_interest(self, excel_file_name: str, pestañas_con_error: List[str], volumenes_sheets: List[str]) -> pd.DataFrame:
        """
        Process the Excel file by filtering, extracting valid sheets, and converting to a DataFrame.
        
        Args:
            excel_file_name (str): Name of the Excel file to process
            pestañas_con_error (List[str]): List of sheet IDs with errors to skip
            volumenes_sheets (List[str]): List of sheet IDs for volume data
        
        Returns:
            pd.DataFrame: The final melted DataFrame containing the requested data.
        """
        # Extract the date from the file name
        fecha = self.extract_date_from_file_name(excel_file_name)

        # Get valid sheets by filtering out the invalid sheets found in the error list
        volumenes_sheets = self._get_valid_sheets(volumenes_sheets, pestañas_con_error)

        # Initialize the excel file to ensure finally block is executed
        volumenes_excel_file = None

        try:
            # Filter the excel file by the valid sheets per market
            print(f"Filtering relevant sheets...")
            volumenes_excel_file = self._filter_sheets(excel_file_name, volumenes_sheets)
            print(f"--------------------------------")

            # Excel file to dataframe 
            print(f"Converting excel file to dataframe...")
            df = self._excel_file_to_df(fecha, volumenes_excel_file)
            print(f"--------------------------------")

            print(f"Successfully processed volumenes for {fecha.date()}:")
            if not df.empty:
                print(df.head())  # Print head for brevity
            else:
                print(f"No data processed into DataFrame for {fecha.date()}")

        finally:
            # Explicitly close the ExcelFile object
            if volumenes_excel_file is not None:
                try:
                    volumenes_excel_file.close()
                except Exception as e:
                    print(f"Warning: Error closing volumenes ExcelFile object: {e}")

        return df  # Return the final DataFrame
    
    def _make_i3_request(self, day: datetime) -> Tuple[str, str]:
        """
        Downloads the I3 zip file for a given day from the ESIOS API, saves it locally, extracts the contained Excel file, and returns their base file names.
        
        Parameters:
            day (datetime): The date for which to download the I3 file.
        
        Returns:
            Tuple[str, str]: The base names of the downloaded zip file and the extracted Excel file.
        """
        # Ensure temporary download directory exists
        os.makedirs(self.temporary_download_path, exist_ok=True)
        
        # Construct API URL for the I3 file
        address = f"https://api.esios.ree.es/archives/32/download?date_type=datos&end_date={day.date()}T23%3A59%3A59%2B00%3A00&locale=es&start_date={day.date()}T00%3A00%3A00%2B00%3A00"
        
        # Download the file
        resp = requests.get(
            address, 
            headers={
                'x-api-key': self.esios_token,
                'Authorization': f'Token token={self.esios_token}'
            }
        )
        resp.raise_for_status()
        
        # Save the downloaded file with a consistent name
        zip_file_name_on_disk = f"I3_{day.strftime('%Y-%m-%d')}.zip"
        zip_path = os.path.join(self.temporary_download_path, zip_file_name_on_disk)
        with open(zip_path, "wb") as fd:
            fd.write(resp.content)
        
        # Extract the zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Find the excel file in the zip archive
            excel_files = [f for f in zip_ref.namelist() if f.lower().endswith('.xls') or f.lower().endswith('.xlsx')]
            if not excel_files:
                raise FileNotFoundError("No Excel file found in the downloaded zip archive for I3.")
            
            # Assuming one excel file per zip
            extracted_excel_name = excel_files[0]
            zip_ref.extract(extracted_excel_name, path=self.temporary_download_path)

        # The excel file name part to be returned (e.g., '20250401')
        excel_file_name_part = os.path.basename(extracted_excel_name).replace('I3DIA_', '').replace('.xls', '')

        return zip_file_name_on_disk, excel_file_name_part
    
    def _excel_file_to_df(self, fecha: datetime, volumenes_excel_file: pd.ExcelFile) -> pd.DataFrame:
        """
        Convert relevant sheets from an Excel file into a cleaned, standardized DataFrame for a given date.
        
        Processes each sheet in the provided Excel file, applies custom filtering and melting logic, adds granularity and date columns, and concatenates all results. Fully empty columns are dropped, and rows with missing or zero values in the main value column are removed. Remaining missing values in other columns are filled with zero, and the date column is formatted as 'YYYY-MM-DD'.
        
        Parameters:
            fecha (datetime): The date associated with the data in the Excel file.
            volumenes_excel_file (pd.ExcelFile): The Excel file object containing volume data.
        
        Returns:
            pd.DataFrame: A DataFrame containing the cleaned and standardized data from all relevant sheets.
        """
        all_dfs = []
        value_col_name = "volumenes"  
        
        # Loop through each sheet
        for sheet_name in volumenes_excel_file.sheet_names:
            # Dynamically find the header row, similar to i90 logic
            df_temp = pd.read_excel(volumenes_excel_file, sheet_name=sheet_name, header=None)
            header_row = 0
            for i, row in df_temp.iterrows():
                # The header in I3 files contains 'Programa' or 'Concepto'
                if 'Total' in row.values or 'Concepto' in row.values:
                    header_row = i
                    break
            
            if header_row == 0:
                raise ValueError(f"Could not find header row in sheet {sheet_name}")

            # Now read the sheet again, skipping the rows before the header
            df = pd.read_excel(volumenes_excel_file, sheet_name=sheet_name, skiprows=header_row)

            # Identify the position of the 'Total' column
            total_col_idx = df.columns.get_loc('Total')
            #hora col name == column straight after total col (used for granularity check)
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
        
        # Drop rows where the value column (volumenes/precios) is NA or 0
        if value_col_name in df_concat.columns:
            # Drop rows with NA values in the volume/price column
            df_concat = df_concat.dropna(subset=[value_col_name])
            # Also drop rows with 0 values to further reduce overhead
            df_concat = df_concat[df_concat[value_col_name] != 0]
            print(f"✅ Filtered out NA and zero values. Remaining rows: {len(df_concat)}")
        else:
            # If no value column, just drop completely empty rows
            df_concat = df_concat.dropna(how='all')
        
        # Only fill remaining NAs with 0 for other columns (not the main value column)
        df_concat = df_concat.fillna(0).infer_objects()

        # Explicitly format the 'fecha' column to ensure consistency in the output
        if not df_concat.empty and 'fecha' in df_concat.columns:
            df_concat['fecha'] = pd.to_datetime(df_concat['fecha']).dt.strftime('%Y-%m-%d')

        return df_concat

    def _get_valid_sheets(self, volumenes_sheets: List[str], pestañas_con_error: List[str]) -> List[str]:
        """
        Get valid sheets from the I3 Excel file.
        """
        if not volumenes_sheets:
            return []
            
        invalid_volumenes = [sheet for sheet in volumenes_sheets if sheet in pestañas_con_error]
        if invalid_volumenes:
            print(f"Warning: Removing invalid volume sheets: {invalid_volumenes}")
        volumenes_sheets = [sheet for sheet in volumenes_sheets if sheet not in pestañas_con_error]
        
        if not volumenes_sheets:
            print("Warning: No valid sheets found after filtering.")
        
        return volumenes_sheets
    
    def _filter_sheets(self, excel_file_name: str, volumenes_sheets: List[str]) -> pd.ExcelFile:
        """
        Process the I3 Excel file by filtering and keeping only specified sheets.
        
        Args:
            excel_file_name (str): The date part of the Excel file name (e.g., '20250401')
            volumenes_sheets (List[str]): List of sheet IDs for volume data
            
        Returns:
            pd.ExcelFile: Filtered ExcelFile with volume data
        """
        full_excel_path = os.path.join(self.temporary_download_path, f"I3DIA_{excel_file_name}.xls")
        if not os.path.exists(full_excel_path):
             raise FileNotFoundError(f"Excel file not found at {full_excel_path}")

        excel_file = pd.ExcelFile(full_excel_path)
        
        print(f"Filtering volumenes sheets: {volumenes_sheets}")
        all_sheets = excel_file.sheet_names
        
        # Match exact sheet numbers 
        filtered_sheets = [sheet for sheet in all_sheets if any(vs in sheet for vs in volumenes_sheets)]
        
        if not filtered_sheets:
            print("Warning: None of the specified volume sheets were found in the I3 file.")
            # Return an empty ExcelFile object if no sheets are found
            empty_path = os.path.join(self.temporary_download_path, f"empty_{excel_file_name}.xls")
            pd.DataFrame().to_excel(empty_path)
            return pd.ExcelFile(empty_path)
        
        # Create a temporary file for volumenes
        volumenes_path = os.path.join(self.temporary_download_path, f"I3DIA_{excel_file_name}_filtered_volumenes.xls")
        with pd.ExcelWriter(volumenes_path, engine='openpyxl') as writer:
            for sheet in filtered_sheets:
                df = pd.read_excel(excel_file, sheet_name=sheet)
                df.to_excel(writer, sheet_name=sheet, index=False)
                
        return pd.ExcelFile(volumenes_path)
    
    def cleanup_files(self, zip_file_name: str, excel_file_name: str) -> None:
        """
        Clean up temporary files after processing, only if they exist.
        
        Args:
            zip_file_name (str): Base file name for the zip file (e.g., '2025-04-01.zip')
            excel_file_name (str): Base file name for the Excel file (e.g., '20250401')
        """
        # List of files to potentially remove (adapted for i3 naming)
        files_to_cleanup = [
            os.path.join(self.temporary_download_path, zip_file_name),
            os.path.join(self.temporary_download_path, f"I3DIA_{excel_file_name}.xls"),
            os.path.join(self.temporary_download_path, f"I3DIA_{excel_file_name}_filtered_volumenes.xls"),
            os.path.join(self.temporary_download_path, f"empty_{excel_file_name}.xls")
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
                    print(f"Error removing file {file_path}: {str(e)}")
                    break  # Exit the retry loop on other exceptions
        
class DiarioDL(I3Downloader):
    """
    Specialized class for downloading and processing diario data from I3 files.
    """
    
    def __init__(self):
        """Initialize the diario downloader"""
        super().__init__()
        self.config = DiarioConfig()
        self.volumenes_sheets = self.config.volumenes_sheets

    def get_i3_volumenes(self, excel_file_name: str, pestañas_con_error: List[str]) -> pd.DataFrame:
        """
        Get diario volume data for a specific day.
        """
        df_volumenes = super().extract_sheets_of_interest(excel_file_name, pestañas_con_error, volumenes_sheets=self.volumenes_sheets)
        return df_volumenes

class SecundariaDL(I3Downloader):
    """
    Specialized class for downloading and processing secundaria volume data from I3 files.
    """
    
    def __init__(self):
        """Initialize the secundaria downloader"""
        super().__init__()
        self.config = SecundariaConfig()
        self.volumenes_sheets = self.config.volumenes_sheets
              
    def get_i3_volumenes(self, excel_file_name: str, pestañas_con_error: List[str]) -> pd.DataFrame:
        """
        Get secundaria volume data for a specific day.
        """
        df_volumenes = super().extract_sheets_of_interest(excel_file_name, pestañas_con_error, volumenes_sheets=self.volumenes_sheets)
        return df_volumenes

class TerciariaDL(I3Downloader):
    """
    Specialized class for downloading and processing tertiary regulation volume data from I3 files.
    """
    
    def __init__(self):
        """Initialize the tertiary regulation downloader"""
        super().__init__()
        self.config = TerciariaConfig()
        self.volumenes_sheets = self.config.volumenes_sheets 

    def get_i3_volumenes(self, excel_file_name: str, pestañas_con_error: List[str]) -> pd.DataFrame:
        """
        Get tertiary regulation volume data.
        """
        df_volumenes = super().extract_sheets_of_interest(excel_file_name, pestañas_con_error, volumenes_sheets=self.volumenes_sheets)
        return df_volumenes

class RRDL(I3Downloader):
    """
    Specialized class for downloading and processing Replacement Reserve (RR) volume data from I3 files.
    """
    
    def __init__(self):
        """Initialize the RR downloader"""
        super().__init__()
        self.config = RRConfig()
        self.volumenes_sheets = self.config.volumenes_sheets

    def get_i3_volumenes(self, excel_file_name: str, pestañas_con_error: List[str]) -> pd.DataFrame:
        """
        Get Replacement Reserve (RR) volume data.
        """
        df_volumenes = super().extract_sheets_of_interest(excel_file_name, pestañas_con_error, volumenes_sheets=self.volumenes_sheets)
        return df_volumenes

class CurtailmentDL(I3Downloader):
    """
    Specialized class for downloading and processing curtailment volume data from I3 files.
    """
    
    def __init__(self):
        """Initialize the curtailment downloader"""
        super().__init__()
        self.config = CurtailmentConfig()
        self.volumenes_sheets = self.config.volumenes_sheets

    def get_i3_volumenes(self, excel_file_name: str, pestañas_con_error: List[str]) -> pd.DataFrame:
        """
        Get curtailment volume data.
        """
        df_volumenes = super().extract_sheets_of_interest(excel_file_name, pestañas_con_error, volumenes_sheets=self.volumenes_sheets)
        return df_volumenes

class RestriccionesDL(I3Downloader):
    """
    Specialized class for downloading and processing restricciones volume data from I3 files.
    """
    
    def __init__(self):
        """Initialize the restricciones downloader"""
        super().__init__()
        self.config = RestriccionesConfig()
        self.volumenes_sheets = self.config.volumenes_sheets
    
    def get_i3_volumenes(self, excel_file_name: str, pestañas_con_error: List[str]) -> pd.DataFrame:
        """
        Get technical restrictions volume data.
        """
        df_volumenes = super().extract_sheets_of_interest(excel_file_name, pestañas_con_error, volumenes_sheets=self.volumenes_sheets)
        return df_volumenes

class P48DL(I3Downloader):
    """
    Specialized class for downloading and processing P48 data from I3 files.
    """
    
    def __init__(self):
        """Initialize the P48 downloader"""
        super().__init__()
        self.config = P48Config()
        self.volumenes_sheets = self.config.volumenes_sheets      

    def get_i3_volumenes(self, excel_file_name: str, pestañas_con_error: List[str]) -> pd.DataFrame:
        """
        Get P48 (Final Program) volume data.
        """
        df_volumenes = super().extract_sheets_of_interest(excel_file_name, pestañas_con_error, volumenes_sheets=self.volumenes_sheets)
        return df_volumenes

class IndisponibilidadesDL(I3Downloader):
    """
    Specialized class for downloading and processing indisponibilidades data from I3 files.
    """
    
    def __init__(self):
        """Initialize the indisponibilidades downloader"""
        super().__init__()
        self.config = IndisponibilidadesConfig()
        self.volumenes_sheets = self.config.volumenes_sheets

    def get_i3_volumenes(self, excel_file_name: str, pestañas_con_error: List[str]) -> pd.DataFrame:
        """
        Get unavailability (indisponibilidades) volume data.
        """
        df_volumenes = super().extract_sheets_of_interest(excel_file_name, pestañas_con_error, volumenes_sheets=self.volumenes_sheets)
        return df_volumenes

class IntradiarioDL(I3Downloader):
    """
    Specialized class for downloading and processing intradiario volume data from I3 files.
    """
    def __init__(self, fecha: datetime = None):
        """
        Initialize the IntradiarioDL downloader for intraday market data.
        
        Parameters:
            fecha (datetime, optional): Date used to configure which intraday market sheets to process.
        """
        super().__init__()
        if fecha is None:
            raise ValueError("A date must be provided for IntradiarioDL")
        self.config = IntraConfig(fecha=fecha) 
        self.volumenes_sheets = self.config.volumenes_sheets  

    def get_i3_volumenes(self, excel_file_name: str, pestañas_con_error: List[str]) -> pd.DataFrame:
        """
        Get intradiario volume data for a specific day.
        """
        df_volumenes = super().extract_sheets_of_interest(excel_file_name, pestañas_con_error, volumenes_sheets=self.volumenes_sheets)
        return df_volumenes 