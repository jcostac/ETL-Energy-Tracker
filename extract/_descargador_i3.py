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
        Initialize the I3Downloader with ESIOS API credentials and configuration.
        
        Loads the API token from environment variables, sets up the configuration object, initializes an empty DataFrame for error tracking, and sets the temporary download path for file operations.
        """
        self.esios_token = os.getenv('ESIOS_TOKEN')
        self.config = I3Config()
        # Note: Error data not implemented in i3_config; setting to empty for now
        self.lista_errores = pd.DataFrame()  # Empty DataFrame; adapt if error handling is added later
        self.temporary_download_path = self.config.temporary_download_path

    @staticmethod
    def extract_date_from_file_name(excel_file_name: str) -> datetime:
        """
        Extracts a date object from an Excel file name based on the '%Y%m%d' format.
        
        Parameters:
            excel_file_name (str): Name of the Excel file, expected to start with a date in 'YYYYMMDD' format.
        
        Returns:
            datetime: The extracted date as a datetime object.
        """
        date_str = excel_file_name.split('.')[0]
        return datetime.strptime(date_str, '%Y%m%d')
    
    def download_i3_file(self, day: datetime) -> Tuple[str, str, List[str]]:
        """
        Downloads the I3 ZIP file for a given day, extracts the Excel file, and identifies any sheets with known errors.
        
        Parameters:
        	day (datetime): The date for which to download the I3 data.
        
        Returns:
        	A tuple containing the ZIP file name, the extracted Excel file name, and a list of sheet names with errors (empty if none are recorded).
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
        Extracts and processes specified sheets from an Excel file, returning a cleaned DataFrame with standardized columns.
        
        Filters out sheets listed as erroneous, loads the relevant sheets, converts them into a unified DataFrame, and ensures proper resource cleanup. The resulting DataFrame contains only the requested volume data, formatted and cleaned for further analysis.
        
        Parameters:
            excel_file_name (str): The name of the Excel file to process.
            pestañas_con_error (List[str]): List of sheet names to exclude due to errors.
            volumenes_sheets (List[str]): List of sheet names to extract and process.
        
        Returns:
            pd.DataFrame: A cleaned DataFrame containing the extracted and standardized data from the specified sheets.
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
        Download the I3 ZIP file for a specified date from the ESIOS API, extract the contained Excel file, and return their base file names.
        
        Parameters:
            day (datetime): The date for which to download and extract the I3 file.
        
        Returns:
            Tuple[str, str]: The base name of the downloaded ZIP file and the extracted Excel file (with identifying prefix and extension removed).
        
        Raises:
            FileNotFoundError: If no Excel file is found in the downloaded ZIP archive.
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
        Convert all relevant sheets from an Excel file into a cleaned and standardized DataFrame for a specific date.
        
        Processes each sheet by dynamically detecting the header row, reshaping the data to long format, adding granularity and date columns, and concatenating the results. Drops fully empty columns and removes rows with missing or zero values in the main value column. Remaining missing values in other columns are filled with zero, and the date column is formatted as 'YYYY-MM-DD'.
        
        Parameters:
            fecha (datetime): The date associated with the data in the Excel file.
            volumenes_excel_file (pd.ExcelFile): The Excel file object containing the sheets to process.
        
        Returns:
            pd.DataFrame: Cleaned and standardized data from all relevant sheets.
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
        Return a list of valid sheet names by excluding those marked as erroneous.
        
        Parameters:
        	volumenes_sheets (List[str]): List of sheet names to consider.
        	pestañas_con_error (List[str]): List of sheet names identified as containing errors.
        
        Returns:
        	List[str]: Sheet names that are not in the error list.
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
        Filter and extract specified sheets from an I3 Excel file, returning a new ExcelFile containing only those sheets.
        
        Parameters:
            excel_file_name (str): The date portion of the Excel file name (e.g., '20250401').
            volumenes_sheets (List[str]): List of sheet identifiers to retain.
        
        Returns:
            pd.ExcelFile: An ExcelFile object containing only the filtered sheets. If none of the specified sheets are found, returns an empty ExcelFile.
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
        Delete temporary ZIP and Excel files associated with a download, retrying on permission errors.
        
        Attempts to remove all relevant temporary files for the specified ZIP and Excel base names, retrying up to five times if a file is in use.
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
        """
        Initialize the DiarioDL downloader with the DiarioConfig settings.
        
        Sets up the configuration and list of volume sheets specific to diario market data extraction.
        """
        super().__init__()
        self.config = DiarioConfig()
        self.volumenes_sheets = self.config.volumenes_sheets

    def get_i3_volumenes(self, excel_file_name: str, pestañas_con_error: List[str]) -> pd.DataFrame:
        """
        Extracts and returns the diario volume data from the specified Excel file, excluding sheets listed as having errors.
        
        Parameters:
            excel_file_name (str): Path to the Excel file to process.
            pestañas_con_error (List[str]): List of sheet names to exclude due to errors.
        
        Returns:
            pd.DataFrame: DataFrame containing the cleaned diario volume data.
        """
        df_volumenes = super().extract_sheets_of_interest(excel_file_name, pestañas_con_error, volumenes_sheets=self.volumenes_sheets)
        return df_volumenes

class SecundariaDL(I3Downloader):
    """
    Specialized class for downloading and processing secundaria volume data from I3 files.
    """
    
    def __init__(self):
        """
        Initialize the downloader for secundaria market data, setting the appropriate configuration and target sheets for extraction.
        """
        super().__init__()
        self.config = SecundariaConfig()
        self.volumenes_sheets = self.config.volumenes_sheets
              
    def get_i3_volumenes(self, excel_file_name: str, pestañas_con_error: List[str]) -> pd.DataFrame:
        """
        Extracts and returns secundaria volume data from the specified Excel file, excluding sheets with known errors.
        
        Parameters:
        	excel_file_name (str): Path to the Excel file to process.
        	pestañas_con_error (List[str]): List of sheet names to exclude due to errors.
        
        Returns:
        	pd.DataFrame: DataFrame containing cleaned secundaria volume data from the valid sheets.
        """
        df_volumenes = super().extract_sheets_of_interest(excel_file_name, pestañas_con_error, volumenes_sheets=self.volumenes_sheets)
        return df_volumenes

class TerciariaDL(I3Downloader):
    """
    Specialized class for downloading and processing tertiary regulation volume data from I3 files.
    """
    
    def __init__(self):
        """
        Initialize the downloader for tertiary regulation market data.
        
        Sets up the configuration and specifies the relevant Excel sheets to process for tertiary regulation volumes.
        """
        super().__init__()
        self.config = TerciariaConfig()
        self.volumenes_sheets = self.config.volumenes_sheets 

    def get_i3_volumenes(self, excel_file_name: str, pestañas_con_error: List[str]) -> pd.DataFrame:
        """
        Extracts and returns tertiary regulation volume data from the specified Excel file, excluding sheets with known errors.
        
        Parameters:
        	excel_file_name (str): Path to the Excel file to process.
        	pestañas_con_error (List[str]): List of sheet names to exclude due to errors.
        
        Returns:
        	pd.DataFrame: DataFrame containing cleaned tertiary regulation volume data from the valid sheets.
        """
        df_volumenes = super().extract_sheets_of_interest(excel_file_name, pestañas_con_error, volumenes_sheets=self.volumenes_sheets)
        return df_volumenes

class RRDL(I3Downloader):
    """
    Specialized class for downloading and processing Replacement Reserve (RR) volume data from I3 files.
    """
    
    def __init__(self):
        """
        Initialize the Replacement Reserve (RR) downloader with the appropriate configuration and sheet selection.
        """
        super().__init__()
        self.config = RRConfig()
        self.volumenes_sheets = self.config.volumenes_sheets

    def get_i3_volumenes(self, excel_file_name: str, pestañas_con_error: List[str]) -> pd.DataFrame:
        """
        Extracts Replacement Reserve (RR) volume data from the specified Excel file, excluding sheets with known errors.
        
        Parameters:
        	excel_file_name (str): Path to the Excel file to process.
        	pestañas_con_error (List[str]): List of sheet names to exclude due to errors.
        
        Returns:
        	pd.DataFrame: DataFrame containing the cleaned RR volume data from the valid sheets.
        """
        df_volumenes = super().extract_sheets_of_interest(excel_file_name, pestañas_con_error, volumenes_sheets=self.volumenes_sheets)
        return df_volumenes

class CurtailmentDL(I3Downloader):
    """
    Specialized class for downloading and processing curtailment volume data from I3 files.
    """
    
    def __init__(self):
        """
        Initialize the CurtailmentDL downloader with the CurtailmentConfig and set the relevant volume sheets for curtailment data extraction.
        """
        super().__init__()
        self.config = CurtailmentConfig()
        self.volumenes_sheets = self.config.volumenes_sheets

    def get_i3_volumenes(self, excel_file_name: str, pestañas_con_error: List[str]) -> pd.DataFrame:
        """
        Extracts curtailment volume data from the specified Excel file, excluding sheets with known errors.
        
        Parameters:
        	excel_file_name (str): Path to the Excel file to process.
        	pestañas_con_error (List[str]): List of sheet names to exclude due to errors.
        
        Returns:
        	pd.DataFrame: DataFrame containing the cleaned curtailment volume data.
        """
        df_volumenes = super().extract_sheets_of_interest(excel_file_name, pestañas_con_error, volumenes_sheets=self.volumenes_sheets)
        return df_volumenes

class RestriccionesDL(I3Downloader):
    """
    Specialized class for downloading and processing restricciones volume data from I3 files.
    """
    
    def __init__(self):
        """
        Initialize the downloader for technical restrictions data, setting the appropriate configuration and target sheets for extraction.
        """
        super().__init__()
        self.config = RestriccionesConfig()
        self.volumenes_sheets = self.config.volumenes_sheets
    
    def get_i3_volumenes(self, excel_file_name: str, pestañas_con_error: List[str]) -> pd.DataFrame:
        """
        Extracts and returns technical restrictions volume data from the specified Excel file, excluding sheets with known errors.
        
        Parameters:
        	excel_file_name (str): Path to the Excel file to process.
        	pestañas_con_error (List[str]): List of sheet names to exclude due to errors.
        
        Returns:
        	pd.DataFrame: DataFrame containing the cleaned technical restrictions volume data.
        """
        df_volumenes = super().extract_sheets_of_interest(excel_file_name, pestañas_con_error, volumenes_sheets=self.volumenes_sheets)
        return df_volumenes

class P48DL(I3Downloader):
    """
    Specialized class for downloading and processing P48 data from I3 files.
    """
    
    def __init__(self):
        """
        Initialize the P48DL downloader with configuration for P48 (Final Program) volume data extraction.
        """
        super().__init__()
        self.config = P48Config()
        self.volumenes_sheets = self.config.volumenes_sheets      

    def get_i3_volumenes(self, excel_file_name: str, pestañas_con_error: List[str]) -> pd.DataFrame:
        """
        Extracts and returns P48 (Final Program) volume data from the specified Excel file, excluding sheets with known errors.
        
        Parameters:
        	excel_file_name (str): Path to the Excel file to process.
        	pestañas_con_error (List[str]): List of sheet names to exclude due to errors.
        
        Returns:
        	pd.DataFrame: DataFrame containing the cleaned and standardized P48 volume data.
        """
        df_volumenes = super().extract_sheets_of_interest(excel_file_name, pestañas_con_error, volumenes_sheets=self.volumenes_sheets)
        return df_volumenes

class IndisponibilidadesDL(I3Downloader):
    """
    Specialized class for downloading and processing indisponibilidades data from I3 files.
    """
    
    def __init__(self):
        """
        Initialize the downloader for indisponibilidades (unavailability) data, setting the appropriate configuration and target sheets for extraction.
        """
        super().__init__()
        self.config = IndisponibilidadesConfig()
        self.volumenes_sheets = self.config.volumenes_sheets

    def get_i3_volumenes(self, excel_file_name: str, pestañas_con_error: List[str]) -> pd.DataFrame:
        """
        Extracts and returns unavailability volume data from the specified Excel file, excluding sheets with known errors.
        
        Parameters:
        	excel_file_name (str): Path to the Excel file to process.
        	pestañas_con_error (List[str]): List of sheet names to exclude due to errors.
        
        Returns:
        	pd.DataFrame: DataFrame containing the cleaned unavailability volume data.
        """
        df_volumenes = super().extract_sheets_of_interest(excel_file_name, pestañas_con_error, volumenes_sheets=self.volumenes_sheets)
        return df_volumenes

class IntradiarioDL(I3Downloader):
    """
    Specialized class for downloading and processing intradiario volume data from I3 files.
    """
    def __init__(self, fecha: datetime = None):
        """
        Initialize the IntradiarioDL downloader for extracting intraday market data for a specific date.
        
        Parameters:
            fecha (datetime, optional): The date used to configure which intraday market sheets to process. Must be provided or a ValueError is raised.
        """
        super().__init__()
        if fecha is None:
            raise ValueError("A date must be provided for IntradiarioDL")
        self.config = IntraConfig(fecha=fecha) 
        self.volumenes_sheets = self.config.volumenes_sheets  

    def get_i3_volumenes(self, excel_file_name: str, pestañas_con_error: List[str]) -> pd.DataFrame:
        """
        Extracts and returns intradiario volume data from the specified Excel file, excluding sheets with known errors.
        
        Parameters:
        	excel_file_name (str): Path to the Excel file to process.
        	pestañas_con_error (List[str]): List of sheet names to exclude due to errors.
        
        Returns:
        	pd.DataFrame: DataFrame containing cleaned intradiario volume data for the specified day.
        """
        df_volumenes = super().extract_sheets_of_interest(excel_file_name, pestañas_con_error, volumenes_sheets=self.volumenes_sheets)
        return df_volumenes 