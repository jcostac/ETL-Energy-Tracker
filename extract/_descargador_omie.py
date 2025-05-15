import requests
import pandas as pd
import zipfile
import os
import time

class OMIEDownloader:
    """Base class for downloading data from OMIE"""
    
    def __init__(self):
        """Initialize the OMIE downloader"""
        # Temporary download path
        self.temporary_download_path = os.path.join(os.path.dirname(__file__), '../tmp')
        os.makedirs(self.temporary_download_path, exist_ok=True)
    
    def _download_file(self, base_url: str, file_name_prefix: str, year: int, month: int) -> str:
        """
        Download a file for a specific month.

        Args:
            base_url (str): The base URL for the download.
            file_name_prefix (str): The prefix for the file name.
            year (int): Year to download.
            month (int): Month to download.

        Returns:
            str: Path to the downloaded zip file.
        """
        file_name = f"{file_name_prefix}_{year}{str(month).zfill(2)}"
        address = f"{base_url}{file_name}.zip"
        zip_path = f"{self.temporary_download_path}/{file_name}.zip"
        
        print(f"Downloading file for {year}-{month:02d} from {address}")
        resp = requests.get(address)
        
        with open(zip_path, "wb") as fd:
            fd.write(resp.content)
        
        print(f"Downloaded zip file to {zip_path}")
        return zip_path

    def _extract_file_from_zip(self, zip_path: str, file_name_to_extract: str) -> str:
        """
        Extract a specific file from a zip archive.

        Args:
            zip_path (str): Path to the zip file.
            file_name_to_extract (str): The name of the file to extract from the zip.
            
        Returns:
            str: Path to the extracted file, or None if extraction fails.
        """
        extracted_file_path = f"{self.temporary_download_path}/{file_name_to_extract}"
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extract(file_name_to_extract, self.temporary_download_path)
            
            print(f"Extracted file: {file_name_to_extract}")
            return extracted_file_path
        except KeyError:
            print(f"File {file_name_to_extract} not found in zip.")
            return None
        except Exception as e:
            print(f"Error extracting file: {e}")
            return None

    def _read_data_from_file(self, file_path: str) -> pd.DataFrame:
        """
        Read data from a CSV file.

        Args:
            file_path (str): Path to the CSV file.

        Returns:
            pd.DataFrame: Raw data from the file.
        """
        if not file_path or not os.path.exists(file_path):
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(file_path, sep=";", skiprows=2, encoding='latin-1')
            return df
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return pd.DataFrame()
    
    def cleanup_files(self, file_name: str) -> None:
        """
        Clean up temporary files after processing
        
        Args:
            file_name (str): Base name of the files to clean up
        """
        # Implementation to be completed based on specific file patterns
        for attempt in range(3):  # Try up to 3 times
            try:
                if os.path.exists(file_name):
                    os.remove(file_name)
                break
            except PermissionError:
                print(f"File {file_name} is in use. Attempt {attempt + 1} of 3.")
                time.sleep(1)  # Wait for 1 second before retrying
            except Exception as e:
                print(f"Error removing file {file_name}: {str(e)}")
                break

class IntraOMIEDownloader(OMIEDownloader):
    """Downloader for intraday market data from OMIE"""
    
    def __init__(self):
        """Initialize the intraday OMIE downloader"""
        super().__init__()
        self.intra_url_base = "https://www.omie.es/es/file-download?parents=curva_pibc_uof&filename="
        self.file_prefix = "curva_pibc_uof"
    
    def download_intra_file(self, year: int, month: int) -> str:
        """
        Download intraday file for a specific month
        
        Args:
            year (int): Year to download
            month (int): Month to download
            
        Returns:
            str: Path to the downloaded zip file
        """
        return self._download_file(self.intra_url_base, self.file_prefix, year, month)
    
    def extract_session_file(self, zip_path: str, year: int, month: int, day: int, session: int) -> str:
        """
        Extract specific session file from zip
        
        Args:
            zip_path (str): Path to zip file
            year (int): Year for the session file
            month (int): Month for the session file
            day (int): Day for the session file
            session (int): Session number (1-6)
            
        Returns:
            str: Path to the extracted session file
        """
        file_name_base = f"{self.file_prefix}_{year}{str(month).zfill(2)}"
        session_file_name = file_name_base + str(day).zfill(2) + str(session).zfill(2) + ".1"
        return self._extract_file_from_zip(zip_path, session_file_name)
    
    def get_intra_data(self, session_path: str) -> pd.DataFrame:
        """
        Get intraday data from session file
        
        Args:
            session_path (str): Path to session file
            
        Returns:
            pd.DataFrame: Raw data from the session file
        """
        return self._read_data_from_file(session_path)

class ContinuoOMIEDownloader(OMIEDownloader):
    """Downloader for continuous market data from OMIE"""
    
    def __init__(self):
        """Initialize the continuous market OMIE downloader"""
        super().__init__()
        self.continuo_url_base = "https://www.omie.es/es/file-download?parents=trades&filename="
        self.file_prefix = "trades"
    
    def download_continuo_file(self, year: int, month: int) -> str:
        """
        Download continuous market file for a specific month
        
        Args:
            year (int): Year to download
            month (int): Month to download
            
        Returns:
            str: Path to the downloaded zip file
        """
        return self._download_file(self.continuo_url_base, self.file_prefix, year, month)
    
    def extract_day_file(self, zip_path: str, year: int, month: int, day: int) -> str:
        """
        Extract specific day file from zip
        
        Args:
            zip_path (str): Path to zip file
            year (int): Year for the day file
            month (int): Month for the day file
            day (int): Day for the day file
            
        Returns:
            str: Path to the extracted day file
        """
        file_name_base = f"{self.file_prefix}_{year}{str(month).zfill(2)}"
        day_file_name = file_name_base + str(day).zfill(2) + ".1"
        return self._extract_file_from_zip(zip_path, day_file_name)
    
    def get_continuo_data(self, day_path: str) -> pd.DataFrame:
        """
        Get continuous market data from day file
        
        Args:
            day_path (str): Path to day file
            
        Returns:
            pd.DataFrame: Raw data from the day file
        """
        return self._read_data_from_file(day_path)
