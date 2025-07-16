import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import shutil
import os
import sys
from typing import Optional
from deprecated import deprecated

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utilidades.storage_file_utils import StorageFileUtils

class RawFileUtils(StorageFileUtils):
    """
    Utility class for processing and saving raw csv files.
    """
    def __init__(self) -> None:
        """
        Initialize RawFileUtils with the base path for raw data storage.
        """
        super().__init__()
        self.raw_path = self.base_path / 'raw'

    @staticmethod
    def drop_raw_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove exact duplicate rows from a DataFrame, keeping only the last occurrence of each duplicate.
        
        Returns:
            pd.DataFrame: DataFrame with exact duplicates removed.
        """
         # First remove exact duplicates across all columns
        df_before = len(df)
        duplicates = df[df.duplicated(keep=False)]  # Get all duplicates
        df_without_duplicates = df.drop_duplicates(keep='last')  
        exact_dups = df_before - len(df_without_duplicates)
        
        if exact_dups > 0:
            print(f"Removed {exact_dups} exact duplicate rows")
            print("Actual duplicates:")
            print(duplicates.head(10))
            print(duplicates.tail(10))
        else:
            print("No duplicates found")
        
        return df_without_duplicates

    #@deprecated(reason="This method is only for development/debugging purposes. Use write_raw_parquet for production code.")
    def write_raw_csv(self, year: int, month: int, df: pd.DataFrame, dataset_type: str, mercado: str) -> None:
        """
        Saves or appends a DataFrame as a raw CSV file in the structured data lake directory for the specified market, year, and month.
        
        If the target CSV file exists, reads and concatenates the new data, optionally removes exact duplicates (except for the "continuo" market), and overwrites the file. If the file does not exist, creates it. Raw data is saved without type conversions.
        
        Parameters:
            year (int): Year for file organization.
            month (int): Month for file organization.
            df (pd.DataFrame): DataFrame to be saved or appended.
            dataset_type (str): Type of dataset (must be valid for the project).
            mercado (str): Market name for file organization.
        
        Raises:
            ValueError: If the dataset type is invalid.
            FileNotFoundError: If the directory structure cannot be created.
            Exception: For errors during file reading, writing, or DataFrame processing.
        """
        # Validate dataset type
        self.validate_dataset_type(dataset_type)
        
        try:
            # Create directory structure
            #ie data/raw/mercado=secundaria/year=2025/month=04/precios.csv
            file_path = self.create_directory_structure(self.raw_path, mercado, year, month)
            filename = f"{dataset_type}.csv"
            full_file_path = file_path / filename
            
            #if file exists, read it and concatenate with new data
            if full_file_path.exists():
                try:
                    # Read existing CSV file
                    existing_df = pd.read_csv(full_file_path)
                    
                    # Concatenate with existing data
                    combined_df = pd.concat([existing_df, df], ignore_index=True)
                    
                    # Drop duplicates using raw string values -this allows for redownloads of the same day
                    #only if mercado is not continuo, because we want to keep all the data for continuo (there can be exact dups)
                    if mercado != "continuo":
                        print("Dropping raw duplicates")
                        combined_df = self.drop_raw_duplicates(combined_df)
                    else:
                        print("Not dropping raw duplicates for continuo market")
                    
                    # Save back to CSV without any type conversions
                    combined_df.to_csv(full_file_path, index=False)
                    print(f"Successfully updated existing file: {filename}")

                except pd.errors.EmptyDataError:
                    # Handle case where existing file is empty
                    df.to_csv(full_file_path, index=False)
                    print(f"Replaced empty file with new data: {filename}")

                except Exception as e:
                    print(f"Error reading existing file {filename}: {str(e)}")
                    raise 
                    
            else: #if file does not exist, create it by saving directly onto filepath
                if mercado != "continuo":
                    print("Dropping raw duplicates")
                    df = self.drop_raw_duplicates(df)
                # Create new file if it doesn't exist
                df.to_csv(full_file_path, index=False)
                print(f"Created new file: {filename}")

        except Exception as e:
            print(f"Error processing file {filename}: {str(e)}")
            raise

    @deprecated(reason="This method is only for development/debugging purposes. Use write_raw_parquet for production code.")
    def write_raw_excel(self, year: int, month: int, excel_file: pd.ExcelFile, dataset_type: str, mercado: str) -> None:
        """
        Processes a pd.ExcelFile and saves it as an Excel file in the appropriate directory structure. This method is used for I90 files.
        """
        try:
            # Create directory structure
            file_path = self.create_directory_structure(self.raw_path, mercado, year, month)
            filename = f"{dataset_type}.xlsx"  # Using the compressed naming format directly
            full_file_path = file_path / filename

            if full_file_path.exists():
                try:
                    # Read existing Excel file
                    existing_df = pd.read_excel(full_file_path)
                    
                    # Concatenate with existing data
                    combined_df = pd.concat([existing_df, excel_file], ignore_index=True)
                    
                    # Save back to Excel
                    combined_df.to_excel(full_file_path, index=False)
                    print(f"Successfully updated existing file: {filename}")
                    
                except Exception as e:
                    print(f"Error reading existing file {filename}: {str(e)}")
                    raise
            
            else:
                # Save the Excel file
                excel_file.to_excel(full_file_path, index=False)
                print(f"Created new file: {filename}")
            
        except Exception as e:
            print(f"Error processing file {filename}: {str(e)}")
            raise

    def write_raw_parquet(self, year: int, month: int, df: pd.DataFrame, dataset_type: str, mercado: str) -> None:
        """
        Saves a DataFrame as a Parquet file in the raw data directory, appending and deduplicating if the file already exists.
        
        If a Parquet file for the specified market, year, month, and dataset type exists, reads the existing file, appends new data, removes duplicate rows, and overwrites the file. If the file does not exist, creates a new Parquet file. Uses Snappy compression and the PyArrow engine. Raises exceptions on failure.
        """
        # Validate dataset type
        self.validate_dataset_type(dataset_type)
        
        try:
            print("\n" + "="*80)
            print(f"🔄 WRITING RAW PARQUET")
            print(f"Market: {mercado.upper()}")
            print(f"Period: {year}-{month:02d}")
            print("="*80)

            # Create directory structure
            file_path = self.create_directory_structure(self.raw_path, mercado, year, month)
            filename = f"{year}_{month:02d}_{dataset_type}.parquet"
            full_file_path = file_path / filename
            
            print("\n📂 FILE OPERATION")
            print("-"*50)
            
            if full_file_path.exists():
                try:
                    print("📌 Reading existing file...")
                    existing_df = pd.read_parquet(full_file_path)
                    print(f"   Records found: {len(existing_df)}")
                    
                    print("\n🔄 Merging data...")
                    combined_df = pd.concat([existing_df, df], ignore_index=True)
                    print(f"   Combined records: {len(combined_df)}")
                    
                    print("\n🧹 Removing duplicates...")
                    combined_df = self.drop_raw_duplicates(combined_df)
                    print(f"   Final records: {len(combined_df)}")
                    
                    print("\n💾 Saving updated file...")
                    combined_df.to_parquet(
                        full_file_path,
                        compression='snappy',
                        engine='pyarrow',
                        index=False
                    )
                    print(f"✅ Successfully updated: {filename}")
                    
                except Exception as e:
                    print(f"❌ Error processing existing file: {str(e)}")
                    raise
                
            else:
                print("📌 Creating new file...")
                print(f"   Records to write: {len(df)}")
                
                df.to_parquet(
                    full_file_path,
                    compression='snappy',
                    engine='pyarrow',
                    index=False
                )
                print(f"✅ Successfully created: {filename}")
            
            print("\n" + "="*80 + "\n")
                
        except Exception as e:
            print("\n❌ OPERATION FAILED")
            print(f"Error processing {filename}")
            print(f"Details: {str(e)}")
            print("="*80 + "\n")
            raise
    
    def delete_raw_files_older_than(self, months: int, mercado: Optional[str] = None, dataset_type: Optional[str] = None) -> None:
        """
        Delete raw CSV data directories older than a specified number of months.
        
        If a market is specified, only directories for that market are considered; otherwise, all standard markets are checked.
        If a dataset type is specified, only directories for that dataset type are considered.
        Removes entire month directories and cleans up empty year directories after deletion.
        """
        
        # Calculate cutoff date
        current_date = datetime.now()
        cutoff_date = current_date - timedelta(days=30 * months)
        
        # Prepare market filter
        markets_to_check = [mercado] if mercado else ["diario", "intra", "secundaria", "terciaria", "rr"]
        
        deleted_count = 0
        # Iterate through markets folders
        for market in markets_to_check:
            market_path = self.raw_path / f"mercado={market}"
            if not market_path.exists():
                continue
                
            # Iterate through year directories
            for year_dir in market_path.glob("year=*"):
                year = int(year_dir.name.split("=")[1])
                
                # Iterate through month directories
                for month_dir in year_dir.glob("month=*"):
                    month = int(month_dir.name.split("=")[1])
                    
                    # Create datetime for comparison (using first day of month)
                    dir_date = datetime(year, month, 1)
                    
                    # Check if dataset type is specified and if the directory matches
                    if dataset_type:
                        dataset_dir = month_dir / f"{dataset_type}.csv"  # Assuming dataset files are named as <dataset_type>.csv
                        if not dataset_dir.exists():
                            continue
                    
                    # Delete if dir date is older than cutoff
                    if dir_date < cutoff_date:
                        try:
                            shutil.rmtree(month_dir)
                            deleted_count += 1
                            print(f"Deleted {month_dir}")
                        except Exception as e:
                            print(f"Error deleting {month_dir}: {e}")
                
                # Remove empty year directories
                if not any(year_dir.iterdir()):
                    try:
                        year_dir.rmdir()
                        print(f"Removed empty directory {year_dir}")
                    except Exception as e:
                        print(f"Error removing empty directory {year_dir}: {e}")
        
        print(f"Deletion complete. Removed {deleted_count} directories older than {months} months.")

    def read_raw_file(self, year: int, month: int, dataset_type: str, mercado: str) -> pd.DataFrame:
        """
        Reads a raw CSV file for the specified market, year, month, and dataset type from the raw data directory.
        
        Parameters:
        	year (int): Year of the data file.
        	month (int): Month of the data file.
        	dataset_type (str): Dataset type identifier.
        	mercado (str): Market name.
        
        Returns:
        	pd.DataFrame: DataFrame containing the contents of the raw CSV file.
        
        Raises:
        	Exception: If the file cannot be read or does not exist.
        """

    
        file_path = os.path.join(self.raw_path, mercado, f"{year}", f"{month:02d}", f"{dataset_type}.{self.raw_file_extension}")

        try:
            return pd.read_csv(file_path)
            
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            raise
       
    def get_raw_folder_list(self, mercado: str = None, year: int = None) -> list[int]:
        """
        Get a list of years from a given market folder based on directory names.
        Assumes directory structure: processed_path/mercado/year/
        
        Args:
            mercado (str): The market name folder. (optional)
            year (int): The year folder. (optional)
        Returns:
            list[int]: A list of folder names found in the directory (it can be mercados, year or months).

        Note:
            -This method is used to batch process raw files (use this to get list of years or months to process, and 
            @read_raw_file to read the files)
        """

        if mercado != None and year == None:
            #when we pass only mercado, we want to get year folder list
            target_path = self.raw_path / mercado 

        elif mercado != None and year != None:
            #when we pass both mercado and year, we want to get month folder list
            target_path = self.raw_path / mercado / str(year)

        elif mercado == None and year == None:
            #when we pass no arguments, we want to get all mercados folder list
            target_path = self.raw_path 

        if target_path.exists() and target_path.is_dir():
            folder_list = []
            for item in target_path.iterdir():
                # Check if the item is a directory and its name represents a year (is numeric)
                if item.is_dir():
                    folder_list.append(int(item.name))

            # Sort the folder list in descending order (oldest year first)
            folder_list.sort()

            return folder_list
        else:
            print(f"Target path {target_path} does not exist or is not a directory.")
            return []

    def get_raw_file_list(self, mercado: str, year: int, month: int) -> list[str]:
        """
        Return a list of raw CSV file names for the specified market, year, and month.
        
        Parameters:
        	mercado (str): The market name.
        	year (int): The year of the files.
        	month (int): The month of the files.
        
        Returns:
        	list[str]: List of raw CSV file names in the specified directory.
        """
        file_path = self.raw_path / mercado / str(year) / f"{month:02d}"

        return list(file_path.glob(f"*.{self.raw_file_extension}"))

    def read_latest_raw_file(self, mercado: str, dataset_type: str) -> Optional[pd.DataFrame]:
        """
        Reads the latest available raw file for a given market and dataset type.

        It identifies the most recent year and month for which data exists and reads
        the corresponding raw file. It prioritizes Parquet format over CSV.

        Args:
            mercado (str): The market name.
            dataset_type (str): The type of dataset.

        Returns:
            Optional[pd.DataFrame]: A DataFrame with the contents of the latest raw file,
                                    or None if no file is found.
        """
        try:
            print(f"🔍 Searching for the latest raw file for market '{mercado}' and dataset '{dataset_type}'...")

            years = self.get_raw_folder_list(mercado=mercado)
            if not years:
                print(f"⚠️ No years found for market '{mercado}'.")
                return None
            
            # Iterate from the most recent year and month backwards
            for year in sorted(years, reverse=True):
                months = self.get_raw_folder_list(mercado=mercado, year=year)
                if not months:
                    continue

                for month in sorted(months, reverse=True):
                    file_path = self.raw_path / mercado / str(year) / f"{month:02d}"
                    
                    # Check for Parquet file first
                    csv_filename = f"{dataset_type}.csv"
                    csv_full_path = file_path / csv_filename
                    if csv_full_path.exists():
                        print(f"  📄 Found CSV file: {csv_full_path}")
                        return pd.read_csv(csv_full_path)

            print(f"⚠️ No raw file found for dataset '{dataset_type}' in market '{mercado}'.")
            return None

        except Exception as e:
            print(f"❌ Error reading the latest raw file: {e}")
            raise
