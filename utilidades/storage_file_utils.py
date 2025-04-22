"""
Utility class for processing and saving parquet files.
"""

__all__ = ['StorageFileUtils', 'RawFileUtils', 'ProcessedFileUtils']

import os 
from typing import Optional
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from datetime import datetime, timedelta
import shutil
import sys
import os
from deprecated import deprecated
# Get the absolute path to the scripts directory
SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(SCRIPTS_DIR))

from configs.storage_config import DATA_LAKE_BASE_PATH, VALID_DATASET_TYPES
from utilidades.env_utils import EnvUtils

class StorageFileUtils:
    """
    Utility class for processing and saving parquet files.
    """
    def __init__(self, use_s3: bool = False) -> None:
        """
        Initializes the ParquetUtils class and sets the base path.

        Sets 

        Args:
            use_s3 (bool): Flag to determine if S3 path should be used.
        """

        #check if dev or prod environment
        env_utils = EnvUtils()
        self.dev, self.prod = env_utils.check_dev_env() 

        #set base path for s3 or local storage
        if self.dev and not self.prod:
            self.base_path = self.set_base_path(False) #use s3 set to false
        else:
            self.base_path = self.set_base_path(True) #use s3 set to true

    def set_base_path(self, use_s3: bool = False) -> Path:
        """
        Sets the base path for data storage.

        Args:
            use_s3 (bool): Flag to determine if S3 path should be used.

        Returns:
            Path: The base path for data storage.
        """
        if use_s3:
            # Add logic for S3 path if needed
            base_path = Path(f"s3://{os.getenv('S3_BUCKET_NAME')}/{base_path}")
            
        else:
            base_path = Path(DATA_LAKE_BASE_PATH)

        # Check if the base path exists
        if not base_path.exists():
            raise FileNotFoundError(f"The base path {base_path} does not exist.")
        
        return base_path

    @staticmethod
    def create_directory_structure(path: str, mercado: str, year: int, month: int, day = None) -> Path:
        """
        Creates the necessary directory structure for storing parquet files.
        
        Args:
            path (str or Path): Base directory path where the folder structure will be created
            mercado (str): Market name for file organization ('diario', 'intra', 'secundaria', 'terciaria', 'rr')
            year (int): Year for the directory structure
            month (int): Month for the directory structure
            day (int): Day for the directory structure, optional if we want to save at the day level
        Returns:
            Path: Path object pointing to the created month directory
    """
        # Convert string path to Path object if necessary
        path = Path(path)

        #create mercado directory
        market_dir = path / f"{mercado}"  # "data/mercado=secundaria"

        # Create year and month directories
        year_dir = market_dir / f"{year}" # "data/mercado=secundaria/year=2025"
    
        month_dir = year_dir / f"{month:02d}" # "data/mercado=secundaria/year=2025/month=04"

        #create directories if they don't exist
        if day is None:
            month_dir.mkdir(parents=True, exist_ok=True)
            return month_dir
        else:
            day_dir = month_dir / f"{day:02d}" # "data/mercado=secundaria/year=2025/month=04/day=01"
            day_dir.mkdir(parents=True, exist_ok=True)
            return day_dir      

    @staticmethod
    def validate_dataset_type(dataset_type: str) -> tuple[str]:
        """
        Checks if data set is a valid type.
        """

        # Validate data_types (WIP - add more types)
        valid_types = VALID_DATASET_TYPES

        if dataset_type not in valid_types:
            raise ValueError(f"data_type must be one of {valid_types}")
    
class RawFileUtils(StorageFileUtils):
    """
    Utility class for processing and saving raw csv files.
    """
    def __init__(self, use_s3: bool = False) -> None:
        super().__init__(use_s3)
        self.raw_path = self.base_path / 'raw'

    @staticmethod
    def drop_raw_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        """
        Drops duplicates from the DataFrame based on the data type and mercado. Used for processing of raw data. 
        This allows for redownloads of the same day since we keep last download of the same day.
        Args:
            df (pd.DataFrame): The DataFrame to drop duplicates from
            dataset_type (str): The type of dataset to drop duplicates from
            WIP redownload (bool): Whether to redownload the data
        Returns:
            pd.DataFrame: The DataFrame with duplicates dropped
        """
         # First remove exact duplicates across all columns
        df_before = len(df)
        df = df.drop_duplicates(keep='last')  
        exact_dups = df_before - len(df)
        
        if exact_dups > 0:
            print(f"Removed {exact_dups} exact duplicate rows")

        return df

    @deprecated(reason="This method is only for development/debugging purposes. Use write_raw_parquet for production code.")
    def write_raw_csv(self, year: int, month: int, df: pd.DataFrame, dataset_type: str, mercado: str) -> None:
        """
        Processes a DataFrame and saves/appends it as a CSV file in the appropriate directory structure.
        Raw data is saved as-is without type conversions.
        
        Args:
            mercado (str): Market name for file organization ('diario', 'intra', 'secundaria', 'terciaria', 'rr')
            year (int): Year for file organization
            month (int): Month for file organization
            df (pd.DataFrame): Input DataFrame to be saved/appended
            dataset_type (str): Type of data ('volumenes_i90', 'volumenes_i3', 'precios', or 'ingresos')
        
        Raises:
            ValueError: If dataset_type is invalid or DataFrame validation fails
            FileNotFoundError: If directory structure cannot be created
        """
        # Validate dataset type
        self.validate_dataset_type(dataset_type)
        
        try:
            # Create directory structure
            #ie data/raw/mercado=secundaria/year=2025/month=04/precios.csv
            file_path = self.create_directory_structure(self.raw_path, mercado, year, month)
            filename = f"{dataset_type}.csv"
            full_file_path = file_path / filename
            
            if full_file_path.exists():
                try:
                    # Read existing CSV file
                    existing_df = pd.read_csv(full_file_path)
                    
                    # Concatenate with existing data
                    combined_df = pd.concat([existing_df, df], ignore_index=True)
                    
                    # Drop duplicates using raw string values -this allows for redownloads of the same day
                    combined_df = self.drop_raw_duplicates(combined_df)
                    
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
                    
            else:
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
        Processes a DataFrame and saves/appends it as a Parquet file in the appropriate directory structure.
        Raw data is saved with proper type conversions and efficient compression.
        
        Args:
            mercado (str): Market name for file organization ('diario', 'intra', 'secundaria', 'terciaria', 'rr')
            year (int): Year for file organization
            month (int): Month for file organization
            df (pd.DataFrame): Input DataFrame to be saved/appended
            dataset_type (str): Type of data ('volumenes_i90', 'volumenes_i3', 'precios', or 'ingresos')
        
        Raises:
            ValueError: If dataset_type is invalid or DataFrame validation fails
            FileNotFoundError: If directory structure cannot be created
        """
        # Validate dataset type
        self.validate_dataset_type(dataset_type)
        
        try:
            # Create directory structure
            file_path = self.create_directory_structure(self.raw_path, mercado, year, month)
            filename = f"{year}_{month:02d}_{dataset_type}.parquet"  # Using the compressed naming format directly
            full_file_path = file_path / filename
            
            if full_file_path.exists():
                try:
                    # Read existing parquet file
                    existing_df = pd.read_parquet(full_file_path)
                    
                    # Concatenate with existing data
                    combined_df = pd.concat([existing_df, df], ignore_index=True)
                    
                    # Drop duplicates using raw string values
                    combined_df = self.drop_raw_duplicates(combined_df)
                    
                    # Save back to parquet with compression
                    combined_df.to_parquet(
                        full_file_path,
                        compression='snappy',
                        engine = 'pyarrow',
                        index=False
                    )
                    print(f"Successfully updated existing file: {filename}")
                    
                except Exception as e:
                    print(f"Error reading existing file {filename}: {str(e)}")
                    raise
                
            else:
                # Create new file if it doesn't exist
                df.to_parquet(
                    full_file_path,
                    compression='snappy',
                    engine = 'pyarrow',
                    index=False
                )
                print(f"Created new file: {filename}")
                
        except Exception as e:
            print(f"Error processing file {filename}: {str(e)}")
            raise
    
    def delete_raw_files_older_than(self, months: int, mercado: Optional[str] = None) -> None:
        """
        Deletes raw CSV files that are older than the specified number of months.
        
        Args:
            months (int): Number of months. Files older than this will be deleted
            mercado (Optional[str]): Optional market filter. If provided, only files from this market are deleted
            
        Returns:
            None
        """
        
        # Calculate cutoff date
        current_date = datetime.now()
        cutoff_date = current_date - timedelta(days=30 * months)
        
        # Prepare market filter
        markets_to_check = [mercado] if mercado else ["diario", "intra", "secundaria", "terciaria", "rr"]
        
        deleted_count = 0
        #iterate through markets folders
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
        Reads a raw file from the appropriate directory structure. 
        Args:
            year (int): The year of the file
            month (int): The month of the file
            dataset_type (str): The type of dataset
            mercado (str): The market name
        Returns:
            pd.DataFrame: The DataFrame containing the data

        Note:
            -This method is used to read raw files that are processed on a daily basis. (Fucntionality reads latest file)
        """

        if self.dev and not self.prod:
            file_extension = 'csv'
        else:
            file_extension = 'parquet'
    
        file_path = os.path.join(self.raw_path, mercado, f"{year}", f"{month:02d}", f"{dataset_type}.{file_extension}")

        try:
            if file_extension == 'csv':
                return pd.read_csv(file_path)
            else:
                return pd.read_parquet(file_path)
            
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            raise
       
    def get_raw_folder_list(self, mercado: str, year: int = None) -> list[int]:
        """
        Get a list of years from a given market folder based on directory names.
        Assumes directory structure: processed_path/mercado/year/
        
        Args:
            mercado (str): The market name folder.
            year (int): The year folder. (optional)
        Returns:
            list[int]: A list of folder names found in the directory (it can be years or months since ).

        Note:
            -This method is used to batch process raw files (use this to get list of years or months to process, and 
            @read_raw_file to read the files)
        """

        if year is None:
            target_path = self.raw_path / mercado 
        else:
            target_path = self.raw_path / mercado / str(year)

        if target_path.exists() and target_path.is_dir():
            folder_list = []
            for item in target_path.iterdir():
                # Check if the item is a directory and its name represents a year (is numeric)
                if item.is_dir():
                    folder_list.append(int(item.name))
            return folder_list
        else:
            print(f"Target path {target_path} does not exist or is not a directory.")
            return []
    

class ProcessedFileUtils(StorageFileUtils):
    """
    Utility class for processing and saving processed parquet files.
    """
    def __init__(self, use_s3: bool = False) -> None:
        super().__init__(use_s3)
        self.processed_path = self.base_path / 'processed'
        self.row_group_size = 2880 

    
    @staticmethod
    def drop_processed_duplicates(df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """
        Drops duplicates from the DataFrame based on the data type and mercado. Used for processing of processed data
        """
        # Then handle subset-based duplicates
        try:
            if dataset_type == 'volumenes_i90':
                df = df.drop_duplicates(subset=['datetime_utc', 'mercado', "UP"], keep='last')
            elif dataset_type == 'volumenes_i3':
                df = df.drop_duplicates(subset=['datetime_utc', 'mercado', "tecnologia"], keep='last')
            else:
                df = df.drop_duplicates(subset=['datetime_utc', 'indicador_id'], keep='last')
        except Exception as e:
            print(f"Error dropping duplicates: {e}")
            raise
        
        return df
    


    def write_processed_parquet(self, df: pd.DataFrame) -> None:
        """
        Processes a DataFrame and saves/appends it as a parquet file in the appropriate directory structure.
        
        Args:
            year (int): The year for the data being processed
            month (int): The month for the data being processed
            df (pd.DataFrame): The DataFrame containing the data to be saved
            dataset_type (str): The type of dataset (e.g., 'precios', 'demanda')
            mercado (str): The market name (e.g., 'diario', 'intradiario')
            
        Returns:
            None
        """

        # Convert to PyArrow Table
        table = pa.Table.from_pandas(df)

        # Define partitioning columns
        partition_cols = ['mercado', 'id_mercado', 'year', 'month']
        partitioned_df= df[partition_cols].drop_duplicates()

        # Output directory
        output_dir = self.processed_path

        # Write partitioned Parquet files
        for _, partition in partitioned_df.iterrows():
            # Create a mask to filter the data for the current partition
            mask = True
            # Iterate over each column defined in partition_cols
            for col in partition_cols:
                # Update the mask to include only rows that match the current partition's value for this column
                mask = mask & (table[col] == partition[col])

            # Apply the mask to filter the table, resulting in a partitioned table
            partition_table = table.filter(mask)

            # Sort by datetime_utc
            partition_table = partition_table.sort_by('datetime_utc')
            
            # Drop partitioning columns (year, month, mercado) to exclude from Parquet to avoid unnecessary storage
            partition_table = partition_table.drop(['year', 'month', 'mercado'])
            
            # Define output path ie mercado/id_mercado/year/month/data.parquet -> secundaria/1/2025/04/data.parquet

            partition_path = os.path.join(
                output_dir,
                f"{partition['mercado']}",
                f"{partition['id_mercado']}",
                f"{partition['year']}",
                f"{partition['month']:02d}"
            ) 
            os.makedirs(partition_path, exist_ok=True)
            
            # Write with ParquetWriter for column-specific statistics
            schema = partition_table.schema
            writer = pq.ParquetWriter(
                os.path.join(partition_path, 'data.parquet'),
                schema,
                use_dictionary=['id_mercado'],  # Dictionary encoding for id_mercado
                compression='zstd',
                write_statistics=['datetime_utc', 'price'],  # Stats for datetime_utc, price only
                data_page_size=64 * 1024,  # 64 KB pages
                data_page_version='2.0',
                row_group_size= self.row_group_size # 2880 rows per partition
            )
            writer.write_table(partition_table)
            writer.close()

        print(f"Partitioned Parquet files written to {self.processed_path}")

        return


if __name__ == "__main__":
    # Example usage
    processed_file_utils = ProcessedFileUtils(use_s3=True)
    processed_file_utils.process_parquet_files(remove=True)