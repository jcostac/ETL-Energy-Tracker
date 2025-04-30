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
import re
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
        self.row_group_size = 122880 # Example: 128k rows per group

    @staticmethod
    def drop_processed_duplicates(df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """
        Drops duplicates from the DataFrame based on the data type and mercado. Used for processing of processed data
        """

        #print duplicate number before dropping
        print(f"Number of duplicates before dropping: {df.duplicated().sum()}")

        # Then handle subset-based duplicates
        try:
            if dataset_type == 'volumenes_i90':
                df = df.drop_duplicates(subset=['datetime_utc', 'mercado', "up"], keep='last')
            elif dataset_type == 'volumenes_i3':
                df = df.drop_duplicates(subset=['datetime_utc', 'mercado', "tecnologia"], keep='last')
            else:
                df = df.drop_duplicates(subset=['datetime_utc', 'id_mercado', 'datetime_utc'], keep='last')
        except Exception as e:
            print(f"Error dropping duplicates: {e}")
            raise
        
        return df
    

    def _add_partition_cols(self, df: pd.DataFrame, mercado: str) -> pd.DataFrame:
        """
        Adds necessary partition columns to the DataFrame.
        """
        df['year'] = df['datetime_utc'].dt.year
        df['month'] = df['datetime_utc'].dt.month
        df['mercado'] = mercado

        return df
        
    def write_processed_parquet(self, df: pd.DataFrame, mercado: str, value_col: str, dataset_type: str) -> None:
        """
        Processes a DataFrame and saves/appends it as partitioned parquet files using Hive partitioning.
        Handles both creating new files and appending to existing files with deduplication.

        Args:
            df (pd.DataFrame): The DataFrame containing the data to be saved/appended.
            mercado (str): The market name (used for partitioning and added as a column).
            value_col (str): The name of the main value column (e.g., 'precio', 'volumenes_i90').
            dataset_type (str): The type of dataset ('precios', 'volumenes_i90', etc.)
        """
        if df.empty:
            print(f"[INFO] Input DataFrame for {mercado} is empty. Skipping parquet write/append.")
            return
        
        # Prepare the input DataFrame ie add partition columns and ensure datetime is UTC naive
        df = self._prepare_input_dataframe(df, mercado)

        # Get unique partitions from the input DataFrame
        partition_cols = ['mercado', 'id_mercado', 'year', 'month']
        if not self._validate_partition_columns(df, partition_cols):
            return

        unique_partitions_df = df[partition_cols].drop_duplicates()
        print("--------------------------------")
        print(f"Processing {len(unique_partitions_df)} partition(s) for market '{mercado}'...")
        print("--------------------------------")

        # Process each partition
        for _, partition in unique_partitions_df.iterrows():
            try:
                self._process_single_partition(df, partition, partition_cols, value_col, dataset_type)
            except Exception as e:
                import traceback
                print(f"❌ Failed to process partition {partition.to_dict()}: {e} ❌")
                print(traceback.format_exc())

        print(f"✅ Finished processing all partitions for market '{mercado}' to {self.processed_path} ✅")

    def _prepare_input_dataframe(self, df: pd.DataFrame, mercado: str) -> pd.DataFrame:
        """
        Prepares the input DataFrame by adding partition columns and ensuring datetime is UTC naive.

        Args:
            df (pd.DataFrame): Raw input DataFrame.
            mercado (str): Market name for partitioning.

        Returns:
            pd.DataFrame: Prepared DataFrame with partition columns and normalized datetime.
        """
        # Add partition columns
        df = self._add_partition_cols(df, mercado)
        # Ensure datetime_utc is timezone-naive
        df = self._ensure_datetime_utc_naive(df)
        return df

    def _validate_partition_columns(self, df: pd.DataFrame, partition_cols: list) -> bool:
        """
        Validates that all required partition columns exist in the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to validate.
            partition_cols (list): List of required partition column names.

        Returns:
            bool: True if all required columns exist, False otherwise.
        """
        missing_cols = [col for col in partition_cols if col not in df.columns]
        if missing_cols:
            print(f"[ERROR] Input DataFrame is missing required partition columns: {missing_cols}. Cannot proceed.")
            return False
        return True

    def _process_single_partition(self, df: pd.DataFrame, partition, partition_cols: list, value_col: str, dataset_type: str) -> None:
        """
        Processes a single partition: filters data, handles existing file if present, and writes result.

        Args:
            df (pd.DataFrame): The full input DataFrame.
            partition: Partition values (from unique_partitions_df row).
            partition_cols (list): List of partition column names.
            value_col (str): The value column name.
            dataset_type (str): The type of dataset.
        """
        # Filter new data for this partition
        new_data_partition_df = self._filter_df_for_partition(df, partition, partition_cols)
        
        if new_data_partition_df.empty:
            print(f"[DEBUG] No new data for partition: {partition.to_dict()}. Checking for existing file.")
        
        # Build file path and check for existing file
        output_file_path = self._build_partition_path(partition, partition_cols, dataset_type)
        
        # Read existing data if file exists
        existing_data_df = self._read_existing_partition_file(output_file_path, partition)
        
        # Combine, deduplicate, and write data
        self._combine_and_write_partition(
            new_data_df=new_data_partition_df,
            existing_data_df=existing_data_df,
            partition=partition,
            partition_cols=partition_cols,
            output_file_path=output_file_path,
            value_col=value_col,
            dataset_type=dataset_type
        )

    def _filter_df_for_partition(self, df: pd.DataFrame, partition, partition_cols: list) -> pd.DataFrame:
        """
        Filters the DataFrame to get the rows that apply to a specific partition.

        Args:
            df (pd.DataFrame): The full DataFrame.
            partition: Partition values (Series or dict).
            partition_cols (list): List of partition column names.

        Returns:
            pd.DataFrame: Filtered DataFrame for the partition.
        """
        # Start with a True mask that will be narrowed down with each partition column
        partition_mask = True
        
        # For each partition column, add a condition to the mask
        # This builds a compound boolean mask that matches all partition criteria
        for col in partition_cols:
            partition_mask &= (df[col] == partition[col])
            
        # Return a copy of the filtered DataFrame to avoid modifying the original
        return df[partition_mask].copy()

    def _read_existing_partition_file(self, file_path: str, partition) -> pd.DataFrame:
        """
        Reads an existing partition file if it exists.
        
        Args:
            file_path (str): Path to the potential existing file.
            partition: Partition values for adding back to the DataFrame.
            
        Returns:
            pd.DataFrame: Existing data or empty DataFrame if no file exists.
        """
        existing_data_df = pd.DataFrame()
        if os.path.exists(file_path):
            print("--------------------------------")
            print(f"[INFO] Existing file found for {partition['mercado'].upper()}, with id_mercado: {partition['id_mercado']}, for year: {partition['year']}, month: {partition['month']}. Reading...")
            try:
                # Read existing data
                existing_data_df = pd.read_parquet(file_path)
                existing_data_df = self._ensure_datetime_utc_naive(existing_data_df)
                print(f"[INFO] Read {len(existing_data_df)} rows from existing file.")
                
                # Add partition columns back if they were dropped
                for col, value in partition.items():
                    if col not in existing_data_df.columns:
                        existing_data_df[col] = value
                        
            except Exception as e:
                print(f"[WARNING] Could not read existing file {file_path}: {e}. Will only write new data if available.")
                existing_data_df = pd.DataFrame()
        else:
            print(f"[INFO] No existing file found at {file_path}. Writing new data.")
            
        return existing_data_df

    def _combine_and_write_partition(self, new_data_df: pd.DataFrame, existing_data_df: pd.DataFrame,
                                    partition, partition_cols: list, output_file_path: str,
                                    value_col: str, dataset_type: str) -> None:
        """
        Combines new and existing data, deduplicates, and writes the result to a parquet file.

        Args:
            new_data_df (pd.DataFrame): New data for the partition.
            existing_data_df (pd.DataFrame): Existing data from the partition file.
            partition: Partition values.
            partition_cols (list): List of partition column names.
            output_file_path (str): Path to write the output file.
            value_col (str): Value column name.
            dataset_type (str): Dataset type.
        """
        # Skip if both DataFrames are empty
        if new_data_df.empty and existing_data_df.empty:
            print(f"[INFO] No data for partition {partition.to_dict()}. Skipping write.")
            return
            
        # Combine data if needed
        if not new_data_df.empty and not existing_data_df.empty:
            print(f"[INFO] Combining new data ({len(new_data_df)} rows) with existing data ({len(existing_data_df)} rows).")
            combined_df = pd.concat([existing_data_df, new_data_df], ignore_index=True)
        elif not new_data_df.empty:
            combined_df = new_data_df
            print(f"[INFO] Using only new data ({len(new_data_df)} rows).")
        else:
            combined_df = existing_data_df
            print(f"[INFO] Using only existing data ({len(existing_data_df)} rows).")
            
        # Deduplicate
        
        print("--------------------------------")
        print(f"[INFO] Deduplicating data ({len(combined_df)} rows) for partition {partition.to_dict()}...")
        final_df = self.drop_processed_duplicates(combined_df, dataset_type)
        print(f"[INFO] Data after deduplication: {len(final_df)} rows.")
        print("--------------------------------")
        
        if final_df.empty:
            print(f"[INFO] Partition {partition.to_dict()} is empty after deduplication. Skipping write.")
            return
            
        # Convert to PyArrow, sort, and prepare for writing
        final_df_sorted = self._prepare_dataframe_for_writing(final_df, partition_cols)
        if final_df_sorted is None:
            return
            
        # Write the file
        row_group_size = min(self.row_group_size, len(final_df_sorted))
        if row_group_size == 0:
            print(f"[WARNING] Final DataFrame to write has 0 rows for partition {partition.to_dict()}. Skipping write.")
            return
            
        
        self._write_final_parquet(final_df_sorted, output_file_path, value_col, row_group_size)

    def _prepare_dataframe_for_writing(self, df: pd.DataFrame, partition_cols: list) -> Optional[pd.DataFrame]:
        """
        Prepares the final DataFrame for writing:  sorts, and drops partition columns.

        Args:
            df (pd.DataFrame): The DataFrame to prepare.
            partition_cols (list): Partition columns to drop.

        Returns:
            pd.DataFrame or None: Prepared DataFrame or None if conversion fails.
        """
        try:
            # Sort by datetime_utc
            df_sorted = df.sort_values(by='datetime_utc').reset_index(drop=True)
            
            # Drop partition columns (they're encoded in the path)
            for col in partition_cols:
                if col in df_sorted.columns:
                    df_sorted = df_sorted.drop(columns=[col])
                    
            return df_sorted
        except Exception as e:
            print(f"[ERROR] Failed to prepare DataFrame for writing: {e}")
            return None

    def _write_final_parquet(self, df: pd.DataFrame, output_file: str, value_col: str, row_group_size: int) -> None:
        """
        Converts the final DataFrame to PyArrow and writes it to a Parquet file.

        Args:
            df (pd.DataFrame): The DataFrame to write.
            output_file (str): Output file path.
            value_col (str): Value column name.
            row_group_size (int): Row group size for Parquet.
        """
        try:
            print("--------------------------------")
            print(f"[INFO] Writing {len(df)} rows to {output_file}...")
            # Convert to PyArrow Table
            table = self._to_pyarrow_table(df)
            if table is None:
                print(f"[ERROR] Could not convert DataFrame to PyArrow Table. Skipping write.")
                return
                
            # Write the file
            schema = table.schema
            stats_cols = self._set_stats_cols(value_col, schema)
            dict_cols = self._set_dict_cols(schema)
            
            writer = pq.ParquetWriter(
                output_file,
                schema,
                compression='zstd', #compression algorithm
                write_statistics=stats_cols, #write statistics for columns where we could easily have value filters ie precios > x amount to optimize queries
                use_dictionary=dict_cols, #dictionary encoding for columns with many repeated values
                data_page_size=64 * 1024, #64k rows per page 
                data_page_version='2.0' #parquet version
            )
            writer.write_table(table, row_group_size)
            writer.close()
            print(f"✅ Successfully wrote {output_file} ✅")
            print(f"================================================")
        except Exception as e:
            print(f"❌ Error writing Parquet file {output_file}: {e} ❌")  
            print(f"================================================")

    def _ensure_datetime_utc_naive(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensures the 'datetime_utc' column is timezone-naive for Parquet compatibility.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with 'datetime_utc' as timezone-naive.
        """
        if pd.api.types.is_datetime64_any_dtype(df['datetime_utc']) and df['datetime_utc'].dt.tz is not None:
            print("Converting datetime_utc to UTC naive for Parquet compatibility...")
            df['datetime_utc'] = df['datetime_utc'].dt.tz_convert(None)
        return df

    def _to_pyarrow_table(self, df: pd.DataFrame) -> Optional[pa.Table]:
        """
        Converts a pandas DataFrame to a PyArrow Table.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pa.Table or None: PyArrow Table if successful, else None.
        """
        try:
            return pa.Table.from_pandas(df, preserve_index=False)
        except Exception as e:
            print(f"[ERROR] Error converting DataFrame to PyArrow Table: {e}")
            print(f"[DEBUG] DataFrame dtypes:\n{df.dtypes}")
            return None

    def _filter_partition(self, table: pa.Table, partition, partition_cols: list) -> pa.Table:
        """
        Filters a PyArrow Table for rows matching the given partition.

        Args:
            table (pa.Table): The full PyArrow Table.
            partition (pd.Series): The partition values.
            partition_cols (list): List of partition column names.

        Returns:
            pa.Table: Filtered PyArrow Table for the partition.
        """
        mask = True
        for col in partition_cols:
            mask = pa.compute.and_(mask, pa.compute.equal(table[col], partition[col]))
        return table.filter(mask)

    def _drop_partition_cols(self, table: pa.Table, partition_cols: list) -> pa.Table:
        """
        Drops partition columns from the PyArrow Table (they are encoded in the Hive path).

        Args:
            table (pa.Table): The PyArrow Table.
            partition_cols (list): List of partition column names to drop.

        Returns:
            pa.Table: Table with partition columns dropped.
        """
        print(f"Dropping partition columns: {partition_cols}")
        cols_to_drop = [col for col in partition_cols if col in table.column_names]
        if not cols_to_drop:
            
            return table
        table = table.drop(cols_to_drop)
        return table

    def _build_partition_path(self, partition, partition_cols: list, dataset_type: str) -> str:
        """
        Builds the output file path for a given partition using Hive-style key=value directories.
        Hive-style: key=value for each partition column
        ie mercado=diario/id_mercado=1/year=2024/month=01

        Hive file partitioning is useful to optimzie queries for engines like Spark. DuckDB also supports it.

        Args:
            partition (pd.Series): The partition values.
            partition_cols (list): List of partition column names.

        Returns:
            str: The full output file path for the partition.
        """
        # Hive-style: key=value for each partition column
        path_segments = [str(self.processed_path)]
        for col in partition_cols:
            value = partition[col] 
            # Format month as two digits, others as is
            if col == 'month':
                segment = f"{col}={value:02d}" #only month requieres 0 padding since there are single digitsfor months
            else:
                segment = f"{col}={value}"
            path_segments.append(segment)
        partition_path_str = os.path.join(*path_segments) #kwargs, join path segments ie> processed/mercado=BTC/id_mercado=BTC/year=2024/month=01
        os.makedirs(partition_path_str, exist_ok=True) #create directory if it doesn't exist

        if dataset_type == "precios_i90":
            print("Naming file as precios.parquet to homogenize processed data naming convention for parquet files")
            dataset_type = "precios"
        
        return os.path.join(partition_path_str, f"{dataset_type}.parquet")

    def _set_stats_cols(self, value_col: str, schema: pa.Schema) -> list[str]:
        """
        Sets the columns to include in statistics.
        """
        stats_cols = ['datetime_utc']
        if value_col in schema.names:
            stats_cols.append(value_col)
        
        return stats_cols
    
    def _set_dict_cols(self, schema: pa.Schema) -> list[str]:
        """
        Sets the columns to include in dictionary encoding.
        """
        print("--------------------------------")
        print(f"Applying dictionary encoding...")
        dict_cols = []
        if "up" in schema.names:
            dict_cols.append("up")
        elif "tecnologia" in schema.names:
            dict_cols.append("tecnologia")
        else:
            print(f"Dictionary encoding not applied for this dataset. Only applied for i90 and i3 datasets.")
        print("--------------------------------")
        return dict_cols


if __name__ == "__main__":
    # Example usage
    processed_file_utils = ProcessedFileUtils(use_s3=True)
    processed_file_utils.process_parquet_files(remove=True)