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
from dotenv import load_dotenv

# Get the absolute path to the scripts directory
SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(SCRIPTS_DIR))

from configs.storage_config import VALID_DATASET_TYPES
from utilidades.env_utils import EnvUtils

class StorageFileUtils:
    """
    Utility class for processing and saving parquet files.
    """
    def __init__(self) -> None:
        """
        Initialize the storage file utility with environment configuration.
        
        Loads environment variables, validates required settings, and sets file extensions and the base data lake path for raw and processed files.
        """
        #load .env file
        load_dotenv()

        #check that all variables needed are present in the environment
        EnvUtils()

        self.raw_file_extension = 'csv'
        self.processed_file_extension = 'parquet'

        #set the base path for data storage
        self.base_path = Path(os.getenv('DATA_LAKE_PATH'))


    @staticmethod
    def create_directory_structure(path: str, mercado: str, year: int, month: int, day = None) -> Path:
        """
        Create and return a nested directory structure for organizing files by market, year, month, and optionally day.
        
        Parameters:
            path (str): Base directory where the structure will be created.
            mercado (str): Market name used as a subdirectory.
            year (int): Year for the directory structure.
            month (int): Month for the directory structure.
            day (int, optional): Day for the directory structure. If not provided, only up to the month level is created.
        
        Returns:
            Path: Path object pointing to the deepest created directory (month or day).
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
    
    #WIP
    def delete_raw_files_older_than(self, months: int, mercado: Optional[str] = None) -> None:
        """
        Delete raw CSV data directories older than a specified number of months.
        
        If a market is specified, only directories for that market are considered; otherwise, all standard markets are checked. Removes entire month directories and cleans up empty year directories after deletion.
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
    

class ProcessedFileUtils(StorageFileUtils):
    """
    Utility class for processing and saving processed parquet files.
    """
    def __init__(self) -> None:
        """
        Initialize the ProcessedFileUtils with the processed data path and default row group size for parquet writing.
        """
        super().__init__()
        self.processed_path = self.base_path / 'processed'
        self.row_group_size = 122880 # Example: 128k rows per group

    @staticmethod
    def drop_processed_duplicates(df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """
        Remove duplicate rows from the DataFrame according to dataset-specific columns for processed data.
        
        Duplicates are dropped based on a subset of columns determined by the dataset type. For 'volumenes_mic', duplicates are not removed. Raises a ValueError if the DataFrame is empty, and a KeyError if required columns are missing.
        
        Parameters:
            df (pd.DataFrame): The DataFrame to deduplicate.
            dataset_type (str): The type of dataset, which determines the deduplication logic.
        
        Returns:
            pd.DataFrame: The deduplicated DataFrame.
        """
        if df.empty:
            raise ValueError("Cannot process empty DataFrame")

        #print duplicate number before dropping
        print(f"Number of duplicates before dropping: {df.duplicated().sum()}")
        print(f"Columns in DataFrame: {df.columns.tolist()}")
        print(f"Dataset type: {dataset_type}")

        # Then handle subset-based duplicates
        try:
            if dataset_type == 'volumenes_i90':
                df = df.drop_duplicates(subset=['datetime_utc', 'mercado', "up"], keep='last')
            elif dataset_type == 'volumenes_i3':
                df = df.drop_duplicates(subset=['datetime_utc', 'mercado', "tecnologia"], keep='last')
            elif dataset_type == 'precios_i90':
                df = df.drop_duplicates(subset=['datetime_utc', 'mercado', "up"], keep='last')
            elif dataset_type == 'volumenes_omie':
                df = df.drop_duplicates(subset=['datetime_utc', 'mercado', "uof"], keep='last')
            elif dataset_type == 'volumenes_mic':
                #we allow duplicates for mic market data, so we don't drop any duplicates
                return df
            else:
                df = df.drop_duplicates(subset=['datetime_utc', 'id_mercado', 'precio'], keep='last')
        except KeyError as e:
            raise KeyError(f"Missing required column for deduplication: {str(e)}")
        except Exception as e:
            raise Exception(f"Error during deduplication: {str(e)}")
        
        return df
    

    def _add_partition_cols(self, df: pd.DataFrame, mercado: str) -> pd.DataFrame:
        """
        Add 'year', 'month', and 'mercado' columns to the DataFrame for partitioning.
        
        The 'year' and 'month' columns are extracted from the 'datetime_utc' column, and 'mercado' is set to the provided value.
        
        Returns:
            pd.DataFrame: DataFrame with added partition columns.
        """
        df['year'] = df['datetime_utc'].dt.year
        df['month'] = df['datetime_utc'].dt.month
        df['mercado'] = mercado

        return df
        
    def write_processed_parquet(self, df: pd.DataFrame, mercado: str, value_cols: list[str], dataset_type: str) -> None:
        """
        Writes a DataFrame as partitioned Parquet files using Hive-style directory structure.
        
        The DataFrame is partitioned by 'mercado', 'id_mercado', 'year', and 'month'. For each unique partition, the method combines new and existing data, removes duplicates, and writes the result as a Parquet file with the specified value columns. Raises an exception if the input DataFrame is empty or if required partition columns are missing.
        """
        if df.empty:
            raise ValueError(f"Input DataFrame for {mercado} is empty. Cannot proceed with write operation.")
        
        try:
            # Prepare the input DataFrame
            df = self._prepare_input_dataframe(df, mercado)

            # Get unique partitions from the input DataFrame
            partition_cols = ['mercado', 'id_mercado', 'year', 'month']
            if not self._validate_partition_columns(df, partition_cols):
                raise ValueError(f"Missing required partition columns in DataFrame")

            # Process partitions
            unique_partitions_df = df[partition_cols].drop_duplicates()
            
            for _, partition in unique_partitions_df.iterrows():
                try:
                    output_file_path = self._build_partition_path(partition, partition_cols, dataset_type)
                    
                    # Handle existing file
                    existing_data_df = self._read_existing_partition_file(output_file_path, partition)
                    
                    # Filter new data
                    new_data_partition_df = self._filter_df_for_partition(df, partition, partition_cols)
                    
                    # Combine and write
                    self._combine_and_write_partition(
                        new_data_df=new_data_partition_df,
                        existing_data_df=existing_data_df,
                        partition_cols=partition_cols,
                        output_file_path=output_file_path,
                        value_cols=value_cols,
                        dataset_type=dataset_type
                    )
                except Exception as e:
                    raise Exception(f"Failed to process partition {partition.to_dict()}: {str(e)}")

        except Exception as e:
            raise Exception(f"Failed to write processed parquet: {str(e)}")

    def _prepare_input_dataframe(self, df: pd.DataFrame, mercado: str) -> pd.DataFrame:
        """
        Prepare a DataFrame for partitioned storage by adding partition columns and converting the `datetime_utc` column to timezone-naive.
        
        Parameters:
            df (pd.DataFrame): Input DataFrame to be prepared.
            mercado (str): Market identifier used for partitioning.
        
        Returns:
            pd.DataFrame: DataFrame with added partition columns and a timezone-naive `datetime_utc` column.
        """
        # Add partition columns
        df = self._add_partition_cols(df, mercado)
        # Ensure datetime_utc is timezone-naive
        df = self._ensure_datetime_utc_naive(df)
        return df

    def _validate_partition_columns(self, df: pd.DataFrame, partition_cols: list) -> bool:
        """
        Check if all specified partition columns are present in the DataFrame.
        
        Returns:
            True if all required partition columns exist; False otherwise.
        """
        missing_cols = [col for col in partition_cols if col not in df.columns]
        if missing_cols:
            print(f"[ERROR] Input DataFrame is missing required partition columns: {missing_cols}. Cannot proceed.")
            return False
        return True

    def _process_single_partition(self, df: pd.DataFrame, partition, partition_cols: list, value_cols: list[str], dataset_type: str) -> None:
        """
        Processes and writes data for a single partition, combining new and existing records and ensuring deduplication.
        
        Filters the input DataFrame for the specified partition, reads any existing partition file, merges and deduplicates the data, and writes the result to the appropriate partitioned Parquet file. Raises exceptions if errors occur during processing.
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
            value_cols=value_cols,
            dataset_type=dataset_type
        )

    def _filter_df_for_partition(self, df: pd.DataFrame, partition, partition_cols: list) -> pd.DataFrame:
        """
        Return a DataFrame containing only the rows matching the specified partition values.
        
        Parameters:
            df (pd.DataFrame): The DataFrame to filter.
            partition: Partition values as a Series or dict.
            partition_cols (list): Names of columns to use for partition filtering.
        
        Returns:
            pd.DataFrame: A copy of the DataFrame filtered to rows matching all partition column values.
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
        Read an existing partition Parquet file and ensure required partition columns are present.
        
        If the file does not exist, returns an empty DataFrame. Ensures the `datetime_utc` column is timezone-naive and adds any missing partition columns with their corresponding values.
        
        Parameters:
            file_path (str): Path to the partition Parquet file.
            partition (dict): Dictionary mapping partition column names to their values.
        
        Returns:
            pd.DataFrame: DataFrame containing the data from the partition file, with all required partition columns.
        
        Raises:
            Exception: If reading the file fails for any reason.
        """
        try:
            if not os.path.exists(file_path):
                return pd.DataFrame()

            existing_data_df = pd.read_parquet(file_path)
            existing_data_df = self._ensure_datetime_utc_naive(existing_data_df)
            
            # Add partition columns back if they were dropped
            for col, value in partition.items():
                if col not in existing_data_df.columns:
                    existing_data_df[col] = value
                    
            return existing_data_df

        except Exception as e:
            raise Exception(f"Failed to read existing partition file {file_path}: {str(e)}")

    def _combine_and_write_partition(self, new_data_df: pd.DataFrame, existing_data_df: pd.DataFrame,
                                    partition_cols: list, output_file_path: str,
                                    value_cols: list[str], dataset_type: str) -> None:
        """
                                    Combine new and existing partition data, remove duplicates, and write the result as a Parquet file.
                                    
                                    Merges new and existing DataFrames for a partition, deduplicates based on dataset type, prepares the data for writing, and writes the final result to the specified Parquet file path. Raises an exception if no data is available or if any processing step fails.
                                    """
        try:
            if new_data_df.empty and existing_data_df.empty:
                raise ValueError("No data available for this partition")

            # Data combination step
            if not new_data_df.empty and not existing_data_df.empty:
                combined_df = pd.concat([existing_data_df, new_data_df], ignore_index=True)
            elif not new_data_df.empty:
                combined_df = new_data_df
            else:
                combined_df = existing_data_df

            # Deduplication step
            final_df = self.drop_processed_duplicates(combined_df, dataset_type)
            if final_df.empty:
                raise ValueError("No data remains after deduplication")

            # Prepare for writing
            final_df_sorted = self._prepare_dataframe_for_writing(final_df, partition_cols)
            if final_df_sorted is None:
                raise ValueError("Failed to prepare data for writing")

            # Write file
            row_group_size = min(self.row_group_size, len(final_df_sorted))
            if row_group_size == 0:
                raise ValueError("No rows to write")

            self._write_final_parquet(final_df_sorted, output_file_path, value_cols, row_group_size)

        except Exception as e:
            raise Exception(f"Failed to combine and write partition: {str(e)}")

    def _prepare_dataframe_for_writing(self, df: pd.DataFrame, partition_cols: list) -> Optional[pd.DataFrame]:
        """
        Sorts the DataFrame by the 'datetime_utc' column and resets its index in preparation for writing.
        
        Parameters:
            df (pd.DataFrame): The DataFrame to be prepared.
            partition_cols (list): List of partition columns (not used in this method but included for interface consistency).
        
        Returns:
            pd.DataFrame: The sorted DataFrame with reset index.
        
        Raises:
            Exception: If sorting or index resetting fails.
        """
        try:
            # Sort by datetime_utc
            df_sorted = df.sort_values(by='datetime_utc').reset_index(drop=True)
                    
            return df_sorted
        except Exception as e:
            print(f"[ERROR] Failed to prepare DataFrame for writing: {e}")
            raise

    def _write_final_parquet(self, df: pd.DataFrame, output_file: str, value_cols: list[str], row_group_size: int) -> None:
        """
        Write a pandas DataFrame to a Parquet file with specified row group size, compression, statistics, and dictionary encoding.
        
        Parameters:
            df (pd.DataFrame): The DataFrame to write.
            output_file (str): Destination file path for the Parquet file.
            value_cols (list[str]): List of value columns to include in Parquet statistics.
            row_group_size (int): Number of rows per Parquet row group.
        
        Raises:
            Exception: If conversion to PyArrow Table fails or if the Parquet write operation encounters an error.
        """
        try:
            print("\n📊 WRITE OPERATION")
            print("-"*40)
            print(f"Records to write: {len(df)}")
            print(f"Row group size: {row_group_size}")
            
            # Convert to PyArrow Table
            print("\n🔄 Converting to PyArrow...")
            table = self._to_pyarrow_table(df)
            if table is None:
                print("❌ Conversion failed")
                raise Exception("Conversion failed")

            # Configure write options
            print("\n⚙️ Configuring write options...")
            schema = table.schema
            stats_cols = self._set_stats_cols(value_cols, schema)
            dict_cols = self._set_dict_cols(schema)

            # Write file
            print("\n💾 Writing to disk...")
            writer = pq.ParquetWriter(
                output_file,
                schema,
                compression='zstd',
                write_statistics=stats_cols,
                use_dictionary=dict_cols,
                data_page_size=64 * 1024,
                data_page_version='2.0'
            )
            writer.write_table(table, row_group_size)
            writer.close()

            print("\n✅ WRITE SUCCESSFUL")
            print(f"File: {output_file}")
            print("-"*40)

        except Exception as e:
            print("\n❌ WRITE FAILED")
            print(f"Error: {str(e)}")
            print("-"*40)
            raise

    def _ensure_datetime_utc_naive(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert the 'datetime_utc' column in the DataFrame to timezone-naive if it contains timezone-aware datetimes.
        
        Returns:
            DataFrame with a timezone-naive 'datetime_utc' column for Parquet compatibility.
        """
        if pd.api.types.is_datetime64_any_dtype(df['datetime_utc']) and df['datetime_utc'].dt.tz is not None:
            print(" Converting datetime_utc to UTC naive for Parquet compatibility...")
            df['datetime_utc'] = df['datetime_utc'].dt.tz_convert(None)
        return df

    def _to_pyarrow_table(self, df: pd.DataFrame) -> Optional[pa.Table]:
        """
        Convert a pandas DataFrame to a PyArrow Table.
        
        Raises:
            Exception: If the conversion fails, an exception is raised with error details.
        
        Returns:
            pa.Table: The resulting PyArrow Table.
        """
        try:
            return pa.Table.from_pandas(df, preserve_index=False)
        except Exception as e:
            print(f"[ERROR] Error converting DataFrame to PyArrow Table: {e}")
            print(f"[DEBUG] DataFrame dtypes:\n{df.dtypes}")
            raise

    def _filter_partition(self, table: pa.Table, partition, partition_cols: list) -> pa.Table:
        """
        Return a PyArrow Table containing only rows that match the specified partition values.
        
        Parameters:
            table (pa.Table): The input PyArrow Table to filter.
            partition (pd.Series): Partition values to match.
            partition_cols (list): Names of columns to use for partition filtering.
        
        Returns:
            pa.Table: A new table with rows matching all partition column values.
        
        Raises:
            Exception: If filtering fails due to missing columns or computation errors.
        """
        mask = True
        try:
            for col in partition_cols:
                mask = pa.compute.and_(mask, pa.compute.equal(table[col], partition[col]))
            return table.filter(mask)
        except Exception as e:
            print(f"[ERROR] Error filtering partition: {e}")
            raise

    def _drop_partition_cols(self, table: pa.Table, partition_cols: list) -> pa.Table:
        """
        Remove specified partition columns from a PyArrow Table.
        
        Parameters:
            table (pa.Table): The input PyArrow Table.
            partition_cols (list): Names of columns to remove.
        
        Returns:
            pa.Table: A new table with the specified partition columns removed.
        """
        try:
            print(f" Dropping partition columns: {partition_cols}")
            cols_to_drop = [col for col in partition_cols if col in table.column_names]
            if not cols_to_drop:
                print(" No partition columns to drop")
                return table
            table = table.drop(cols_to_drop)
            return table
        except Exception as e:
            print(f"[ERROR] Error dropping partition columns: {e}")
            raise

    def _build_partition_path(self, partition, partition_cols: list, dataset_type: str) -> str:
        """
        Constructs the full output file path for a partitioned parquet file using Hive-style key=value directories.
        
        The resulting path is structured as `<processed_path>/<key1>=<value1>/<key2>=<value2>/.../<dataset_type>.parquet`, with the month value zero-padded. If the dataset type is "precios_i90", the file is named "precios.parquet" for consistency. Creates the partition directory if it does not exist.
        
        Parameters:
            partition: Partition values, typically a pandas Series.
            partition_cols (list): Names of columns used for partitioning.
            dataset_type (str): Type of dataset, used for the output file name.
        
        Returns:
            str: Full file path for the partitioned parquet file.
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

        if dataset_type == "precios_i90" or dataset_type == "precios_esios":
            print(" Naming file as precios.parquet to homogenize processed data naming convention for precios parquet files")
            dataset_type = "precios"
        
        return os.path.join(partition_path_str, f"{dataset_type}.parquet")

    def _set_stats_cols(self, value_cols: list[str], schema: pa.Schema) -> list[str]:
        """
        Determine which columns should be included in Parquet file statistics, including 'datetime_utc' and any value columns present in the schema.
        
        Parameters:
            value_cols (list[str]): List of value column names to include if present in the schema.
            schema (pa.Schema): PyArrow schema to check for column presence.
        
        Returns:
            list[str]: List of column names to include in Parquet statistics.
        """
        stats_cols = ['datetime_utc']
        
        # Add all valid value columns to statistics
        for value_col in value_cols:
            if value_col in schema.names:
                stats_cols.append(value_col)
            else:
                print(f"Warning: Value column '{value_col}' not found in schema")
        
        return stats_cols
    
    def _set_dict_cols(self, schema: pa.Schema) -> list[str]:
        """
        Determine which columns should use dictionary encoding for Parquet writing based on the schema.
        
        Returns:
            dict_cols (list[str]): List of column names to apply dictionary encoding, such as "up", "tecnologia", or "uof", depending on the dataset type.
        """
        print("--------------------------------")
        print(f" Applying dictionary encoding...")
        dict_cols = []
        if "up" in schema.names: #for i90 market data
            dict_cols.append("up")
        elif "tecnologia" in schema.names: #for i3 market data
            dict_cols.append("tecnologia")
        elif "uof" in schema.names: #for omie market data
            dict_cols.append("uof")
        else:
            print(f" Dictionary encoding not applied for this dataset. Only applied for i90 and i3 datasets.")
        print("--------------------------------")
        return dict_cols
