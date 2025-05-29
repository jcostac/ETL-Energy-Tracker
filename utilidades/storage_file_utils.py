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
        duplicates = df[df.duplicated(keep=False)]  # Get all duplicates
        df = df.drop_duplicates(keep='last')  
        exact_dups = df_before - len(df)
        
        if exact_dups > 0:
            print(f"Removed {exact_dups} exact duplicate rows")
            print("Actual duplicates:")
            print(duplicates.head(10))
            print(duplicates.tail(10))

        return df

    #@deprecated(reason="This method is only for development/debugging purposes. Use write_raw_parquet for production code.")
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
        """Processes a DataFrame and saves/appends it as a Parquet file."""
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

            # Sort the folder list in descending order (oldest year first)
            folder_list.sort()

            return folder_list
        else:
            print(f"Target path {target_path} does not exist or is not a directory.")
            return []

    def get_raw_file_list(self, mercado: str, year: int, month: int) -> list[str]:
        """
        Get a list of raw files for a given market, year, and month.
        """
        file_path = self.raw_path / mercado / str(year) / str(month)

        if self.dev and not self.prod:
            file_extension = 'csv'
        else:
            file_extension = 'parquet'

        return list(file_path.glob(f"*.{file_extension}"))
    

class ProcessedFileUtils(StorageFileUtils):
    """
    Utility class for processing and saving processed parquet files.
    """
    def __init__(self, use_s3: bool = False, ) -> None:
        super().__init__(use_s3)
        self.processed_path = self.base_path / 'processed'
        self.row_group_size = 122880 # Example: 128k rows per group

    @staticmethod
    def drop_processed_duplicates(df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """
        Drops duplicates from the DataFrame based on the data type and mercado. Used for processing of processed data
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
            else:
                df = df.drop_duplicates(subset=['datetime_utc', 'id_mercado', 'precio'], keep='last')
        except KeyError as e:
            raise KeyError(f"Missing required column for deduplication: {str(e)}")
        except Exception as e:
            raise Exception(f"Error during deduplication: {str(e)}")
        
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
                        partition=partition,
                        partition_cols=partition_cols,
                        output_file_path=output_file_path,
                        value_col=value_col,
                        dataset_type=dataset_type
                    )
                except Exception as e:
                    raise Exception(f"Failed to process partition {partition.to_dict()}: {str(e)}")

        except Exception as e:
            raise Exception(f"Failed to write processed parquet: {str(e)}")

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
        """Reads an existing partition file if it exists."""
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
                                    partition, partition_cols: list, output_file_path: str,
                                    value_col: str, dataset_type: str) -> None:
        """Combines new and existing data, deduplicates, and writes the result."""
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

            self._write_final_parquet(final_df_sorted, output_file_path, value_col, row_group_size)

        except Exception as e:
            raise Exception(f"Failed to combine and write partition: {str(e)}")

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
                    
            return df_sorted
        except Exception as e:
            print(f"[ERROR] Failed to prepare DataFrame for writing: {e}")
            raise

    def _write_final_parquet(self, df: pd.DataFrame, output_file: str, value_col: str, row_group_size: int) -> None:
        """Writes the final Parquet file."""
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
            stats_cols = self._set_stats_cols(value_col, schema)
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
        Ensures the 'datetime_utc' column is timezone-naive for Parquet compatibility.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with 'datetime_utc' as timezone-naive.
        """
        if pd.api.types.is_datetime64_any_dtype(df['datetime_utc']) and df['datetime_utc'].dt.tz is not None:
            print(" Converting datetime_utc to UTC naive for Parquet compatibility...")
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
            raise

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
        try:
            for col in partition_cols:
                mask = pa.compute.and_(mask, pa.compute.equal(table[col], partition[col]))
            return table.filter(mask)
        except Exception as e:
            print(f"[ERROR] Error filtering partition: {e}")
            raise


    def _drop_partition_cols(self, table: pa.Table, partition_cols: list) -> pa.Table:
        """
        Drops partition columns from the PyArrow Table (they are encoded in the Hive path).

        Args:
            table (pa.Table): The PyArrow Table.
            partition_cols (list): List of partition column names to drop.

        Returns:
            pa.Table: Table with partition columns dropped.
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
            print(" Naming file as precios.parquet to homogenize processed data naming convention for parquet files")
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
        print(f" Applying dictionary encoding...")
        dict_cols = []
        if "up" in schema.names:
            dict_cols.append("up")
        elif "tecnologia" in schema.names:
            dict_cols.append("tecnologia")
        else:
            print(f" Dictionary encoding not applied for this dataset. Only applied for i90 and i3 datasets.")
        print("--------------------------------")
        return dict_cols


if __name__ == "__main__":
    # Example usage
    processed_file_utils = ProcessedFileUtils(use_s3=True)
    processed_file_utils.process_parquet_files(remove=True)