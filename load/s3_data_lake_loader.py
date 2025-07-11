"""
S3DataLakeLoader implementation for storing data in AWS S3.
"""

from typing import Optional, List, Dict, Any
import pandas as pd
import os
import sys
from pathlib import Path
import tempfile
import io
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# Get the absolute path to the scripts directory
SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(SCRIPTS_DIR))

from configs.storage_config import DATA_LAKE_BASE_PATH
from load._data_lake_loader import DataLakeLoader

class S3DataLakeLoader(DataLakeLoader):
    """
    Implementation of DataLakeLoader for storing data in AWS S3.
    
    This class handles processing and saving parquet files to an S3 bucket following 
    the same directory structure pattern as the local implementation.
    """
    
    def __init__(
        self, 
        bucket_name: str,
        base_prefix: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        region_name: Optional[str] = None
    ):
        """
        Initialize the S3 data lake loader.
        
        Args:
            bucket_name (str): Name of the S3 bucket
            base_prefix (Optional[str]): Base prefix (folder path) within the bucket
            aws_access_key_id (Optional[str]): AWS access key ID. If None, uses environment variables or credentials file
            aws_secret_access_key (Optional[str]): AWS secret access key. If None, uses environment variables or credentials file
            region_name (Optional[str]): AWS region name. If None, uses environment variables or configuration file
        """
        super().__init__(base_prefix)
        load_dotenv() #load environment variables from .env file
        self.bucket_name = bucket_name
        self.base_prefix = base_prefix or ''
        
        # Remove leading/trailing slashes from prefix for consistency
        if self.base_prefix:
            self.base_prefix = self.base_prefix.strip('/')
            if self.base_prefix:
                self.base_prefix += '/'
        
        # Set up paths
        self.raw_prefix = f"{self.base_prefix}raw/"
        self.processed_prefix = f"{self.base_prefix}processed/"
        
        # Initialize S3 client
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id if aws_access_key_id else os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=aws_secret_access_key if aws_secret_access_key else os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=region_name if region_name else os.getenv('AWS_REGION')
        )
        
        # Ensure the bucket exists and we have access
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                raise ValueError(f"Bucket {self.bucket_name} does not exist")
            elif error_code == '403':
                raise ValueError(f"No permission to access bucket {self.bucket_name}")
            else:
                raise

    def create_processed_directory_structure(self, year: int, month: int) -> str:
        """
        Creates the necessary prefix structure for storing parquet files in S3.
        
        In S3, directories don't need to be explicitly created, they are 
        implied by the key prefix. This method just returns the appropriate prefix.
        
        Args:
            year (int): Year for the directory structure
            month (int): Month for the directory structure
    
        Returns:
            str: S3 key prefix for the specified year and month
        """
        # Create year and month prefix path
        month_prefix = f"{self.processed_prefix}year={year}/month={month:02d}/"
        return month_prefix

    def process_and_save_parquet(
        self,
        df: pd.DataFrame,
        data_type: str,
        mercado: str,
        year: int,
        month: int,
    ) -> None:
        """
        Processes a DataFrame and saves it as a parquet file in the appropriate S3 location.
        If a parquet file already exists, the new data will be appended to it.
        
        Args:
            df (pd.DataFrame): Input DataFrame to be saved as a parquet file
            data_type (str): Type of data ('volumenes', 'precios', or 'ingresos')
            mercado (str): Market identifier
            year (int): Year for file organization
            month (int): Month for file organization
        
        Raises:
            ValueError: If data_type is not one of the allowed values
        """
        # Validate data_types
        valid_types = ('volumenes', 'precios', 'ingresos')
        if data_type not in valid_types:
            raise ValueError(f"data_type must be one of {valid_types}")
            
        # Get the prefix for the specified year and month
        month_prefix = self.create_processed_directory_structure(year, month)
        
        # Create filename and full S3 key
        filename = f"{data_type}_{mercado}.parquet"
        s3_key = f"{month_prefix}{filename}"
        
        try:
            # Check if file already exists in S3
            file_exists = False
            try:
                self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
                file_exists = True
            except ClientError as e:
                if e.response['Error']['Code'] != '404':
                    raise
            
            # Ensure datetime column is in datetime format
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
            
            if file_exists:
                # Download existing file from S3
                with tempfile.NamedTemporaryFile() as temp_file:
                    self.s3_client.download_file(
                        Bucket=self.bucket_name,
                        Key=s3_key,
                        Filename=temp_file.name
                    )
                    
                    # Read existing parquet file
                    existing_df = pd.read_parquet(temp_file.name)
                    
                    # Ensure datetime column is in datetime format for both DataFrames
                    if 'datetime' in df.columns and 'datetime' in existing_df.columns:
                        existing_df['datetime'] = pd.to_datetime(existing_df['datetime'])
                        
                        # Concatenate the DataFrames
                        combined_df = pd.concat([existing_df, df], ignore_index=True)
                        
                        # Drop duplicates based on datetime column
                        combined_df = combined_df.drop_duplicates(subset=['datetime'], keep='last')
                        
                        # Sort by datetime
                        combined_df = combined_df.sort_values('datetime')
                        
                        # Save to temporary file and upload to S3
                        with tempfile.NamedTemporaryFile() as upload_file:
                            combined_df.to_parquet(upload_file.name)
                            self.s3_client.upload_file(
                                Filename=upload_file.name,
                                Bucket=self.bucket_name,
                                Key=s3_key
                            )
                        print(f"Updated existing file {filename} in {month_prefix}")
                    else:
                        raise ValueError("Both DataFrames must contain 'datetime' column for proper merging")
            else:
                # If file doesn't exist, create a new one
                with tempfile.NamedTemporaryFile() as temp_file:
                    df.to_parquet(temp_file.name)
                    self.s3_client.upload_file(
                        Filename=temp_file.name,
                        Bucket=self.bucket_name,
                        Key=s3_key
                    )
                print(f"Created new file {filename} in {month_prefix}")
                
        except Exception as e:
            print(f"Error processing file {filename}: {str(e)}")
            raise

    def _list_s3_files(self, prefix: str, extension: str = None) -> List[Dict[str, Any]]:
        """
        List files in an S3 bucket with the given prefix and extension.
        
        Args:
            prefix (str): Prefix to filter by
            extension (str, optional): File extension to filter by
            
        Returns:
            List[Dict[str, Any]]: List of file information dictionaries with 'Key' and 'LastModified' fields
        """
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=prefix)
            
            all_objects = []
            for page in pages:
                if 'Contents' in page:
                    all_objects.extend(page['Contents'])
            
            if extension:
                return [obj for obj in all_objects if obj['Key'].endswith(extension)]
            return all_objects
        
        except ClientError as e:
            print(f"Error listing files in S3: {str(e)}")
            return []

    def process_parquet_files(self, remove: bool = False) -> List[str]:
        """
        Processes all csv files in the raw S3 directory and saves them as parquet files in the processed directory.
        
        Args:
            remove (bool): Whether to remove the raw file after processing
        
        Returns:
            List[str]: List of files that were not processed
        """
        bad_files = []
        
        # List all CSV files in the raw prefix
        raw_files = self._list_s3_files(self.raw_prefix, '.csv')
        
        for file_obj in raw_files:
            s3_key = file_obj['Key']
            filename = os.path.basename(s3_key)
            
            try:
                # Download the CSV file to a temporary file
                with tempfile.NamedTemporaryFile() as temp_file:
                    self.s3_client.download_file(
                        Bucket=self.bucket_name,
                        Key=s3_key,
                        Filename=temp_file.name
                    )
                    
                    # Read the csv file
                    df = pd.read_csv(temp_file.name, sep=';')
                
                if df.empty:
                    raise ValueError(f"Warning: Nothing to process in {s3_key} - empty DataFrame")

                # Assuming the DataFrame has a datetime column to extract year and month
                if 'datetime' in df.columns:
                    # Get the first date in the DataFrame to determine year and month
                    first_date = pd.to_datetime(df['datetime'].iloc[0])
                    year = first_date.year
                    month = first_date.month

                    # Extract the data type and market from the filename
                    # e.g., 'volumenes_secundaria.csv'
                    parts = os.path.splitext(filename)[0].split('_')
                    if len(parts) >= 2:
                        data_type = parts[0]
                        market = parts[1]
                        
                        # Process and save the file
                        self.process_and_save_parquet(
                            df=df,
                            data_type=data_type,
                            market=market,
                            year=year,
                            month=month
                        )

                        # Remove raw file after processing (optional)
                        if remove:
                            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
                            print(f"Processed and deleted {s3_key}")
                    else:
                        raise ValueError(f"Warning: Could not extract data_type and market from filename {s3_key}")
                else:
                    raise ValueError(f"Warning: Could not process {s3_key} - missing datetime column")
                
            except ValueError as e:
                print(f"Error processing file {s3_key}: {str(e)}")
                bad_files.append(s3_key)
                continue
            except Exception as e:
                print(f"Error processing file {s3_key}: {str(e)}")
                bad_files.append(s3_key)
                continue
            
        if bad_files:
            print(f"Warning: The following files were not processed: {bad_files}")
            return bad_files
        else:
            print("All files were processed successfully!")
            return []


if __name__ == "__main__":
    # Example usage
    s3_loader = S3DataLakeLoader(
        bucket_name="my-data-lake-bucket",
        base_prefix="energy-data",
        region_name="us-east-1"
    )
    s3_loader.process_parquet_files(remove=True) 