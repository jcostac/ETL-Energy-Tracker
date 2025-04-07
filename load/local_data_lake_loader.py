"""
LocalDataLakeLoader implementation for storing data in the local filesystem.
"""

from typing import Optional, List, Union
from pathlib import Path
import pandas as pd
import os
import sys

# Get the absolute path to the scripts directory
SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(SCRIPTS_DIR))

from config import DATA_LAKE_BASE_PATH
from load.data_lake_loader import DataLakeLoader


class LocalDataLakeLoader(DataLakeLoader):
    """
    Implementation of DataLakeLoader for storing data in the local filesystem.
    
    This class handles processing and saving parquet files to a local directory structure.
    """
    
    def __init__(self, base_path: Optional[Union[str, Path]] = None):
        """
        Initialize the local data lake loader with a base path.
        
        Args:
            base_path (Optional[Path object or str]): Base directory path where data will be stored.
                                      If None, uses the default from config.
        """
        super().__init__(base_path)

        self.type = 'local'

        if base_path is None:
            self.base_path =  Path(DATA_LAKE_BASE_PATH)
        else: 
            self.base_path = Path(base_path) if isinstance(base_path, str) else base_path

        self.raw_path = self.base_path / 'raw'
        self.processed_path = self.base_path / 'processed'
        
        # Ensure directories exist
        self.raw_path.mkdir(parents=True, exist_ok=True)
        self.processed_path.mkdir(parents=True, exist_ok=True)

    def create_processed_directory_structure(self, year: int, month: int) -> Path:
        """
        Creates the necessary directory structure for storing parquet files.
        
        Args:
            year (int): Year for the directory structure
            month (int): Month for the directory structure
    
        Returns:
            Path: Path object pointing to the created month directory
        """
        # Convert string path to Path object if necessary
        processed_dir = Path(self.processed_path)
    
        # Create year and month directories
        year_dir = processed_dir / f"year={year}"
        month_dir = year_dir / f"month={month:02d}"
        
        # Create directories if they don't exist
        month_dir.mkdir(parents=True, exist_ok=True)
        
        return month_dir

    def save_df_to_parquet(self, df: pd.DataFrame, data_type: str, mercado: str, year: int, month: int) -> None:
        """
        Processes a DataFrame and saves it as a parquet file in the appropriate directory structure.
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
            
        # Create directory structure
        month_dir = self.create_processed_directory_structure(year, month)
        
        # Create filename
        filename = f"{data_type}_{mercado}.parquet"
        
        # Full path for the processed parquet file
        full_processed_file_path = month_dir / filename
        
        try:
            if full_processed_file_path.exists():
                # Read existing parquet file
                existing_df = pd.read_parquet(full_processed_file_path)
                
                # Ensure datetime column is in datetime format for both DataFrames
                if 'datetime' in df.columns and 'datetime' in existing_df.columns:
                    # Convert to datetime if not already
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    existing_df['datetime'] = pd.to_datetime(existing_df['datetime'])
                    
                    # Concatenate the DataFrames
                    combined_df = pd.concat([existing_df, df], ignore_index=True)
                    
                    # Drop duplicates based on datetime column (or other relevant columns)
                    combined_df = combined_df.drop_duplicates(subset=['datetime'], keep='last')
                    
                    # Sort by datetime
                    combined_df = combined_df.sort_values('datetime')
                    
                    # Save the combined DataFrame
                    combined_df.to_parquet(full_processed_file_path)
                    print(f"Updated existing file {filename} in {month_dir}")
                else:
                    raise ValueError("Both DataFrames must contain 'datetime' column for proper merging")
            else:
                # If file doesn't exist, save as new
                df.to_parquet(full_processed_file_path)
                print(f"Created new file {filename} in {month_dir}")
                
        except Exception as e:
            print(f"Error processing file {filename}: {str(e)}")
            raise
    
    def process_parquet_files(self, remove: bool = False) -> List[Path]:
        """
        Processes all csv files in the raw directory and saves them as parquet files in the processed directory.
        
        Args:
            remove (bool): Whether to remove the raw file after processing
        
        Returns:
            List[Path]: List of files that were not processed
        """
        bad_files = []
        
        for file in self.raw_path.glob('*.csv'):  # iterate over all csv files in the directory
            try:
                # Read the csv file
                df = pd.read_csv(file, sep=';')

                if df.empty:
                    raise ValueError(f"Warning: Nothing to process in {file} - empty DataFrame")

            except ValueError as e:
                print(f"Error processing file {file}: {str(e)}")
                bad_files.append(file)
                continue

            # Assuming the DataFrame has a datetime column to extract year and month
            try:
                if 'datetime' in df.columns:
                    # Get the first date in the DataFrame to determine year and month
                    first_date = pd.to_datetime(df['datetime'].iloc[0])
                    year = first_date.year
                    month = first_date.month

                    # Extract the market from the filename ie 'volumenes_secundaria.csv'
                    data_type = file.stem.split('_')[0]
                    market = file.stem.split('_')[1]
                    
                    # Process and save the file
                    self.save_df_to_parquet(
                        df=df,
                        data_type=data_type,
                        market=market,
                        year=year,
                        month=month
                    )

                    # Remove raw file after processing (optional)
                    if remove:
                        os.remove(file)
                        print(f"Processed and deleted {file}")

                else:
                    raise ValueError(f"Warning: Could not process {file} - missing datetime column")
                
            except ValueError as e:
                print(f"Error processing file {file}: {str(e)}")
                continue

            except Exception as e:
                # Handle any exceptions that might occur during processing
                print(f"Error processing file {file}: {str(e)}")
                bad_files.append(file)
                continue
            
        if bad_files:
            print(f"Warning: The following files were not processed: {bad_files}")
            return bad_files
        else:
            print("All files were processed successfully!")
            return []


if __name__ == "__main__":
    # Example usage
    local_loader = LocalDataLakeLoader(base_path="data")
    local_loader.process_parquet_files(remove=True) 