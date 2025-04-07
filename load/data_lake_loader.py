"""
Data Lake Loader base class and implementations.
This module provides abstract base class and concrete implementations for loading data to various data lake solutions.
"""

import os
from typing import Optional, List, Union, Protocol, runtime_checkable, Dict, Any
from pathlib import Path
import pandas as pd
from abc import ABC, abstractmethod
import sys
from datetime import datetime

# Get the absolute path to the scripts directory
SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(SCRIPTS_DIR))

from config import DATA_LAKE_BASE_PATH


class DataLakeLoader(ABC):
    """
    Abstract base class for data lake loaders.
    
    This class defines the interface that all data lake loaders must implement,
    regardless of the underlying storage mechanism (local filesystem, S3, etc.).
    """
    
    def __init__(self):
        pass
        
    
    @abstractmethod
    def create_processed_directory_structure(self, year: int, month: int) -> Any:
        """
        Creates the necessary directory structure for storing data.
        
        Args:
            year (int): Year for the directory structure
            month (int): Month for the directory structure
        
        Returns:
            Any: Path or identifier for the created directory structure
        """
        pass
    
    @abstractmethod
    def process_and_save_parquet(
        self,
        df: pd.DataFrame,
        data_type: str,
        mercado: str,
        year: int,
        month: int,
    ) -> None:
        """
        Processes a DataFrame and saves it as a parquet file.
        
        Args:
            df (pd.DataFrame): Input DataFrame to be saved
            data_type (str): Type of data ('volumenes', 'precios', or 'ingresos')
            mercado (str): Market identifier
            year (int): Year for file organization
            month (int): Month for file organization
        
        Raises:
            ValueError: If data_type is not one of the allowed values
        """
        pass
        
    @abstractmethod
    def process_parquet_files(self, remove: bool = False) -> List[Any]:
        """
        Processes all csv files in the raw directory and saves them as parquet files.
        
        Args:
            remove (bool): Whether to remove the original file after processing
            
        Returns:
            List[Any]: List of files that were not processed successfully
        """
        pass


if __name__ == "__main__":
    # Example usage
    from load.local_data_lake_loader import LocalDataLakeLoader
    local_loader = LocalDataLakeLoader(base_path="data")
    local_loader.process_parquet_files(remove=True)


