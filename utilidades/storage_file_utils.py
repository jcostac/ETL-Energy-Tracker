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
    
