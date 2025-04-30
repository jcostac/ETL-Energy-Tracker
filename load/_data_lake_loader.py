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
# Avoid adding path if already present, standard practice
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

from configs.storage_config import DATA_LAKE_BASE_PATH # Keep for potential use in subclasses


class DataLakeLoader(ABC):
    """
    Abstract base class for data lake loaders.

    This class defines the interface for saving processed dataframes
    to a data lake storage layer (local filesystem, S3, etc.).
    """

    def __init__(self):
        """Initializes the base loader."""
        pass

    @abstractmethod
    def save_processed_data(
        self,
        df: pd.DataFrame,
        mercado: str,
        value_col: str,
        dataset_type: str
    ) -> None:
        """
        Saves the processed DataFrame to the data lake.

        Implementations should handle partitioning, file formats (e.g., Parquet),
        and potential updates/appends based on the storage system.

        Args:
            df (pd.DataFrame): The processed DataFrame to save.
            mercado (str): Market identifier (used for partitioning/organization).
            value_col (str): The name of the main value column in the DataFrame.
            dataset_type (str): The type of dataset (e.g., 'precios', 'volumenes_i90').

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
            Exception: Implementation-specific errors during the save operation.
        """
        pass


# Example usage can be removed or updated later when integrating with Transform
# if __name__ == "__main__":
#     # Example usage (requires concrete implementation and sample data)
#     # from load.local_data_lake_loader import LocalDataLakeLoader
#     # local_loader = LocalDataLakeLoader(base_path="data")
#     # Sample DataFrame creation would go here
#     # local_loader.save_processed_data(sample_df, 'diario', 'precio', 'precios')
#     pass


