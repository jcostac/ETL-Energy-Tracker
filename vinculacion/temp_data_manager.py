import shutil
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from typing import Optional, List
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from vinculacion.configs.vinculacion_config import VinculacionConfig
from utilidades.storage_file_utils import ProcessedFileUtils

class TemporaryDataManager:
    """Manages temporary data storage for vinculacion operations"""
    
    def __init__(self):
        self.config = VinculacionConfig()
        self.processed_file_utils = ProcessedFileUtils()
        
    def setup_temp_directory(self) -> Path:
        """Creates and returns the temporary directory path"""
        temp_path = self.config.TEMP_DATA_BASE_PATH
        temp_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Temporary directory ready: {temp_path}")
        return temp_path
        
    def cleanup_temp_directory(self) -> None:
        """Removes the temporary directory and all its contents"""
        if self.config.TEMP_DATA_BASE_PATH.exists():
            shutil.rmtree(self.config.TEMP_DATA_BASE_PATH)
            print(f"🗑️  Cleaned up temporary directory: {self.config.TEMP_DATA_BASE_PATH}")