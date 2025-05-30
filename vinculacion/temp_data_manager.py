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
            
    def copy_processed_data_to_temp(self, start_date: str, end_date: str, 
                                   markets: List[str], dataset_type: str) -> bool:
        """
        Copies processed data files to temporary directory for the specified date range
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD) 
            markets: List of market names
            dataset_type: Type of dataset (volumenes_omie, volumenes_i90)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            temp_path = self.setup_temp_directory()
            
            # Generate date range
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            date_range = pd.date_range(start_dt, end_dt, freq='D')
            
            copied_files = 0
            
            for market in markets:
                market_temp_path = temp_path / dataset_type / market
                market_temp_path.mkdir(parents=True, exist_ok=True)
                
                for date in date_range:
                    try:
                        # Try to read processed file for this date
                        df = self.processed_file_utils.read_processed_file(
                            date.year, date.month, dataset_type, market
                        )
                        
                        if not df.empty:
                            # Filter for specific date
                            if 'datetime_utc' in df.columns:
                                daily_df = df[df['datetime_utc'].dt.date == date.date()]
                                
                                if not daily_df.empty:
                                    # Save to temp directory
                                    temp_file_path = (market_temp_path / 
                                                    f"{date.strftime('%Y-%m-%d')}.parquet")
                                    daily_df.to_parquet(temp_file_path, index=False)
                                    copied_files += 1
                                    
                    except FileNotFoundError:
                        print(f"⚠️  No processed file found for {market} on {date.date()}")
                        continue
                    except Exception as e:
                        print(f"❌ Error copying data for {market} on {date.date()}: {e}")
                        continue
                        
            print(f"✅ Copied {copied_files} processed files to temporary directory")
            return copied_files > 0
            
        except Exception as e:
            print(f"❌ Error setting up temporary data: {e}")
            return False
            
    def read_temp_data(self, market: str, dataset_type: str, 
                      date: Optional[str] = None) -> pd.DataFrame:
        """
        Reads data from temporary directory
        
        Args:
            market: Market name
            dataset_type: Dataset type
            date: Specific date (YYYY-MM-DD), if None reads all available
            
        Returns:
            pd.DataFrame: Combined data
        """
        temp_path = self.config.TEMP_DATA_BASE_PATH / dataset_type / market
        
        if not temp_path.exists():
            print(f"⚠️  No temporary data found for {market} - {dataset_type}")
            return pd.DataFrame()
            
        dfs = []
        
        if date:
            # Read specific date
            file_path = temp_path / f"{date}.parquet"
            if file_path.exists():
                dfs.append(pd.read_parquet(file_path))
        else:
            # Read all files
            for file_path in temp_path.glob("*.parquet"):
                dfs.append(pd.read_parquet(file_path))
                
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        else:
            return pd.DataFrame() 