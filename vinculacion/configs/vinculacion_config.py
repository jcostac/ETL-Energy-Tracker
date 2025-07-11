from pathlib import Path
from datetime import datetime, timedelta

class VinculacionConfig:
    """Configuration for the vinculacion module"""
    
    def __init__(self):
        # Time windows
        """
        Initialize configuration parameters for the vinculacion module, including time windows, market types, database table names, and dataset type identifiers.
        """
        self.DATA_DOWNLOAD_WINDOW = 93  # days back for data availability
        self.HISTORICAL_CHECK_WINDOW = 94  # days for ambiguous matches
        
        # Markets for linking
        self.OMIE_MARKETS = ['diario', 'intra']
        self.I90_MARKETS = ['diario', 'intra']
        
        # Database tables
        self.DATABASE_NAME = "energy_tracker"
        self.UP_LISTADO_TABLE = "up_listado"
        self.UP_CHANGE_LOG_TABLE = "up_change_log"
        self.UP_UOF_VINCULACION_TABLE = "up_uof_vinculacion"  # Target table
        self.VINCULACION_CHANGE_LOG_TABLE = "vinculacion_change_log"
        
        # Dataset types
        self.OMIE_DATASET_TYPE = "volumenes_omie"
        self.I90_DATASET_TYPE = "volumenes_i90"
        
    def get_linking_target_date(self) -> str:
        """
        Return the target date for data linking, calculated as 93 days before the current date.
        
        Returns:
            str: The target date formatted as 'YYYY-MM-DD'.
        """
        target_date = datetime.now() - timedelta(days=self.DATA_DOWNLOAD_WINDOW)
        return target_date.strftime('%Y-%m-%d')
        
 