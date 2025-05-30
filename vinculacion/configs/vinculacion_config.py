from pathlib import Path
from datetime import datetime, timedelta

class VinculacionConfig:
    """Configuration for the vinculacion module"""
    
    def __init__(self):
        # Time windows
        self.DATA_DOWNLOAD_WINDOW = 93  # days back for data availability
        self.HISTORICAL_CHECK_WINDOW = 94  # days for ambiguous matches
        
        # Data paths (only for temporary extraction)
        self.TEMP_DATA_BASE_PATH = Path("data/temporary/vinculacion")
        
        # Markets for linking
        self.OMIE_MARKETS = ['diario', 'intra']
        self.I90_MARKETS = ['diario', 'intra']
        
        # Database tables
        self.UP_LISTADO_TABLE = "up_listado"
        self.UP_CHANGE_LOG_TABLE = "up_change_log"
        self.UP_UOF_VINCULACION_TABLE = "up_uof_vinculacion"  # Target table
        
        # Linking parameters
        self.VOLUME_TOLERANCE = 0.001  # Tolerance for volume matching (MWh)
        self.MIN_MATCHING_HOURS = 23   # Minimum hours that must match for a valid link
        
        # Dataset types
        self.OMIE_DATASET_TYPE = "volumenes_omie"
        self.I90_DATASET_TYPE = "volumenes_i90"
        
    def get_linking_target_date(self) -> str:
        """
        Calculate the target date for linking (always 93 days back)
        
        Returns:
            str: Target date in YYYY-MM-DD format
        """
        target_date = datetime.now() - timedelta(days=self.DATA_DOWNLOAD_WINDOW)
        return target_date.strftime('%Y-%m-%d')
        
 