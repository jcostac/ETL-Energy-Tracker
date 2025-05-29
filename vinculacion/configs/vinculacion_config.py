from pathlib import Path
from datetime import timedelta

class VinculacionConfig:
    """Configuration for the vinculacion module"""
    
    def __init__(self):
        # Time windows
        self.DEFAULT_DOWNLOAD_WINDOW = 93  # days
        self.HISTORICAL_CHECK_WINDOW = 94  # days for ambiguous matches
        
        # Data paths (only for temporary extraction)
        self.TEMP_DATA_BASE_PATH = Path("data/temp_vinculacion")
        
        # Markets for linking
        self.OMIE_MARKETS = ['diario']
        self.I90_MARKETS = ['diario']
        self.INTRA_MARKETS = ['intra']  # For I90 intra 1, 2, 3
        
        # Database tables
        self.UP_LISTADO_TABLE = "up_listado"
        self.UP_CHANGE_LOG_TABLE = "up_change_log"
        self.UP_UOF_VINCULACION_TABLE = "up_uof_vinculacion"  # Target table
        
        # Linking parameters
        self.VOLUME_TOLERANCE = 0.001  # Tolerance for volume matching (MWh)
        self.MIN_MATCHING_HOURS = 20   # Minimum hours that must match for a valid link
        
        # Dataset types
        self.OMIE_DATASET_TYPE = "volumenes_omie"
        self.I90_DATASET_TYPE = "volumenes_i90" 