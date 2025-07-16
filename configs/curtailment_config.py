import sys
import os
from sqlalchemy import text
from dotenv import load_dotenv
from pathlib import Path

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from utilidades.db_utils import DatabaseUtils

class CurtailmentConfig:
    def __init__(self):
        """
        Initialize CurtailmentConfig with mappings loaded from the database.
        """
        load_dotenv()
        self.rt1_redespacho_filter = ["UPLPVPV", "UPLPVPCBN"] #RT1
        self.rt5_redespacho_filter = ["Restricciones Técnicas"] #RT5 

        #not used but useful to have it here as reference
        self.rtx_mapping = {
            "RT1": "03", #sheet num
            "RT5": "08" 
        }
        
    

