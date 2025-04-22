import pandas as pd
from datetime import datetime
from utilidades.etl_date_utils import DateUtilsETL
from utilidades.db_utils import DatabaseUtils
import pytz
import pretty_errors
import sys

# Add necessary imports
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Ensure the project root is added to sys.path if necessary
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

class TransformadorI90:
    def __init__(self):
        pass

    def transform_diario(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
    