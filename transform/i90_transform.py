import pandas as pd
from datetime import datetime
from utilidades.etl_date_utils import DateUtilsETL
from utilidades.db_utils import DatabaseUtils
import pytz
import pretty_errors

class TransformadorI90:
    def __init__(self):
        pass

    def transform_diario(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
    