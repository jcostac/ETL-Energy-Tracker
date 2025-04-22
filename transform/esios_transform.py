import pandas as pd
from datetime import datetime
from utilidades.etl_date_utils import DateUtilsETL
from utilidades.db_utils import DatabaseUtils
import pytz
import pretty_errors


class TransformadorESIOS:
    
    def __init__(self):
        pass

    def transform_diario(self, day: datetime) -> pd.DataFrame:
        #iterate through all the years and months of the "diario" folder
        for year in range(day.year, 2025):
            for month in range(1, 13):
                #iterate through all the files in the "diario" folder
                for file in os.listdir(f"diario/{year}/{month}"):
                    #read the file
                    df = pd.read_csv(f"diario/{year}/{month}/{file}")
        pass

    def transform_intra(self, day: datetime) -> pd.DataFrame:
        pass

    def transform_secundaria(self, day: datetime) -> pd.DataFrame:
        pass

    def transform_terciaria(self, day: datetime) -> pd.DataFrame:
        pass

    def transform_rr(self, day: datetime) -> pd.DataFrame:
        pass
    
    

    