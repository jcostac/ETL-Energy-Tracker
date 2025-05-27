import os
import pandas as pd
import pretty_errors
import sys
from pathlib import Path
 
# Get the absolute path to the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
 
from utilidades.db_utils import DatabaseUtils
 
 
class OMIEConfig:
 
    def __init__(self):
        self._base_url = ""
        self._mercado = ""
        self._filename_pattern = ""
        self._engine = ""
 
    @property
    def base_url(self):
        return self._base_url
 
    @base_url.setter
    def base_url(self, value):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("base_url must be a non-empty string")
        self._base_url = value
 
    @property
    def mercado(self):
        return self._mercado
 
    @mercado.setter
    def mercado(self, value):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("mercado must be a non-empty string")
        self._mercado = value
 
    @property
    def filename_pattern(self):
        return self._filename_pattern
 
    @filename_pattern.setter
    def filename_pattern(self, value):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("filename_pattern must be a non-empty string")
        if "{year_month}" not in value:
            raise ValueError("filename_pattern must contain '{year_month}' placeholder")
        self._filename_pattern = value
 
    @property
    def engine(self):
        return self._engine
 
    @engine.setter
    def engine(self, engine: str):
        if engine is None:
            raise ValueError("Error setting engine: engine cannot be None")
        self._engine = engine
       
 
    def get_error_data(self) -> pd.DataFrame:
        """
        Get error data for OMIE files from the database.
       
        Returns:
            pd.DataFrame: DataFrame with error data containing dates and error types
        """
       
        self.engine = DatabaseUtils.create_engine('pruebas_BT')
           
        # Use DatabaseUtils.read_table to fetch error data
        df_errores = DatabaseUtils.read_table(
            self.engine,
            table_name="Errores_i90_OMIE",
            columns=["fecha", "tipo_error"],
            where_clause='fuente_error = "omie-intra"'
        )
        # Convert 'fecha' column to date type
        df_errores['fecha'] = pd.to_datetime(df_errores['fecha']).dt.date
 
        return df_errores
 
class DiarioConfig(OMIEConfig):
    def __init__(self):
        super().__init__()
        self.base_url = "https://www.omie.es/es/file-download?parents=curva_pbc_uof&filename="
        self.mercado = "diario"
        self.filename_pattern = "curva_pbc_uof_{year_month}.zip"
 
 
class IntraConfig(OMIEConfig):
    def __init__(self):
        super().__init__()
        self.base_url = "https://www.omie.es/es/file-download?parents=curva_pibc_uof&filename="
        self.mercado = "intra"
        self.filename_pattern = "curva_pibc_uof_{year_month}.zip"
 
 
class IntraContinuoConfig(OMIEConfig):
    def __init__(self):
        super().__init__()
        self.base_url = f"https://www.omie.es/es/file-download?parents=trades&filename="
        self.mercado = "continuo"
        self.filename_pattern = "trades_{year_month}.zip"
 
 
 
if __name__ == "__main__":
    config = IntraConfig()
    config.engine = DatabaseUtils.create_engine('pruebas_BT')
    print(config.get_error_data())
 