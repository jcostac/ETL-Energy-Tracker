import os
import pandas as pd
import pretty_errors
import sys
from pathlib import Path
from sqlalchemy import text
 
# Get the absolute path to the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
 
from utilidades.db_utils import DatabaseUtils
 
 
class OMIEConfig:
 
    def __init__(self):
        """
        Initialize the OMIEConfig instance with empty configuration values for base URL, market type, and filename pattern.
        """
        self._base_url = ""
        self._mercado = ""
        self._filename_pattern = ""
 
    @property
    def base_url(self):
        """
        Get the base URL used for OMIE data file downloads.
        
        Returns:
            str: The base URL for downloading OMIE market data files.
        """
        return self._base_url
 
    @base_url.setter
    def base_url(self, value):
        """
        Set the base URL for OMIE data downloads.
        
        Raises:
            ValueError: If the provided value is not a non-empty string.
        """
        if not isinstance(value, str) or not value.strip():
            raise ValueError("base_url must be a non-empty string")
        self._base_url = value
 
    @property
    def mercado(self):
        """
        Get the market type identifier for the OMIE configuration.
        
        Returns:
            str: The market type (e.g., "diario", "intra", or "continuo").
        """
        return self._mercado
 
    @mercado.setter
    def mercado(self, value):
        """
        Set the market type identifier after validating it is a non-empty string.
        
        Raises:
            ValueError: If the provided value is not a non-empty string.
        """
        if not isinstance(value, str) or not value.strip():
            raise ValueError("mercado must be a non-empty string")
        self._mercado = value
 
    @property
    def filename_pattern(self):
        """
        Get the filename pattern used for OMIE data files.
        
        Returns:
            str: The filename pattern, which must include the '{year_month}' placeholder.
        """
        return self._filename_pattern
 
    @filename_pattern.setter
    def filename_pattern(self, value):
        """
        Set the filename pattern for OMIE data files, ensuring it is a non-empty string containing the '{year_month}' placeholder.
        
        Raises:
            ValueError: If the pattern is empty, not a string, or missing the '{year_month}' placeholder.
        """
        if not isinstance(value, str) or not value.strip():
            raise ValueError("filename_pattern must be a non-empty string")
        if "{year_month}" not in value:
            raise ValueError("filename_pattern must contain '{year_month}' placeholder")
        self._filename_pattern = value
   
 
    def get_error_data(self) -> pd.DataFrame:
        """
        Retrieve OMIE error records from the database as a DataFrame.
        
        Returns:
            pd.DataFrame: A DataFrame containing 'fecha' (as date objects) and 'tipo_error' columns for errors with source 'omie-intra'.
        """
        engine = None
        try:
            # Create engine for this operation
            engine = DatabaseUtils.create_engine('pruebas_BT')
            
            # Use DatabaseUtils.read_table to fetch error data
            df_errores = DatabaseUtils.read_table(
                engine,
                table_name="Errores_i90_OMIE",
                columns=["fecha", "tipo_error"],
                where_clause='fuente_error = "omie-intra"'
            )
            # Convert 'fecha' column to date type
            df_errores['fecha'] = pd.to_datetime(df_errores['fecha']).dt.date

            return df_errores
            
        except Exception as e:
            print(f"Error getting error data: {e}")
            raise e
        
        finally:
            # Clean up the engine connection to avoid a connection pool error
            if engine:
                engine.dispose()
 
class DiarioConfig(OMIEConfig):
    def __init__(self):
        """
        Initialize the DiarioConfig with OMIE daily market file settings.
        """
        super().__init__()
        self.base_url = "https://www.omie.es/es/file-download?parents=curva_pbc_uof&filename="
        self.mercado = "diario"
        self.filename_pattern = "curva_pbc_uof_{year_month}.zip"
 
 
class IntraConfig(OMIEConfig):
    def __init__(self):
        """
        Initialize the IntraConfig with OMIE intraday market file settings.
        
        Sets the base URL, market identifier, and filename pattern for downloading intraday OMIE data files.
        """
        super().__init__()
        self.base_url = "https://www.omie.es/es/file-download?parents=curva_pibc_uof&filename="
        self.mercado = "intra"
        self.filename_pattern = "curva_pibc_uof_{year_month}.zip"
 
 
class IntraContinuoConfig(OMIEConfig):
    def __init__(self):
        """
        Initialize the configuration for OMIE continuous intraday market data downloads with predefined URL, market identifier, and filename pattern.
        """
        super().__init__()
        self.base_url = f"https://www.omie.es/es/file-download?parents=trades&filename="
        self.mercado = "continuo"
        self.filename_pattern = "trades_{year_month}.zip"
 
 
 
if __name__ == "__main__":
    config = IntraConfig()
    config.engine = DatabaseUtils.create_engine('pruebas_BT')
    print(config.get_error_data())
 