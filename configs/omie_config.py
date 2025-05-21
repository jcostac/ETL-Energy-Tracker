import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from utilidades.db_utils import DatabaseUtils
import pandas as pd
from typing import Optional, List, Tuple, Dict

class OMIEConfig:
    """
    Configuration for OMIE data extraction
    """
    
    def __init__(self):
        """Initialize OMIE configuration"""
        self._bbdd_engine = None
        
        # ID mercado mapping
        self.id_mercado_map = {
            "diario": "1",
            "continuo": "21",
            # Intra sessions will be mapped based on session number
            # e.g., "intra_1": "2", "intra_2": "3", etc.
        }
        
        # Define the base URLs for OMIE data
        self.intra_url_base = "https://www.omie.es/es/file-download?parents=curva_pibc_uof&filename="
        self.continuo_url_base = "https://www.omie.es/es/file-download?parents=trades&filename="
        
        # Temporary download path
        self.temporary_download_path = os.path.join(os.path.dirname(__file__), '../tmp')
        os.makedirs(self.temporary_download_path, exist_ok=True)

    @property
    def bbdd_engine(self):
        if not self._bbdd_engine:
            self._bbdd_engine = DatabaseUtils.create_engine('pruebas_BT')
        return self._bbdd_engine
    

    def get_lista_UPs(self, UP_ids: Optional[List[int]] = None) -> Tuple[List[str], Dict[str, int]]:
        """
        Get the list of programming units from the database.
        
        Args:
            UP_ids (Optional[List[int]]): List of programming unit IDs to filter
            
        Returns:
            Tuple[List[str], Dict[str, int]]: List of programming unit names and dictionary mapping names to IDs
        """
        self.bbdd_engine = DatabaseUtils.create_engine('pruebas_BT')

        # Build the WHERE clause for filtering by region and optionally by UP_ids
        where_clause = 'a.region = "ES"'
        if UP_ids:
            UP_list = ", ".join([str(item) for item in UP_ids])
            where_clause += f' AND u.id IN ({UP_list})'

        # Use DatabaseUtils.read_table to fetch the data
        df_up = DatabaseUtils.read_table(
            self.bbdd_engine,
            table_name="UPs u INNER JOIN Activos a ON u.activo_id = a.id",
            columns=["u.id as id", "UP", "UOF"],
            where_clause=where_clause
        )
        
        # Extract the list of UP names and a dictionary mapping names to IDs
        unidades = df_up['UOF'].tolist()
        dict_unidades = dict(zip(unidades, df_up['id']))

        return unidades, dict_unidades

    def get_error_data(self) -> pd.DataFrame:
        """
        Get error data for OMIE files from the database.
        
        Returns:
            pd.DataFrame: DataFrame with error data containing dates and error types
        """
        self.bbdd_engine = DatabaseUtils.create_engine('pruebas_BT')

        # Use DatabaseUtils.read_table to fetch error data
        df_errores = DatabaseUtils.read_table(
            self.bbdd_engine,
            table_name="Errores_i90_OMIE",
            columns=["fecha", "tipo_error"],
            where_clause='fuente_error = "omie-intra"'
        )
        # Convert 'fecha' column to date type
        df_errores['fecha'] = pd.to_datetime(df_errores['fecha']).dt.date

        return df_errores
