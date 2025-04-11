import pandas as pd
from typing import Optional
import sys
import os
from pathlib import Path
import pretty_errors
# Get the absolute path to the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
# Use absolute imports
from utilidades.db_utils import DatabaseUtils
from configs.esios_config import ESIOSConfig
from utilidades.parquet_utils import RawFileUtils


class ReadOps:
    """
    Clase para leer datos de la base de datos o de un fichero parquet.
    """
    def __init__(self):
        self.bbdd_engine = DatabaseUtils.create_engine('pruebas_BT')
        self.duckdb_engine = "implementar"
        self.esios_config = ESIOSConfig()

    def validate_mercado(self, mercado: str) -> bool:
        """
        Valida si el mercado es válido.
        """
        #check if mercado is a string
        if not isinstance(mercado, str):
            raise ValueError(f"Mercado {mercado} need to be passed as a string")
        
        #get market keys from indicator id_map ie. {'Intra 4': 600, 'Intra 5': 612, 'Intra 6': 613, 'Intra 7': 614}
        valid_mercados = list(self.esios_config.indicator_id_map.keys())
        valid_mercados_lower = [mercado.lower().strip() for mercado in valid_mercados]

        #check if mercado is in valid_mercados_lower
        if mercado.lower().strip() not in valid_mercados_lower:
            raise ValueError(f"El mercado {mercado} no es válido")
        else:
            return True
        
    def validate_indicator_ids(self, indicator_ids: list[int]) -> bool:
        """
        Valida si los IDs de indicador son válidos.
        """
        if not isinstance(indicator_ids, list):
            raise ValueError(f"Indicator ids {indicator_ids} need to be passed as a list")
        else:
            #check if all items in indicator_ids are ints
            if not all(isinstance(item, int) for item in indicator_ids):
                raise ValueError(f"Indicator ids {indicator_ids} need to be a list of ints")
           
        #get indicator ids from indicator id_map ie. {'Intra 4': 600, 'Intra 5': 612, 'Intra 6': 613, 'Intra 7': 614}
        valid_indicator_ids = list(self.esios_config.indicator_id_map.values())
        if indicator_ids not in valid_indicator_ids:
            raise ValueError(f"Indicator ids {indicator_ids} are not valid")
        
        return True
        
    def read_db_data(self, indicator_ids: list[int], table_name: str, fecha_inicio_carga: Optional[str] = None, fecha_fin_carga: Optional[str] = None) -> pd.DataFrame:
        """
        Obtiene los datos de la base de datos.
        
        Args:
            fecha_inicio_carga: Fecha inicial en formato YYYY-MM-DD
            fecha_fin_carga: Fecha final en formato YYYY-MM-DD
            indicator_ids: Lista de IDs de indicadores ESIOS para filtrar
            table_name: Nombre de la tabla en la base de datos
            
        Returns:
            pd.DataFrame: DataFrame con los datos de la base de datos
        """
        #validate indicator ids
        try:
            self.validate_indicator_ids(indicator_ids)
        except ValueError as e:
            raise ValueError(f"Error validating indicator ids: {e}")

        #cehck if fecha inicio carga is older than fecha fin carga
        if fecha_inicio_carga and fecha_fin_carga:
            if fecha_inicio_carga > fecha_fin_carga:
                raise ValueError("La fecha de inicio de carga no puede ser mayor que la fecha de fin de carga")
            else:
                # Construct where clause for dates
                where_clause = ""
                where_clause = f'fecha between "{fecha_inicio_carga}" and "{fecha_fin_carga}"'
                
        # Add market filter if indicator_ids provided
        if indicator_ids:
            mercados_list = ", ".join([str(item) for item in indicator_ids])
            market_filter = f'id_mercado in ({mercados_list})'
            where_clause = f'{where_clause} and {market_filter}' if where_clause else market_filter

        try:
            # Get data from database
            df = DatabaseUtils.read_table(
                self.bbdd_engine, 
                table_name,
                where_clause=where_clause
            )
        except Exception as e:
            raise ValueError(f"Error reading data from database: {e}")
        
        return df

    def read_parquet_data(self, fecha_inicio_carga: str, fecha_fin_carga: str, mercado: str, indicator_ids: Optional[list[int]] = None) -> pd.DataFrame:
        """
        Lee un fichero parquet y lo devuelve como un DataFrame.
        """
        #check if fecha inicio carga is older than fecha fin carga
        if fecha_inicio_carga and fecha_fin_carga:
            if fecha_inicio_carga > fecha_fin_carga:
                raise ValueError("La fecha de inicio de carga no puede ser mayor que la fecha de fin de carga")
            else:
                pass
        else:
            pass
        
        #validate mercado
        self.validate_mercado(mercado)
        