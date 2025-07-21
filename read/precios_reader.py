from typing import Optional
import sys
import os
import pretty_errors
from pathlib import Path
# Get the absolute path to the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
# Use absolute imports
from configs.esios_config import ESIOSConfig, DiarioConfig, IntraConfig, SecundariaConfig, TerciariaConfig, RRConfig
from utilidades.db_utils import DatabaseUtils

class ConsultasESIOS:

    def __init__(self):
        """
        Initialize the ConsultasESIOS class with configuration settings.
        """
        self.processed_dir = "" #to be configured in storage.config.py

    def consulta_precios(self, fecha_inicio: Optional[str] = None, fecha_fin: Optional[str] = None, 
                    indicator_ids: Optional[list[int]] = None, is_quinceminutal: bool = False):
        """
        TODO: Obtiene los datos del parquet procesado (actualmente se obtiene de la base de datos)
        
        Args:
            fecha_inicio (Optional[str]): Fecha inicial en formato YYYY-MM-DD
            fecha_fin (Optional[str]): Fecha final en formato YYYY-MM-DD
            indicator_ids (Optional[list[int]]): Lista de IDs de indicadores ESIOS para filtrar
            is_quinceminutal (bool): Si True, obtiene datos de Precios_quinceminutales, si False, de Precios_horarios
            
        Returns:
            pd.DataFrame: DataFrame con los datos de la base de datos
        """
        # Check if fecha inicio carga is older than fecha fin carga
        if fecha_inicio and fecha_fin:
            if fecha_inicio > fecha_fin:
                raise ValueError("La fecha de inicio de carga no puede ser mayor que la fecha de fin de carga")
            else:
                # Construct where clause for dates
                where_clause = f'fecha between "{fecha_inicio}" and "{fecha_fin}"'
                
        else:
            where_clause = ""
                
        # Add market filter if indicator_ids provided
        if indicator_ids:
            mercados_list = ", ".join([str(item) for item in indicator_ids])
            market_filter = f'id_mercado in ({mercados_list})'
            where_clause = f'{where_clause} and {market_filter}' if where_clause else market_filter

        # Select appropriate table based on is_quinceminutal parameter
        table_name = 'Precios_quinceminutales' if is_quinceminutal else 'Precios_horarios'

        # Get data from database
        df = DatabaseUtils.read_table(
            self.bbdd_engine, 
            table_name,
            where_clause=where_clause
        )

        return df


class ConsultasDiario(ConsultasESIOS):

    def __init__(self):
        """
        Initialize the ConsultasDiario class with the daily market indicator ID.
        """
        super().__init__()

        config = DiarioConfig()
        self.indicator_id = config.indicator_id

    def consulta_precios(self, fecha_inicio: str, fecha_fin: str):
        """
        Consulta los precios del mercado diario para un rango de fechas.
        
        Args:
            fecha_inicio (str): Fecha inicial en formato YYYY-MM-DD
            fecha_fin (str): Fecha final en formato YYYY-MM-DD
            
        Returns:
            pd.DataFrame: DataFrame con los precios del mercado diario
        """
        return super().consulta_precios(fecha_inicio, fecha_fin, indicator_ids=[self.indicator_id], is_quinceminutal=False)


class ConsultasIntra(ConsultasESIOS):
    
    def __init__(self):
        """
        Initialize the ConsultasIntra class with intraday market configuration.
        """
        super().__init__()

        config = IntraConfig()
        self.intra_name_map = config.intra_name_map
        self.intra_reduccion_fecha = config.intra_reduccion_fecha
        self.cambio_granularidad_fecha = config.cambio_granularidad_fecha
    
    def consulta_precios(self, fecha_inicio: str, fecha_fin: str, intra_lst: list[int]):
        """
        Consulta los precios de las diferentes sesiones del mercado intradiario para un rango de fechas.
        
        Args:
            fecha_inicio (str): Fecha inicial en formato YYYY-MM-DD
            fecha_fin (str): Fecha final en formato YYYY-MM-DD
            intra_lst (list[int]): Lista de sesiones de intradiario [1-7]
            
        Returns:
            pd.DataFrame: DataFrame con los precios del mercado intradiario
        """
        # Validate input
        invalid_intra_nums = [intra_num for intra_num in intra_lst if intra_num not in self.intra_name_map]
        if invalid_intra_nums:
            raise ValueError(f"Invalid intraday sessions: {invalid_intra_nums}. Valid sessions are 1-7")
        
        # Get indicator IDs for requested intraday sessions
        indicator_ids = [ESIOSConfig().get_indicator_id(self.intra_name_map[intra_num]) for intra_num in intra_lst]
        
        # Query database with granularity consideration
        return super().consulta_precios(fecha_inicio, fecha_fin, indicator_ids=indicator_ids, is_quinceminutal=True)


class ConsultasSecundaria(ConsultasESIOS):
    
    def __init__(self):
        """
        Initialize the ConsultasSecundaria class with secondary regulation market configuration.
        """
        super().__init__()

        config = SecundariaConfig()
        self.secundaria_name_map = config.secundaria_name_map
        self.precio_dual_fecha = config.precio_dual_fecha
        self.cambio_granularidad_fecha = config.cambio_granularidad_fecha
    
    def consulta_precios(self, fecha_inicio: str, fecha_fin: str, secundaria_lst: list[int]):
        """
        Consulta los precios del mercado de regulación secundaria para un rango de fechas,
        teniendo en cuenta el cambio regulatorio.
        
        Args:
            fecha_inicio (str): Fecha inicial en formato YYYY-MM-DD
            fecha_fin (str): Fecha final en formato YYYY-MM-DD
            secundaria_lst (list[int]): Lista de tipos de secundaria [1: subir, 2: bajar]
            
        Returns:
            pd.DataFrame: DataFrame con los precios de regulación secundaria
        """
        from datetime import datetime
        
        # Validate input
        invalid_sec_nums = [sec_num for sec_num in secundaria_lst if sec_num not in self.secundaria_name_map]
        if invalid_sec_nums:
            raise ValueError(f"Invalid secondary regulation types: {invalid_sec_nums}. Valid types are 1-2")
        
        fecha_inicio_dt = datetime.strptime(fecha_inicio, '%Y-%m-%d')
        
        indicator_ids = []
        if fecha_inicio_dt < self.precio_dual_fecha:
            # Before change: only use bajar indicator
            indicator_ids.append(ESIOSConfig().get_indicator_id("Secundaria a bajar"))
        else:
            # After change: use requested indicators
            for sec_id in secundaria_lst:
                indicator_ids.append(ESIOSConfig().get_indicator_id(self.secundaria_name_map[sec_id]))

        return super().consulta_precios(fecha_inicio, fecha_fin, indicator_ids=indicator_ids, is_quinceminutal=True)


class ConsultasTerciaria(ConsultasESIOS):
    
    def __init__(self):
        """
        Initialize the ConsultasTerciaria class with tertiary regulation market configuration.
        """
        super().__init__()

        config = TerciariaConfig()
        self.terciaria_name_map = config.terciaria_name_map
        self.precio_unico_fecha = config.precio_unico_fecha
        self.cambio_granularidad_fecha = config.cambio_granularidad_fecha
    
    def consulta_precios(self, fecha_inicio: str, fecha_fin: str, terciaria_lst: list[int]):
        """
        Consulta los precios del mercado de regulación terciaria para un rango de fechas,
        teniendo en cuenta el cambio regulatorio.
        
        Args:
            fecha_inicio (str): Fecha inicial en formato YYYY-MM-DD
            fecha_fin (str): Fecha final en formato YYYY-MM-DD
            terciaria_lst (list[int]): Lista de tipos de terciaria [1: subir, 2: bajar, 3: directa subir, 
                                                                   4: directa bajar, 5: programada único]
            
        Returns:
            pd.DataFrame: DataFrame con los precios de regulación terciaria
        """
        from datetime import datetime
        
        # Validate input
        invalid_ter_nums = [ter_num for ter_num in terciaria_lst if ter_num not in self.terciaria_name_map]
        if invalid_ter_nums:
            raise ValueError(f"Invalid tertiary regulation types: {invalid_ter_nums}. Valid types are 1-5")
        
        fecha_inicio_dt = datetime.strptime(fecha_inicio, '%Y-%m-%d')
        
        indicator_ids = []

        
        # Always include unaffected markets
        for ter_num in [ter_num for ter_num in terciaria_lst if ter_num in [3, 4]]:
            indicator_ids.append(ESIOSConfig().get_indicator_id(self.terciaria_name_map[ter_num]))

        # Handle affected markets based on date
        if fecha_inicio_dt < self.precio_unico_fecha:
            # Before change: use dual prices
            for ter_num in [ter_num for ter_num in terciaria_lst if ter_num in [1, 2]]:
                indicator_ids.append(ESIOSConfig().get_indicator_id(self.terciaria_name_map[ter_num]))
        else:
            # After change: use single price
            if any(ter_num in terciaria_lst for ter_num in [1, 2, 5]):
                indicator_ids.append(self.terciaria_config.get_indicator_id("Terciaria programada unico"))

        return super().consulta_precios(fecha_inicio, fecha_fin, indicator_ids=indicator_ids, is_quinceminutal=True)


class ConsultasRR(ConsultasESIOS):
    
    def __init__(self):
        """
        Initialize the ConsultasRR class with Replacement Reserve market configuration.
        """
        super().__init__()

        config = RRConfig()
        self.indicator_id = config.indicator_id
        self.cambio_granularidad_fecha = config.cambio_granularidad_fecha
    
    def consulta_precios(self, fecha_inicio: str, fecha_fin: str):
        """
        Consulta los precios del mercado de Reservas de Sustitución (RR) para un rango de fechas.
        
        Args:
            fecha_inicio (str): Fecha inicial en formato YYYY-MM-DD
            fecha_fin (str): Fecha final en formato YYYY-MM-DD
            
        Returns:
            pd.DataFrame: DataFrame con los precios de RR
        """
        return super().consulta_precios(fecha_inicio, fecha_fin, indicator_ids=[self.indicator_id], is_quinceminutal=True)
    

if __name__ == "__main__":
    consulta_diario = ConsultasDiario()
    print(consulta_diario.indicator_id)