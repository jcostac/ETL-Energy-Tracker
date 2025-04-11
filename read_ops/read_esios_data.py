from read_ops.read_data import ReadOps
from configs.esios_config import ESIOSConfig, IntraConfig, SecundariaConfig, TerciariaConfig, RRConfig
from utilidades.db_utils import DatabaseUtils
import pandas as pd
from datetime import datetime
from deprecated import deprecated

class ESIOS_Price_Reader(ReadOps):
    """
    Base class for ESIOS price data operations.
    Handles common configuration and indicator mappings used by both DB and Parquet operations.
    """
    def __init__(self):
        super().__init__()
        # Common configurations used by both DB and Parquet operations
        self.esios_config = ESIOSConfig()
        self.intra_config = IntraConfig()
        self.secundaria_config = SecundariaConfig()
        self.terciaria_config = TerciariaConfig()
        self.rr_config = RRConfig()

    def _get_indicator_ids_for_intra(self, intra_lst: list[int]) -> list[int]:
        """
        Common method to convert intra market numbers to indicator IDs.
        
        Args:
            intra_lst (list[int]): List of intraday market numbers (1-7)
            
        Returns:
            list[int]: List of corresponding indicator IDs
        """
        indicator_ids = []
        for intra_num in intra_lst:
            intra_name = f'Intra {intra_num}'
            indicator_id = self.esios_config.indicator_id_map.get(intra_name)
            if indicator_id:
                indicator_ids.append(indicator_id)
            else:
                raise ValueError(f"Invalid intra market: {intra_num}")
        return indicator_ids

    def _get_indicator_ids_for_secundaria(self, secundaria_lst: list[int]) -> list[int]:
        """
        Common method to convert secondary regulation types to indicator IDs.
        
        Args:
            secundaria_lst (list[int]): List of secondary regulation types [1: up, 2: down]
            
        Returns:
            list[int]: List of corresponding indicator IDs
        """
        indicator_ids = []
        for sec_id in secundaria_lst:
            if sec_id not in self.secundaria_config.secundaria_name_map:
                raise ValueError(f"Invalid secundaria type: {sec_id}")
            
            indicator_id = self.esios_config.indicator_id_map.get(
                self.secundaria_config.secundaria_name_map[sec_id]
            )
            if indicator_id:
                indicator_ids.append(indicator_id)
        return indicator_ids

@deprecated(action="default", reason="Class associated in old ETL pipeline and old DB structure, now deprecated")
class ESIOS_Price_DB_Reader(ESIOS_Price_Reader):
    """
    Handles all database read operations for ESIOS price data.
    """
    def __init__(self):
        super().__init__()
        self.bbdd_engine = DatabaseUtils.create_engine('pruebas_BT')

    @staticmethod
    def _determine_table_by_date(fecha_inicio: str, fecha_fin: str, 
                                cambio_granularidad_fecha) -> tuple[str, str] | str:
        """
        DB-specific method to determine appropriate table(s) based on date range.
        """
        fecha_inicio_dt = datetime.strptime(fecha_inicio, '%Y-%m-%d')
        fecha_fin_dt = datetime.strptime(fecha_fin, '%Y-%m-%d')
        
        if fecha_fin_dt < cambio_granularidad_fecha:
            return 'Precios_horarios'
        elif fecha_inicio_dt >= cambio_granularidad_fecha:
            return 'Precios_quinceminutales'
        else:
            return ('Precios_horarios', 'Precios_quinceminutales')

    def db_intra_data(self, fecha_inicio_carga: str, fecha_fin_carga: str, 
                      intra_lst: list[int]) -> pd.DataFrame:
        """
        Gets intraday market data from database.
        """
        # Get indicator IDs using common parent method
        indicator_ids = self._get_indicator_ids_for_intra(intra_lst)
        
        # DB-specific operations
        tables = self._determine_table_by_date(
            fecha_inicio_carga, 
            fecha_fin_carga, 
            self.intra_config.cambio_granularidad_fecha
        )
        
        if isinstance(tables, tuple):
            df_before = self.read_db_data(indicator_ids, tables[0], 
                                         fecha_inicio_carga, 
                                         self.intra_config.cambio_granularidad_fecha.strftime('%Y-%m-%d'))
            
            df_after = self.read_db_data(indicator_ids, tables[1],
                                        self.intra_config.cambio_granularidad_fecha.strftime('%Y-%m-%d'),
                                        fecha_fin_carga)
            
            return pd.concat([df_before, df_after], axis=0).reset_index(drop=True)
        else:
            return self.read_db_data(indicator_ids, tables, fecha_inicio_carga, fecha_fin_carga)
    
    def db_secundaria_data(self, fecha_inicio_carga: str, fecha_fin_carga: str, secundaria_lst: list[int]) -> pd.DataFrame:
        """
        Gets secondary regulation data from database, handling granularity change gracefully.
        
        Args:
            fecha_inicio_carga (str): Start date in YYYY-MM-DD format
            fecha_fin_carga (str): End date in YYYY-MM-DD format
            secundaria_lst (list[int]): List of secondary regulation types [1: up, 2: down]
            
        Returns:
            pd.DataFrame: Combined DataFrame with secondary regulation data
        """
        indicator_ids = self._get_indicator_ids_for_secundaria(secundaria_lst)
        
        # Get table name(s)
        tables = self._determine_table_by_date(fecha_inicio_carga, fecha_fin_carga, 
                                             self.secundaria_config.cambio_granularidad_fecha)
        
        if isinstance(tables, tuple):
            # Handle date range that spans granularity change
            df_before = self.read_db_data(indicator_ids, tables[0], 
                                         fecha_inicio_carga, 
                                         self.secundaria_config.cambio_granularidad_fecha.strftime('%Y-%m-%d'))
            
            df_after = self.read_db_data(indicator_ids, tables[1],
                                        self.secundaria_config.cambio_granularidad_fecha.strftime('%Y-%m-%d'),
                                        fecha_fin_carga)
            
            return pd.concat([df_before, df_after], axis=0).reset_index(drop=True)
        else:
            return self.read_db_data(indicator_ids, tables, fecha_inicio_carga, fecha_fin_carga)
    
    def db_terciaria_data(self, fecha_inicio_carga: str, fecha_fin_carga: str, terciaria_lst: list[int]) -> pd.DataFrame:
        """
        Gets tertiary regulation data from database, handling granularity change gracefully.
        
        Args:
            fecha_inicio_carga (str): Start date in YYYY-MM-DD format
            fecha_fin_carga (str): End date in YYYY-MM-DD format
            terciaria_lst (list[int]): List of tertiary types [1: scheduled up, 2: scheduled down, 
                                      3: direct up, 4: direct down, 5: scheduled single]
            
        Returns:
            pd.DataFrame: Combined DataFrame with tertiary regulation data
        """
        indicator_ids = self._get_indicator_ids_for_terciaria(terciaria_lst)
        
        # Get table name(s)
        tables = self._determine_table_by_date(fecha_inicio_carga, fecha_fin_carga, 
                                             self.terciaria_config.cambio_granularidad_fecha)
        
        if isinstance(tables, tuple):
            # Handle date range that spans granularity change
            df_before = self.read_db_data(indicator_ids, tables[0], 
                                         fecha_inicio_carga, 
                                         self.terciaria_config.cambio_granularidad_fecha.strftime('%Y-%m-%d'))
            
            df_after = self.read_db_data(indicator_ids, tables[1],
                                        self.terciaria_config.cambio_granularidad_fecha.strftime('%Y-%m-%d'),
                                        fecha_fin_carga)
            
            return pd.concat([df_before, df_after], axis=0).reset_index(drop=True)
        else:
            return self.read_db_data(indicator_ids, tables, fecha_inicio_carga, fecha_fin_carga)
    
    def db_rr_data(self, fecha_inicio_carga: str, fecha_fin_carga: str) -> pd.DataFrame:
        """
        Gets restoration reserve (RR) data from database, handling granularity change gracefully.
        
        Args:
            fecha_inicio_carga (str): Start date in YYYY-MM-DD format
            fecha_fin_carga (str): End date in YYYY-MM-DD format
            
        Returns:
            pd.DataFrame: Combined DataFrame with RR data
        """
        indicator_id = self.esios_config.indicator_id_map.get('RR')
        cambio_granularidad_fecha = self.rr_config.cambio_granularidad_fecha
        
        # Get table name(s)
        tables = self._determine_table_by_date(fecha_inicio_carga, fecha_fin_carga, 
                                             cambio_granularidad_fecha)
        
        if isinstance(tables, tuple):
            # Handle date range that spans granularity change
            df_before = self.read_db_data([indicator_id], tables[0], 
                                         fecha_inicio_carga, 
                                         cambio_granularidad_fecha.strftime('%Y-%m-%d'))
            
            df_after = self.read_db_data([indicator_id], tables[1],
                                        cambio_granularidad_fecha.strftime('%Y-%m-%d'),
                                        fecha_fin_carga)
            
            return pd.concat([df_before, df_after], axis=0).reset_index(drop=True)
        else:
            return self.read_db_data([indicator_id], tables, fecha_inicio_carga, fecha_fin_carga)

class ESIOS_Price_Parquet_Reader(ESIOS_Price_Reader):
    """
    Handles all parquet read operations for ESIOS price data.
    """
    def __init__(self):
        super().__init__()
        
        
