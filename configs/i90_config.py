from datetime import datetime
from typing import Optional, List, Tuple, Dict
import pandas as pd
from utilidades.db_utils import DatabaseUtils
from configs.storage_config import DATA_LAKE_BASE_PATH
import os

class I90Config:
    def __init__(self):
        self.dia_inicio_SRS = datetime(2024, 11, 20)  # Regulatory change date
        self.bbdd_engine = DatabaseUtils.create_engine('pruebas_BT')
        self.indicator_id_map, self.precios_sheet, self.volumenes_sheet = self.get_id_sheet_mapping()
        self.temporary_download_path = os.path.join(DATA_LAKE_BASE_PATH, 'temp_downloads')

    def get_id_sheet_mapping(self) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
        """
        Obtiene el mapping de los IDs de los mercados de ESIOS.
        Returns:
            dict: 1. indicator_id_map: Un diccionario con los nombres de los mercados y sus respectivos IDs de ESIOS.
                        i.e {'Diario': 1, 'Intra 1': 2, 'Intra 2': 3, 'Intra 3': 4} (nombre de mercado como key, id de ESIOS como value)
                    2. volumenes_id_map: Un diccionario con los IDs de los mercados de ESIOS y sus IDs de mercado en la BBDD.
                        i.e {1: 1, 2: 2, 3: 3, 4: 4} (id de ESIOS como key, id de mercado como value)
                    3. precios_id_map: Un diccionario con los IDs de los mercados de ESIOS y sus IDs de mercado en la BBDD.
                        i.e {1: 1, 2: 2, 3: 3, 4: 4} (id de ESIOS como key, id de mercado como value)
        """

        #get all market ids with indicator_esios_precios != 0
        df_mercados = DatabaseUtils.read_table(self.bbdd_engine, 'Mercados', columns=['id', 'mercado', 'sheet_i90_precios', 'sheet_i90_volumenes', 'is_quinceminutal'], 
                                            where_clause='(sheet_i90_volumenes != 0 OR sheet_i90_precios != 0)')
        
        
        #get idnicator map with mercado as key and indicator as value i.e {'Intra 4': 600, 'Intra 5': 612, 'Intra 6': 613, 'Intra 7': 614}
        indicator_map = dict(zip(df_mercados['mercado'], df_mercados['id']))

        #filter where sheet_i90_volumenes != 0
        df_mercados_volumenes = df_mercados[df_mercados['sheet_i90_volumenes'] != 0]
        
        #convert int to str "00" -> 3 to "03"
        df_mercados_volumenes['sheet_i90_volumenes'] = df_mercados_volumenes['sheet_i90_volumenes'].astype(str).str.zfill(2)
        volumenes_id_map = dict(zip(df_mercados_volumenes["id"], df_mercados_volumenes['sheet_i90_volumenes']))

        #filter where sheet_i90_precios != 0
        df_mercados_precios = df_mercados[df_mercados['sheet_i90_precios'] != 0]
        df_mercados_precios['sheet_i90_precios'] = df_mercados_precios['sheet_i90_precios'].astype(str).str.zfill(2)
        precios_id_map = dict(zip(df_mercados_precios["id"], df_mercados_precios['sheet_i90_precios']))

        #convert indicator value to str to avoid type errors
        indicator_id_map = {str(key): str(value) for key, value in indicator_map.items()}
        volumenes_id_map = {str(key): str(value) for key, value in volumenes_id_map.items()}
        precios_id_map = {str(key): str(value) for key, value in precios_id_map.items()}

        return indicator_id_map, volumenes_id_map, precios_id_map
    
    def get_lista_UPs(self, UP_ids: Optional[List[int]] = None) -> Tuple[List[str], Dict[str, int]]:
        """
        Get the list of programming units from the database.
        
        Args:
            bbdd_engine: Database engine connection
            UP_ids (Optional[List[int]]): List of programming unit IDs to filter
            
        Returns:
            Tuple[List[str], Dict[str, int]]: List of programming unit names and dictionary mapping names to IDs
        """
        # Query to get programming units
        query = '''SELECT u.id as id, UP FROM UPs u inner join Activos a on u.activo_id = a.id where a.region = "ES"'''
        
        # Add filter for specific programming units if provided
        if UP_ids:
            UP_list = """, """.join([str(item) for item in UP_ids])
            query += f' and u.id in ({UP_list})'
        
        # Execute query and process results
        df_up = pd.read_sql_query(query, con=self.bbdd_engine)
        unidades = df_up['UP'].tolist()
        dict_unidades = dict(zip(unidades, df_up['id']))
        
        return dict_unidades
    
    def get_market_data(self, mercados_ids: Optional[List[int]] = None) -> Tuple[pd.DataFrame, List[int], List[int]]:
        """
        Get market data from the database.
        
        Args:
            mercados_ids (Optional[List[int]]): List of market IDs to filter
            
        Returns:
            Tuple[pd.DataFrame, List[int], List[int]]: DataFrame with market data, list of volume sheet IDs, and list of all relevant sheet numbers
        """
        # Query to get markets with I90 volume data
        query = '''SELECT * FROM Mercados
                where sheet_i90_volumenes != 0'''
        
        # Add filter for specific markets if provided
        if mercados_ids:
            mercados_list = """, """.join([str(item) for item in mercados_ids])
            query += f' and id in ({mercados_list})'
        
        # Execute query and process results
        df_mercados = pd.read_sql_query(query, con=self.bbdd_engine) 

        #prestañas relacionadas con volumenes
        pestañas_volumenes = df_mercados['sheet_i90_volumenes'].unique().tolist()

        #prestañas relacionadas con precios y volumenes (entire thing)
        pestañas = pestañas_volumenes + df_mercados['sheet_i90_precios'].unique().tolist()
        
        # Convert to integers and filter out NaN values
        pestañas = [int(item) for item in pestañas if item is not None and not (isinstance(item, float) and pd.isna(item))]
        
        return df_mercados, pestañas_volumenes, pestañas
    
    def get_error_data(self) -> pd.DataFrame:
        """
        Get error data for I90 files from the database.
        
        Returns:
            pd.DataFrame: DataFrame with error data containing dates and error types
        """
        # Query to get error data
        query = '''SELECT fecha, tipo_error FROM Errores_i90_OMIE where fuente_error = "i90"'''
        
        # Execute query and process results
        df_errores = pd.read_sql_query(query, con=self.bbdd_engine)
        df_errores['fecha'] = pd.to_datetime(df_errores['fecha']).dt.date
        
        return df_errores

    def _get_sheet_num(self, market_id: int, sheet_type: str) -> Optional[int]:
        """
        Get the sheet number for a given market ID.
        
        Args:
            market_id (int): The ID of the market to get the sheet number for
            sheet_type (str): The type of sheet to get ('precios' or 'volumenes')
            
        Returns:
            Optional[int]: The sheet number for the given market ID and sheet type, or None if not found
            
        Raises:
            ValueError: If an invalid sheet_type is provided
        """
        try:
            if sheet_type == "precios":
                return self.precios_sheet[market_id]
            elif sheet_type == "volumenes":
                return self.volumenes_sheet[market_id]
            else:
                raise ValueError(f"Invalid sheet_type: {sheet_type}. Must be 'precios' or 'volumenes'")
            
        except KeyError:
            # Handle case where market_id is not found in the dictionary
            print(f"Warning: No {sheet_type} sheet found for market ID {market_id}")
            return None

    def _get_precios_sheets(self, market_ids: List[int]) -> List[int]:
        """
        Get the sheet numbers for prices for the specified market IDs.
        
        This method retrieves the sheet numbers in the I90 Excel file that contain
        price data for the given market IDs. It filters out any market IDs that
        don't have corresponding price sheets.
        
        Args:
            market_ids (List[int]): List of market IDs to get price sheets for
            
        Returns:
            List[int]: List of sheet numbers containing price data for the specified markets
        """
        sheet_nums = []
        for id in market_ids:
            sheet_num = self._get_sheet_num(id, "precios")
            if sheet_num is not None:
                sheet_nums.append(sheet_num)
        return sheet_nums
    
    def _get_volumenes_sheets(self, market_ids: List[int]) -> List[int]:
        """
        Get the sheet numbers for volumes for the specified market IDs.
        
        This method retrieves the sheet numbers in the I90 Excel file that contain
        volume data for the given market IDs. It filters out any market IDs that
        don't have corresponding volume sheets.
        
        Args:
            market_ids (List[int]): List of market IDs to get price sheets for
            
        Returns:
            List[int]: List of sheet numbers containing volume data for the specified markets
        """
        sheet_nums = []
        for id in market_ids:
            sheet_num = self._get_sheet_num(id, "volumenes")
            if sheet_num is not None:
                sheet_nums.append(sheet_num)
        return sheet_nums

    def get_sheets_of_interest(self) -> List[str]:
        """
        Get the sheets of interest (by combining volumenes and precios sheets for a particular indicator).
        """
        #get volumnes and precios sheets and merge them (dropping duplicates)
        volumenes_sheets = self._get_volumenes_sheets(self.market_ids)
        precios_sheets = self._get_precios_sheets(self.market_ids)
        sheets_of_interest = volumenes_sheets + precios_sheets
        sheets_of_interest = list(set(sheets_of_interest))

        return volumenes_sheets, precios_sheets, sheets_of_interest

class DiariaConfig(I90Config):
    def __init__(self):
        super().__init__()
        #get individual id
        self.diaria_id = self.indicator_id_map["Diario"]

        #group id onto a single var (to be used in get_sheets_of_interest)
        self.market_ids = [self.diaria_id]

        #get sheets of interest
        self.volumenes_sheets, self.precios_sheets, self.sheets_of_interest = self.get_sheets_of_interest()

class SecundariaConfig(I90Config):
    def __init__(self):
        super().__init__()
        #get individual ids for subir and bajar
        self.secundaria_subir_id = self.indicator_id_map["Secundaria a subir"]
        self.secundaria_bajar_id = self.indicator_id_map["Secundaria a bajar"]

        #group ids onto a single var (to be used in get_sheets_of_interest)
        self.market_ids = [self.secundaria_subir_id, self.secundaria_bajar_id]

        #get sheets of interest
        self.volumenes_sheets, self.precios_sheets, self.sheets_of_interest = self.get_sheets_of_interest()

class TerciariaConfig(I90Config):
    def __init__(self):
        super().__init__()
        #get individual ids for subir and bajar
        self.terciaria_subir_id = self.indicator_id_map["Terciaria a subir"]
        self.terciaria_bajar_id = self.indicator_id_map["Terciaria a bajar"]

        #group ids onto a single var (to be used in get_sheets_of_interest)
        self.market_ids = [self.terciaria_subir_id, self.terciaria_bajar_id]

        #get sheets of interest
        self.volumenes_sheets, self.precios_sheets, self.sheets_of_interest = self.get_sheets_of_interest()

class RRConfig(I90Config):
    def __init__(self):
        super().__init__()
        #get individual ids for subir and bajar
        self.rr_subir_id = self.indicator_id_map["RR a subir"]
        self.rr_bajar_id = self.indicator_id_map["RR a bajar"]
        
        #group ids onto a single var (to be used in get_sheets_of_interest)
        self.market_ids = [self.rr_subir_id, self.rr_bajar_id]

        #get sheets of interest
        self.volumenes_sheets, self.precios_sheets, self.sheets_of_interest = self.get_sheets_of_interest()

class CurtailmentConfig(I90Config):
    def __init__(self):
        super().__init__()
        #get individual id
        self.curtailment_id = self.indicator_id_map["Curtailment"]

        #group id onto a single var (to be used in get_sheets_of_interest)
        self.market_ids = [self.curtailment_id]

        #get sheets of interest
        self.volumenes_sheets, self.precios_sheets, self.sheets_of_interest = self.get_sheets_of_interest()

class P48Config(I90Config):
    def __init__(self):
        super().__init__()
        #get individual id
        self.p48_id = self.indicator_id_map["P48"]

        #group id onto a single var (to be used in get_sheets_of_interest)
        self.market_ids = [self.p48_id]

        #get sheets of interest
        self.volumenes_sheets, _, self.sheets_of_interest = self.get_sheets_of_interest()

    