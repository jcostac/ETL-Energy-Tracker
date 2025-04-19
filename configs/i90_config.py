from datetime import datetime
from typing import Optional, List, Tuple, Dict
import pandas as pd
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from utilidades.db_utils import DatabaseUtils
from configs.storage_config import DATA_LAKE_BASE_PATH
import os

class I90Config:
    def __init__(self):

        self.dia_inicio_SRS = datetime(2024, 11, 20)  # Regulatory change date

        self.bbdd_engine = DatabaseUtils.create_engine('pruebas_BT')

        self.indicator_id_map, self.precios_sheet, self.volumenes_sheet = self.get_id_sheet_mapping()

        self.temporary_download_path = os.path.join(DATA_LAKE_BASE_PATH, 'temp_downloads')

    def get_id_sheet_mapping(self) -> tuple[dict[str, str], dict[str, Optional[str]], dict[str, Optional[str]]]:
        """
        Obtiene el mapping de los IDs de los mercados de ESIOS y sus números de hoja correspondientes.

        Returns:
            tuple[dict[str, str], dict[str, Optional[str]], dict[str, Optional[str]]]:
                1. indicator_id_map: Mapping de nombre de mercado a ID de mercado (str -> str).
                   Ej: {'Diario': '1', 'Intra 1': '2', ...}
                2. precios_id_map: Mapping de ID de mercado a número de hoja de precios (str -> Optional[str]).
                   El número de hoja es un string con padding ('01', '09', etc.) o None si no aplica.
                   Ej: {'1': '01', '2': None, ...}
                3. volumenes_id_map: Mapping de ID de mercado a número de hoja de volúmenes (str -> Optional[str]).
                   El número de hoja es un string con padding ('01', '03', etc.) o None si no aplica.
                   Ej: {'1': '03', '2': '05', ...}
        """
        # Get all market ids where either sheet number is specified (not 0)
        df_mercados = DatabaseUtils.read_table(self.bbdd_engine, 'Mercados',
                                                columns=['id', 'mercado', 'sheet_i90_precios', 'sheet_i90_volumenes', 'is_quinceminutal'],
                                                where_clause='(sheet_i90_volumenes != 0 OR sheet_i90_precios != 0)')

        # Create indicator map {mercado_name: market_id}
        indicator_map_raw = dict(zip(df_mercados['mercado'], df_mercados['id']))

        # Initialize maps for sheet numbers {market_id: sheet_num_or_None}
        volumenes_id_map_raw = {}
        precios_id_map_raw = {}

        # Populate sheet maps, handling NaNs before string conversion
        for index, row in df_mercados.iterrows():
            market_id: int = row['id']
            sheet_vol = row['sheet_i90_volumenes']
            sheet_pre = row['sheet_i90_precios']

            # Process volumenes sheet number
            if pd.notna(sheet_vol) and sheet_vol != 0:
                # Convert valid number to padded string
                volumenes_id_map_raw[market_id] = str(int(sheet_vol)).zfill(2)
            else:
                # Assign None if NaN or 0
                volumenes_id_map_raw[market_id] = None

            # Process precios sheet number
            if pd.notna(sheet_pre) and sheet_pre != 0:
                # Convert valid number to padded string
                precios_id_map_raw[market_id] = str(int(sheet_pre)).zfill(2)
            else:
                # Assign None if NaN or 0
                precios_id_map_raw[market_id] = None

        # Convert all keys and non-None values to strings for final output
        # Ensuring consistent string keys across all maps
        indicator_id_map = {str(key): str(value) for key, value in indicator_map_raw.items()}
        
        # Create final volumenes map with string keys
        volumenes_id_map = {str(key): value for key, value in volumenes_id_map_raw.items()}
        
        # Create final precios map with string keys
        precios_id_map = {str(key): value for key, value in precios_id_map_raw.items()}

        # Return the maps with string keys and Optional[str] values for sheets
        return indicator_id_map, precios_id_map, volumenes_id_map
    
    def get_lista_UPs(self, UP_ids: Optional[List[int]] = None) -> Tuple[List[str], Dict[str, int]]:
        """
        Get the list of programming units from the database.
        
        Args:
            UP_ids (Optional[List[int]]): List of programming unit IDs to filter
            
        Returns:
            Tuple[List[str], Dict[str, int]]: List of programming unit names and dictionary mapping names to IDs
        """
        # Build the WHERE clause for filtering by region and optionally by UP_ids
        where_clause = 'a.region = "ES"'
        if UP_ids:
            UP_list = ", ".join([str(item) for item in UP_ids])
            where_clause += f' AND u.id IN ({UP_list})'

        # Use DatabaseUtils.read_table to fetch the data
        df_up = DatabaseUtils.read_table(
            self.bbdd_engine,
            table_name="UPs u INNER JOIN Activos a ON u.activo_id = a.id",
            columns=["u.id as id", "UP"],
            where_clause=where_clause
        )
        # Extract the list of UP names and a dictionary mapping UP names to IDs
        unidades = df_up['UP'].tolist()
        dict_unidades = dict(zip(unidades, df_up['id']))

        return unidades, dict_unidades
    
    def get_market_data(self, mercados_ids: Optional[List[int]] = None) -> Tuple[pd.DataFrame, List[int], List[int]]:
        """
        Get market data from the database.
        
        Args:
            mercados_ids (Optional[List[int]]): List of market IDs to filter
            
        Returns:
            Tuple[pd.DataFrame, List[int], List[int]]: DataFrame with market data, list of volume sheet IDs, and list of all relevant sheet numbers
        """
        # Build the WHERE clause for filtering by sheet_i90_volumenes and optionally by mercados_ids
        where_clause = 'sheet_i90_volumenes != 0'
        if mercados_ids:
            mercados_list = ", ".join([str(item) for item in mercados_ids])
            where_clause += f' AND id IN ({mercados_list})'

        # Use DatabaseUtils.read_table to fetch the data
        df_mercados = DatabaseUtils.read_table(
            self.bbdd_engine,
            table_name="Mercados",
            columns=None,  # None means select all columns
            where_clause=where_clause
        )

        # Get unique volume sheet numbers
        pestañas_volumenes = df_mercados['sheet_i90_volumenes'].unique().tolist()
        # Combine volume and price sheet numbers, filter out NaN and convert to int
        pestañas = pestañas_volumenes + df_mercados['sheet_i90_precios'].unique().tolist()
        pestañas = [int(item) for item in pestañas if item is not None and not (isinstance(item, float) and pd.isna(item))]

        return df_mercados, pestañas_volumenes, pestañas
    
    def get_error_data(self) -> pd.DataFrame:
        """
        Get error data for I90 files from the database.
        
        Returns:
            pd.DataFrame: DataFrame with error data containing dates and error types
        """
        # Use DatabaseUtils.read_table to fetch error data
        df_errores = DatabaseUtils.read_table(
            self.bbdd_engine,
            table_name="Errores_i90_OMIE",
            columns=["fecha", "tipo_error"],
            where_clause='fuente_error = "i90"'
        )
        # Convert 'fecha' column to date type
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

class DiarioConfig(I90Config):
    def __init__(self):
        super().__init__()
        #get individual id
        self.diaria_id = self.indicator_id_map["Diario"]

        # group id onto a single var (this is the right way to do it in order to beused in get_sheets_of_interest)
        self.market_ids: List[str] = [self.diaria_id]

        # get sheets of interest
        self.volumenes_sheets: List[str]
        self.precios_sheets: List[str]
        self.sheets_of_interest: List[str]
        self.volumenes_sheets, self.precios_sheets, self.sheets_of_interest = self.get_sheets_of_interest()

class SecundariaConfig(I90Config):
    """
    Config for secundaria market data
    """
    def __init__(self):
        """
        Configuration specific to the Secundaria market.
        Initializes Secundaria specific settings including market IDs and relevant sheet filters.
        """
        super().__init__()
        # get individual ids for subir and bajar
        self.secundaria_subir_id: str = self.indicator_id_map["Secundaria a subir"]
        self.secundaria_bajar_id: str = self.indicator_id_map["Secundaria a bajar"]

        # group ids onto a single var (to be used in get_sheets_of_interest)
        self.market_ids: List[str] = [self.secundaria_subir_id, self.secundaria_bajar_id]

        # get sheets of interest
        self.volumenes_sheets: List[str]
        self.precios_sheets: List[str]
        self.sheets_of_interest: List[str]
        self.volumenes_sheets, self.precios_sheets, self.sheets_of_interest = self.get_sheets_of_interest()

class TerciariaConfig(I90Config):
    """
    Config for terciaria market data
    """
    def __init__(self):
        super().__init__()
        # get individual ids for subir and bajar
        self.terciaria_subir_id: str = self.indicator_id_map["Terciaria a subir"]
        self.terciaria_bajar_id: str = self.indicator_id_map["Terciaria a bajar"]

        # group ids onto a single var (to be used in get_sheets_of_interest)
        self.market_ids: List[str] = [self.terciaria_subir_id, self.terciaria_bajar_id]

        # get sheets of interest
        self.volumenes_sheets: List[str]
        self.precios_sheets: List[str]
        self.sheets_of_interest: List[str]
        self.volumenes_sheets, self.precios_sheets, self.sheets_of_interest = self.get_sheets_of_interest()

class RRConfig(I90Config):
    def __init__(self):
        """
        Configuration specific to the RR (Reserva de Regulación) market.
        Initializes RR specific settings including market IDs and relevant sheet filters.
        """
        super().__init__()
        # get individual ids for subir and bajar
        self.rr_subir_id: str = self.indicator_id_map["RR a subir"]
        self.rr_bajar_id: str = self.indicator_id_map["RR a bajar"]
        
        # group ids onto a single var (to be used in get_sheets_of_interest)
        self.market_ids: List[str] = [self.rr_subir_id, self.rr_bajar_id]

        # get sheets of interest
        self.volumenes_sheets: List[str]
        self.precios_sheets: List[str]
        self.sheets_of_interest: List[str]
        self.volumenes_sheets, self.precios_sheets, self.sheets_of_interest = self.get_sheets_of_interest()
        # No specific Redespacho filters defined for RR sheets ('06') in the provided snippet


class CurtailmentConfig(I90Config):
    def __init__(self):
        """
        Configuration specific to the Curtailment market.
        Initializes Curtailment specific settings including market ID and relevant sheet filters.
        """
        super().__init__()
        # get individual id
        self.curtailment_id: str = self.indicator_id_map["Curtailment"]
        self.curtailment_demanda_id: str = self.indicator_id_map["Curtailment demanda"]

        # group id onto a single var (to be used in get_sheets_of_interest)
        self.market_ids: List[str] = [self.curtailment_id]

        # get sheets of interest
        self.volumenes_sheets: List[str]
        self.precios_sheets: List[str] # Curtailment typically only has volumes
        self.sheets_of_interest: List[str]
        self.volumenes_sheets, self.precios_sheets, self.sheets_of_interest = self.get_sheets_of_interest()

        # Define Redespacho filter for volumenes sheet ('03')
        # Renamed from curtailment_volumenes_sheets for clarity
        self.redespacho_filter_volumenes: List[str] = ['UPLPVPV', 'UPLPVPCBN', 'UPOPVPB']


class P48Config(I90Config):
    def __init__(self):
        """
        Configuration specific to the P48 market.
        Initializes P48 specific settings including market ID and relevant sheet filters.
        """
        super().__init__()
        # get individual id
        self.p48_id: str = self.indicator_id_map["P48"]

        # group id onto a single var (to be used in get_sheets_of_interest)
        self.market_ids: List[str] = [self.p48_id]

        # get sheets of interest
        self.volumenes_sheets: List[str]
        _ : List[str] # P48 typically only has volumes
        self.sheets_of_interest: List[str]
        self.volumenes_sheets, _, self.sheets_of_interest = self.get_sheets_of_interest()
        # No specific Redespacho filters defined for P48 sheets ('12') in the provided snippet

class IndisponibilidadesConfig(I90Config):
    def __init__(self):
        """
        Configuration specific to the Indisponibilidades market.
        Initializes Indisponibilidades specific settings including market ID and relevant sheet filters.
        """
        super().__init__()
        # get individual id
        self.indisponibilidades_id: str = self.indicator_id_map["Indisponibilidades"]

        self.market_ids: List[str] = [self.indisponibilidades_id]

        # get sheets of interest
        self.volumenes_sheets: List[str]
        _ : List[str] # Indisponibilidades typically only has volumes
        self.sheets_of_interest: List[str]
        self.volumenes_sheets, _, self.sheets_of_interest = self.get_sheets_of_interest()

        # Define Redespacho filter for volumenes sheet ('08')
        self.redespacho_filter_volumenes: List[str] = ["Indisponibilidad"]


class RestriccionesConfig(I90Config):
    def __init__(self):
        """
        Configuration specific to the Restricciones markets (MD, TR, RT2).
        Initializes Restricciones specific settings including market IDs and relevant sheet filters.
        """
        super().__init__()

        # get restricciones mercado diario subir y bajar
        self.restricciones_md_subir_id: str = self.indicator_id_map["Restricciones MD a subir"]
        self.restricciones_md_bajar_id: str = self.indicator_id_map["Restricciones MD a bajar"]

        # get restricciones mercado tiempo real subir y bajar
        self.restricciones_tr_subir_id: str = self.indicator_id_map["Restricciones TR a subir"]
        self.restricciones_tr_bajar_id: str = self.indicator_id_map["Restricciones TR a bajar"]

        # get restricciones rt subir y bajar
        self.restricciones_rt2_subir_id: str = self.indicator_id_map["RT2 a subir"]
        self.restricciones_rt2_bajar_id: str = self.indicator_id_map["RT2 a bajar"]

        # total market ids 
        self.market_ids: List[str] = [self.restricciones_md_subir_id, self.restricciones_md_bajar_id, self.restricciones_tr_subir_id,
                            self.restricciones_tr_bajar_id, self.restricciones_rt2_subir_id, self.restricciones_rt2_bajar_id]

        # get sheets of interest (uses attribute market ids)
        self.volumenes_sheets: List[str]
        self.precios_sheets: List[str]
        self.sheets_of_interest: List[str]
        self.volumenes_sheets, self.precios_sheets, self.sheets_of_interest = self.get_sheets_of_interest()

        # --- Define Redespacho filters based on market type and sheet ---
        # Filter for MD volumenes (sheet '03') and precios (sheet '09')
        self.redespacho_filter_md: List[str] = ['ECO', 'ECOCB', 'UPOPVPV', 'UPOPVPVCB']
        # Filter for RT2 volumenes (sheet '03' - based on transform logic)
        self.redespacho_filter_rt2_vol: List[str] = ['ECOBSO', 'ECOBCBSO']

        # Filter for TR volumenes (sheet '08') and TR precios (sheet '10') - "Restricciones Técnicas"
        self.redespacho_filter_tr: List[str] = ["Restricciones Técnicas"]
        
if __name__ == "__main__":
    config = DiarioConfig()

    print(config.volumenes_sheets)
    
    if config.precios_sheets is None:
        print("precios_sheets is None")
    else:
        print(config.precios_sheets)

    print(config.sheets_of_interest)
