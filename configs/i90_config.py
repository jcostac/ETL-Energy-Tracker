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

        self.id_mercado_map, self.precios_sheet, self.volumenes_sheet, self.sentido_map = self.get_id_mercado_sheet_mapping()

        self.temporary_download_path = os.path.join(DATA_LAKE_BASE_PATH, 'temporary')

    def get_id_mercado_sheet_mapping(self) -> tuple[dict[str, str], dict[str, Optional[str]], dict[str, Optional[str]], dict[str, str]]:
        """
        Obtiene el mapping de los IDs de los mercados de ESIOS y sus números de hoja correspondientes.

        Returns:
            tuple[dict[str, str], dict[str, Optional[str]], dict[str, Optional[str]]]:
                1. id_mercado_map: Mapping de nombre de mercado a ID de mercado (str -> str).
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
                                                columns=['id', 'mercado', 'sheet_i90_precios', 'sheet_i90_volumenes', 'is_quinceminutal', 'sentido'],
                                                where_clause='(sheet_i90_volumenes != 0 OR sheet_i90_precios != 0)')

        # Create indicator map {mercado_name: market_id, sentido: subir/bajar}
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
        id_mercado_map = {str(key): str(value) for key, value in indicator_map_raw.items()}
        
        # Create final volumenes map with string keys
        volumenes_id_map = {str(key): value for key, value in volumenes_id_map_raw.items()}
        
        # Create final precios map with string keys
        precios_id_map = {str(key): value for key, value in precios_id_map_raw.items()}

        #sentido map {market_id: subir/bajar}
        sentido_map = dict(zip(df_mercados['id'], df_mercados['sentido']))
        sentido_map = {str(key): value for key, value in sentido_map.items()}
    

        # Return the maps with string keys and Optional[str] values for sheets
        return id_mercado_map, precios_id_map, volumenes_id_map, sentido_map
    
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
    
    def _get_sheets(self, market_ids: List[int], sheet_type: str) -> List[int]:
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
            sheet_num = self._get_sheet_num(id, sheet_type)
            if sheet_num is not None:
                sheet_nums.append(sheet_num)
        sheet_nums = list(set(sheet_nums))
        return sheet_nums

    def get_redespacho_filter(self, market_id: str) -> Optional[List[str]]:
        """
        Get the redespacho filter for a given market ID. If the particular class does not have a get_redespacho_filter method,
        it will call the parent class's get_redespacho_filter method, which will return None.
        """
        return None

    def get_sheets_of_interest(self) -> Tuple[List[str], List[str], List[str]]:
        """
        Get the lists of relevant sheet numbers (volumenes, precios, and combined unique)
        based on the market_ids defined in the specific config instance.

        Returns:
            Tuple[List[str], List[str], List[str]]:
                - List of unique volume sheet numbers (as strings).
                - List of unique price sheet numbers (as strings).
                - List of unique combined sheet numbers (as strings).
        """
        # Ensure market_ids attribute exists in the subclass instance calling this
        if not hasattr(self, 'market_ids'):
             # This should ideally not happen if subclasses always define self.market_ids
             # Defaulting to empty list, but consider raising an error if it indicates a setup issue.
             print("Warning: 'market_ids' attribute not found in config instance. Returning empty sheet lists.")
             self.market_ids = []

        # Get unique volume and price sheets based on the instance's market_ids
        volumenes_sheets = self._get_sheets(self.market_ids, "volumenes")
        precios_sheets = self._get_sheets(self.market_ids, "precios")

        # Combine and get unique sheets from both lists
        sheets_of_interest = list(set(volumenes_sheets + precios_sheets))

        return volumenes_sheets, precios_sheets, sheets_of_interest

class DiarioConfig(I90Config):
    def __init__(self):
        super().__init__()
        #get individual id
        self.diaria_id = self.id_mercado_map["Diario"]

        # group id onto a single var (this is the right way to do it in order to be used in get_sheets_of_interest)
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
        self.secundaria_subir_id: str = self.id_mercado_map["Secundaria a subir"]
        self.secundaria_bajar_id: str = self.id_mercado_map["Secundaria a bajar"]

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
        self.terciaria_subir_id: str = self.id_mercado_map["Terciaria a subir"]
        self.terciaria_bajar_id: str = self.id_mercado_map["Terciaria a bajar"]

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
        self.rr_subir_id: str = self.id_mercado_map["RR a subir"]
        self.rr_bajar_id: str = self.id_mercado_map["RR a bajar"]
        
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
        self.curtailment_id: str = self.id_mercado_map["Curtailment"]
        self.curtailment_demanda_id: str = self.id_mercado_map["Curtailment demanda"]

        # group id onto a single var (to be used in get_sheets_of_interest)
        # Include both relevant market IDs if processing logic needs them
        self.market_ids: List[str] = [self.curtailment_id, self.curtailment_demanda_id]

        # get sheets of interest
        self.volumenes_sheets: List[str]
        self.precios_sheets: List[str] # Curtailment typically only has volumes ('03'), demand ('23') might have others
        self.sheets_of_interest: List[str]
        self.volumenes_sheets, self.precios_sheets, self.sheets_of_interest = self.get_sheets_of_interest()

        # Define Redespacho filter for volumenes sheet ('03') associated with the main Curtailment ID
        # Renamed from curtailment_volumenes_sheets for clarity
        self.redespacho_filter_curtailment: List[str] = ['UPLPVPV', 'UPLPVPCBN', 'UPOPVPB']

    def get_redespacho_filter(self, market_id: str) -> Optional[List[str]]:
        """
        Returns the redespacho filter list applicable to the Curtailment market ID.
        This filter is intended for the 'Redespacho' column when processing data
        associated with the main Curtailment market (ID '13').
        """
        if market_id == self.curtailment_id:
            # This filter applies when dealing with the primary Curtailment market ID ('13')
            return self.redespacho_filter_curtailment
        # elif market_id == self.curtailment_demanda_id:
            # Define filters for curtailment_demanda (ID '23') if needed, e.g.:
            # return ['Some', 'Demand', 'Filters']
        # Otherwise, no specific filter defined for this ID in this config
        return super().get_redespacho_filter(market_id)


class P48Config(I90Config):
    def __init__(self):
        """
        Configuration specific to the P48 market.
        Initializes P48 specific settings including market ID and relevant sheet filters.
        """
        super().__init__()
        # get individual id
        self.p48_id: str = self.id_mercado_map["P48"]

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
        self.indisponibilidades_id: str = self.id_mercado_map["Indisponibilidades"]

        self.market_ids: List[str] = [self.indisponibilidades_id]

        # get sheets of interest
        self.volumenes_sheets: List[str]
        _ : List[str] # Indisponibilidades typically only has volumes ('08')
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
        self.restricciones_md_subir_id: str = self.id_mercado_map["Restricciones MD a subir"] # 24
        self.restricciones_md_bajar_id: str = self.id_mercado_map["Restricciones MD a bajar"] # 25

        # get restricciones mercado tiempo real subir y bajar
        self.restricciones_tr_subir_id: str = self.id_mercado_map["Restricciones TR a subir"] # 26
        self.restricciones_tr_bajar_id: str = self.id_mercado_map["Restricciones TR a bajar"] # 27

        # get restricciones rt subir y bajar (assuming RT2 means RT from context)
        self.restricciones_rt2_subir_id: str = self.id_mercado_map["RT2 a subir"] # 18
        self.restricciones_rt2_bajar_id: str = self.id_mercado_map["RT2 a bajar"] # 19

        # total market ids
        self.market_ids: List[str] = [self.restricciones_md_subir_id, self.restricciones_md_bajar_id,
                                      self.restricciones_tr_subir_id, self.restricciones_tr_bajar_id,
                                      self.restricciones_rt2_subir_id, self.restricciones_rt2_bajar_id]

        # get sheets of interest (uses attribute market ids)
        self.volumenes_sheets: List[str]
        self.precios_sheets: List[str]
        self.sheets_of_interest: List[str]
        self.volumenes_sheets, self.precios_sheets, self.sheets_of_interest = self.get_sheets_of_interest()

        # --- Define Redespacho filters based on market type ---
        # Filter for MD markets (IDs 24, 25) - applies potentially to sheets '03', '09'
        self.redespacho_filter_md: List[str] = ['ECO', 'ECOCB', 'UPOPVPV', 'UPOPVPVCB']

        # Filter for RT2/RT markets (IDs 18, 19) - applies potentially to sheet '03' (Vol)
        # Renaming variable for clarity based on market name 'RT'
        self.redespacho_filter_rt_vol: List[str] = ['ECOBSO', 'ECOBCBSO']

        # Filter for TR markets (IDs 26, 27) - applies potentially to sheets '08' (Vol), '10' (Pre)
        self.redespacho_filter_tr: List[str] = ["Restricciones Técnicas"]

    def get_redespacho_filter(self, market_id: str) -> Optional[List[str]]:
        """
        Returns the appropriate redespacho filter list based on the market ID for Restricciones.
        The returned list is intended to filter the 'Redespacho' column in your DataFrame.
        """
        if market_id in [self.restricciones_md_subir_id, self.restricciones_md_bajar_id]:
            # Market IDs 24, 25 correspond to MD filters
            return self.redespacho_filter_md
        elif market_id in [self.restricciones_tr_subir_id, self.restricciones_tr_bajar_id]:
            # Market IDs 26, 27 correspond to TR filters
            return self.redespacho_filter_tr
        elif market_id in [self.restricciones_rt2_subir_id, self.restricciones_rt2_bajar_id]:
            # Market IDs 18, 19 correspond to RT volume filters
            return self.redespacho_filter_rt_vol
        # Otherwise, no specific filter defined for this ID in this config
        return super().get_redespacho_filter(market_id)

def print_config_info():
    # --- Base I90Config Info ---
    print("\n=== BASE I90CONFIG INFORMATION ===")
    base_config = I90Config()
    print("\nID -> Mercado Map:")
    for id, mercado in base_config.id_mercado_map.items():
        print(f"  {id}: {mercado}")
    
    print("\nID -> Precios Sheet Map:")
    for id, sheet in base_config.precios_sheet.items():
        print(f"  {id}: {sheet}")
    
    print("\nID -> Volumenes Sheet Map:")
    for id, sheet in base_config.volumenes_sheet.items():
        print(f"  {id}: {sheet}")
    
    print("\nID -> Sentido Map:")
    for id, sentido in base_config.sentido_map.items():
        print(f"  {id}: {sentido}")

    # --- Test Specific Configs ---
    # Get all subclasses of I90Config
    configs_to_test = {cls.__name__: cls() for cls in I90Config.__subclasses__()}

    print("\n\n=== SPECIFIC CONFIG CLASSES ===")
    for config_name, config_instance in configs_to_test.items():
        print(f"\n{'=' * 50}")
        print(f"  {config_name}")
        print(f"{'=' * 50}")
        
        if not isinstance(config_instance, I90Config):
            print(f"ERROR: Expected an I90Config instance for {config_name}, got {type(config_instance)}")
            continue

        # Use getattr for safety, in case an attribute is missing in a specific config
        market_ids = getattr(config_instance, 'market_ids', [])
        volumenes_sheets = getattr(config_instance, 'volumenes_sheets', 'N/A')
        precios_sheets = getattr(config_instance, 'precios_sheets', 'N/A')
        sheets_of_interest = getattr(config_instance, 'sheets_of_interest', 'N/A')

        print("\nConfiguration Summary:")
        print(f"  • Market IDs: {', '.join(market_ids) if market_ids else 'None'}")
        print(f"  • Volume Sheets: {', '.join(volumenes_sheets) if isinstance(volumenes_sheets, list) else volumenes_sheets}")
        print(f"  • Price Sheets: {', '.join(precios_sheets) if isinstance(precios_sheets, list) else precios_sheets}")
        print(f"  • All Sheets: {', '.join(sheets_of_interest) if isinstance(sheets_of_interest, list) else sheets_of_interest}")

        if not market_ids:
            print("\n  No specific market IDs defined for this config.")
            continue

        print("\nDetailed Market Information:")
        print(f"{'ID':<6} {'Sentido':<10} {'Redespacho Filter'}")
        print(f"{'-'*6} {'-'*10} {'-'*50}")
        for mid in market_ids:
            # Access sentido_map from the instance (inherited or overridden)
            # Ensure 'sentido' is a string, defaulting to 'N/A' if None or key missing
            sentido_val = config_instance.sentido_map.get(mid)
            sentido = str(sentido_val) if sentido_val is not None else 'N/A'
            # Call the instance's specific get_redespacho_filter method
            filter_list = config_instance.get_redespacho_filter(mid)
            filter_str = ', '.join(filter_list) if filter_list else 'None'
            print(f"{mid:<6} {sentido:<10} {filter_str}")

            
if __name__ == "__main__":
    print_config_info()