from datetime import datetime
from typing import Optional, List, Tuple, Dict
import pandas as pd
import sys
import os
from sqlalchemy import text
from dotenv import load_dotenv
from pathlib import Path

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from utilidades.db_utils import DatabaseUtils

class I90Config:
    def __init__(self):

        """
        Initialize the I90Config base class by loading environment variables, setting the regulatory change date, preparing database engine management, retrieving market and sheet mappings, and configuring the temporary download path.
        """
        load_dotenv()

        self.dia_inicio_SRS = datetime(2024, 11, 20)  # Regulatory change date

        self._bbdd_engine = None

        self.id_mercado_map, self.precios_sheet, self.volumenes_sheet, self.sentido_map = self.get_id_mercado_sheet_mapping()

        self.temporary_download_path = Path(os.getenv('DATA_LAKE_PATH')) / 'temporary'
      

    @property
    def bbdd_engine(self):
        """
        Returns the current database engine instance.
        
        Raises:
            ValueError: If the database engine has not been set.
        """
        if not self._bbdd_engine:
            raise ValueError("Engine not set")
        return self._bbdd_engine
    
    @bbdd_engine.setter
    def bbdd_engine(self, engine):
        """
        Set the database engine and verify its connectivity.
        
        Attempts to establish a connection and execute a simple query to ensure the provided engine is valid. Raises an exception if the connection or query fails.
        """
        self._bbdd_engine = engine
        # Test if the engine is working by simply connecting and checking
        try:
            with self._bbdd_engine.connect() as connection:
                connection.execute(text("SELECT 1"))
        except Exception as e:
            print(f"Error in engine setting: {e}")
            raise e
    
    def get_id_mercado_sheet_mapping(self) -> tuple[dict[str, str], dict[str, Optional[str]], dict[str, Optional[str]], dict[str, str]]:
        """
        Retrieve mappings between market names, market IDs, and their associated I90 sheet numbers and directions from the database.
        
        Returns:
            tuple: A tuple containing four dictionaries:
                - id_mercado_map (dict[str, str]): Maps market names to their corresponding market IDs as strings.
                - precios_id_map (dict[str, Optional[str]]): Maps market IDs to their price sheet numbers as zero-padded strings, or None if not applicable.
                - volumenes_id_map (dict[str, Optional[str]]): Maps market IDs to their volume sheet numbers as zero-padded strings, or None if not applicable.
                - sentido_map (dict[str, str]): Maps market IDs to their direction ("subir" or "bajar").
        """

        self.bbdd_engine = DatabaseUtils.create_engine('energy_tracker')

        # Get all market ids where either sheet number is specified (not 0)
        df_mercados = DatabaseUtils.read_table(self.bbdd_engine, 'mercados_mapping',
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
        Retrieve programming unit names and their corresponding IDs from the database, optionally filtered by a list of unit IDs.
        
        Parameters:
            UP_ids (Optional[List[int]]): If provided, only programming units with these IDs are included.
        
        Returns:
            Tuple[List[str], Dict[str, int]]: A list of programming unit names and a dictionary mapping each name to its ID.
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
            columns=["u.id as id", "UP"],
            where_clause=where_clause
        )
        # Extract the list of UP names and a dictionary mapping UP names to IDs
        unidades = df_up['UP'].tolist()
        dict_unidades = dict(zip(unidades, df_up['id']))

        return unidades, dict_unidades
    
    def get_market_data(self, mercados_ids: Optional[List[int]] = None) -> Tuple[pd.DataFrame, List[int], List[int]]:
        """
        Retrieve market data and associated sheet numbers from the database, optionally filtered by market IDs.
        
        Parameters:
        	mercados_ids (Optional[List[int]]): List of market IDs to filter the query. If None, all markets with volume sheets are included.
        
        Returns:
        	Tuple containing:
        	- DataFrame with market data for the selected markets.
        	- List of unique volume sheet numbers.
        	- List of all relevant sheet numbers (volume and price), with duplicates removed.
        """
        # Build the WHERE clause for filtering by sheet_i90_volumenes and optionally by mercados_ids
        where_clause = 'sheet_i90_volumenes != 0'
        if mercados_ids:
            mercados_list = ", ".join([str(item) for item in mercados_ids])
            where_clause += f' AND id IN ({mercados_list})'

        self.bbdd_engine = DatabaseUtils.create_engine('energy_tracker')

        # Use DatabaseUtils.read_table to fetch the data
        df_mercados = DatabaseUtils.read_table(
            self.bbdd_engine,
            table_name="mercados_mapping",
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
        Retrieve error records for I90 files from the database, including error dates and types.
        
        Returns:
            pd.DataFrame: DataFrame containing 'fecha' (date) and 'tipo_error' columns for errors with source "i90".
        """
        self.bbdd_engine = DatabaseUtils.create_engine('pruebas_BT')

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
        Return lists of relevant sheet numbers for volumes, prices, and their combined unique set based on the instance's market IDs.
        
        Returns:
            Tuple[List[str], List[str], List[str]]: 
                A tuple containing:
                - List of unique volume sheet numbers.
                - List of unique price sheet numbers.
                - List of all unique sheet numbers from both categories.
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

    @classmethod
    def has_volumenes_sheets(cls) -> bool:
        """
        Determine whether the configuration class provides volume sheet numbers.
        
        Returns:
            bool: True if the class instance has volume sheets; False otherwise or if instantiation fails.
        """
        try:
            if cls.__name__ == "IntraConfig":
                instance = cls(fecha=datetime.today())
            else:
                instance = cls()
            return bool(instance.volumenes_sheets)
        except TypeError:
            # If instantiation fails due to missing parameters, return False
            return False
        except Exception:
            # Any other error during instantiation
            return False
    
    @classmethod
    def has_precios_sheets(cls) -> bool:
        """
        Determine whether the configuration class provides price sheet numbers.
        
        Returns:
            bool: True if the class instance has price sheets; False otherwise or if instantiation fails.
        """
        try:
            instance = cls()
            if cls.__name__ == "IntraConfig":
                instance = cls(fecha=datetime.today())
            return bool(instance.precios_sheets)
        except TypeError:
            # If instantiation fails due to missing parameters, return False
            return False
        except Exception:
            # Any other error during instantiation
            return False

class DiarioConfig(I90Config):
    def __init__(self):
        super().__init__()
        #get individual i.e. {'Diario': 1}
        self.diaria_id = self.id_mercado_map["Diario"]

        # group id onto a single var (this is the right way to do it in order to be used in get_sheets_of_interest)
        self.market_ids: List[str] = [self.diaria_id]

        # get sheets of interest
        self.volumenes_sheets: List[str]
        self.precios_sheets: List[str]
        self.sheets_of_interest: List[str]
        self.volumenes_sheets, self.precios_sheets, self.sheets_of_interest = self.get_sheets_of_interest()

class IntraConfig(I90Config):
    def __init__(self, fecha):
        """
        Initialize the IntraConfig with market IDs and sheet selections based on the provided date.
        
        Parameters:
            fecha (datetime): The date used to determine which intra markets to include. Must not be None.
        
        Raises:
            ValueError: If `fecha` is not provided.
        """
        super().__init__()
        
        # Set the cutoff date for intra market reduction
        self.intra_reduction_date = datetime(2024, 6, 13)
        
        # Use provided date or current date
        if fecha is None:
            raise ValueError("Fecha is required for IntraConfig")
        
        # Get individual IDs for each Intra market (Intra 1 through Intra 7)
        self.intra_1_id: str = self.id_mercado_map["Intra 1"]  # ID: 2
        self.intra_2_id: str = self.id_mercado_map["Intra 2"]  # ID: 3  
        self.intra_3_id: str = self.id_mercado_map["Intra 3"]  # ID: 4
        self.intra_4_id: str = self.id_mercado_map["Intra 4"]  # ID: 5
        self.intra_5_id: str = self.id_mercado_map["Intra 5"]  # ID: 6
        self.intra_6_id: str = self.id_mercado_map["Intra 6"]  # ID: 7
        self.intra_7_id: str = self.id_mercado_map["Intra 7"]  # ID: 8

        # Determine which markets to include based on the date
        if fecha >= self.intra_reduction_date:
            # After June 13, 2024: only use Intra 1, 2, and 3
            self.market_ids: List[str] = [
                self.intra_1_id, self.intra_2_id, self.intra_3_id
            ]
            print(f"Using intra markets 1-3 (after intra reduction date) for date {fecha.date()}")
        else:
            # Before June 13, 2024: use all 7 intra markets
            self.market_ids: List[str] = [
                self.intra_1_id, self.intra_2_id, self.intra_3_id, 
                self.intra_4_id, self.intra_5_id, self.intra_6_id, self.intra_7_id
            ]
            print(f"Using intra markets 1-7 (before intra reduction date) for date {fecha.date()}")

        # Get sheets of interest based on the selected markets
        self.volumenes_sheets: List[str]
        self.precios_sheets: List[str]
        self.sheets_of_interest: List[str]
        self.volumenes_sheets, self.precios_sheets, self.sheets_of_interest = self.get_sheets_of_interest()

class SecundariaConfig(I90Config):
    """
    Config for secundaria market data
    """
    def __init__(self):
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
        """
        Initialize the configuration for the Terciaria market, setting market IDs and retrieving relevant sheet numbers for volume and price data.
        """
        super().__init__()
        # get individual ids for subir and bajar
        self.terciaria_subir_id: str = self.id_mercado_map["Terciaria a subir"]
        self.terciaria_bajar_id: str = self.id_mercado_map["Terciaria a bajar"]
        
        self.terciaria_directa_subir_id: str = self.id_mercado_map["Terciaria directa a subir"]
        self.terciaria_directa_bajar_id: str = self.id_mercado_map["Terciaria directa a bajar"]

        # group ids onto a single var (to be used in get_sheets_of_interest)
        self.market_ids: List[str] = [self.terciaria_subir_id, self.terciaria_bajar_id, self.terciaria_directa_subir_id, self.terciaria_directa_bajar_id]

        # get sheets of interest
        self.volumenes_sheets: List[str]
        self.precios_sheets: List[str]
        self.sheets_of_interest: List[str]
        self.volumenes_sheets, self.precios_sheets, self.sheets_of_interest = self.get_sheets_of_interest()

        self.redespacho_filter_terciaria_dir: List[str] = ['TERDIR', "TERMER"]
        self.redespacho_filter_terciaria: List[str] = ['!TERDIR', "!TERMER"] #the ! serves as an identifier for the exclude filter

    def get_redespacho_filter(self, market_id: str) -> Optional[List[str]]:
        """
        Return the redespacho filter list for the specified Terciaria market ID.
        """
        if market_id in [self.terciaria_subir_id, self.terciaria_bajar_id]: #programada subir y bajar
            return self.redespacho_filter_terciaria #everything except TERDIR
        elif market_id in [self.terciaria_directa_subir_id, self.terciaria_directa_bajar_id]: #directa subir y bajar
            return self.redespacho_filter_terciaria_dir #only TERDIR
        else:
            return super().get_redespacho_filter(market_id)

class RRConfig(I90Config):
    def __init__(self):
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
        
class P48Config(I90Config):
    def __init__(self):
        """
        Initializes configuration for the P48 market, setting the market ID and retrieving relevant volume sheet numbers.
        """
        super().__init__()
        # get individual id
        self.p48_id: str = self.id_mercado_map["P48"]

        # group id onto a single var (to be used in get_sheets_of_interest)
        self.market_ids: List[str] = [self.p48_id]

        # get sheets of interest
        self.volumenes_sheets: List[str]
        self.precios_sheets: List[str]
        self.sheets_of_interest: List[str]
        self.volumenes_sheets, self.precios_sheets, self.sheets_of_interest = self.get_sheets_of_interest()
        # No specific Redespacho filters defined for P48 sheets ('12') in the provided snippet

class IndisponibilidadesConfig(I90Config):
    def __init__(self):
        """
        Initializes configuration for the Indisponibilidades market, setting up market IDs, relevant volume sheets, and redespacho filters.
        """
        super().__init__()
        # get individual id
        self.indisponibilidades_id: str = self.id_mercado_map["Indisponibilidades"]

        self.market_ids: List[str] = [self.indisponibilidades_id]

        # get sheets of interest
        self.volumenes_sheets: List[str]
        self.precios_sheets: List[str]
        self.sheets_of_interest: List[str]
        self.volumenes_sheets, self.precios_sheets, self.sheets_of_interest = self.get_sheets_of_interest()

        # Define Redespacho filter for volumenes sheet ('08')
        self.redespacho_filter_volumenes: List[str] = ["Indisponibilidad"]

class RestriccionesTRConfig(I90Config):
    def __init__(self):
        """
        Initializes configuration for Restricciones markets, setting market IDs, relevant sheet numbers, and redespacho filters for each market type.
        
        Defines market IDs for Restricciones MD, TR, and RT2 (both subir and bajar), retrieves associated volume and price sheets, and sets up filters used to process redespacho data for each market category.
        """
        super().__init__()

        # get restricciones mercado tiempo real subir y bajar sheet 08
        self.restricciones_tr_subir_id: str = self.id_mercado_map["Restricciones TR a subir"] # 11
        self.restricciones_tr_bajar_id: str = self.id_mercado_map["Restricciones TR a bajar"] # 12

        # total market ids
        self.market_ids: List[str] = [self.restricciones_tr_subir_id, self.restricciones_tr_bajar_id]

        # get sheets of interest (uses attribute market ids)
        self.volumenes_sheets: List[str]
        self.precios_sheets: List[str]
        self.sheets_of_interest: List[str]
        self.volumenes_sheets, self.precios_sheets, self.sheets_of_interest = self.get_sheets_of_interest()

        # Filter for TR (tiempo real) markets (IDs 11/12) - applies to sheets '08' (Vol), '10' (Pre)
        self.redespacho_filter_tr: List[str] = ["Restricciones Técnicas"] #meter restriccion para solo descargar precios 

    def get_redespacho_filter(self, market_id: str) -> Optional[List[str]]:
        """
        Return the redespacho filter list for a given market ID in the Restricciones configuration.
        
        Parameters:
            market_id (str): The market ID for which to retrieve the redespacho filter.
        
        Returns:
            Optional[List[str]]: The filter list for the specified market ID, or None if no filter is defined.
        """
        if market_id in [self.restricciones_tr_subir_id, self.restricciones_tr_bajar_id]:
            return self.redespacho_filter_tr
        return super().get_redespacho_filter(market_id)

class RestriccionesMDConfig(I90Config):
    def __init__(self):
        super().__init__()
        # get restricciones mercado diario subir y bajar sheet 03
        self.restricciones_md_subir_id: str = self.id_mercado_map["Restricciones MD a subir"] # 9
        self.restricciones_md_bajar_id: str = self.id_mercado_map["Restricciones MD a bajar"] # 10

        # group ids onto a single var (to be used in get_sheets_of_interest)
        self.market_ids: List[str] = [self.restricciones_md_subir_id, self.restricciones_md_bajar_id]

        # get sheets of interest
        self.volumenes_sheets: List[str]
        self.precios_sheets: List[str]
        self.sheets_of_interest: List[str]
        self.volumenes_sheets, self.precios_sheets, self.sheets_of_interest = self.get_sheets_of_interest()

        #no redespacho filter defined for this config (for sheets 03 and 09 we take all the data)

class DesviosConfig(I90Config):
    def __init__(self):
        super().__init__()
        # get desvios subir y bajar sheet 08
        self.desvios_subir_id: str = self.id_mercado_map["Desvios a subir"] # 30
        self.desvios_bajar_id: str = self.id_mercado_map["Desvios a bajar"] # 31

        # group ids onto a single var (to be used in get_sheets_of_interest)
        self.market_ids: List[str] = [self.desvios_subir_id, self.desvios_bajar_id]

        # get sheets of interest
        self.volumenes_sheets: List[str]
        self.precios_sheets: List[str]
        self.sheets_of_interest: List[str]
        self.volumenes_sheets, self.precios_sheets, self.sheets_of_interest = self.get_sheets_of_interest()

        self.redespacho_filter_desvios: List[str] = ["Desvíos"]

    def get_redespacho_filter(self, market_id: str) -> Optional[List[str]]:
        """
        Return the redespacho filter list for a given market ID in the Desvios configuration.
        """
        if market_id in [self.desvios_subir_id, self.desvios_bajar_id]:
            return self.redespacho_filter_desvios
        else:
            return super().get_redespacho_filter(market_id)

def print_config_info():
    # --- Base I90Config Info ---
    """
    Prints detailed configuration information for the base I90Config and all its subclasses.
    
    Displays mappings of market IDs to market names, sheet numbers, and directions for the base configuration. For each subclass, prints the configured market IDs, associated sheet numbers, and any redespacho filters, along with the temporary download path. Intended for diagnostic or inspection purposes.
    """
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
    configs_to_test = {}
    for cls in I90Config.__subclasses__():
        if cls.__name__ == 'IntraConfig':
            # IntraConfig requires a fecha parameter - use a date before 2024-06
            configs_to_test[cls.__name__] = cls(fecha=datetime(2024, 5, 1))
        else:
            configs_to_test[cls.__name__] = cls()

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