import os
import pytz
from datetime import datetime
import sys
import os
import pretty_errors
from dotenv import load_dotenv
load_dotenv()
import pytz
from pathlib import Path
# Get the absolute path to the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
# Use absolute imports
from utilidades.db_utils import DatabaseUtils
from sqlalchemy import text

class ESIOSConfig:

    def __init__(self):
        self._bbdd_engine = None
        self.indicator_id_map, self.market_id_map = self.get_market_id_mapping() #returns a tuple of two dictionaries

        #valid markets for which we get esios precios data --> useful to check if ESIOS data is available for a market in processing
        self.esios_precios_markets = ['diario', 'intra', 'secundaria', 'terciaria', 'rr']

    @property
    def bbdd_engine(self):
        if not self._bbdd_engine:
            raise ValueError("BBDD engine not set")
        return self._bbdd_engine
    
    @bbdd_engine.setter
    def bbdd_engine(self, engine):
        self._bbdd_engine = engine
        #test if the engine is working
        try:
            with self._bbdd_engine.connect() as connection:
                connection.execute(text("SELECT 1"))
        except Exception as e:
            print(f"Error in engine setting: {e}")
            raise e

    def get_market_id_mapping(self) -> tuple[dict[str, str], dict[str, str]]:
        """
        Obtiene el mapping de los IDs de los mercados de ESIOS.
        Returns:
            dict: 1. indicator_id_map: Un diccionario con los nombres de los mercados y sus respectivos IDs de ESIOS.
                        i.e {'Intra 4': 600, 'Intra 5': 612, 'Intra 6': 613, 'Intra 7': 614} (nombre de mercado como key, id de ESIOS como value)
                    2. market_id_map: Un diccionario con los IDs de los mercados de ESIOS y sus IDs de mercado en la BBDD.
                        i.e {600: 1, 612: 2, 613: 3, 614: 4} (id de ESIOS como key, id de mercado como value)
        """
        self.bbdd_engine = DatabaseUtils.create_engine('energy_tracker')

        #get all market ids with indicator_esios_precios != 0
        df_mercados = DatabaseUtils.read_table(self.bbdd_engine, 'mercados_mapping', columns=['id', 'mercado', 'indicador_esios_precios as indicador', 'is_quinceminutal'], 
                                            where_clause='indicador_esios_precios != 0')
        
        
        #get idnicator map with mercado as key and indicator as value i.e {'Intra 4': 600, 'Intra 5': 612, 'Intra 6': 613, 'Intra 7': 614}
        indicator_map = dict(zip(df_mercados['mercado'], df_mercados['indicador']))
        market_id_map = dict(zip(df_mercados['indicador'], df_mercados['id']))

        
        #convert indicator value to str to avoid type errors
        indicator_id_map = {str(key): str(value) for key, value in indicator_map.items()}
        market_id_map = {str(key): str(value) for key, value in market_id_map.items()}

        return indicator_id_map, market_id_map

    def get_indicator_id(self, mercado: str) -> str:
        """
        Obtiene el ID del indicador de ESIOS para un mercado específico.

        Args:
            indicator_id (int): El ID del indicador de ESIOS.
            
        Returns:
            str: El ID del indicador de ESIOS para el mercado especificado.

        Raises:
            ValueError: Si el mercado no se encuentra en el mapping.
        """

        try: 
            return self.indicator_id_map[mercado]
        except KeyError:
            raise ValueError(f"Mercado no encontrado en el mapping: {mercado}")
        
class DiarioConfig(ESIOSConfig):

    def __init__(self):
        super().__init__() 
        self.indicator_id = self.get_indicator_id("Diario") # ie. 600

class IntraConfig(ESIOSConfig):

    def __init__(self):
        super().__init__()
        self.intra_name_map = {
            1: "Intra 1",
            2: "Intra 2",
            3: "Intra 3",
            4: "Intra 4",
            5: "Intra 5",
            6: "Intra 6",
            7: "Intra 7"
        }

        self.intra_reduccion_fecha = datetime.strptime('2024-06-13', '%Y-%m-%d')
        self.cambio_granularidad_fecha = datetime.strptime('2025-03-19', '%Y-%m-%d')

class SecundariaConfig(ESIOSConfig):

    def __init__(self):
        super().__init__()
        self.secundaria_name_map = {
            1: "Secundaria a subir",  # indicator 2130
            2: "Secundaria a bajar",  # indicator 634
        }
        self.precio_dual_fecha = datetime.strptime('2024-11-20', '%Y-%m-%d')
        self.cambio_granularidad_fecha = datetime.strptime('2022-05-24', '%Y-%m-%d')

class TerciariaConfig(ESIOSConfig):

    def __init__(self):
        super().__init__()
        self.terciaria_name_map = {
            1: "Terciaria a subir",          # 677 -> changes to 2197
            2: "Terciaria a bajar",          # 676 -> changes to 2197
            3: "Terciaria directa a subir",  # 10400 (not affected)
            4: "Terciaria directa a bajar",  # 10401 (not affected)
            5: "Terciaria programada unico", # 2197 (new indicator after change)
        }
        self.precio_unico_fecha = datetime.strptime('2024-12-10', '%Y-%m-%d')
        self.cambio_granularidad_fecha = datetime.strptime('2022-05-24', '%Y-%m-%d')

class RRConfig(ESIOSConfig):
    def __init__(self):
        super().__init__()
        self.indicator_id = self.get_indicator_id("RR a subir") #precio unico que se guarda en el indicador de RR a subir
        self.cambio_granularidad_fecha = datetime.strptime('2022-05-24', '%Y-%m-%d')
