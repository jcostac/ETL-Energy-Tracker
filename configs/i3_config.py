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

class I3Config:
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()
        self.dia_inicio_SRS = datetime(2024, 11, 20)  # Regulatory change date (adapt if needed for i3)
        self._bbdd_engine = None
        self.id_mercado_map, self.volumenes_sheet, self.sentido_map = self.get_id_mercado_sheet_mapping()
        self.temporary_download_path = Path(os.getenv('DATA_LAKE_PATH')) / 'temporary'
        self.tech_map = self._get_technology_map()

    @property
    def bbdd_engine(self):
        if not self._bbdd_engine:
            raise ValueError("Engine not set")
        return self._bbdd_engine
    
    @bbdd_engine.setter
    def bbdd_engine(self, engine):
        self._bbdd_engine = engine
        try:
            with self._bbdd_engine.connect() as connection:
                connection.execute(text("SELECT 1"))
        except Exception as e:
            print(f"Error in engine setting: {e}")
            raise e
    
    def get_id_mercado_sheet_mapping(self) -> Tuple[Dict[str, str], Dict[str, Optional[str]], Dict[str, str]]:
        """
        Similar to i90, but using sheet_i3_* columns.
        Only handles volumenes sheets since no precios are downloaded for i3.
        """
        self.bbdd_engine = DatabaseUtils.create_engine('energy_tracker')
        df_mercados = DatabaseUtils.read_table(self.bbdd_engine, 'mercados_mapping',
                                               columns=['id', 'mercado', 'sheet_i3_volumenes', 'is_quinceminutal', 'sentido'],
                                               where_clause='(sheet_i3_volumenes != 0)')

        indicator_map_raw = dict(zip(df_mercados['mercado'], df_mercados['id']))
        volumenes_id_map_raw = {}

        for index, row in df_mercados.iterrows():
            market_id: int = row['id']
            sheet_vol = row['sheet_i3_volumenes']

            if pd.notna(sheet_vol) and sheet_vol != 0:
                volumenes_id_map_raw[market_id] = str(int(sheet_vol)).zfill(2)
            else:
                volumenes_id_map_raw[market_id] = None

        id_mercado_map = {str(key): str(value) for key, value in indicator_map_raw.items()}
        volumenes_id_map = {str(key): value for key, value in volumenes_id_map_raw.items()}
        sentido_map = {str(key): value for key, value in dict(zip(df_mercados['id'], df_mercados['sentido'])).items()}

        return id_mercado_map, volumenes_id_map, sentido_map
    
    def _get_technology_map(self) -> Dict[int, str]:
        """Map from tecnologias_generacion table for energy programs by technology."""
        self.bbdd_engine = DatabaseUtils.create_engine('energy_tracker')
        df = DatabaseUtils.read_table(self.bbdd_engine, 'tecnologias_generacion', columns=['id', 'tecnologia'])
        return dict(zip(df['id'], df['tecnologia']))
    
    def get_technologies(self) -> Dict[int, str]:
        return self.tech_map

    def _get_sheet_num(self, market_id: str) -> Optional[str]:
        try:
            return self.volumenes_sheet[market_id]
        except KeyError:
            print(f"Warning: No volumenes sheet found for market ID {market_id}")
            return None
    
    def _get_sheets(self, market_ids: List[str]) -> List[str]:
        sheet_nums = []
        for id in market_ids:
            sheet_num = self._get_sheet_num(id)
            if sheet_num is not None:
                sheet_nums.append(sheet_num)
        return list(set(sheet_nums))

    def get_redespacho_filter(self, market_id: str) -> Optional[List[str]]:
        """Base method - overridden in subclasses."""
        return None

    def get_sheets_of_interest(self) -> List[str]:
        if not hasattr(self, 'market_ids'):
            print("Warning: 'market_ids' not found. Returning empty list.")
            self.market_ids = []
        volumenes_sheets = self._get_sheets(self.market_ids)
        return volumenes_sheets

    @classmethod
    def has_volumenes_sheets(cls) -> bool:
        try:
            instance = cls()
            return bool(instance.volumenes_sheet)
        except:
            return False

class DiarioConfig(I3Config):
    def __init__(self):
        super().__init__()
        self.diario_id = self.id_mercado_map.get("Diario")
        self.market_ids: List[str] = [self.diario_id] if self.diario_id else []
        self.volumenes_sheets = self.get_sheets_of_interest()

    def get_redespacho_filter(self, market_id: str) -> Optional[List[str]]:
        return super().get_redespacho_filter(market_id)  # Returns None

class IntraConfig(I3Config): 
    
    def __init__(self, fecha: Optional[datetime] = None):
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
        self.intra_1_id: str = self.id_mercado_map.get("Intra 1")
        self.intra_2_id: str = self.id_mercado_map.get("Intra 2")
        self.intra_3_id: str = self.id_mercado_map.get("Intra 3")
        self.intra_4_id: str = self.id_mercado_map.get("Intra 4")
        self.intra_5_id: str = self.id_mercado_map.get("Intra 5")
        self.intra_6_id: str = self.id_mercado_map.get("Intra 6")
        self.intra_7_id: str = self.id_mercado_map.get("Intra 7")

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
        
        # Filter out None values from market_ids
        self.market_ids = [id for id in self.market_ids if id]
        
        # Get sheets of interest based on the selected markets
        self.volumenes_sheets = self.get_sheets_of_interest()

        self.phf_intra_map = {
            'PHF-1': 'Intra 1',
            'PHF-2': 'Intra 2',
            'PHF-3': 'Intra 3',
            'PHF-4': 'Intra 4',
            'PHF-5': 'Intra 5',
            'PHF-6': 'Intra 6',
            'PHF-7': 'Intra 7',
        }

    def get_redespacho_filter(self, market_id: str) -> Optional[List[str]]:
        return super().get_redespacho_filter(market_id)  # Returns None

class SecundariaConfig(I3Config):
    def __init__(self):
        super().__init__()
        self.secundaria_subir_id = self.id_mercado_map.get("Secundaria a subir")
        self.secundaria_bajar_id = self.id_mercado_map.get("Secundaria a bajar")
        self.market_ids: List[str] = [id for id in [self.secundaria_subir_id, self.secundaria_bajar_id] if id]
        self.volumenes_sheets = self.get_sheets_of_interest()
        self.redespacho_filter = ['RR']

    def get_redespacho_filter(self, market_id: str) -> Optional[List[str]]:
        if market_id in self.market_ids:
            return self.redespacho_filter
        return super().get_redespacho_filter(market_id)

class TerciariaConfig(I3Config):
    def __init__(self):
        super().__init__()
        self.terciaria_subir_id = self.id_mercado_map.get("Terciaria a subir")
        self.terciaria_bajar_id = self.id_mercado_map.get("Terciaria a bajar")
        self.market_ids: List[str] = [id for id in [self.terciaria_subir_id, self.terciaria_bajar_id] if id]
        self.volumenes_sheets = self.get_sheets_of_interest()
        # No filter list defined (handle exclusion in processing)

    def get_redespacho_filter(self, market_id: str) -> Optional[List[str]]:
        if market_id in self.market_ids:
            return None  # Logic: != 'TERDIR' (handle as exclusion in processing)
        return super().get_redespacho_filter(market_id)

class RRConfig(I3Config):
    def __init__(self):
        super().__init__()
        self.rr_subir_id = self.id_mercado_map.get("RR a subir")
        self.rr_bajar_id = self.id_mercado_map.get("RR a bajar")
        self.market_ids: List[str] = [id for id in [self.rr_subir_id, self.rr_bajar_id] if id]
        self.volumenes_sheets = self.get_sheets_of_interest()
        self.redespacho_filter = ['RR']

    def get_redespacho_filter(self, market_id: str) -> Optional[List[str]]:
        if market_id in self.market_ids:
            return self.redespacho_filter
        return super().get_redespacho_filter(market_id)

class CurtailmentConfig(I3Config):
    def __init__(self):
        super().__init__()
        self.curtailment_id = self.id_mercado_map.get("Curtailment")
        self.curtailment_demanda_id = self.id_mercado_map.get("Curtailment demanda")
        self.market_ids: List[str] = [id for id in [self.curtailment_id, self.curtailment_demanda_id] if id]
        self.volumenes_sheets = self.get_sheets_of_interest()
        self.redespacho_filter = ['UPLPVPV', 'UPLPVPCBN', 'UPOPVPB']

    def get_redespacho_filter(self, market_id: str) -> Optional[List[str]]:
        if market_id in self.market_ids:
            return self.redespacho_filter
        return super().get_redespacho_filter(market_id)

class P48Config(I3Config):
    def __init__(self):
        super().__init__()
        self.p48_id = self.id_mercado_map.get("P48")
        self.market_ids: List[str] = [self.p48_id] if self.p48_id else []
        self.volumenes_sheets = self.get_sheets_of_interest()
        # Removed redespacho_filter - not needed for this market

    def get_redespacho_filter(self, market_id: str) -> Optional[List[str]]:
        return super().get_redespacho_filter(market_id)  # Returns None

class IndisponibilidadesConfig(I3Config):
    def __init__(self):
        super().__init__()
        self.indisponibilidades_id = self.id_mercado_map.get("Indisponibilidades")
        self.market_ids: List[str] = [self.indisponibilidades_id] if self.indisponibilidades_id else []
        self.volumenes_sheets = self.get_sheets_of_interest()
        self.redespacho_filter = ['Indisponibilidad']

    def get_redespacho_filter(self, market_id: str) -> Optional[List[str]]:
        if market_id == self.indisponibilidades_id:
            return self.redespacho_filter
        return super().get_redespacho_filter(market_id)

class RestriccionesConfig(I3Config):
    def __init__(self):
        super().__init__()
        self.restricciones_md_subir_id = self.id_mercado_map.get("Restricciones MD a subir")
        self.restricciones_md_bajar_id = self.id_mercado_map.get("Restricciones MD a bajar")
        self.restricciones_tr_subir_id = self.id_mercado_map.get("Restricciones TR a subir")
        self.restricciones_tr_bajar_id = self.id_mercado_map.get("Restricciones TR a bajar")
        self.restricciones_rt2_subir_id = self.id_mercado_map.get("RT2 a subir")
        self.restricciones_rt2_bajar_id = self.id_mercado_map.get("RT2 a bajar")

        self.market_ids: List[str] = [
            self.restricciones_md_subir_id, self.restricciones_md_bajar_id,
            self.restricciones_tr_subir_id, self.restricciones_tr_bajar_id,
            self.restricciones_rt2_subir_id, self.restricciones_rt2_bajar_id
        ]
        
        self.volumenes_sheets = self.get_sheets_of_interest()
        self.redespacho_filter_md = ['ECO', 'ECOCB', 'UPOPVPV', 'UPOPVPVCB']
        self.redespacho_filter_tr = ['Restricciones Técnicas']
        self.redespacho_filter_rt = ['ECOBSO', 'ECOBCBSO']

    def get_redespacho_filter(self, market_id: str) -> Optional[List[str]]:
        if market_id in [self.restricciones_md_subir_id, self.restricciones_md_bajar_id]:
            return self.redespacho_filter_md
        elif market_id in [self.restricciones_tr_subir_id, self.restricciones_tr_bajar_id]:
            return self.redespacho_filter_tr
        elif market_id in [self.restricciones_rt2_subir_id, self.restricciones_rt2_bajar_id]:
            return self.redespacho_filter_rt
        return super().get_redespacho_filter(market_id)

def print_config_info():
    base_config = I3Config()
    print("\n=== BASE I3CONFIG INFORMATION ===")
    print("\nID -> Mercado Map:", base_config.id_mercado_map)
    print("\nID -> Volumenes Sheet Map:", base_config.volumenes_sheet)
    print("\nID -> Sentido Map:", base_config.sentido_map)
    print("\nTechnology Map:", base_config.get_technologies())

    configs_to_test = {}
    for cls in I3Config.__subclasses__():
        if cls.__name__ == 'IntraConfig':
            configs_to_test[cls.__name__] = cls(fecha=datetime(2024, 5, 1))
        else:
            configs_to_test[cls.__name__] = cls()

    print("\n\n=== SPECIFIC CONFIG CLASSES ===")
    for config_name, config_instance in configs_to_test.items():
        print(f"\n{config_name}")
        market_ids = getattr(config_instance, 'market_ids', [])
        volumenes_sheets = getattr(config_instance, 'volumenes_sheets', 'N/A')
        print(f"  Market IDs: {market_ids}")
        print(f"  Volume Sheets: {volumenes_sheets}")
        for mid in market_ids:
            filter_list = config_instance.get_redespacho_filter(mid)
            print(f"  Filter for {mid}: {filter_list}")

if __name__ == "__main__":
    print_config_info() 