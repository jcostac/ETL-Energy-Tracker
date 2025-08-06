import sys
import os
from sqlalchemy import text
from dotenv import load_dotenv
from pathlib import Path

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from utilidades.db_utils import DatabaseUtils
from i90_config import I90Config


class RestriccionesConfig:
    def __init__(self):
        """
        Initialize CurtailmentConfig with mappings loaded from the database.
        """
        load_dotenv()
       
        
        #sheet 03
        self.md_redespacho_filter_map = {
            "Fase1": ["UPL", "UPO", "VPB"], #have prefixes UPL and UPO
            "Fase2": ["ECO", "UPA", "VPA"] #have prefix ECO
        }

        #sheet 08
        self.tr_redespacho_filter_map = {
            "RT5": ["Restricciones Técnicas"]
        }

        """
        # get restricciones mercado diario subir y bajar
        self.restricciones_md_subir_id: str = self.id_mercado_map["Restricciones MD a subir"] # 9
        self.restricciones_md_bajar_id: str = self.id_mercado_map["Restricciones MD a bajar"] # 10

        # get restricciones mercado tiempo real subir y bajar
        self.restricciones_tr_subir_id: str = self.id_mercado_map["Restricciones TR a subir"] # 11
        self.restricciones_tr_bajar_id: str = self.id_mercado_map["Restricciones TR a bajar"] # 12

        # get restricciones rt subir y bajar 
        self.restricciones_rt2_subir_id: str = self.id_mercado_map["RT2 a subir"] # 24
        self.restricciones_rt2_bajar_id: str = self.id_mercado_map["RT2 a bajar"] # 25

        # total market ids
        self.market_ids: List[str] = [self.restricciones_md_subir_id, self.restricciones_md_bajar_id,
                                      self.restricciones_tr_subir_id, self.restricciones_tr_bajar_id,
                                      self.restricciones_rt2_subir_id, self.restricciones_rt2_bajar_id]


        # --- Define Redespacho filters based on market type ---
        # Filter for MD (mercado diario) markets (IDs 9, 10) - applies  to sheets '03', '09' 
        #restriccions tecnicas fase 2 
        self.redespacho_filter_md: List[str] = ['ECO', 'ECOCB', 'UPOPVPV', 'UPOPVPVCB']

        # Filter for RT2 markets (IDs 24/25) - applies to sheet '03' (Vol)
        self.redespacho_filter_rt_vol: List[str] = ['ECOBSO', 'ECOBCBSO']

        # Filter for TR (tiempo real) markets (IDs 11/12) - applies to sheets '08' (Vol), '10' (Pre)
        self.redespacho_filter_tr: List[str] = ["Restricciones Técnicas"] #meter restriccion para solo descargar precios 

        """
    

