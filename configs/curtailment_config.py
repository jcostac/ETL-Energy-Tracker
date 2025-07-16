import sys
import os
from sqlalchemy import text
from dotenv import load_dotenv
from pathlib import Path

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from utilidades.db_utils import DatabaseUtils

class CurtailmentConfig:
    def __init__(self):
        """
        Initialize CurtailmentConfig with mappings loaded from the database.
        """
        load_dotenv()
        self._bbdd_engine = None
        self.rtx_mapping = {"RT1": "03", "RT5": "08"}
        self.rt1_redespacho_filter = ["UPLPVPV", "UPLPVPCBN"]
        self.rt5_redespacho_filter = ["Restricciones Técnicas"]
        self.i90_sheet_to_market_id, self.i3_sheet_to_market_id = self.get_sheet_to_market_mappings()

        #mapping useful so we can search in the raw file of the correct market folder
        self.market_id_to_name = {
            9: "restricciones", #a subir no se va a usar
            10: "restricciones", 
            11: "restricciones", #a subir no se va a usar
            12: "restricciones", 
            13: "curtailment",
            22: "indisponibilidades",
            23: "curtailment", #a subir no se va a usar (ie curtailment demanda)
            24: "restricciones", #a subir no se va a usar
            25: "restricciones"#a subir no se va a usar
        }

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

    def get_sheet_to_market_mappings(self):
        """
        Retrieve mappings from sheet numbers to market names for i90 and i3 from the mercados_mapping table.
        
        Returns:
            tuple: Two dictionaries - i90_sheet_to_market and i3_sheet_to_market.
        """
        self.bbdd_engine = DatabaseUtils.create_engine('energy_tracker')

        # Query for i90 sheets 03 and 08
        df_i90 = DatabaseUtils.read_table(
            self.bbdd_engine,
            table_name="mercados_mapping",
            columns=['mercado', 'sheet_i90_volumenes'],
            where_clause="sheet_i90_volumenes IN (3, 8)"
        )
        i90_sheet_to_market_id = dict(zip(df_i90['sheet_i90_volumenes'].astype(str).str.zfill(2), df_i90['id']))

        # Query for i3 sheets 03 and 08
        df_i3 = DatabaseUtils.read_table(
            self.bbdd_engine,
            table_name="mercados_mapping",
            columns=['mercado', 'sheet_i3_volumenes'],
            where_clause="sheet_i3_volumenes IN (3, 8)"
        )
        i3_sheet_to_market_id= dict(zip(df_i3['sheet_i3_volumenes'].astype(str).str.zfill(2), df_i3['id']))

        return i90_sheet_to_market_id, i3_sheet_to_market_id

    def get_market_folder(self, market_id: int):
        """
        Get the market folder name associated with a given market id.

        Args:
            market_id (int): The market id to get the folder name for.

        Returns:
            str: The market folder name.
        """
        return self.market_id_to_name[market_id]


        
    

