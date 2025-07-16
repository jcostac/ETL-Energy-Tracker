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
        self.i90_sheet_to_market, self.i3_sheet_to_market = self.get_sheet_to_market_mappings()

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
        i90_sheet_to_market = dict(zip(df_i90['sheet_i90_volumenes'].astype(str).str.zfill(2), df_i90['mercado']))

        # Query for i3 sheets 03 and 08
        df_i3 = DatabaseUtils.read_table(
            self.bbdd_engine,
            table_name="mercados_mapping",
            columns=['mercado', 'sheet_i3_volumenes'],
            where_clause="sheet_i3_volumenes IN (3, 8)"
        )
        i3_sheet_to_market = dict(zip(df_i3['sheet_i3_volumenes'].astype(str).str.zfill(2), df_i3['mercado']))

        return i90_sheet_to_market, i3_sheet_to_market

    def get_market_name(self, sheet_num, source='i90'):
        """
        Get the market name associated with a given sheet number for i90 or i3.
        
        Args:
            sheet_num (str): Sheet number like '03' or '08'
            source (str): 'i90' or 'i3'
            
        Returns:
            str: Market name or None if not found
        """
        if source == 'i90':
            return self.i90_sheet_to_market.get(sheet_num)
        elif source == 'i3':
            return self.i3_sheet_to_market.get(sheet_num)
        return None
    

