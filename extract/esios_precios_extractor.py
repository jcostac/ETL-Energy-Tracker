from typing import Dict, List, Optional, Union
import pandas as pd
from datetime import datetime
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from extract.descargador_esios import Diario, Intra, Secundaria, Terciaria, RR

class ESIOSPreciosExtractor:
    """
    Wrapper class for extracting price data from ESIOS API.
    Provides a unified interface for extracting data from different markets.
    """
    
    def __init__(self):
        """Initialize market extractors"""
        self.diario = Diario()
        self.intra = Intra()
        self.secundaria = Secundaria()
        self.terciaria = Terciaria()
        self.rr = RR()

    def extract_diario(self, fecha_inicio: str, fecha_fin: str) -> pd.DataFrame:
        """
        Extract daily market prices from ESIOS.
        
        Args:
            fecha_inicio (str): Start date in YYYY-MM-DD format
            fecha_fin (str): End date in YYYY-MM-DD format
            
        Returns:
            pd.DataFrame: DataFrame with daily market prices containing columns:
                - fecha: date
                - hora: hour
                - precio: price
                - id_mercado: market ID
        """
        return self.diario.get_prices(fecha_inicio_carga=fecha_inicio, 
                                    fecha_fin_carga=fecha_fin)

    def extract_intra(self, 
                     fecha_inicio: str, 
                     fecha_fin: str, 
                     intra_lst: List[int]) -> pd.DataFrame:
        """
        Extract intraday market prices from ESIOS.
        
        Args:
            fecha_inicio (str): Start date in YYYY-MM-DD format
            fecha_fin (str): End date in YYYY-MM-DD format
            intra_lst (List[int]): List of intraday market IDs (1-7)
            
        Returns:
            pd.DataFrame: DataFrame with intraday market prices
        
        Note:
            After 2024-06-13, only Intra 1-3 are available due to regulatory changes
        """
        return self.intra.get_prices(fecha_inicio_carga=fecha_inicio,
                                   fecha_fin_carga=fecha_fin,
                                   intra_lst=intra_lst)

    def extract_secundaria(self, 
                         fecha_inicio: str, 
                         fecha_fin: str, 
                         secundaria_lst: List[int]) -> pd.DataFrame:
        """
        Extract secondary regulation prices from ESIOS.
        
        Args:
            fecha_inicio (str): Start date in YYYY-MM-DD format
            fecha_fin (str): End date in YYYY-MM-DD format
            secundaria_lst (List[int]): List of secondary regulation types [1: up, 2: down]
            
        Returns:
            pd.DataFrame: DataFrame with secondary regulation prices
            
        Note:
            After 2024-11-20, prices are split into up/down regulation
        """
        return self.secundaria.get_prices(fecha_inicio_carga=fecha_inicio,
                                        fecha_fin_carga=fecha_fin,
                                        secundaria_lst=secundaria_lst)

    def extract_terciaria(self, 
                         fecha_inicio: str, 
                         fecha_fin: str, 
                         terciaria_lst: List[int]) -> pd.DataFrame:
        """
        Extract tertiary regulation prices from ESIOS.
        
        Args:
            fecha_inicio (str): Start date in YYYY-MM-DD format
            fecha_fin (str): End date in 
            YYYY-MM-DD format
            terciaria_lst (List[int]): List of tertiary types 
                [1: up, 2: down, 3: direct up, 4: direct down, 5: programmed single]
            
        Returns:
            pd.DataFrame: DataFrame with tertiary regulation prices
            
        Note:
            After 2024-12-10, programmed tertiary uses single price (type 5)
        """
        return self.terciaria.get_prices(fecha_inicio_carga=fecha_inicio,
                                       fecha_fin_carga=fecha_fin,
                                       terciaria_lst=terciaria_lst)

    def extract_rr(self, fecha_inicio: str, fecha_fin: str) -> pd.DataFrame:
        """
        Extract Replacement Reserve (RR) prices from ESIOS.
        
        Args:
            fecha_inicio (str): Start date in YYYY-MM-DD format
            fecha_fin (str): End date in YYYY-MM-DD format
            
        Returns:
            pd.DataFrame: DataFrame with RR prices
        """
        return self.rr.get_rr_data(fecha_inicio_carga=fecha_inicio,
                                  fecha_fin_carga=fecha_fin)
