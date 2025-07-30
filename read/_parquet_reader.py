import pandas as pd
from typing import Optional, Union
import sys
import os
from pathlib import Path
import pretty_errors
from datetime import datetime
from deprecated import deprecated
from dotenv import load_dotenv

# Get the absolute path to the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Use absolute imports
from configs.esios_config import ESIOSConfig
# Get the processed folder path



class ParquetReader:
    """
    Clase para leer datos de la base de datos o de un fichero parquet.
    """
    def __init__(self):
        """
        Initializes the ParquetReader by loading environment variables, setting up configuration, and determining valid markets from the processed data directory.
        """
        load_dotenv()
        self.duckdb_engine = "implementar"
        self.esios_config = ESIOSConfig()
        self.processed_path = Path(os.getenv('DATA_LAKE_PATH')) / "processed"
        self.valid_mercados = self.get_valid_mercados()
        

    def get_valid_mercados(self) -> list[str]:
        """
        Returns the list of valid mercados extracted from the processed folder mercado names
        
        Returns:
            list[str]: A list of valid market names that can be used in the system
        """
        if not self.processed_path.exists() or not self.processed_path.is_dir():
            # If directory doesn't exist, return the default list
            return self.valid_mercados
        
        # Get all subdirectories in the processed folder that follow the pattern "mercado={name}"
        valid_mercados = []
        for folder in self. processed_path.iterdir():
            if folder.is_dir() and folder.name.startswith('mercado='):
                # Extract the market name from the "mercado={name}" format
                market_name = folder.name.split('=')[1].lower()
                valid_mercados.append(market_name)
    
        
        return valid_mercados
    
    def validate_mercado(self, mercados: list[str]) -> bool:
        """
        Validates if all markets names are valid market names
        """
        #check if mercados is a list 
        if not isinstance(mercados, list):
            raise ValueError(f"Mercado {mercados} need to be a list")
        
        for mercado in mercados:
            try:
                #check if mercado is a string
                if not isinstance(mercado, str):
                    raise ValueError(f"Mercado {mercado} need to be a string")
            
                
                #check if all items in mercados are in valid_mercados
                if not all(mercado in self.valid_mercados for mercado in mercados):
                    raise ValueError(f"Mercado {mercados} are not valid")
                else:
                    return True
            except Exception as e:
                print(f"Error: {str(e)}")
                print(f"Valid market names are: {self.valid_mercados}")
                raise

    def validate_mercado_ids_for_market(self, mercado: list[str], mercado_id_lst: Optional[list[int]] = None) -> dict[str, list[int]]:
        """
        Validates that the provided mercado_ids are valid for the given market types and
        returns all valid IDs for each market if no specific IDs are provided.
        
        Args:
            mercado (list[str]): List of market types (e.g., ["intra", "secundaria"])
            mercado_id_lst (Optional[list[int]]): List of specific market IDs to validate, or None to get all IDs
            
        Returns:
            dict[str, list[int]]: Dictionary mapping each market to its list of valid IDs.
                                  Only includes markets that have at least one valid ID.
            
        Raises:
            ValueError: If mercado is invalid or if no valid data is found for any market
        """
        # Check if mercado is a list
        if not isinstance(mercado, list):
            raise ValueError(f"mercado must be a list, got {type(mercado)}")
        
        validated_ids = {}
        
        for market in mercado:
            # Navigate to the appropriate market folder
            market_folder_name = f"mercado={market.lower()}"
            market_folder = self.processed_path / market_folder_name
            
            # Check if the market folder exists and is a directory
            if not market_folder.exists() or not market_folder.is_dir():
                print(f"Warning: No processed data folder found for market: {market}. Skipping.")
                continue
            
            # Get all valid IDs from the folder structure
            valid_ids = []
            for id_folder in market_folder.iterdir():
                if id_folder.is_dir() and id_folder.name.startswith("id_mercado="):
                    try:
                        id_value = int(id_folder.name.split("=")[1])
                        valid_ids.append(id_value)
                    except (IndexError, ValueError):
                        continue
            
            valid_ids.sort()  # Sort for consistent output
            
            if not valid_ids:
                print(f"Warning: No valid market IDs found in processed folder for market: {market}. Skipping.")
                continue
            
            # If no specific IDs provided, use all valid IDs for this market
            if mercado_id_lst is None:
                validated_ids[market] = valid_ids
                print(f"Validated IDs for mercado {market}: all available ({len(valid_ids)} IDs)")
                continue
            
            # Find intersection of provided IDs and valid IDs for this market
            market_valid_ids = [id_ for id_ in mercado_id_lst if id_ in valid_ids]
            
            if market_valid_ids:
                validated_ids[market] = market_valid_ids
                print(f"Validated IDs for mercado {market}: {market_valid_ids}")
            else:
                print(f"Warning: None of the provided IDs {mercado_id_lst} are valid for market {market}. Available IDs: {valid_ids}. Skipping this market.")
        
        # Only raise an error if NO markets have valid data
        if not validated_ids:
            if mercado_id_lst is None:
                raise ValueError(f"No valid data found for any of the specified markets: {mercado}")
            else:
                raise ValueError(f"None of the provided IDs {mercado_id_lst} are valid for any of the specified markets: {mercado}")
        
        return validated_ids

    def read_parquet_data(self, fecha_inicio_lectura: str, fecha_fin_lectura: str, mercado_lst: Union[str, list[str]], dataset_type: str, mercado_id_lst: Optional[Union[int, list[int]]] = None) -> pd.DataFrame:
        """
        Lee un fichero parquet y lo devuelve como un DataFrame.
        
        Args:
            fecha_inicio_lectura (str): Fecha inicial en formato YYYY-MM-DD
            fecha_fin_lectura (str): Fecha final en formato YYYY-MM-DD
            mercado_lst (Union[str, list[str]]): Tipo(s) de mercado (e.g., "intra", "secundaria", etc.)
                                                 Puede ser un string único o una lista de strings
            dataset_type (str): The type of data to read (e.g., "precios", "volumenes")
            mercado_id_lst (Optional[Union[int, list[int]]]): ID(s) específicos de mercado. 
                                                              Puede ser un int único, una lista de ints, o None.
                                                              Si no se proporciona, se utilizarán todos los IDs válidos para ese mercado.
        """
        # Convert single values to lists for internal processing
        if isinstance(mercado_lst, str):
            mercado_lst = [mercado_lst]
        
        if mercado_id_lst is not None:
            if isinstance(mercado_id_lst, int):
                mercado_id_lst = [mercado_id_lst]
        
        # Check dates
        if fecha_inicio_lectura and fecha_fin_lectura:
            if fecha_inicio_lectura > fecha_fin_lectura:
                raise ValueError("La fecha de inicio de lectura no puede ser mayor que la fecha de fin de lectura")

        # Validate mercado and get appropriate mercado_ids
        self.validate_mercado(mercado_lst)
        validated_mercado_ids = self.validate_mercado_ids_for_market(mercado_lst, mercado_id_lst)
        print(validated_mercado_ids)

        fecha_inicio_lectura_dt = pd.to_datetime(fecha_inicio_lectura)
        fecha_fin_lectura_dt = pd.to_datetime(fecha_fin_lectura)

        all_days_in_range = pd.date_range(fecha_inicio_lectura_dt, fecha_fin_lectura_dt, freq='D')

        if all_days_in_range.empty:
            raise ValueError("No days in range")

        year_months = sorted(list(set([(d.year, d.month) for d in all_days_in_range])))

        all_dfs = []
        for year, month in year_months:
            for mercado in mercado_lst:
                mercado_ids = validated_mercado_ids.get(mercado, [])
                for id_mercado in mercado_ids:
                    path = self.processed_path / f"mercado={mercado}" / f"id_mercado={id_mercado}" / f"year={year:04d}" / f"month={month:02d}" / f"{dataset_type}.parquet"
                    if path.exists():
                        print(f"Reading parquet file for {mercado} (id_mercado={id_mercado})")
                        print("--------------------------------")
                        df = pd.read_parquet(path)
                        print(f"Read {df.shape[0]} rows")
                        print(f"Read {df.shape[1]} columns")
                        print(f"Columns: {df.columns}")
                        print(f"Dtypes: {df.dtypes}")
                        print(f"First 10 rows: {df.head(10)}")
                        print(f"Last 10 rows: {df.tail(10)}")
                        print(f"--------------------------------")
                        all_dfs.append(df)

        df_concat = pd.concat(all_dfs, ignore_index=True)
        print(df_concat.head())
        print(df_concat.tail())
        print(df_concat.shape)
        return df_concat
        


if __name__ == "__main__":
    read_ops = ParquetReader()
    read_ops.read_parquet_data(fecha_inicio_lectura="2025-03-31", fecha_fin_lectura="2025-04-01", mercado_lst=["diario"], mercado_id_lst=None, dataset_type="volumenes_i90")
