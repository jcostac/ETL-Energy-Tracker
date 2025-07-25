import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from transform.omie_transform import TransformadorOMIE
from transform.i90_transform import TransformadorI90
from extract.omie_extractor import OMIEExtractor
from extract.i90_extractor import I90VolumenesExtractor
from vinculacion.configs.vinculacion_config import VinculacionConfig

class VinculacionDataExtractor:
    """Extracts and transforms data needed for vinculacion process"""
    
    def __init__(self):
        """
        Initializes the VinculacionDataExtractor with configuration, extractor, and transformer instances for OMIE and I90 market data sources.
        """
        self.config = VinculacionConfig()
        
        # Extractors (for downloading)
        self.omie_extractor = OMIEExtractor()
        self.i90_extractor = I90VolumenesExtractor()
        
        # Transformers (for processing)
        self.omie_transformer = TransformadorOMIE()
        self.i90_transformer = TransformadorI90()
        
    def extract_data_for_matching(self, target_date: str) -> Dict[str, pd.DataFrame]:
        """
        Extracts raw OMIE and I90 market data for the specified date for use in the matching process.
        
        Parameters:
            target_date (str): The date for which to extract data, in 'YYYY-MM-DD' format.
        
        Returns:
            Dict[str, pd.DataFrame]: A dictionary containing the extracted dataframes for each market and dataset combination.
        
        Raises:
            Exception: If data extraction for either OMIE or I90 fails.
        """
        print(f"\n🔄 STARTING DATA EXTRACTION FOR VINCULACION")
        print(f"Target Date: {target_date}")
        print(f"Download Window: {self.config.DATA_DOWNLOAD_WINDOW} days")
        print("="*60)

        try:
            #-- Downloading OMIE raw data for both diario and intra
            print(f"\n📥 DOWNLOADING OMIE DATA")
            try:
                omie_download_result = self.omie_extractor.extract_data_for_all_markets(fecha_inicio=target_date, fecha_fin=target_date, mercados_lst=self.config.OMIE_MARKETS) 
                
                if omie_download_result['success']:
                    print(f"✅ OMIE raw data download successful")
                else:
                    raise Exception(f"⚠️  OMIE raw data download had issues: {omie_download_result.get('details', {})}")
                    
            except Exception as e:
                print(f"❌ OMIE raw data download failed: {e}")
                raise e 
            
            #-- Downloading I90 raw data (volumenes only) for both diario and intra
            print(f"\n📥 DOWNLOADING I90 DATA")
            try:
                i90_download_result = self.i90_extractor.extract_data_for_all_markets(fecha_inicio=target_date, fecha_fin=target_date,
                                                                                       mercados_lst=self.config.I90_MARKETS)
                
                if i90_download_result['success']:
                    print(f"✅ I90 raw data download successful")
                    return i90_download_result, omie_download_result
                else:
                    raise Exception(f"⚠️  I90 raw data download had issues: {i90_download_result.get('details', {})}")
                
            except Exception as e:
                print(f"❌ I90 raw data download failed: {e}")
                raise e
            
        except Exception as e:
            print(f"❌ Error during data extraction: {e}")
            raise e
        
    def transform_diario_data_for_matching(self, target_date: str) -> Dict[str, pd.DataFrame]:
        """
        Transform and retrieve daily ("diario") OMIE and I90 market data for the specified date.
        
        Attempts to transform diario data for both OMIE and I90 sources using their respective transformers. Returns a dictionary containing the resulting dataframes under keys `'i90_diario'` and `'omie_diario'` if transformation is successful and data is present. Raises an exception if I90 diario transformation fails.
        
        Parameters:
            target_date (str): The date for which to transform diario data, in 'YYYY-MM-DD' format.
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary with keys `'i90_diario'` and `'omie_diario'` containing the transformed diario dataframes.
        """
        print(f"\n🔍 TRANSFORMING DIARIO DATA FOR MATCHING")
        print("-"*50)

        # Step 3: Transform OMIE data (diario)
        print(f"\n🔄 TRANSFORMING OMIE DATA")
        print("-"*40)

        transformed_diario_data = {}

        # Step 4: Transform I90 data (diario)
        print(f"\n🔄 TRANSFORMING I90 DATA")
        print("-"*40)
        i90_result = self.i90_transformer.transform_data_for_all_markets(
            mercados_lst= ['diario'], #only transfrom diario
            fecha_inicio=target_date,
            dataset_type='volumenes_i90'
        )

        
        if i90_result['status']['success'] and 'diario' in i90_result['data']:
            i90_data = i90_result['data']['diario']
            if i90_data is not None and not i90_data.empty:
                transformed_diario_data['i90_diario'] = i90_data
                print(f"✅ I90 diario: {len(i90_data)} records extracted")
            else:
                print("⚠️  I90 diario: No data extracted")
        else:
            print("❌ I90 diario transformation failed")
            raise Exception("I90 diario transformation failed")

        omie_result = self.omie_transformer.transform_data_for_all_markets(
            mercados_lst= ['diario'], #only transfrom diario
            fecha_inicio=target_date,
        )
        
        if omie_result['status']['success'] and 'diario' in omie_result['data']:
            omie_data = omie_result['data']['diario']
            if omie_data is not None and not omie_data.empty:
                transformed_diario_data['omie_diario'] = omie_data
                print(f"✅ OMIE diario: {len(omie_data)} records extracted")
            else:
                print("⚠️  OMIE diario: No data extracted")
        else:
            print("❌ OMIE diario transformation failed")
            
        print(f"\n✅ DATA TRANSFORMATION COMPLETE")
        print(f"Total datasets extracted: {len(transformed_diario_data)}")
        print("="*60)
        
        return transformed_diario_data
            
    def transform_intra_data_for_matching(self, target_date: str) -> Dict[str, pd.DataFrame]:
        """
        Transforms and splits OMIE and I90 intraday ("intra") market data for a given date into session-specific dataframes.
        
        For the specified target date, transforms OMIE and I90 intra data, then separates each by session based on `id_mercado` values (mapping: 2→session 1, 3→session 2, 4→session 3). Returns a dictionary with keys like `'i90_intra_1'`, `'omie_intra_2'`, each containing the corresponding session dataframe. If transformation fails or no data is found for a session, that session is omitted from the result.
        
        Parameters:
            target_date (str): The date for which to transform intra data, in 'YYYY-MM-DD' format.
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of session-specific intra dataframes for OMIE and I90 sources.
        """
        print(f"\n🔍 TRANSFORMING INTRA DATA FOR AMBIGUOUS MATCHES")
        print(f"Target Date: {target_date}")
        print("-"*50)
        
        transformed_intra_data = {}
        
        try:
            # Transform I90 intra data
            print(f"🔄 Transforming I90 intra data for {target_date}")
            i90_result = self.i90_transformer.transform_data_for_all_markets(
                mercados_lst=['intra'],
                dataset_type='volumenes_i90',
                fecha_inicio=target_date,
            )
            
            if i90_result['status']['success'] and 'intra' in i90_result['data']:
                intra_transformed = i90_result['data']['intra']
                if intra_transformed is not None and not intra_transformed.empty:
                    # Split by intra session using id_mercado
                    # Session mapping: session 1 = id_mercado 2, session 2 = id_mercado 3, session 3 = id_mercado 4
                    session_mapping = {2: 1, 3: 2, 4: 3}  # id_mercado: session_number
                    
                    for id_mercado, session_num in session_mapping.items():
                        #filter the data by the id_mercado
                        session_data = intra_transformed[intra_transformed['id_mercado'] == id_mercado]

                        #if the data is not empty, add the data to the transformed_intra_data dictionary with the key "i90_intra_{session_num}"
                        if not session_data.empty:
                            transformed_intra_data[f'i90_intra_{session_num}'] = session_data
                            print(f"✅ I90 Intra Session {session_num} (id_mercado={id_mercado}): {len(session_data)} records")
                        else:
                            print(f"⚠️  I90 Intra Session {session_num} (id_mercado={id_mercado}): No data found")
                        
                else:
                    print("⚠️  No I90 intra data extracted")
            else:
                print("❌ I90 intra data transformation failed")
            
            # Transform OMIE intra data
            print(f"🔄 Transforming OMIE intra data for {target_date}")
            omie_result = self.omie_transformer.transform_data_for_all_markets(
                fecha_inicio=target_date,
                mercados_lst=['intra']
            )
            
            #if if omie result of transformation was a success and we have intra data in the result
            if omie_result['status']['success'] and 'intra' in omie_result['data']:
                #get the intra data from the result
                omie_intra_data = omie_result['data']['intra']

                #if the data is not empty or None, split the data by sessions using id_mercado
                if omie_intra_data is not None and not omie_intra_data.empty:
                    # Split OMIE intra data by sessions using id_mercado
                    session_mapping = {2: 1, 3: 2, 4: 3}  # id_mercado: session_number
                    
                    for id_mercado, session_num in session_mapping.items():
                        #filter the data by the id_mercado
                        session_data = omie_intra_data[omie_intra_data['id_mercado'] == id_mercado]

                        if not session_data.empty:
                            #add the session data to the transformed_intra_data dictionary wiht the key "omie_intra_{session_num}"
                            transformed_intra_data[f'omie_intra_{session_num}'] = session_data
                            print(f"✅ OMIE Intra Session {session_num} (id_mercado={id_mercado}): {len(session_data)} records")
                        else:
                            print(f"⚠️  OMIE Intra Session {session_num} (id_mercado={id_mercado}): No data found")
                        
                else:
                    print("⚠️  No OMIE intra data extracted")
            else:
                print("❌ OMIE intra data transformation failed")
                
            print(f"\n✅ INTRA DATA TRANSFORMATION COMPLETE")
            print(f"Total intra datasets (OMIE + i90): {len(transformed_intra_data)}")
            print("-"*50)
            
            return transformed_intra_data
            
        except Exception as e:
            print(f"❌ Error transforming intra data: {e}")
            return {}
            
    def transform_and_combine_data_for_linking(self, target_date: str) -> Dict[str, pd.DataFrame]:
        """
        Transforms and combines OMIE and I90 diario and intra data for the specified date into consolidated DataFrames.
        
        Parameters:
            target_date (str): The date for which to process and combine data.
        
        Returns:
            Dict[str, pd.DataFrame]: A dictionary with keys 'omie_combined' and 'i90_combined', each containing a DataFrame with all relevant diario and intra session data for OMIE and I90, respectively.
        """
        print(f"\n🔄 TRANSFORMING & COMBINING ALL DATA FOR {target_date}")
        print("="*60)
        
        # 1. Transform diario data
        diario_data = self.transform_diario_data_for_matching(target_date)
        
        # 2. Transform intra data
        intra_data = self.transform_intra_data_for_matching(target_date)
        
        print("🔄 COMBINING OMIE DATA")
        # 3. Combine OMIE data
        omie_dfs = []
        if 'omie_diario' in diario_data and not diario_data['omie_diario'].empty:
            df = diario_data['omie_diario'].copy()
            df['id_mercado'] = 1 # Diario is market 1
            omie_dfs.append(df)
        
        omie_intra_keys = [k for k in intra_data.keys() if k.startswith('omie_intra_')]
        for key in sorted(omie_intra_keys): # Sort to keep session order (1, 2, 3)
            if intra_data.get(key) is not None and not intra_data[key].empty:
                #append intra df
                omie_dfs.append(intra_data[key])
            
        omie_combined = pd.concat(omie_dfs, ignore_index=True) if omie_dfs else pd.DataFrame()

        if not omie_combined.empty:
            print(f"✅ Combined OMIE data: {len(omie_combined)} records, markets: {sorted(omie_combined.id_mercado.unique())}")

        print("🔄 COMBINING I90 DATA")
        # 4. Combine I90 data
        i90_dfs = []
        if 'i90_diario' in diario_data and not diario_data['i90_diario'].empty:
            df = diario_data['i90_diario'].copy()
            df['id_mercado'] = 1 # Diario is market 1
            i90_dfs.append(df)
            
        i90_intra_keys = [k for k in intra_data.keys() if k.startswith('i90_intra_')]
        for key in sorted(i90_intra_keys): # Sort to keep session order (1, 2, 3)
            if intra_data.get(key) is not None and not intra_data[key].empty:
                i90_dfs.append(intra_data[key])
            
        i90_combined = pd.concat(i90_dfs, ignore_index=True) if i90_dfs else pd.DataFrame()
        
        if not i90_combined.empty:
            print(f"✅ Combined I90 data: {len(i90_combined)} records, markets: {sorted(i90_combined.id_mercado.unique())}")

        return {'omie_combined': omie_combined, 'i90_combined': i90_combined}
        

def example_usage():
    """
    Demonstrates how to use VinculacionDataExtractor to transform and extract market data for a specific date.
    """
    data_extractor = VinculacionDataExtractor()
    data_extractor.transform_intra_data_for_matching("2025-03-07")
    data_extractor.extract_data_for_matching("2025-03-07")


if __name__ == "__main__":
    example_usage()