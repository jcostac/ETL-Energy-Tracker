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
        self.config = VinculacionConfig()
        
        # Extractors (for downloading)
        self.omie_extractor = OMIEExtractor()
        self.i90_extractor = I90VolumenesExtractor()
        
        # Transformers (for processing)
        self.omie_transformer = TransformadorOMIE()
        self.i90_transformer = TransformadorI90()
        
    def extract_data_for_matching(self, target_date: str) -> Dict[str, pd.DataFrame]:
        """
        Downloads and transforms data for the matching process
        
        Args:
            target_date: Target date for linking (YYYY-MM-DD) - this should be the actual date to download data for
            
        Returns:
            Dict with extracted dataframes for each market/dataset combination
        """
        print(f"\n🔄 STARTING DATA EXTRACTION FOR VINCULACION")
        print(f"Target Date: {target_date}")
        print(f"Download Window: {self.config.DATA_DOWNLOAD_WINDOW} days")
        print("="*60)

        try:
            
            # Step 1: Download OMIE raw data
            print(f"\n📥 DOWNLOADING OMIE RAW DATA")
            print(f"Date range: {target_date}")
            print("-"*40)
            
            try:
                omie_download_result = self.omie_extractor.extract_data_for_all_markets(fecha_inicio_carga=target_date, fecha_fin_carga=target_date, mercados_lst=self.config.OMIE_MARKETS) 
                
                if omie_download_result['success']:
                    print(f"✅ OMIE raw data download successful")
                else:
                    print(f"⚠️  OMIE raw data download had issues: {omie_download_result.get('details', {})}")
                    
            except Exception as e:
                print(f"❌ OMIE raw data download failed: {e}")
                raise e 
            
            # Step 2: Download I90 raw data (volumenes only)
            print(f"\n📥 DOWNLOADING I90 RAW DATA")
            print(f"Date range: {target_date}")
            print("-"*40)
            
            try:
                i90_download_result = self.i90_extractor.extract_data_for_all_markets(fecha_inicio_carga=target_date, fecha_fin_carga=target_date,
                                                                                       mercados_lst=self.config.I90_MARKETS)
                
                if i90_download_result['success']:
                    print(f"✅ I90 raw data download successful")
                else:
                    print(f"⚠️  I90 raw data download had issues: {i90_download_result.get('details', {})}")

                return i90_download_result, omie_download_result
            
            except Exception as e:
                print(f"❌ I90 raw data download failed: {e}")
                raise e
            
        except Exception as e:
            print(f"❌ Error during data extraction: {e}")
            raise e
        
    def transform_diario_data_for_initial_matching(self, target_date: str) -> Dict[str, pd.DataFrame]:
        """
        Transforms diario data for initial matching
        """
        print(f"\n🔍 TRANSFORMING DIARIO DATA FOR INITIAL MATCHING")
        print("-"*50)

        # Step 3: Transform OMIE data (diario)
        print(f"\n🔄 TRANSFORMING OMIE DATA")
        print("-"*40)

        transformed_diario_data = {}

        omie_result = self.omie_transformer.transform_data_for_all_markets(
            mercados_lst= ['diario'], #only transfrom diario
            transform_type='single',
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
            
        # Step 4: Transform I90 data (diario)
        print(f"\n🔄 TRANSFORMING I90 DATA")
        print("-"*40)
        i90_result = self.i90_transformer.transform_data_for_all_markets(
            mercados_lst= ['diario'], #only transfrom diario
            transform_type='single',
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
            
        print(f"\n✅ DATA TRANSFORMATION COMPLETE")
        print(f"Total datasets extracted: {len(transformed_diario_data)}")
        print("="*60)
        
        return transformed_diario_data
            
    def transform_intra_data_for_ambiguous_matches(self, target_date: str) -> Dict[str, pd.DataFrame]:
        """
        Transforms I90 intra data and OMIE intra data for resolving ambiguous matches
        Note: Raw data should already be downloaded by extract_data_for_matching method
        
        Args:
            target_date: Target date (YYYY-MM-DD)
            
        Returns:
            Dict with intra dataframes split by sessions
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
                transform_type='single',
                fecha_inicio=target_date,
            )
            
            if i90_result['status']['success'] and 'intra' in i90_result['data']:
                intra_transformed = i90_result['data']['intra']
                if intra_transformed is not None and not intra_transformed.empty:
                    # Split by intra session using id_mercado
                    # Session mapping: session 1 = id_mercado 2, session 2 = id_mercado 3, session 3 = id_mercado 4
                    session_mapping = {"2": 1, "3": 2, "4": 3}  # id_mercado: session_number
                    
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
                mercados_lst=['intra'],
                transform_type='single'
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
            print(f"Total intra datasets by session: {len(transformed_intra_data)}")
            print("-"*50)
            
            return transformed_intra_data
            
        except Exception as e:
            print(f"❌ Error transforming intra data: {e}")
            return {}
            