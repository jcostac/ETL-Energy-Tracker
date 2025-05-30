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
from vinculacion.temp_data_manager import TemporaryDataManager

class VinculacionDataExtractor:
    """Extracts and transforms data needed for vinculacion process"""
    
    def __init__(self):
        self.config = VinculacionConfig()
        self.temp_manager = TemporaryDataManager()
        
        # Extractors (for downloading)
        self.omie_extractor = OMIEExtractor()
        self.i90_extractor = I90VolumenesExtractor()
        
        # Transformers (for processing)
        self.omie_transformer = TransformadorOMIE()
        self.i90_transformer = TransformadorI90()
        
    def extract_data_for_matching(self, target_date: str) -> Dict[str, pd.DataFrame]:
        """
        Downloads and transforms data for the matching process
        Always uses the full 93-day window ending on target_date
        
        Args:
            target_date: Target date for linking (YYYY-MM-DD)
            
        Returns:
            Dict with extracted dataframes for each market/dataset combination
        """
        print(f"\n🔄 STARTING DATA EXTRACTION FOR VINCULACION")
        print(f"Target Date: {target_date}")
        print(f"Download Window: {self.config.DATA_DOWNLOAD_WINDOW} days")
        print("="*60)

        # Calculate date range (93 days back from target date)
        target_dt = pd.to_datetime(target_date)
        target_date = (target_dt - timedelta(days=self.config.DATA_DOWNLOAD_WINDOW)).strftime('%Y-%m-%d')


        
        try:
            # Clean and setup temp directory
            self.temp_manager.cleanup_temp_directory()
            self.temp_manager.setup_temp_directory()
            
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
                    
            except Exception as e:
                print(f"❌ I90 raw data download failed: {e}")
                raise e
            
        except Exception as e:
            print(f"❌ Error during data extraction: {e}")
            return {}
        
    def transform_diario_data_for_initial_matching(self, i90_download_result: Dict[str, pd.DataFrame], omie_download_result: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Transforms diario data for initial matching
        """
        print(f"\n🔍 TRANSFORMING DIARIO DATA FOR INITIAL MATCHING")
        print("-"*50)

        # Step 3: Transform OMIE data (diario)
        print(f"\n🔄 TRANSFORMING OMIE DATA")
        print("-"*40)

        transformed_data = {}

        omie_result = self.omie_transformer.transform_data_for_all_markets(
            mercados_lst= ['diario'], #only transfrom diario
            mode='latest'
        )
        
        if omie_result['status']['success'] and 'diario' in omie_result['data']:
            omie_data = omie_result['data']['diario']
            if omie_data is not None and not omie_data.empty:
                transformed_data['omie_diario'] = omie_data
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
            mode='latest'
        )
        
        if i90_result['status']['success'] and 'diario' in i90_result['data']:
            i90_data = i90_result['data']['diario']

            if i90_data is not None and not i90_data.empty:
                transformed_data['i90_diario'] = i90_data
                print(f"✅ I90 diario: {len(i90_data)} records extracted")
            else:
                print("⚠️  I90 diario: No data extracted")
        else:
            print("❌ I90 diario transformation failed")
            
        print(f"\n✅ DATA TRANSFORMATION COMPLETE")
        print(f"Total datasets : {len(transformed_data)}")
        print("="*60)
        
        return transformed_data
            
    def transform_intra_data_for_ambiguous_matches(self, target_date: str) -> Dict[str, pd.DataFrame]:
        """
        Transforms I90 intra data (sessions 1, 2, 3) and OMIE intra data for resolving ambiguous matches
        Note: Raw data should already be downloaded by extract_data_for_linking method
        
        Args:
            target_date: Target date (YYYY-MM-DD)
            
        Returns:
            Dict with intra dataframes
        """
        print(f"\n🔍 TRANSFORMING INTRA DATA FOR AMBIGUOUS MATCHES")
        print(f"Target Date: {target_date}")
        print("-"*50)
        
        intra_data = {}
        
        try:
            # Transform I90 intra data (raw data should already be available)
            print(f"🔄 Transforming I90 intra data for {target_date}")
            i90_result = self.i90_transformer.transform_data_for_all_markets(
                start_date=target_date,
                end_date=target_date,
                mercados_lst=['intra'],
                dataset_type='volumenes_i90',
                transform_type='single'
            )
            
            if i90_result['status']['success'] and 'intra' in i90_result['data']:
                intra_raw = i90_result['data']['intra']
                if intra_raw is not None and not intra_raw.empty:
                    # Split by intra session (assuming there's a session identifier)
                    # This might need adjustment based on your actual I90 intra data structure
                    for session in [1, 2, 3]:
                        session_data = intra_raw[intra_raw.get('session', 1) == session]
                        if not session_data.empty:
                            intra_data[f'i90_intra_{session}'] = session_data
                            print(f"✅ I90 Intra {session}: {len(session_data)} records")
                        else:
                            print(f"⚠️  I90 Intra {session}: No data found")
                else:
                    print("⚠️  No I90 intra data extracted")
            else:
                print("❌ I90 intra data transformation failed")
            
            # Transform OMIE intra data (raw data should already be available)
            print(f"🔄 Transforming OMIE intra data for {target_date}")
            omie_result = self.omie_transformer.transform_data_for_all_markets(
                fecha_inicio=target_date,
                mercados_lst=['intra'],
                mode='single'
            )
            
            if omie_result['status']['success'] and 'intra' in omie_result['data']:
                omie_intra_data = omie_result['data']['intra']
                if omie_intra_data is not None and not omie_intra_data.empty:
                    intra_data['omie_intra'] = omie_intra_data
                    print(f"✅ OMIE Intra: {len(omie_intra_data)} records")
                else:
                    print("⚠️  No OMIE intra data extracted")
            else:
                print("❌ OMIE intra data transformation failed")
            
            print(f"\n✅ INTRA DATA TRANSFORMATION COMPLETE")
            print(f"Total intra datasets: {len(intra_data)}")
            print("-"*50)
            
            return intra_data
            
        except Exception as e:
            print(f"❌ Error transforming intra data: {e}")
            return {}
            
    def cleanup_extraction(self) -> None:
        """Cleanup temporary data after extraction"""
        self.temp_manager.cleanup_temp_directory()
        print("🗑️  Extraction cleanup complete") 