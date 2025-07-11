import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import sys
from pathlib import Path
from sqlalchemy import text
import asyncio

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from utilidades.db_utils import DatabaseUtils
from vinculacion._linking_algorithm import UOFUPLinkingAlgorithm
from vinculacion._vinculacion_monitoring import UPChangeMonitor
from vinculacion.configs.vinculacion_config import VinculacionConfig

class VinculacionOrchestrator:
    """Main orchestrator for the vinculacion module"""
    
    def __init__(self):
        
        """
        Initialize the VinculacionOrchestrator with configuration, linking algorithm, change monitoring, and database utility components.
        """
        self.config = VinculacionConfig()
        self.linking_algorithm = UOFUPLinkingAlgorithm()
        self.change_monitor = UPChangeMonitor()
        self.db_utils = DatabaseUtils()

        
    def _create_engine(self, database_name: str):
        """
        Create and return a database engine for the specified database.
        
        Parameters:
            database_name (str): Name of the target database.
        
        Returns:
            Engine object for database operations.
        """
        return self.db_utils.create_engine(database_name)
        
    async def perform_full_linking(self) -> pd.DataFrame:
        """
        Perform a full linking process between all active UPs and UOFs as of a target date 93 days prior to today.
        
        Returns:
            pd.DataFrame: DataFrame containing the linked UP-UOF pairs with columns ['UP', 'UOF', 'date_updated']. Returns an empty DataFrame if linking fails or an error occurs.
        """
        target_date = self.config.get_linking_target_date()
        
        print(f"\n🚀 STARTING FULL UOF-UP LINKING")
        print(f"Target Date: {target_date} (93 days back)")
        print("="*80)
        
        try:
            # Perform linking
            results = await self.linking_algorithm.link_uofs_to_ups(target_date, save_to_db = True)
            
            if results['success']:
                links_df = results['links_df']
                print(f"\n📊 LINKING RESULTS SUMMARY")
                print("-"*40)
                print(f"Total links created: {len(links_df)}")
                print(f"Unique UPs linked: {links_df['up'].nunique()}")
                print(f"Unique UOFs linked: {links_df['uof'].nunique()}")
                
                db_links = pd.DataFrame({
                    'UP': links_df['up'].str.upper(),
                    'UOF': links_df['uof'].str.upper(),
                    'date_updated': links_df['date_updated']
                })
                
                print(f"\n📋 SAMPLE LINKS (First 10):")
                print(db_links.head(10).to_string(index=False))

                return db_links
            
            else:
                print(f"\n⚠️  {results['message']}")
                return pd.DataFrame(columns=['UP', 'UOF', 'date_updated'])
                
        except Exception as e:
            print(f"\n❌ ERROR IN FULL LINKING: {e}")
            return pd.DataFrame(columns=['UP', 'UOF', 'date_updated'])
            
    async def perform_new_ups_linking(self) -> Dict:
        """
        Performs incremental linking for newly enabled UPs as of the configured target date.
        
        Returns:
            dict: A dictionary containing the results of the incremental linking process, including success status, any new UPs found, and new links created.
        """
        return #TODO: Implement incremental linking
        check_date = self.config.get_linking_target_date() #93 days back from today|    
        
        print(f"\n🔄 STARTING INCREMENTAL UOF-UP LINKING")
        print(f"Checking for UPs enabled on: {check_date} (93 days back)")
        print("="*80)
        
        try:
            engine = self._create_engine(database_name = self.config.DATABASE_NAME)
            
            # Run daily check for UPs enabled 93 days ago
            results = await self.change_monitor.run_incremental_check(engine, check_date)
            
            if results['success']:
                print(f"\n📊 INCREMENTAL LINKING RESULTS")
                print("-"*45)
                print(f"UPs enabled 93 days ago: {len(results['new_ups_found'])}")
                print(f"New links created: {len(results['new_links_created'])}")
                
                if not results['new_links_created'].empty:
                    print(f"\n📋 NEW LINKS CREATED:")
                    print(results['new_links_created'][['UP', 'UOF']].to_string(index=False))
                    
            return results
            
        except Exception as e:
            print(f"\n❌ ERROR IN INCREMENTAL LINKING: {e}")
            return {
                'success': False,
                'message': f"Error: {e}",
                'new_ups_found': [],
                'new_links_created': pd.DataFrame(columns=['UP', 'UOF', 'date_updated'])
            }
        
    async def perform_vinculacion_monitoring(self) -> Dict:
        """
        Monitor existing UP-UOF links for changes and new associations.
        
        Returns:
            dict: A dictionary containing the monitoring results, including detected changes and new links. On error, returns a dictionary with success status and error message.
        """
        print(f"\n🔄 STARTING EXISTING LINKS MONITORING")
        print("="*80)
        
        try:
            results = await self.change_monitor.monitor_existing_links()
            
            if results['success']:
                print(f"\n📊 EXISTING LINKS MONITORING RESULTS")
                print("-"*45)
                print(f"Changes found: {len(results['changes_found'])}")
                print(f"New links found: {len(results['new_links_found'])}")

                if not results['changes_found'].empty:
                    print(f"\n📋 CHANGES DETECTED:")
                    print(results['changes_found'].to_string(index=False))

                if not results['new_links_found'].empty:
                    print(f"\n📋 NEW LINKS FOUND:")
                    print(results['new_links_found'].to_string(index=False))

            return results
            
        except Exception as e:
            print(f"\n❌ ERROR IN EXISTING LINKS MONITORING: {e}")
            return {'success': False, 'message': str(e)}

