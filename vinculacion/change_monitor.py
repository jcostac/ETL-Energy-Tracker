import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from utilidades.db_utils import DatabaseUtils
from vinculacion.configs.vinculacion_config import VinculacionConfig
from vinculacion.linking_algorithm import UOFUPLinkingAlgorithm

class UPChangeMonitor:
    """Monitors up_change_log table for new enabled UPs"""
    
    def __init__(self):
        self.config = VinculacionConfig()
        self.linking_algorithm = UOFUPLinkingAlgorithm()
        
    def check_for_new_enabled_ups(self, engine, check_date: Optional[str] = None) -> List[str]:
        """
        Check for new UPs that were enabled on a specific date
        
        Args:
            engine: Database engine
            check_date: Date to check (YYYY-MM-DD), defaults to today
            
        Returns:
            List[str]: List of newly enabled UPs
        """
        if check_date is None:
            check_date = datetime.now().strftime('%Y-%m-%d')
            
        print(f"\n🔍 CHECKING FOR NEW ENABLED UPs ON {check_date}")
        print("-"*50)
        
        try:
            query = f"""
            SELECT DISTINCT up
            FROM {self.config.UP_CHANGE_LOG_TABLE}
            WHERE field_changed = 'habilitada'
            AND CAST(date_updated AS DATE) = CAST(? AS DATE)
            AND new_value = '1'
            """
            
            with engine.connect() as conn:
                result = conn.execute(query, (check_date,))
                new_ups = [row[0] for row in result.fetchall()]
                
            if new_ups:
                print(f"✅ Found {len(new_ups)} newly enabled UPs:")
                for up in new_ups:
                    print(f"   - {up}")
            else:
                print("ℹ️  No newly enabled UPs found")
                
            return new_ups
            
        except Exception as e:
            print(f"❌ Error checking for new enabled UPs: {e}")
            return []
            
    def trigger_linking_for_new_ups(self, new_ups: List[str], 
                                  target_date: str, engine) -> pd.DataFrame:
        """
        Trigger linking process for newly enabled UPs
        
        Args:
            new_ups: List of newly enabled UPs
            target_date: Date to perform linking for
            engine: Database engine
            
        Returns:
            pd.DataFrame: New links created
        """
        if not new_ups:
            print("ℹ️  No new UPs to link")
            return pd.DataFrame(columns=['up', 'uof'])
            
        print(f"\n🔄 TRIGGERING LINKING FOR {len(new_ups)} NEW UPs")
        print("-"*50)
        
        try:
            # Perform full linking process (this will include new UPs)
            all_links = self.linking_algorithm.link_uofs_to_ups(target_date, engine)
            
            # Filter to only new UPs
            new_links = all_links[all_links['up'].isin(new_ups)]
            
            print(f"✅ Created {len(new_links)} new links for enabled UPs")
            return new_links
            
        except Exception as e:
            print(f"❌ Error triggering linking for new UPs: {e}")
            return pd.DataFrame(columns=['up', 'uof'])
            
    def run_daily_check(self, engine, check_date: Optional[str] = None) -> Dict:
        """
        Run daily check for new enabled UPs and trigger linking if needed
        
        Args:
            engine: Database engine  
            check_date: Date to check (defaults to today)
            
        Returns:
            Dict: Results of the daily check
        """
        if check_date is None:
            check_date = datetime.now().strftime('%Y-%m-%d')
            
        print(f"\n🗓️  RUNNING DAILY UP CHANGE CHECK FOR {check_date}")
        print("="*60)
        
        results = {
            'check_date': check_date,
            'new_ups_found': [],
            'new_links_created': pd.DataFrame(columns=['up', 'uof']),
            'success': False,
            'message': ''
        }
        
        try:
            # Check for new enabled UPs
            new_ups = self.check_for_new_enabled_ups(engine, check_date)
            results['new_ups_found'] = new_ups
            
            if new_ups:
                # Trigger linking for new UPs
                new_links = self.trigger_linking_for_new_ups(new_ups, check_date, engine)
                results['new_links_created'] = new_links
                results['success'] = True
                results['message'] = f"Successfully processed {len(new_ups)} new UPs, created {len(new_links)} links"
            else:
                results['success'] = True
                results['message'] = "No new enabled UPs found"
                
            print(f"\n✅ DAILY CHECK COMPLETE: {results['message']}")
            
        except Exception as e:
            results['message'] = f"Error during daily check: {e}"
            print(f"\n❌ DAILY CHECK FAILED: {results['message']}")
            
        print("="*60)
        return results 