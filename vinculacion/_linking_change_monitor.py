import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import sys
from pathlib import Path
from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from utilidades.db_utils import DatabaseUtils
from vinculacion.configs.vinculacion_config import VinculacionConfig
from vinculacion._linking_algorithm import UOFUPLinkingAlgorithm

class UPChangeMonitor:
    """Monitors up_change_log table for UPs enabled at least 93 days ago so we can link them to UOFs"""
    
    def __init__(self):
        self.config = VinculacionConfig()
        self.linking_algorithm = UOFUPLinkingAlgorithm()
        
    def _check_for_ups_enabled_93_days_ago(self, engine, check_date: str) -> List[str]:
        """
        Check for UPs that were enabled exactly 93 days ago
        
        Args:
            engine: Database engine
            check_date: Date to check (93 days back from today)
            
        Returns:
            List[str]: List of UPs enabled on that date
        """
        print(f"\n🔍 CHECKING FOR UPs ENABLED ON {check_date}")
        print("-"*50)
        
        try:
            query = f"""
            SELECT DISTINCT UP
            FROM {self.config.UP_CHANGE_LOG_TABLE}
            WHERE field_changed = 'habilitada'
            AND CAST(date_updated AS DATE) = CAST(? AS DATE)
            AND new_value = '1'
            """
            
            with engine.connect() as conn:
                result = conn.execute(text(query), (check_date,))
                enabled_ups = [row[0] for row in result.fetchall()]
                
            if enabled_ups:
                print(f"✅ Found {len(enabled_ups)} UPs enabled on {check_date}:")
                for up in enabled_ups:
                    print(f"   - {up}")
            else:
                print(f"ℹ️  No UPs were enabled on {check_date}")
                
            return enabled_ups
            
        except Exception as e:
            print(f"❌ Error checking for UPs enabled on {check_date}: {e}")
            return []
    
    def _check_if_ups_already_linked(self, engine, ups_to_check: List[str], target_date: str) -> List[str]:
        """
        Check which UPs from the list are not already linked in the database
        
        Args:
            engine: Database engine
            ups_to_check: List of UPs to check
            target_date: Target linking date
            
        Returns:
            List[str]: UPs that are not yet linked
        """
        if not ups_to_check:
            return []
            
        print(f"\n🔍 CHECKING EXISTING LINKS FOR {len(ups_to_check)} UPs")
        print("-"*50)
        
        try:
            # Create placeholders for the IN clause
            placeholders = ', '.join(['?' for _ in ups_to_check])
            
            query = f"""
            SELECT DISTINCT UP
            FROM {self.config.UP_UOF_VINCULACION_TABLE}
            WHERE UP IN ({placeholders})
            AND date_updated = ?
            """
            
            params = ups_to_check + [target_date]
            
            with engine.connect() as conn:
                result = conn.execute(text(query), params)
                already_linked_ups = [row[0] for row in result.fetchall()]
                
            # Find UPs that are not yet linked
            unlinked_ups = [up for up in ups_to_check if up not in already_linked_ups]
            
            print(f"📊 Link status check:")
            print(f"   - Already linked: {len(already_linked_ups)}")
            print(f"   - Need linking: {len(unlinked_ups)}")
            
            if unlinked_ups:
                print(f"📋 UPs needing linking:")
                for up in unlinked_ups:
                    print(f"   - {up}")
                    
            return unlinked_ups
            
        except Exception as e:
            print(f"❌ Error checking existing links: {e}")
            return ups_to_check  # Return all if check fails
            
    def _trigger_linking_for_specific_ups(self, ups_to_link: List[str], 
                                       target_date: str, engine) -> pd.DataFrame:
        """
        Trigger linking process for specific UPs
        
        Args:
            ups_to_link: List of UPs to link
            target_date: Date to perform linking for
            engine: Database engine
            
        Returns:
            pd.DataFrame: New links created (formatted for database)
        """
        if not ups_to_link:
            print("ℹ️  No UPs to link")
            return pd.DataFrame(columns=['UP', 'UOF', 'date_updated'])
            
        print(f"\n🔄 TRIGGERING LINKING FOR {len(ups_to_link)} SPECIFIC UPs")
        print("-"*50)
        
        try:
            # Perform full linking process (this will include all UPs)
            all_matches_df = self.linking_algorithm.link_uofs_to_ups(target_date, ups_to_link)
            
            if not all_matches_df.empty:
                # Filter to only the specific UPs we want to link
                new_links = all_matches_df[all_matches_df['up'].isin(ups_to_link)].copy()
                
                if not new_links.empty:
                    # Format for database
                    db_new_links = pd.DataFrame({
                        'UP': new_links['up'].str.upper(),
                        'UOF': new_links['uof'].str.upper(),
                        'date_updated': pd.to_datetime(target_date).date()
                    })
                    
                    print(f"✅ Created {len(db_new_links)} new links for specific UPs")
                    return db_new_links
                else:
                    print("⚠️  No links found for the specified UPs")
                    return pd.DataFrame(columns=['UP', 'UOF', 'date_updated'])
            else:
                print("⚠️  No links created in linking process")
                return pd.DataFrame(columns=['UP', 'UOF', 'date_updated'])
            
        except Exception as e:
            print(f"❌ Error triggering linking for specific UPs: {e}")
            return pd.DataFrame(columns=['UP', 'UOF', 'date_updated'])

    #main method to run the incremental check       
    def run_incremental_check(self, engine, check_date: str) -> Dict:
        """
        Run incremental check for UPs that were enabled 93 days ago
        
        Args:
            engine: Database engine  
            check_date: Date to check (93 days back)
            
        Returns:
            Dict: Results of the incremental check
        """
        print(f"\n🗓️  RUNNING INCREMENTAL LINKING CHECK")
        print(f"Checking UPs enabled on: {check_date} (93 days back)")
        print("="*60)
        
        results = {
            'check_date': check_date,
            'new_ups_found': [],
            'new_links_created': pd.DataFrame(columns=['UP', 'UOF', 'date_updated']),
            'success': False,
            'message': ''
        }
        
        try:
            # Step 1: Find UPs that were enabled 93 days ago
            enabled_ups = self._check_for_ups_enabled_93_days_ago(engine, check_date)
            results['new_ups_found'] = enabled_ups
            
            if enabled_ups:
                # Step 2: Check which of these UPs are not already linked
                target_date = self.config.get_linking_target_date()
                unlinked_ups = self._check_if_ups_already_linked(engine, enabled_ups, target_date)
                
                if unlinked_ups:
                    # Step 3: Trigger linking for unlinked UPs
                    new_links = self._trigger_linking_for_specific_ups(unlinked_ups, target_date, engine)
                    results['new_links_created'] = new_links
                    results['success'] = True
                    results['message'] = f"Successfully processed {len(enabled_ups)} UPs enabled 93 days ago. Created {len(new_links)} new links for {len(unlinked_ups)} previously unlinked UPs."
                else:
                    results['success'] = True
                    results['message'] = f"Found {len(enabled_ups)} UPs enabled 93 days ago, but all are already linked."
            else:
                results['success'] = True
                results['message'] = "No UPs were enabled 93 days ago."
                
            print(f"\n✅ INCREMENTAL CHECK COMPLETE: {results['message']}")
            
        except Exception as e:
            results['message'] = f"Error during incremental check: {e}"
            print(f"\n❌ INCREMENTAL CHECK FAILED: {results['message']}")
            
        print("="*60)
        return results 