import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from utilidades.db_utils import DatabaseUtils
from vinculacion.linking_algorithm import UOFUPLinkingAlgorithm
from vinculacion.change_monitor import UPChangeMonitor
from vinculacion.configs.vinculacion_config import VinculacionConfig

class VinculacionOrchestrator:
    """Main orchestrator for the vinculacion module"""
    
    def __init__(self, database_name: str = "energy_tracker"):
        self.database_name = database_name
        self.config = VinculacionConfig()
        self.linking_algorithm = UOFUPLinkingAlgorithm()
        self.change_monitor = UPChangeMonitor()
        
    def create_engine(self):
        """Create database engine"""
        return DatabaseUtils.create_engine(self.database_name)
        
    def perform_full_linking(self, target_date: Optional[str] = None) -> pd.DataFrame:
        """
        Perform full UOF-UP linking for all active UPs
        
        Args:
            target_date: Target date (YYYY-MM-DD), defaults to yesterday
            
        Returns:
            pd.DataFrame: Complete linking results
        """
        if target_date is None:
            target_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            
        print(f"\n🚀 STARTING FULL UOF-UP LINKING")
        print(f"Target Date: {target_date}")
        print("="*80)
        
        try:
            engine = self.create_engine()
            
            # Perform linking
            links = self.linking_algorithm.link_uofs_to_ups(target_date, engine)
            
            if not links.empty:
                print(f"\n📊 LINKING RESULTS SUMMARY")
                print("-"*40)
                print(f"Total links created: {len(links)}")
                print(f"Unique UPs linked: {links['up'].nunique()}")
                print(f"Unique UOFs linked: {links['uof'].nunique()}")
                
                # Display sample results
                print(f"\n📋 SAMPLE LINKS (First 10):")
                print(links.head(10).to_string(index=False))
                
            else:
                print("\n⚠️  No links were created")
                
            return links
            
        except Exception as e:
            print(f"\n❌ ERROR IN FULL LINKING: {e}")
            return pd.DataFrame(columns=['up', 'uof'])
            
    def perform_incremental_linking(self, check_date: Optional[str] = None) -> Dict:
        """
        Perform incremental linking for newly enabled UPs
        
        Args:
            check_date: Date to check for new UPs (defaults to today)
            
        Returns:
            Dict: Results of incremental linking
        """
        if check_date is None:
            check_date = datetime.now().strftime('%Y-%m-%d')
            
        print(f"\n🔄 STARTING INCREMENTAL UOF-UP LINKING")
        print(f"Check Date: {check_date}")
        print("="*80)
        
        try:
            engine = self.create_engine()
            
            # Run daily check
            results = self.change_monitor.run_daily_check(engine, check_date)
            
            if results['success']:
                print(f"\n📊 INCREMENTAL LINKING RESULTS")
                print("-"*45)
                print(f"New UPs found: {len(results['new_ups_found'])}")
                print(f"New links created: {len(results['new_links_created'])}")
                
                if not results['new_links_created'].empty:
                    print(f"\n📋 NEW LINKS CREATED:")
                    print(results['new_links_created'].to_string(index=False))
                    
            return results
            
        except Exception as e:
            print(f"\n❌ ERROR IN INCREMENTAL LINKING: {e}")
            return {
                'success': False,
                'message': f"Error: {e}",
                'new_ups_found': [],
                'new_links_created': pd.DataFrame(columns=['up', 'uof'])
            }
            
    def save_links_to_database(self, links_df: pd.DataFrame, 
                             table_name: str = "uof_up_links") -> bool:
        """
        Save linking results to database
        
        Args:
            links_df: DataFrame with links
            table_name: Target table name
            
        Returns:
            bool: Success status
        """
        try:
            engine = self.create_engine()
            
            if links_df.empty:
                print("⚠️  No links to save")
                return False
                
            # Add metadata
            links_with_metadata = links_df.copy()
            links_with_metadata['created_at'] = datetime.now()
            links_with_metadata['link_type'] = 'automatic'
            
            # Save to database
            DatabaseUtils.write_table(
                engine, 
                links_with_metadata, 
                table_name, 
                if_exists='append'
            )
            
            print(f"✅ Saved {len(links_df)} links to {table_name}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving links to database: {e}")
            return False

def main():
    """Main execution function"""
    orchestrator = VinculacionOrchestrator()
    
    # Example usage:
    
    # 1. Perform full linking for yesterday
    print("="*100)
    print("🎯 VINCULACION MODULE - UOF TO UP LINKING")
    print("="*100)
    
    # Full linking
    full_links = orchestrator.perform_full_linking()
    
    # Save results
    if not full_links.empty:
        orchestrator.save_links_to_database(full_links)
    
    # Incremental check
    incremental_results = orchestrator.perform_incremental_linking()
    
    print("\n" + "="*100)
    print("🏁 VINCULACION PROCESS COMPLETE")
    print("="*100)

if __name__ == "__main__":
    main() 