import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import sys
from pathlib import Path
from sqlalchemy import text

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
        
    def perform_full_linking(self) -> pd.DataFrame:
        """
        Perform full UOF-UP linking for all active UPs
        Target date is automatically set to 93 days back from today
        
        Returns:
            pd.DataFrame: Complete linking results ready for database
        """
        target_date = self.config.get_linking_target_date()
        
        print(f"\n🚀 STARTING FULL UOF-UP LINKING")
        print(f"Target Date: {target_date} (93 days back)")
        print("="*80)
        
        try:
            engine = self.create_engine()
            
            # Perform linking
            links = self.linking_algorithm.link_uofs_to_ups(target_date, engine)
            
            if not links.empty:
                # Format for database insertion
                db_links = self._format_links_for_database(links, target_date)
                
                print(f"\n📊 LINKING RESULTS SUMMARY")
                print("-"*40)
                print(f"Total links created: {len(db_links)}")
                print(f"Unique UPs linked: {db_links['UP'].nunique()}")
                print(f"Unique UOFs linked: {db_links['UOF'].nunique()}")
                
                # Display sample results
                print(f"\n📋 SAMPLE LINKS (First 10):")
                print(db_links.head(10)[['UP', 'UOF', 'date_updated']].to_string(index=False))
                
                return db_links
                
            else:
                print("\n⚠️  No links were created")
                return pd.DataFrame(columns=['UP', 'UOF', 'date_updated'])
                
        except Exception as e:
            print(f"\n❌ ERROR IN FULL LINKING: {e}")
            return pd.DataFrame(columns=['UP', 'UOF', 'date_updated'])
            
    def perform_incremental_linking(self) -> Dict:
        """
        Perform incremental linking for UPs that were enabled at least 93 days ago
        
        Returns:
            Dict: Results of incremental linking
        """
        check_date = self.config.get_linking_target_date()
        
        print(f"\n🔄 STARTING INCREMENTAL UOF-UP LINKING")
        print(f"Checking for UPs enabled on: {check_date} (93 days back)")
        print("="*80)
        
        try:
            engine = self.create_engine()
            
            # Run daily check for UPs enabled 93 days ago
            results = self.change_monitor.run_incremental_check(engine, check_date)
            
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
    
    def _format_links_for_database(self, links_df: pd.DataFrame, target_date: str) -> pd.DataFrame:
        """
        Format links DataFrame to match database schema
        
        Args:
            links_df: Raw links DataFrame with columns [up, uof]
            target_date: Date for the links
            
        Returns:
            pd.DataFrame: Formatted for up_uof_vinculacion table
        """
        if links_df.empty:
            return pd.DataFrame(columns=['UP', 'UOF', 'date_updated'])
            
        # Create formatted DataFrame matching database schema
        db_links = pd.DataFrame({
            'UP': links_df['up'].str.upper(),  # Ensure uppercase
            'UOF': links_df['uof'].str.upper(),  # Ensure uppercase  
            'date_updated': pd.to_datetime(target_date).date()
        })
        
        # Remove duplicates
        db_links = db_links.drop_duplicates(subset=['UP', 'UOF'])
        
        return db_links
        
    def save_links_to_database(self, links_df: pd.DataFrame, 
                             replace_existing: bool = False) -> bool:
        """
        Save linking results to up_uof_vinculacion table
        
        Args:
            links_df: DataFrame with links (formatted for database)
            replace_existing: Whether to replace existing links for the same date
            
        Returns:
            bool: Success status
        """
        try:
            engine = self.create_engine()
            
            if links_df.empty:
                print("⚠️  No links to save")
                return False
                
            table_name = self.config.UP_UOF_VINCULACION_TABLE
            
            if replace_existing:
                # Delete existing records for this date first
                target_date = links_df['date_updated'].iloc[0]
                delete_query = f"DELETE FROM {table_name} WHERE date_updated = ?"
                
                with engine.connect() as conn:
                    conn.execute(text(delete_query), (target_date,))
                    conn.commit()
                    print(f"🗑️  Deleted existing links for {target_date}")
            
            # Save new links
            DatabaseUtils.write_table(
                engine, 
                links_df, 
                table_name, 
                if_exists='append',
                index=False
            )
            
            print(f"✅ Saved {len(links_df)} links to {table_name}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving links to database: {e}")
            return False
            
    def get_existing_links(self, target_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get existing links from database for a specific date
        
        Args:
            target_date: Date to query (defaults to 93 days back)
            
        Returns:
            pd.DataFrame: Existing links
        """
        if target_date is None:
            target_date = self.config.get_linking_target_date()
            
        try:
            engine = self.create_engine()
            
            query = f"""
            SELECT UP, UOF, date_updated
            FROM {self.config.UP_UOF_VINCULACION_TABLE}
            WHERE date_updated = ?
            ORDER BY UP, UOF
            """
            
            with engine.connect() as conn:
                result = conn.execute(text(query), (target_date,))
                df = pd.DataFrame(result.fetchall(), columns=['UP', 'UOF', 'date_updated'])
                
            print(f"📋 Found {len(df)} existing links for {target_date}")
            return df
            
        except Exception as e:
            print(f"❌ Error retrieving existing links: {e}")
            return pd.DataFrame(columns=['UP', 'UOF', 'date_updated'])

def main():
    """Main execution function"""
    orchestrator = VinculacionOrchestrator()
    
    print("="*100)
    print("🎯 VINCULACION MODULE - UOF TO UP LINKING")
    print("🗓️  Target Date: Automatically set to 93 days back")
    print("="*100)
    
    # Full linking (all active UPs)
    full_links = orchestrator.perform_full_linking()
    
    # Save results to database
    if not full_links.empty:
        orchestrator.save_links_to_database(full_links, replace_existing=True)
    
    # Incremental check (UPs enabled 93 days ago)
    incremental_results = orchestrator.perform_incremental_linking()
    
    # Save incremental results if any
    if (incremental_results['success'] and 
        not incremental_results['new_links_created'].empty):
        orchestrator.save_links_to_database(
            incremental_results['new_links_created']
        )
    
    print("\n" + "="*100)
    print("🏁 VINCULACION PROCESS COMPLETE")
    print("="*100)

if __name__ == "__main__":
    main() 