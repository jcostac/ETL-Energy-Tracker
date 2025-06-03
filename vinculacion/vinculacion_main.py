import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import sys
from pathlib import Path
from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from utilidades.db_utils import DatabaseUtils
from vinculacion._linking_algorithm import UOFUPLinkingAlgorithm
from vinculacion._linking_change_monitor import UPChangeMonitor
from vinculacion.configs.vinculacion_config import VinculacionConfig

class VinculacionOrchestrator:
    """Main orchestrator for the vinculacion module"""
    
    def __init__(self, database_name: str = "energy_tracker"):
        self.database_name = database_name
        self.config = VinculacionConfig()
        self.linking_algorithm = UOFUPLinkingAlgorithm()
        self.change_monitor = UPChangeMonitor()
        self.db_utils = DatabaseUtils()
        
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
                print(f"\n📊 LINKING RESULTS SUMMARY")
                print("-"*40)
                print(f"Total links created: {len(links)}")
                print(f"Unique UPs linked: {links['UP'].nunique()}")
                print(f"Unique UOFs linked: {links['UOF'].nunique()}")
                
                # Display sample results
                print(f"\n📋 SAMPLE LINKS (First 10):")
                print(links.head(10)[['UP', 'UOF', 'date_updated']].to_string(index=False))
                
                return links
            
            else:
                print("\n⚠️  No links were created")
                raise Exception("No links were created")
                
        except Exception as e:
            print(f"\n❌ ERROR IN FULL LINKING: {e}")
            return pd.DataFrame(columns=['UP', 'UOF', 'date_updated'])
            
    def perform_new_ups_linking(self) -> Dict:
        """
        Perform incremental linking for UPs that were enabled at least 93 days ago
        
        Returns:
            Dict: Results of incremental linking
        """
        check_date = self.config.get_linking_target_date() #93 days back from today|    
        
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
        try:
            # Create formatted DataFrame matching database schema
            db_links = pd.DataFrame({
                'UP': links_df['up'].str.upper(),  # Ensure uppercase
                'UOF': links_df['uof'].str.upper(),  # Ensure uppercase  
                'date_updated': pd.to_datetime(target_date).date()
            })

            # Remove duplicates
            check_duplicates = db_links.duplicated(subset=['UP', 'UOF'])
            if check_duplicates.any():
                print(f"⚠️  Duplicates found in {target_date}")
                print(db_links[check_duplicates])
                raise Exception("Duplicates found in links")
            
        except Exception as e:
            print(f"❌ Error formatting links for database: {e}")
            raise e
        
        return db_links
            
    def _get_existing_links(self, target_date: Optional[str] = None) -> pd.DataFrame:
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

    def _save_links_to_database(self, links_df: pd.DataFrame, table_name: str, 
                              if_exists: str = 'append') -> bool:
        """
        Save the generated links to database using DatabaseUtils
        
        Args:
            links_df: DataFrame with UOF-UP links
            table_name: Target table name
            if_exists: How to behave if table exists ('fail', 'replace', 'append') (default: append)
            
        Returns:
            bool: Success status
        """
        try:
            engine = self.create_engine()

            if links_df.empty:
                print("⚠️  No links to save")
                return False
                
            self.db_utils.write_table(
                engine=engine,
                df=links_df,
                table_name=table_name,
                if_exists=if_exists,
                index=False
            )
            
            print(f"✅ Successfully saved {len(links_df)} links to {table_name}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving links to database: {e}")
            return False
        
    def update_links_in_database(self, links_df: pd.DataFrame, table_name: str,
                               key_columns: List[str] = ['up']) -> bool:
        """
        Update existing links in database using DatabaseUtils
        
        Args:
            links_df: DataFrame with updated UOF-UP links
            table_name: Target table name
            key_columns: Columns to match for updates
            
        Returns:
            bool: Success status
        """
        try:
            engine = self.create_engine()
                
            if links_df.empty:
                print("⚠️  No links to update")
                return False
                
            self.db_utils.update_table(
                engine=engine,
                df=links_df,
                table_name=table_name,
                key_columns=key_columns
            )
            
            print(f"✅ Successfully updated {len(links_df)} links in {table_name}")
            return True
            
        except Exception as e:
            print(f"❌ Error updating links in database: {e}")
            return False 
    

def example_usage():
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
        orchestrator._save_links_to_database(full_links, replace_existing=True)
    
    # Incremental check (UPs enabled 93 days ago)
    incremental_results = orchestrator.perform_new_ups_linking()

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
    example_usage() 