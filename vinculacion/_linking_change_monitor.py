import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple
import sys
from pathlib import Path
from sqlalchemy import text, create_engine
import asyncio

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from utilidades.db_utils import DatabaseUtils
from vinculacion.configs.vinculacion_config import VinculacionConfig
from vinculacion._linking_algorithm import UOFUPLinkingAlgorithm

class UPChangeMonitor:
    """Monitors up_change_log table for UPs enabled at least 93 days ago so we can link them to UOFs"""
    
    def __init__(self, database_name: str = "energy_tracker"):
        self.config = VinculacionConfig()
        self.linking_algorithm = UOFUPLinkingAlgorithm()
        self.db_utils = DatabaseUtils()
        self.engine = self.db_utils.create_engine(database_name)

    def _get_current_links(self) -> pd.DataFrame:
        """Fetches current UP-UOF links from the database."""

        print("1. Fetching current UP-UOF links from database...")
        try:

            df = self.db_utils.read_table(self.engine, self.config.UP_UOF_VINCULACION_TABLE)
            df.rename(columns={'UP': 'up', 'UOF': 'uof_old'}, inplace=True)
            print(f"   Found {len(df)} existing links.")
            return df
        
        except Exception:
            print(f"   Table '{self.config.UP_UOF_VINCULACION_TABLE}' not found or empty. Assuming no existing links.")
            return pd.DataFrame(columns=['up', 'uof_old'])

    async def _run_initial_linking(self, target_date: str) -> None:
        """Runs an initial full linking round and populates the database. 
        Only used if the table is empty
        Args:
            target_date: The date to link to
        """

        print("\n⚠️ No existing links found. Running initial full linking round...")
        results = await self.linking_algorithm.link_uofs_to_ups(target_date)
        if results['success']:  #if the linking process was successful, we can add the new links to the database
            all_matches_df = results['links_df']
            db_new_links = pd.DataFrame({
                'UP': all_matches_df['up'].str.upper(),
                'UOF': all_matches_df['uof'].str.upper(),
                'date_updated': all_matches_df['date_updated']
            })
            self.db_utils.write_table(self.engine, db_new_links, self.config.UP_UOF_VINCULACION_TABLE, if_exists='append')
            print(f"✅ Successfully inserted {len(db_new_links)} new links.")
        else:
            print("ℹ️ No links found in the initial run.")

    async def _get_new_matches(self, target_date: str) -> pd.DataFrame:
        """Runs a full linking round to get new matches.
        Args:
            target_date: The date to link to
        """

        print("\n2. Running full linking round to get new matches...")
        results = await self.linking_algorithm.link_uofs_to_ups(target_date)

        if results['success']:
            new_matches_df = results['links_df']
        else:
            print("⚠️ New linking round did not produce any matches.")
            return pd.DataFrame()
        
        new_matches_df.rename(columns={'uof': 'uof_new'}, inplace=True)

        print(f"   Found {len(new_matches_df)} potential links.")

        return new_matches_df

    def _find_changes(self, current_links_df: pd.DataFrame, new_matches_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Compares current and new links to find changes and new links.
        Args:
            current_links_df: The current links dataframe (UP-UOF) from the database
            new_matches_df: The "new" matches dataframe ie the dataframe with the UP matches found in the full linking round
        """

        print("\n3. Identifying changes and new/unlinked UPs...")
        merged_df = pd.merge(current_links_df, new_matches_df, on='up', how='outer', indicator=True)
        
        # In new macthes but not current matches -> new links
        new_links_df = merged_df[merged_df['_merge'] == 'right_only'][['up', 'uof_new']].copy()

        # In current matches but not new matches -> unlinked
        unlinked_df = merged_df[merged_df['_merge'] == 'left_only'].copy()
        unlinked_df['uof_new'] = 'UNKNOWN'
        
        # In both -> check for changes
        changed_df = merged_df[merged_df['_merge'] == 'both'].copy()
        changed_df = changed_df[changed_df['uof_old'] != changed_df['uof_new']]
        
        all_changes_df = pd.concat([changed_df, unlinked_df])
        
        return all_changes_df[['up', 'uof_old', 'uof_new']], new_links_df

    def _filter_valid_unlinked_ups(self, changes_df: pd.DataFrame) -> pd.DataFrame:
        """Filters out changes for UPs that are now unlinked and obsolete.
        Args:
            changes_df: The changes dataframe
        """

        unlinked_ups = changes_df[changes_df['uof_new'] == 'UNKNOWN']['up'].tolist()
        if not unlinked_ups:
            return changes_df

        print(f"   Verifying {len(unlinked_ups)} unlinked UPs for validity...")
        placeholders = ', '.join([f"'{up}'" for up in unlinked_ups])
        where_clause = f"UP IN ({placeholders})"
        
        validity_df = self.db_utils.read_table(self.engine, self.config.UP_LISTADO_TABLE, columns=['UP', 'obsoleta'], where_clause=where_clause)
        valid_ups = validity_df[validity_df['obsoleta'] == 0]['UP'].tolist()
        invalid_ups = set(unlinked_ups) - set(valid_ups)
        
        if invalid_ups:
            print(f"   - Found {len(invalid_ups)} obsolete/invalid UPs. They will be ignored.")
        
        return changes_df[~changes_df['up'].isin(list(invalid_ups))].copy()

    def _log_and_update_changes(self, changes_to_apply: pd.DataFrame) -> None:
        """Logs changes and updates the main linking table."""
        if changes_to_apply.empty:
            return
            
        print(f"\n4. Found {len(changes_to_apply)} changes to apply.")
        
        change_log_df = pd.DataFrame({
            'UP': changes_to_apply['up'],
            'field_changed': 'UOF',
            'old_value': changes_to_apply['uof_old'],
            'new_value': changes_to_apply['uof_new'],
            'date_updated': datetime.now().date()
        })
        self.db_utils.update_table(self.engine, change_log_df, self.config.VINCULACION_CHANGE_LOG_TABLE, key_columns=['UP'])
        print(f"   Logged {len(change_log_df)} changes to `{self.config.VINCULACION_CHANGE_LOG_TABLE}`.")

        update_df = changes_to_apply[['up', 'uof_new']].rename(columns={'uof_new': 'UOF', 'up': 'UP'})
        update_df['date_updated'] = datetime.now().date()
        self.db_utils.update_table(self.engine, update_df, self.config.UP_UOF_VINCULACION_TABLE, key_columns=['UP'])
        print(f"   Updated {len(changes_to_apply)} records in `{self.config.UP_UOF_VINCULACION_TABLE}`.")

    def _add_new_links(self, new_links_df: pd.DataFrame, target_date: str) -> None:
        """Adds newly identified links to the database.
        Args:
            new_links_df: The new links dataframe
            target_date: The date to link to
        """

        if new_links_df.empty:
            return
            
        print(f"\n5. Adding {len(new_links_df)} new links...")
        db_new_links = pd.DataFrame({
            'UP': new_links_df['up'].str.upper(),
            'UOF': new_links_df['uof_new'].str.upper(),
            'date_updated': pd.to_datetime(target_date).date()
        })
        self.db_utils.write_table(self.engine, db_new_links, self.config.UP_UOF_VINCULACION_TABLE, if_exists='append')
        print(f"   Successfully inserted {len(db_new_links)} new links.")

    async def monitor_existing_links(self) -> Dict:
        """Monitors existing UP-UOF links for changes, updates the database, and logs changes."""
        print("\n🔍 STARTING EXISTING LINK MONITORING")
        print("="*60)
        
        target_date = self.config.get_linking_target_date()

        results = {
            'target_date': target_date,
            'changes_found': pd.DataFrame(),
            'new_links_found': pd.DataFrame(),
            'success': False,
            'message': ''
        }

        try:
            current_links_df = self._get_current_links()

            if current_links_df.empty:
                await self._run_initial_linking(target_date)
                results['success'] = True
                results['message'] = "Initial linking run completed."
                return results

            
            new_matches_df = await self._get_new_matches(target_date)
            if new_matches_df.empty:
                results['message'] = "New linking round did not produce any matches. Aborting."
                print(f"⚠️ {results['message']}")
                return results
            
            all_changes_df, new_links_df = self._find_changes(current_links_df, new_matches_df)
            
            if not all_changes_df.empty:
                changes_to_apply = self._filter_valid_unlinked_ups(all_changes_df)
                self._log_and_update_changes(changes_to_apply)
                results['changes_found'] = changes_to_apply
            else:
                print("\n✅ No changes detected in existing UP-UOF links.")

            self._add_new_links(new_links_df, target_date)
            results['new_links_found'] = new_links_df
            
            results['success'] = True
            num_changes = len(results['changes_found'])
            num_new = len(results['new_links_found'])
            results['message'] = f"Monitoring complete. Found {num_changes} changes and {num_new} new links."

        except Exception as e:
            results['message'] = f"An error occurred during link monitoring: {e}"
            print(f"❌ {results['message']}")
        
        finally:
            print(f"\n✅ LINK MONITORING COMPLETE: {results['message']}")
            print("="*60)
            return results
        
if __name__ == "__main__":
    monitor = UPChangeMonitor()
    asyncio.run(monitor.monitor_existing_links())

