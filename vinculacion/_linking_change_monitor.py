import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple
import sys
from pathlib import Path
from sqlalchemy import text, create_engine
from sqlalchemy.exc import IntegrityError
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
        unlinked_df['uof_new'] = 'unknown'
        
        # In both -> check for changes
        changed_df = merged_df[merged_df['_merge'] == 'both'].copy()
        changed_df = changed_df[changed_df['uof_old'] != changed_df['uof_new']]
        
        all_changes_df = pd.concat([changed_df, unlinked_df])
        
        return all_changes_df[['up', 'uof_old', 'uof_new']], new_links_df

    def _filter_valid_unlinked_ups(self, all_changes_df: pd.DataFrame) -> pd.DataFrame:
        """Filters out changes for UPs that are now unlinked and obsolete.
        Args:
            all_changes_df: The dataframe with the all the detected changes in UP

        Returns:
            * note: all_changes_df = uof_changes_df + unknown_changes_df + obsolete_invalid_ups_df
            uof_changes_df: The dataframe with the changes in UOF
            unknown_changes_df: The dataframe with the changes in UP where UP no longer is in mathcing but still active
            obsolete_invalid_ups_df: The data frame with the changes in UP where UP is obsolete hence it does not appear in matching 
        """
        uof_changes_df = all_changes_df[all_changes_df['uof_new'] != 'unknown']
        missing_vinculaciones_df = all_changes_df[all_changes_df['uof_new'] == 'unknown']
        missing_vinculaciones_list = missing_vinculaciones_df['up'].tolist()
        if not missing_vinculaciones_list:
            return all_changes_df
        

        print(f"   Checking {len(missing_vinculaciones_list)} already matched UPs that were not found in current matching...")
        placeholders = ', '.join([f"'{up}'" for up in missing_vinculaciones_list])
        where_clause = f"UP IN ({placeholders})"
        
        listado_ups = self.db_utils.read_table(self.engine, self.config.UP_LISTADO_TABLE, columns=['UP', 'obsoleta'], where_clause=where_clause)
        valid_ups = listado_ups[listado_ups['obsoleta'] == 0]['UP'].tolist()
        invalid_ups = set(missing_vinculaciones_list) - set(valid_ups)
        
        if invalid_ups:
            #change "unknown" field to None in the all_changes_df
            all_changes_df.loc[all_changes_df['up'].isin(list(invalid_ups)), 'uof_new'] = None
            print(f"   - Found {len(invalid_ups)} obsolete/invalid UPs. Their UOF status will be set to None.")


        obsolete_invalid_ups_df = missing_vinculaciones_df[missing_vinculaciones_df['up'].isin(list(invalid_ups))]
        unknown_changes_df = missing_vinculaciones_df[~missing_vinculaciones_df['up'].isin(list(invalid_ups))]

        print(f"   Found {len(uof_changes_df)} changes in UOF, {len(unknown_changes_df)} missing UPs that were previously matched are missing from current matching and {len(obsolete_invalid_ups_df)} obsolete/invalid UPs.")
             
        
        return unknown_changes_df, obsolete_invalid_ups_df, uof_changes_df

    def _log_and_update_changes(self, uof_changes_df: pd.DataFrame, unknown_changes_df: pd.DataFrame, obsolete_invalid_ups_df: pd.DataFrame) -> None:
        """Logs changes and updates the main linking table."""
        if unknown_changes_df.empty and obsolete_invalid_ups_df.empty and uof_changes_df.empty:
            return
            
        print(f"\n4. Logging changes and updating the main linking table...")
        
        change_log_dfs = []
        
        if not uof_changes_df.empty:
            uof_change_log_df = pd.DataFrame({
                'UP': uof_changes_df['up'],
                'field_changed': 'UOF',
                'old_value': uof_changes_df['uof_old'],
                'new_value': uof_changes_df['uof_new'],
                'date_updated': datetime.now().date()
            })
        
            #print the changes in a nice format
            print(f" UP-UOF changes: ")
            for index, row in uof_change_log_df.iterrows():
                print(f"   {row['UP']}: {row['old_value']} -> {row['new_value']}")
            change_log_dfs.append(uof_change_log_df)

        if not obsolete_invalid_ups_df.empty:
            obsolete_change_log_df = pd.DataFrame({
                'UP': obsolete_invalid_ups_df['up'],
                'field_changed': 'obsoleta',
                'old_value': obsolete_invalid_ups_df['uof_old'],
                'new_value': None,
                'date_updated': datetime.now().date()
            })

            #print the changes in a nice format
            print(f" Obsolete/invalid UPs: ")
            for index, row in obsolete_change_log_df.iterrows():
                print(f"   {row['UP']}: {row['old_value']} -> {row['new_value']}")
            change_log_dfs.append(obsolete_change_log_df)

        if not unknown_changes_df.empty: #this type of changes would mean that there is something wrong with the matching algorithm
            unknown_change_log_df = pd.DataFrame({
                'UP': unknown_changes_df['up'],
                'field_changed': 'UOF',
                'old_value': unknown_changes_df['uof_old'],
                'new_value': 'unknown',
                'date_updated': datetime.now().date()
            })
            #change_log_dfs.append(unknown_change_log_df)
            #we will not log these changes, only recorded for purposes of debugging

        # Combine all change logs
        if change_log_dfs:
            change_log_df = pd.concat(change_log_dfs, ignore_index=True)
            print(f"   Logging {len(change_log_dfs)} changes to `{self.config.VINCULACION_CHANGE_LOG_TABLE}`.")
        
        #update the main UP-UOF linking table  
        all_changes_df = pd.concat([uof_changes_df, obsolete_invalid_ups_df])
        up_uof_to_update_df = all_changes_df[['up', 'uof_new']].rename(columns={'uof_new': 'UOF', 'up': 'UP'})
        up_uof_to_update_df['date_updated'] = datetime.now().date()

        return change_log_df, up_uof_to_update_df

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

        return db_new_links
        

    def _check_if_already_updated_today(self) -> bool:
        """Checks if the UP-UOF vinculacion table was already updated today."""
        latest_dates_df = self.db_utils.read_table(
            self.engine,
            self.config.UP_UOF_VINCULACION_TABLE,
            columns=['date_updated']
        )
        last_update_date = pd.to_datetime(latest_dates_df['date_updated']).max().date()
        today = datetime.now().date()
        if last_update_date == today:
            return True
        return False
    
    def _write_operations_to_db(self, change_log_df: pd.DataFrame, up_uof_to_update_df: pd.DataFrame, db_new_links: pd.DataFrame) -> None:
        """Writes operations to the database."""
        try:
            if not change_log_df.empty:
                self.db_utils.update_table(self.engine, change_log_df, self.config.VINCULACION_CHANGE_LOG_TABLE, key_columns=['UP'])
                print(f"   Logged {len(change_log_df)} changes to `{self.config.VINCULACION_CHANGE_LOG_TABLE}`.")

            if not up_uof_to_update_df.empty:
                self.db_utils.update_table(self.engine, up_uof_to_update_df, self.config.UP_UOF_VINCULACION_TABLE, key_columns=['UP'])
                print(f"   Updated {len(up_uof_to_update_df)} records in `{self.config.UP_UOF_VINCULACION_TABLE}`.")

            if not db_new_links.empty:
                self.db_utils.write_table(self.engine, db_new_links, self.config.UP_UOF_VINCULACION_TABLE, if_exists='append')
                print(f"   Successfully inserted {len(db_new_links)} new links.")

        except IntegrityError as e:
            print(f"⚠️ Warning: Encountered duplicate entries when inserting new links: {e}")
            print("   Identifying which of the new links are duplicates...")
            
            existing_links = self.db_utils.read_table(self.engine, self.config.UP_UOF_VINCULACION_TABLE, columns=['UP', 'UOF'])
            
            # Use a merge to find the rows in db_new_links that are already in existing_links
            duplicate_rows = pd.merge(db_new_links, existing_links, on=['UP', 'UOF'], how='inner')
            
            if not duplicate_rows.empty:
                print("   The following entries are duplicates and were not inserted:")
                print(duplicate_rows.to_string())

            raise Exception("Duplicate entries found in the database. Please check the logs for more details.")

        except Exception as e:
            print(f"❌ Error writing operations to database: {e}")
            raise e

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
            if self._check_if_already_updated_today():
                print(f"⏩ Skipping monitoring: UP-UOF vinculacion table already updated today ({datetime.now().date()}).")
                results['success'] = True
                results['message'] = f"Table already updated today ({datetime.now().date()}); skipping."
                return results
        except Exception as e:
            print(f"⚠️ Could not check last update date: {e}")

        try:
            current_links_df = self._get_current_links()

            if current_links_df.empty: #if the table is empty, we run the initial linking and return
                await self._run_initial_linking(target_date)
                results['success'] = True
                results['message'] = "Initial linking run completed."
                return results
            

            
            new_matches_df = await self._get_new_matches(target_date) #get the full linking round for the target date
            if new_matches_df.empty: #never unless there is an error in the linking algorithm will we have empty 
                results['message'] = "New linking round did not produce any matches. Aborting."
                print(f"⚠️ {results['message']}")
                return results
            
            
            
            #compare the current links with the new matches to find changes and new links
            all_changes_df, new_links_df = self._find_changes(current_links_df, new_matches_df)
            

            if not all_changes_df.empty:
                #we might have UPs that are now unlinked and obsolete, so we filter them out
                unknown_changes_df, obsolete_invalid_ups_df, uof_changes_df = self._filter_valid_unlinked_ups(all_changes_df)
                change_log_df, up_uof_to_update_df = self._log_and_update_changes(uof_changes_df, unknown_changes_df, obsolete_invalid_ups_df) #log the changes to the change log table
                results['changes_found'] = pd.concat([up_uof_to_update_df, change_log_df], ignore_index=True)

            else: #if changes df is empty
                print("\n✅ No changes detected in existing UP-UOF links.")

            #add new up-uof links to the db (if any)
            if not new_links_df.empty:
                db_new_links = self._add_new_links(new_links_df, target_date)
                results['new_links_found'] = db_new_links

            
            #write operations in database (ups-uof linkings that changed, obsolete ups and new up-uof links)
            self._write_operations_to_db(change_log_df, up_uof_to_update_df, db_new_links)

            
            
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

