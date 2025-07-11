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
        """
        Initialize the UPChangeMonitor with configuration, linking algorithm, database utilities, and a database engine for the specified database.
        
        Parameters:
            database_name (str): Name of the database to connect to. Defaults to "energy_tracker".
        """
        self.config = VinculacionConfig()
        self.linking_algorithm = UOFUPLinkingAlgorithm()
        self.db_utils = DatabaseUtils()
        self.engine = self.db_utils.create_engine(database_name)

    def _get_current_links(self) -> pd.DataFrame:
        """
        Retrieve the current UP-UOF linking records from the database.
        
        Returns:
            pd.DataFrame: A DataFrame containing existing UP-UOF links with columns 'up' and 'uof_old'. Returns an empty DataFrame if the linking table is missing or empty.
        """

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
        """
        Performs an initial full linking of UPs to UOFs for the specified date and populates the database if no existing links are present.
        
        Parameters:
            target_date (str): The date for which to perform the initial linking.
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
        """
        Asynchronously runs a full linking round for the specified date and returns new UP-UOF matches.
        
        Parameters:
            target_date (str): The date for which to perform the linking operation.
        
        Returns:
            pd.DataFrame: DataFrame containing new UP-UOF matches with columns renamed for consistency. Returns an empty DataFrame if no matches are found.
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
        """
        Compare current and new UP-UOF links to identify changes and new links.
        
        This method merges the current and newly generated UP-UOF link DataFrames to detect:
        - New links (UPs present only in the new matches).
        - Unlinked UPs (UPs present only in the current links).
        - Changed links (UPs present in both, but with a different UOF).
        
        Parameters:
            current_links_df (pd.DataFrame): DataFrame of current UP-UOF links from the database.
            new_matches_df (pd.DataFrame): DataFrame of new UP-UOF matches from the latest linking round.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: 
                - DataFrame of all detected changes (including changed and unlinked UPs) with columns ['up', 'uof_old', 'uof_new'].
                - DataFrame of new links to be added, with columns ['up', 'uof_new'].
        """
        all_changes_df = pd.DataFrame()
        new_links_df = pd.DataFrame()

        try:
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

        except Exception as e:
            print(f"❌ Error finding changes: {e}")
            raise e
        
        return all_changes_df[['up', 'uof_old', 'uof_new']], new_links_df

    def _filter_valid_unlinked_ups(self, all_changes_df: pd.DataFrame) -> pd.DataFrame:
        """
        Categorizes unlinked UPs from detected changes into active, obsolete/invalid, and UOF-changed groups.
        
        Parameters:
            all_changes_df (pd.DataFrame): DataFrame containing all detected UP changes, including UOF changes and unlinked UPs.
        
        Returns:
            unknown_changes_df (pd.DataFrame): UPs that are no longer matched but remain active (not obsolete).
            obsolete_invalid_ups_df (pd.DataFrame): UPs that are unlinked and marked as obsolete or invalid.
            uof_changes_df (pd.DataFrame): UPs with detected changes in their associated UOF.
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

    def _log_changes(self, uof_changes_df: pd.DataFrame, unknown_changes_df: pd.DataFrame, obsolete_invalid_ups_df: pd.DataFrame) -> None:
        """
        Log detected changes in UP-UOF links and prepare updates for the main linking table.
        
        Creates change log entries for UOF changes and obsolete or invalid UPs, printing details for each. Unknown changes are not logged but are available for debugging. Returns DataFrames for the change log and for updating the main linking table.
        
        Parameters:
            uof_changes_df (pd.DataFrame): DataFrame of UPs with changed UOF assignments.
            unknown_changes_df (pd.DataFrame): DataFrame of UPs with unknown or ambiguous changes.
            obsolete_invalid_ups_df (pd.DataFrame): DataFrame of UPs identified as obsolete or invalid.
        
        Returns:
            change_log_df (pd.DataFrame): DataFrame containing the changes to be logged.
            up_uof_to_update_df (pd.DataFrame): DataFrame with updates for the main UP-UOF linking table.
        """
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
        change_log_df = pd.DataFrame()
        up_uof_to_update_df = pd.DataFrame()

        if change_log_dfs:
            change_log_df = pd.concat(change_log_dfs, ignore_index=True)
            print(f"   Changes to log: {len(change_log_df)}")
        
        #update the main UP-UOF linking table  
        all_changes_df = pd.concat([uof_changes_df, obsolete_invalid_ups_df])
        up_uof_to_update_df = all_changes_df[['up', 'uof_new']].rename(columns={'uof_new': 'UOF', 'up': 'UP'})
        up_uof_to_update_df['date_updated'] = datetime.now().date()

        return change_log_df, up_uof_to_update_df

    def _add_new_links(self, new_links_df: pd.DataFrame, target_date: str) -> None:
        """
        Prepare a DataFrame of new UP-UOF links with standardized formatting for database insertion.
        
        Parameters:
            new_links_df (pd.DataFrame): DataFrame containing new UP-UOF link information.
            target_date (str): Date to assign as the update date for the new links.
        
        Returns:
            pd.DataFrame: Formatted DataFrame of new links ready for database insertion.
        """
       
        print(f"\n5. Adding {len(new_links_df)} new links...")
        db_new_links = pd.DataFrame({
            'UP': new_links_df['up'].str.upper(),
            'UOF': new_links_df['uof_new'].str.upper(),
            'date_updated': pd.to_datetime(target_date).date()
        })

        return db_new_links
        
    def _check_if_already_updated_today(self) -> bool:
        """
        Determine whether the UP-UOF linking table has already been updated on the current date.
        
        Returns:
            bool: True if the table was updated today; otherwise, False.
        """
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
        """
        Write change logs, updates, and new UP-UOF links to the database, handling duplicates and errors.
        
        This method appends change logs, updates existing UP-UOF links, and inserts new links into the appropriate database tables. If duplicate entries are detected during insertion, it identifies and reports the duplicates before raising an exception. Any other database errors are also raised after logging.
        
        Parameters:
            change_log_df (pd.DataFrame): DataFrame containing change log entries to append.
            up_uof_to_update_df (pd.DataFrame): DataFrame of UP-UOF records to update.
            db_new_links (pd.DataFrame): DataFrame of new UP-UOF links to insert.
        """
        
        try:
            if not change_log_df.empty:
                self.db_utils.write_table(self.engine, change_log_df, self.config.VINCULACION_CHANGE_LOG_TABLE, if_exists='append')
                print(f"   Logged {len(change_log_df)} changes to `{self.config.VINCULACION_CHANGE_LOG_TABLE}`.")

            if not up_uof_to_update_df.empty:
                self.db_utils.update_table(self.engine, up_uof_to_update_df, self.config.UP_UOF_VINCULACION_TABLE, key_columns=['UP'])
                print(f"   Updated {len(up_uof_to_update_df)} records (obsolete + UOF changes) in `{self.config.UP_UOF_VINCULACION_TABLE}`.")

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
        """
        Asynchronously monitors and updates UP-UOF links, detecting changes, updating the database, and logging modifications.
        
        Returns:
            Dict: A summary of the monitoring operation, including the target date, detected changes, new links, success status, and a message.
        """
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
            #skip if the table was already updated today
            if self._check_if_already_updated_today():
                print(f"⏩ Skipping monitoring: UP-UOF vinculacion table already updated today ({datetime.now().date()}).")
                results['success'] = True
                results['message'] = f"Table already updated today ({datetime.now().date()}); skipping."
                return results
        except Exception as e:
            print(f"⚠️ Could not check last update date: {e}")

        try:
            current_links_df = self._get_current_links()

            #if the table is empty, we run the initial linking and return
            if current_links_df.empty: 
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
                change_log_df, up_uof_to_update_df = self._log_changes(uof_changes_df, unknown_changes_df, obsolete_invalid_ups_df) #log the changes to the change log table
                results['changes_found'] = change_log_df
                

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
        
        
        print(f"\n✅ LINK MONITORING COMPLETE: {results['message']}")
        print("="*60)
        return results
        
if __name__ == "__main__":
    monitor = UPChangeMonitor()
    asyncio.run(monitor.monitor_existing_links())

