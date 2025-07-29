"""
Module for tracking and updating UOFs from OMIE.
"""
import pandas as pd
import sqlalchemy
from pathlib import Path
import sys
import os
from datetime import datetime
from typing import List, Dict, Tuple

# Add the root directory to Python path to allow imports from utilidades and config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', )))
from utilidades.db_utils import DatabaseUtils
from descarga_uofs_omie import download_uofs_from_omie 
import pretty_errors

class UOFDatabaseManager:
    """Handles all database operations for UOFs"""
    def __init__(self, bbdd_name="energy_tracker"):
        """
        Initializes the UOFDatabaseManager with the specified database name and default table names for UOFs and change logs.
        
        Parameters:
            bbdd_name (str): Name of the database to connect to. Defaults to "energy_tracker".
        """
        self.bbdd_name = bbdd_name
        self._engine = None
        self.uof_table_name = "uof_listado"
        self.change_log_table_name = "uof_change_log"

    @property
    def engine(self):
        """
        Returns the SQLAlchemy engine instance for the configured database, creating it if it does not already exist.
        """
        if self._engine is None:
            self._engine = DatabaseUtils.create_engine(self.bbdd_name)
        return self._engine
    
    @engine.setter
    def engine(self, engine):
        """
        Sets the SQLAlchemy engine instance for database operations.
        """
        self._engine = engine

    def load_uofs(self) -> pd.DataFrame:
        """
        Loads all UOF records from the database table and returns them as a DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame containing all UOF records from the database.
        
        Raises:
            Exception: If there is an error during database access or reading.
        """
        try:
            print(f"Loading UOFs from database table: {self.uof_table_name}...")
            engine = DatabaseUtils.create_engine(self.bbdd_name)
            db_df = DatabaseUtils.read_table(
                engine=engine,
                table_name=self.uof_table_name
            )
            engine.dispose()
            return db_df
        except Exception as e:
            print(f"Error loading UOFs from database: {e}")
            raise

    def save_new_uofs(self, new_uofs_df: pd.DataFrame) -> bool:
        """
        Saves new UOF records to the database, removing duplicates within the batch before insertion.
        
        Returns:
            bool: True if new UOFs were successfully saved or if there were no new UOFs to add; False if a duplicate entry error occurred during insertion.
        """
        if not new_uofs_df.empty:
            try:
                print(f"Adding {len(new_uofs_df)} new UOFs to the database...")
                
                # Find duplicates within the batch (before dropping) -> uof con propiedad compartida (misma uof 2+ agente propietarios)
                duplicates = new_uofs_df[new_uofs_df.duplicated(subset=['UOF', 'obsoleta'], keep="first")]
                if not duplicates.empty:
                    print("Duplicates found in new_uofs_df (will be dropped):")
                    print(duplicates)

                # Remove duplicates within the batch to avoid unique constraint violation
                new_uofs_df = new_uofs_df.drop_duplicates(subset=['UOF', 'obsoleta'])

                if new_uofs_df.empty:
                    print("No new UOFs to add after filtering duplicates.")
                    return True
                    
                print(f"Adding {len(new_uofs_df)} new UOFs after filtering duplicates...")
                engine = DatabaseUtils.create_engine(self.bbdd_name)
                DatabaseUtils.write_table(
                    engine, 
                    new_uofs_df, 
                    self.uof_table_name, 
                    if_exists='append', 
                    index=False
                )
                engine.dispose()
                return True
                
            except sqlalchemy.exc.IntegrityError as e:
                if "Duplicate entry" in str(e):
                    print(f"Skipping duplicate entry error: {e}")
                    return False
                else:
                    print(f"Database integrity error when adding new UOFs: {e}")
                    raise
            except Exception as e:
                print(f"Error saving new UOFs to database: {e}")
                raise
        return True

    def save_obsolete_uofs(self, obsolete_uofs_df: pd.DataFrame) -> bool:
        """
        Marks specified UOFs as obsolete in the database.
        
        Returns:
            bool: True if the operation succeeds, or False if a duplicate entry error occurs.
        """
        if not obsolete_uofs_df.empty:
            try:
                print(f"Marking {len(obsolete_uofs_df)} UOFs as obsolete...")
                engine = DatabaseUtils.create_engine(self.bbdd_name)
                DatabaseUtils.update_table(
                    engine=engine,
                    df=obsolete_uofs_df,
                    table_name=self.uof_table_name,
                    key_columns=['UOF']
                )
                engine.dispose()
                return True
            except sqlalchemy.exc.IntegrityError as e:
                if "Duplicate entry" in str(e):
                    print(f"Skipping duplicate entry error: {e}")
                    return False
                else:
                    print(f"Database integrity error when marking UOFs as obsolete: {e}")
                    raise
            except Exception as e:
                print(f"Error marking UOFs as obsolete: {e}")
                raise
        return True

    def save_change_log(self, change_log: List[Dict]) -> bool:
        """
        Saves a list of change log entries to the database change log table.
        
        Parameters:
            change_log (List[Dict]): List of change log entries to be saved.
        
        Returns:
            bool: True if the entries were saved successfully or if there were no changes to log; False if a duplicate entry error occurred.
        """
        if len(change_log) == 0:
            print("No changes to log.")
            return True

        try:
            print(f"Saving {len(change_log)} entries to {self.change_log_table_name}...")
            final_log_entries = self._process_change_log(change_log)
            
            if not final_log_entries:
                print("No changes to log after processing.")
                return True

            final_log_df = pd.DataFrame(final_log_entries)
            final_log_df = final_log_df.rename(columns={'UOF': 'UOF'})

            engine = DatabaseUtils.create_engine(self.bbdd_name)
            DatabaseUtils.write_table(
                engine, 
                final_log_df, 
                self.change_log_table_name, 
                if_exists='append', 
                index=False
            )
            engine.dispose()

            self._print_change_log_summary(final_log_df)
            return True

        except sqlalchemy.exc.IntegrityError as e:
            if "Duplicate entry" in str(e):
                print(f"Skipping duplicate entry error in change log: {e}")
                return False
            else:
                print(f"Database integrity error when saving change log: {e}")
                raise
        except Exception as e:
            print(f"Error saving change log: {e}")
            raise

    def _process_change_log(self, change_log: List[Dict]) -> List[Dict]:
        """
        Formats change log entries for database insertion, standardizing field values and handling special cases.
        
        Parameters:
            change_log (List[Dict]): List of raw change log entries to process.
        
        Returns:
            List[Dict]: List of formatted change log entries ready for storage.
        """
        final_log_entries = []
       
        for entry in change_log:
            if entry['field_changed'] == 'habilitada':
                final_log_entries.append({
                    'UOF': entry.get('UOF'),
                    'field_changed': 'habilitada',
                    'old_value': False,
                    'new_value': True,
                    'date_updated': entry['date_updated']
                })
            else:
                final_log_entries.append({
                    'UOF': entry.get('UOF'),
                    'field_changed': entry['field_changed'],
                    'old_value': str(entry['old_value']) if pd.notna(entry['old_value']) else None,
                    'new_value': str(entry['new_value']) if pd.notna(entry['new_value']) else None,
                    'date_updated': entry['date_updated']
                })
        return final_log_entries

    def _print_change_log_summary(self, log_df: pd.DataFrame) -> None:
        """
        Prints a summary of the changes recorded in the change log DataFrame.
        
        Each change entry includes the UOF identifier, the field changed, previous and new values, and the date of the update.
        """
        print("\nSummary of logged changes:")
        for _, row in log_df.iterrows():
            print(f"  - UOF: {row['UOF']}, Field: {row['field_changed']}, Old: {row['old_value']}, New: {row['new_value']}, Date: {row['date_updated']}")

class UOFChangeDetector:
    """Handles detection of changes in UOFs"""
    def __init__(self):
        """
        Initialize the UOFChangeDetector with the current date as a string in 'YYYY-MM-DD' format.
        """
        self.current_date_str = datetime.now().strftime('%Y-%m-%d')

    def identify_changes(self, omie_uofs_df: pd.DataFrame, db_uofs_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[Dict]]:
        """
        Detects and categorizes changes between OMIE UOF data and database UOF data.
        
        Compares the provided OMIE and database UOF DataFrames to identify new UOFs, obsolete UOFs, and updates to existing UOFs. Generates a change log detailing all detected modifications.
        
        Parameters:
            omie_uofs_df (pd.DataFrame): DataFrame containing the latest UOF data from OMIE.
            db_uofs_df (pd.DataFrame): DataFrame containing the current UOF data from the database.
        
        Returns:
            new_uofs_df (pd.DataFrame): DataFrame of UOFs present in OMIE but not in the database.
            obsolete_uofs_df (pd.DataFrame): DataFrame of UOFs present in the database but no longer in OMIE.
            updates_df (pd.DataFrame): DataFrame of UOFs with updated attributes (e.g., owner changes).
            change_log (List[Dict]): List of dictionaries describing each detected change.
        """
        change_log = []

        # Set up indices for comparison
        omie_uofs_df, db_uofs_df = self._prepare_dataframes(omie_uofs_df, db_uofs_df)
        
        # Get sets of UOF IDs
        omie_uof_ids = set(omie_uofs_df.index)
        db_uof_ids = set(db_uofs_df.index)

        # Identify new UOFs
        new_uofs_df, change_log = self._identify_new_uofs(omie_uofs_df, omie_uof_ids, db_uof_ids, change_log)
        
        # Identify obsolete UOFs
        obsolete_uofs_df, change_log = self._identify_obsolete_uofs(db_uofs_df, omie_uof_ids, change_log)
        
        # Identify reactivated and updated UOFs
        updates_df = self._identify_updates(omie_uofs_df, db_uofs_df, change_log)

        return new_uofs_df, obsolete_uofs_df, updates_df, change_log

    def _prepare_dataframes(self, omie_df: pd.DataFrame, db_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Sets the 'UOF' column as the index for both OMIE and database DataFrames while retaining it as a column.
        
        Returns:
            A tuple containing the modified OMIE and database DataFrames, ready for comparison.
        """
        if not omie_df.empty:
            omie_df = omie_df.set_index('UOF', drop=False)
        if not db_df.empty:
            db_df = db_df.set_index('UOF', drop=False)
        return omie_df, db_df

    def _identify_new_uofs(self, omie_df: pd.DataFrame, omie_ids: set, db_ids: set, change_log: List[Dict]) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Identify UOFs present in the OMIE dataset but missing from the database, marking them as new and updating the change log.
        
        Parameters:
            omie_df (pd.DataFrame): DataFrame of OMIE UOFs indexed by UOF identifier.
            omie_ids (set): Set of UOF identifiers from OMIE data.
            db_ids (set): Set of UOF identifiers from the database.
            change_log (List[Dict]): List to append change log entries for new UOFs.
        
        Returns:
            Tuple[pd.DataFrame, List[Dict]]: DataFrame of new UOFs with 'obsoleta' set to False, and the updated change log.
        """
        new_uof_ids = omie_ids - db_ids
        #create a new dataframe with the new uofs based on the new uof_ids
        new_uofs_df = omie_df[omie_df.index.isin(new_uof_ids)].copy()
        #set obsoleta to false
        new_uofs_df['obsoleta'] = False

        if not new_uofs_df.empty:
            for uof_id, row in new_uofs_df.iterrows():
                change_log.append({
                    'UOF': uof_id,
                    'field_changed': 'habilitada',
                    'old_value': False,
                    'new_value': True,
                    'date_updated': self.current_date_str
                })

        return new_uofs_df, change_log  # Return the list directly, not a DataFrame

    def _identify_obsolete_uofs(self, db_df: pd.DataFrame, omie_ids: set, change_log: List[Dict]) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Identifies active UOFs in the database that are no longer present in the OMIE dataset and prepares them to be marked as obsolete.
        
        Returns:
            obsolete_uof_df (pd.DataFrame): DataFrame of UOFs to be marked as obsolete.
            change_log (List[Dict]): Updated change log with entries for each UOF marked obsolete.
        """
        # Handle empty database case
        if db_df.empty:
            return pd.DataFrame(columns=db_df.columns), change_log
        
        # Get active UOFs
        active_db_uofs = db_df[db_df['obsoleta'] == False]
        
        # If no active UOFs, return empty DataFrame
        if active_db_uofs.empty:
            return pd.DataFrame(columns=db_df.columns), change_log
        
        # Now we can safely perform the set operation
        obsolete_uof_ids = set(active_db_uofs.index) - omie_ids
        obsolete_uof_df = active_db_uofs[active_db_uofs.index.isin(obsolete_uof_ids)].copy()
        
        uofs_to_mark_obsolete_list = []
        for uof_id in obsolete_uof_ids:
            uofs_to_mark_obsolete_list.append({'UOF': uof_id, 'obsoleta': True})
            change_log.append({
                'UOF': uof_id,
                'field_changed': 'obsoleta',
                'old_value': False,
                'new_value': True,
                'date_updated': self.current_date_str
            })

        return obsolete_uof_df, change_log  # Return the list directly, not a DataFrame

    def _identify_updates(self, omie_df: pd.DataFrame, db_df: pd.DataFrame, change_log: List[Dict]) -> pd.DataFrame:
        """
        Detects and returns updates to the 'agente_propietario' field for active UOFs present in both OMIE and the database.
        
        Compares active UOFs in the database with those from OMIE, identifies changes in ownership, logs these changes, and prepares a DataFrame of updated records.
        
        Parameters:
            omie_df (pd.DataFrame): DataFrame of UOFs from OMIE, indexed by 'UOF'.
            db_df (pd.DataFrame): DataFrame of UOFs from the database, indexed by 'UOF'.
            change_log (List[Dict]): List to which detected changes will be appended.
        
        Returns:
            pd.DataFrame: DataFrame containing updated UOFs with new 'agente_propietario', 'zona', and 'tecnologia' values.
        """
        updates_list = []
        
        # Handle field updates for active UOFs
        if not db_df.empty:
            active_db_uofs = db_df[db_df['obsoleta'] == False]
            common_active_ids = set(omie_df.index).intersection(set(active_db_uofs.index))
        else:
            print("No active updates to perform since db schema is empty")
            return pd.DataFrame(updates_list)
        
        #drop omie df uof duplicates ->>> duplicates occur ij uof with shared ownership (nucleares)
        omie_df = omie_df.drop_duplicates(subset=['UOF'], keep='first')
        
        for uof_id in common_active_ids:
            omie_row = omie_df.loc[uof_id]
            db_row = active_db_uofs.loc[uof_id]
            
            if db_row['agente_propietario'] != omie_row['agente_propietario']:
                change_log.append({
                    'UOF': uof_id,
                    'field_changed': 'agente_propietario',
                    'old_value': db_row['agente_propietario'],
                    'new_value': omie_row['agente_propietario'],
                    'date_updated': self.current_date_str
                })
                
                updates_list.append({
                    'UOF': uof_id,
                    'agente_propietario': omie_row['agente_propietario'],
                    'zona': omie_row['zona'],
                    'tecnologia': omie_row['tecnologia']
                })

        return pd.DataFrame(updates_list)

class UOFTracker:
    """Main class for tracking UOFs"""
    def __init__(self, bbdd_name="energy_tracker"):
        """
        Initialize the UOFTracker with a database manager and change detector.
        
        Parameters:
        	bbdd_name (str): Name of the database to use for UOF tracking. Defaults to "energy_tracker".
        """
        self.db_manager = UOFDatabaseManager(bbdd_name)
        self.change_detector = UOFChangeDetector()

    def _filter_omie_df(self, omie_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filters and normalizes the OMIE UOF DataFrame for further processing.
        
        The method standardizes the 'tipo_unidad' column, filters rows by allowed unit types and zones, maps zone names to short codes, drops unnecessary columns, and validates the presence of required columns. Raises a ValueError if any required columns are missing after processing.
        
        Parameters:
            omie_df (pd.DataFrame): Raw OMIE UOF DataFrame to be filtered.
        
        Returns:
            pd.DataFrame: Filtered and normalized DataFrame ready for tracking.
        """
        #strip tipo_unidad and convert to uppercase for consistent comparison
        omie_df['tipo_unidad'] = omie_df['tipo_unidad'].str.strip().str.upper()
        allowed_tipo_unidad = ['ALMACENAMIENTO', 'BOMBEO', 'GENERACION', 'COMERCIALIZADOR']
        
        # Debug: Print counts for each tipo_unidad
        print("Counts by tipo_unidad:")
        print(omie_df['tipo_unidad'].value_counts())
        
        omie_df = omie_df[omie_df['tipo_unidad'].isin(allowed_tipo_unidad)]

        # Filter by zona
        allowed_zonas = ['ZONA ESPAÑOLA', 'ZONA PORTUGUESA']
        omie_df = omie_df[omie_df['zona'].isin(allowed_zonas)]
       
        # Rename zona values
        zone_mapping = {
            'ZONA ESPAÑOLA': 'ES',
            'ZONA PORTUGUESA': 'PT'
        }
        omie_df['zona'] = omie_df['zona'].replace(zone_mapping)

        # Drop tipo_unidad column
        omie_df = omie_df.drop(columns=['tipo_unidad'])

        # Ensure required columns are present for the tracker
        expected_cols = ['UOF', 'agente_propietario', 'zona', 'tecnologia']
        if not all(col in omie_df.columns for col in expected_cols):
            missing_cols = [col for col in expected_cols if col not in omie_df.columns] 
            raise ValueError(f"Missing expected columns after processing for tracker: {missing_cols}")
        
        return omie_df
    
    def process_uofs(self, omie_path: str) -> None:
        """
        Processes and updates UOF records by comparing OMIE data with the database, applying detected changes, and logging results.
        
        Reads UOF data from the specified OMIE Excel file, filters and normalizes it, loads the current database state, identifies new, obsolete, and updated UOFs, and applies these changes to the database. Handles duplicate entry errors gracefully and prints a summary of operations performed. Raises an exception if critical errors occur during processing.
        
        Parameters:
            omie_path (str): Path to the OMIE UOF Excel file to process.
        """
        try:
            print("Starting UOF processing...")
            omie_df = pd.read_excel(omie_path)
            processed_omie_df = self._filter_omie_df(omie_df)
            
            if processed_omie_df.empty:
                print("Provided OMIE UOF DataFrame is empty after filtering. Aborting.")
                raise ValueError("Provided OMIE UOF DataFrame is empty after filtering.")

            # Load current state
            db_uofs_df = self.db_manager.load_uofs()
            initial_stats = self._get_database_stats(db_uofs_df)

            # Identify changes
            new_uofs_df, obsolete_uofs_df, updates_df, change_log = self.change_detector.identify_changes(
                processed_omie_df, 
                db_uofs_df
            )

            # Update database with error handling for integrity errors
            print("Updating database...")
            
            # Add new UOFs
            add_success = self.db_manager.save_new_uofs(new_uofs_df)
            if not add_success:
                print("⚠️ Skipped adding UOFs due to duplicate entries - continuing with next steps")
            
            # Mark obsolete UOFs
            obsolete_success = self.db_manager.save_obsolete_uofs(obsolete_uofs_df)
            if not obsolete_success:
                print("⚠️ Skipped marking obsolete UOFs due to duplicate entries - continuing with next steps")
            
            # Save change log
            log_success = self.db_manager.save_change_log(change_log)
            if not log_success:
                print("⚠️ Skipped saving change log due to duplicate entries - continuing with next steps")

            # Print summary
            final_db_uofs_df = self.db_manager.load_uofs()
            final_stats = self._get_database_stats(final_db_uofs_df)
            self._print_summary(initial_stats, final_stats, 
                               new_uofs_df if add_success else pd.DataFrame(), 
                               obsolete_uofs_df if obsolete_success else pd.DataFrame(), 
                               updates_df, 
                               change_log if log_success else [])

        except Exception as e:
            print(f"\n❌ Error during UOF processing: {e}")
            raise

    def _get_database_stats(self, db_df: pd.DataFrame) -> Dict[str, int]:
        """
        Compute the total, active, and obsolete UOF counts from the provided database DataFrame.
        
        Parameters:
        	db_df (pd.DataFrame): DataFrame containing UOF records, expected to include an 'obsoleta' column indicating obsolete status.
        
        Returns:
        	Dict[str, int]: Dictionary with counts for 'total', 'active', and 'obsolete' UOFs.
        """
        total = len(db_df)
    
        # More explicit way to count active UOFs
        active = 0
        if not db_df.empty and 'obsoleta' in db_df.columns:
            active = len(db_df[db_df['obsoleta'] == False])  # Count where obsoleta is False
        
        obsolete = total - active
        
        return {'total': total, 'active': active, 'obsolete': obsolete}

    def _print_summary(self, initial_stats: Dict[str, int], final_stats: Dict[str, int],
                      new_uofs_df: pd.DataFrame, obsolete_uofs_df: pd.DataFrame,
                      updates_df: pd.DataFrame, change_log: List[Dict]) -> None:
        """
                      Prints a summary of UOF database changes and operations performed during processing.
                      
                      Displays before-and-after statistics for total, active, and obsolete UOFs, as well as counts of new, obsolete, and updated UOFs and the total number of changes logged.
                      """
        print("\n📊 UOF Database Summary (Before -> After):")
        print(f"- Total UOFs: {initial_stats['total']} -> {final_stats['total']}")
        print(f"- Active UOFs: {initial_stats['active']} -> {final_stats['active']}")
        print(f"- Obsolete UOFs: {initial_stats['obsolete']} -> {final_stats['obsolete']}")

        print("\n📊 Summary of operations performed:")
        print(f"- New UOFs added: {len(new_uofs_df)}")
        print(f"- UOFs marked obsolete: {len(obsolete_uofs_df)}")
        print(f"- UOFs updated: {len(updates_df)}")
        print(f"- Total changes logged: {len(change_log)}")

        print("\n✅ UOF processing completed successfully.")
