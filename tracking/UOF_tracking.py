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
        self.bbdd_name = bbdd_name
        self._engine = None
        self.uof_table_name = "uof_listado"
        self.change_log_table_name = "uof_change_log"

    @property
    def engine(self):
        if self._engine is None:
            self._engine = DatabaseUtils.create_engine(self.bbdd_name)
        return self._engine
    
    @engine.setter
    def engine(self, engine):
        self._engine = engine

    def load_uofs(self) -> pd.DataFrame:
        """Load UOFs from database"""
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
        """Save new UOFs to database
        
        Returns:
            bool: True if operation was successful, False if it failed with a duplicate entry error
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
        """Mark UOFs as obsolete in database
        
        Returns:
            bool: True if operation was successful, False if it failed with a duplicate entry error
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
        """Save change log entries to database
        
        Returns:
            bool: True if operation was successful, False if it failed with a duplicate entry error
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
        """Process and format change log entries"""
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
        """Print summary of logged changes"""
        print("\nSummary of logged changes:")
        for _, row in log_df.iterrows():
            print(f"  - UOF: {row['UOF']}, Field: {row['field_changed']}, Old: {row['old_value']}, New: {row['new_value']}, Date: {row['date_updated']}")

class UOFChangeDetector:
    """Handles detection of changes in UOFs"""
    def __init__(self):
        self.current_date_str = datetime.now().strftime('%Y-%m-%d')

    def identify_changes(self, omie_uofs_df: pd.DataFrame, db_uofs_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[Dict]]:
        """Identify all types of changes in UOFs"""
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
        """Prepare dataframes for comparison. Set UOF as index, but keep it as column"""
        if not omie_df.empty:
            omie_df = omie_df.set_index('UOF', drop=False)
        if not db_df.empty:
            db_df = db_df.set_index('UOF', drop=False)
        return omie_df, db_df

    def _identify_new_uofs(self, omie_df: pd.DataFrame, omie_ids: set, db_ids: set, change_log: List[Dict]) -> Tuple[pd.DataFrame, List[Dict]]:
        """Identify new UOFs"""
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
        """Identify obsolete UOFs"""
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
        """Identify updates to existing UOFs"""
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
        self.db_manager = UOFDatabaseManager(bbdd_name)
        self.change_detector = UOFChangeDetector()

    def _filter_omie_df(self, omie_df: pd.DataFrame) -> pd.DataFrame:
        """Filter OMIE UOF DataFrame"""
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
        """Main method to process and update UOFs"""
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
        """Get database statistics"""
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
        """Print summary of changes"""
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

def main():
    """
    Main execution function for UOF tracking.
    This function now simulates the daemon's pre-processing steps.
    """
    print("--- UOF Tracker Script ---")
    # Resolve the path properly using abspath
    download_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "downloads", "tracking"))
    download_uofs_from_omie(download_dir)
    tracker = UOFTracker()
    tracker.process_uofs(omie_path=os.path.join(download_dir, "listado_unidades.xlsx"))
   
   

if __name__ == "__main__":
    main()
