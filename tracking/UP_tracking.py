"""
Script to track and update UPs between the ESIOS UP list and the Optimize Energy database
"""
__all__ = ['UPTracker']

import pandas as pd
import sqlalchemy
from pathlib import Path
import sys
import os
# Add the root directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
# Now you can import from config
from config.db_config import DB_URL
import pretty_errors

class UPTracker:
    """
    Class to track and update UPs between the ESIOS UP list and the Optimize Energy database
    """
    def __init__(self):
        """
        Initialize UPTracker with database connection
        
        Args:
            bbdd_name (str): Name of the database to connect to
        """
        self._bbdd_name = None
        self._bbdd_url = None
        self._engine = None
        self.table_name = "up_listado"
        self.change_log_table_name = "up_change_log"

    @property
    def bbdd_name(self):
        return self._bbdd_name
    
    @bbdd_name.setter
    def bbdd_name(self, bbdd_name):
        self._bbdd_name = bbdd_name
        self._bbdd_url = DB_URL(self._bbdd_name)
        self._engine = sqlalchemy.create_engine(self._bbdd_url)

    @property
    def bbdd_url(self):
        return self._bbdd_url
    
    @bbdd_url.setter
    def bbdd_url(self, bbdd_url):
        self._bbdd_url = bbdd_url

    @property
    def engine(self):
        return self._engine
    
    @engine.setter
    def engine(self, engine):
        try:
            self._engine = engine
            with self.engine.connect() as connection:
                pass
            print(f"Successfully connected to database: {self.bbdd_name}")
        except sqlalchemy.exc.SQLAlchemyError as e:
            print(f"Database connection error: {e}")
            raise
    
    def load_csv_ups(self, csv_path: str) -> pd.DataFrame:
        """
        Load and process UPs from CSV file
        
        Args:
            csv_path (str): Path to csv file containing UP data
            
        Returns:
            pd.DataFrame: Processed DataFrame with required columns
            
        Raises:
            FileNotFoundError: If the CSV file does not exist
            pd.errors.EmptyDataError: If the CSV file is empty
            pd.errors.ParserError: If the CSV file cannot be parsed
            ValueError: If required columns are missing
        """
        try:
            # Check if file exists
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV file not found: {csv_path}")
                
            # Read csv file
            df = pd.read_csv(csv_path, sep=';')
            
            # Check if DataFrame is empty
            if df.empty:
                raise pd.errors.EmptyDataError(f"CSV file is empty: {csv_path}")
            
            # Check for required columns
            required_cols = {
                'Código de UP': 'UP',
                'Potencia máxima MW': 'potencia',
                'Tipo de producción': 'tipo_produccion',
                'Zona de Regulación': 'zona_regulacion',
                'Tipo de UP': 'tipo_up' 
            }
            
            missing_cols = [col for col in required_cols.keys() if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns in CSV: {', '.join(missing_cols)}")
            
            # Select and rename required columns
            df = df[list(required_cols.keys())].rename(columns=required_cols)
            
            # Filter for generation type UPs
            df = df[df['tipo_up'] == 'Generación']
            #no es necesario mantener la columna tipo_up
            df = df.drop(columns=['tipo_up'])

            #add obsoleta column with default value 0
            df['obsoleta'] = 0
            
            # Convert potencia to numeric, handling any non-numeric values
            df['potencia'] = pd.to_numeric(df['potencia'].str.replace(',', '.'), errors='coerce')
            
            # Ensure all required columns are present and have the correct data types
            df['UP'] = df['UP'].astype(str)
            df['tipo_produccion'] = df['tipo_produccion'].astype(str)
            df['zona_regulacion'] = df['zona_regulacion'].astype(str)

            # Add date_updated column with current date
            df['date_updated'] = pd.Timestamp.now().strftime('%Y-%m-%d')
            
            print(f"Successfully loaded {len(df)} UPs from CSV file")
            return df
            
        except pd.errors.EmptyDataError as e:
            print(f"Error: CSV file is empty: {csv_path}")
            raise
        except pd.errors.ParserError as e:
            print(f"Error: Could not parse CSV file: {csv_path}")
            print(f"Details: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error loading CSV file: {e}")
            raise
    
    def load_db_ups(self) -> pd.DataFrame:
        """
        Load UPs from database
    
        Returns:
            pd.DataFrame: DataFrame containing UP data from database
            
        Raises:
            sqlalchemy.exc.SQLAlchemyError: If there is a database error
        """
        try:
            #set bbdd name, which sets the bbdd_url and engine with the setter methods
            self.bbdd_name = "energy_tracker"

            query = sqlalchemy.text(f"SELECT * FROM {self.table_name}")

            with self.engine.connect() as connection:
                df = pd.read_sql(query, connection)
                
            if df.empty:
                print(f"Warning: No UPs found in database table {self.table_name}")
            else:
                print(f"Successfully loaded {len(df)} UPs from database table {self.table_name}")
                
            return df
            
        except sqlalchemy.exc.SQLAlchemyError as e:
            print(f"Database error when loading UPs: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error when loading UPs from database: {e}")
            raise
    
    def load_tecnologias(self) -> pd.DataFrame:
        """
        Load technology mapping (tipo tecnolgia/id table) from database
        
        Returns:
            pd.DataFrame: DataFrame containing technology mappings
            
        Raises:
            sqlalchemy.exc.SQLAlchemyError: If there is a database error
            ValueError: If the technology mapping table is empty
        """
        try:
            #set bbdd name, which sets the bbdd_url and engine with the setter methods
            self.bbdd_name = "Optimize_Energy"

            query = sqlalchemy.text("SELECT * FROM Tecnologias_generacion")
            with self.engine.connect() as connection:
                df = pd.read_sql(query, connection)
                
            if df.empty:
                raise ValueError("Technology mapping table is empty")
                
            print(f"Successfully loaded {len(df)} technology mappings from database")
            return df
            
        except sqlalchemy.exc.SQLAlchemyError as e:
            print(f"Database error when loading technology mappings: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error when loading technology mappings: {e}")
            raise
    
    def load_change_log(self) -> pd.DataFrame:
        """
        Load UP historical data from database
        
        Returns:
            pd.DataFrame: DataFrame containing UP historical data
            
        Raises:
            sqlalchemy.exc.SQLAlchemyError: If there is a database error
        """
        try:
            #set bbdd name, which sets the bbdd_url and engine with the setter methods
            self.bbdd_name = "energy_tracker"

            #load the UP_change_log table
            query = sqlalchemy.text(f"SELECT * FROM {self.change_log_table_name}")
            with self.engine.connect() as connection:
                df = pd.read_sql(query, connection)
                
            print(f"Successfully loaded {len(df)} change log entries from database")
            return df
            
        except sqlalchemy.exc.SQLAlchemyError as e:
            print(f"Database error when loading change log: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error when loading change log: {e}")
            raise

    def map_tecnologia_id(self, csv_df: pd.DataFrame, tecnologias_df: pd.DataFrame) -> pd.DataFrame:
        """
        Map tipo_produccion to tecnologia_id using the Tecnologias_generacion table
        
        Args:
            csv_df (pd.DataFrame): DataFrame containing UP data from csv
            tecnologias_df (pd.DataFrame): DataFrame containing technology mappings
            
        Returns:
            pd.DataFrame: DataFrame with tecnologia_id mapped
            
        Raises:
            ValueError: If all technologies cannot be mapped
        """
        try:
            # Create mapping dictionary from tecnologias_df
            #ie: {'Hidroelectrica': 1, 'Termica': 2, 'Nuclear': 3, 'Eolica': 4, etc   
            tecnologia_mapping = dict(zip(tecnologias_df['tecnologia'], tecnologias_df['id']))
            
            # Map tipo_produccion to tecnologia_id, drop tipo_produccion column
            csv_df['tecnologia_id'] = csv_df['tipo_produccion'].map(tecnologia_mapping)
            
            # Log any unmapped technologies
            unmapped = csv_df[csv_df['tecnologia_id'].isna()]['tipo_produccion'].unique() 

            #if unmapped technologies, print warning
            if len(unmapped) > 0:
                raise ValueError(f"\nWarning: someproduction types could not be mapped to technology IDs:")
                
            else: #if no unmapped technologies, drop tipo_produccion column
                csv_df = csv_df.drop(columns=['tipo_produccion'])
                print("All technologies successfully mapped to technology IDs")
            
            return csv_df
            
        except Exception as e:
            print(f"Error mapping technology IDs: {e}")

            if isinstance(e, ValueError): #print unmapped technologies that caused the error
                for tipo in unmapped:
                    print(f"- {tipo}")
            raise
    
    def extract_new_and_obsolete_ups(self, csv_df: pd.DataFrame, db_df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
        """
        Compare UPs between csv and database, identify new and obsolete UPs
        and add new UPs to the database

        Args:
            csv_df (pd.DataFrame): DataFrame from csv
            db_df (pd.DataFrame): DataFrame from database
            
        Returns:
            tuple: (new_ups_df, obsolete_ups)
                - new_ups_df: DataFrame containing new UPs to be added
                - obsolete_ups: List of UPs to be marked as obsolete
        """
        try:
            # Get sets of UPs from both sources
            csv_ups = set(csv_df['UP'])
            db_ups = set(db_df['UP'])
            
            # Find added UPs
            new_ups = csv_ups - db_ups

            # Find obsolete UPs to mark as changed
            total_obsolete_ups = db_ups - csv_ups
            
            # Filter to get only UPs that are not already marked as obsolete
            # This gives us the delta - only the newly obsolete UPs
            already_obsolete = set(db_df[db_df['obsoleta'] == 1]['UP'])
            delta_obsolete_ups = total_obsolete_ups - already_obsolete

            # Create DataFrame for new UPs
            new_ups_df = csv_df[csv_df['UP'].isin(new_ups)].copy()
            
            print(f"Found {len(new_ups)} new UPs and {len(delta_obsolete_ups)} obsolete UPs")
            return new_ups_df, list(delta_obsolete_ups)
            
        except Exception as e:
            print(f"Error extracting new and obsolete UPs: {e}")
            raise
    
    def add_new_ups(self, new_ups_df: pd.DataFrame) -> bool:
        """
        Add new UPs to database
        
        Args:
            new_ups_df (pd.DataFrame): DataFrame containing new UPs to add
            
        Returns:
            bool: True if operation was successful, False if it failed with a duplicate entry error
            
        Raises:
            sqlalchemy.exc.SQLAlchemyError: If there is a database error (except for IntegrityError with duplicate entry)
        """
        try:
            #if there are new ups in the csv file, add them to the database
            if not new_ups_df.empty:
                #set bbdd name, which sets the bbdd_url and engine with the setter methods
                self.bbdd_name = "energy_tracker"

                # Ensure we only include columns that exist in the database table
                columns_to_include = ['UP', 'potencia', 'tecnologia_id', 'zona_regulacion', 'obsoleta', "date_updated"]

                new_ups_df = new_ups_df[columns_to_include]
                new_ups_df['date_updated'] = pd.Timestamp.now().strftime('%Y-%m-%d')
                
                # Save to database
                new_ups_df.to_sql(self.table_name, con=self.engine, if_exists='append', index=False)
                
                print(f"Added new UPs in {self.table_name} in database table")
            else:
                print("-No new UPs to add")
            
            return True
                
        except sqlalchemy.exc.IntegrityError as e:
            if "Duplicate entry" in str(e):
                print(f"Skipping duplicate entry error: {e}")
                return False
            else:
                print(f"Database integrity error when adding new UPs: {e}")
                raise
        except sqlalchemy.exc.SQLAlchemyError as e:
            print(f"Database error when adding new UPs: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error when adding new UPs: {e}")
            raise
    
    def mark_obsolete_ups(self, delta_obsolete_ups: list) -> None:
        """
        Mark obsolete UPs in database
        
        Args:
            obsolete_ups (list): List of UP codes to mark as obsolete
            
        Raises:
            sqlalchemy.exc.SQLAlchemyError: If there is a database error
        """
        try:
            if delta_obsolete_ups:
                #set bbdd name, which sets the bbdd_url and engine with the setter methods
                self.bbdd_name = "energy_tracker"

                # Update obsoleta column and date_updated for obsolete UPs
                current_date = pd.Timestamp.now().strftime('%Y-%m-%d')
                update_query = sqlalchemy.text(
                    f"UPDATE {self.table_name} SET obsoleta = 1, date_updated = :date_updated WHERE UP IN :ups"
                )
                with self.engine.connect() as connection:
                    connection.execute(update_query, {'ups': delta_obsolete_ups, 'date_updated': current_date})
                    connection.commit()
                    
                print(f"Marked UPs as obsolete in {self.table_name} database table")
            else:
                print("-No obsolete UPs to mark")
                
        except sqlalchemy.exc.SQLAlchemyError as e:
            print(f"Database error when marking obsolete UPs: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error when marking obsolete UPs: {e}")
            raise
    
    def check_up_changes(self, csv_df: pd.DataFrame, new_ups_df: pd.DataFrame, delta_obsolete_ups: list, db_df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
        """
        Check for changes in potencia and zona_regulacion fields for existing UPs
        
        Args:
            csv_df (pd.DataFrame): DataFrame from CSV with UPs to check
            new_ups_df (pd.DataFrame): DataFrame with the list of new UPs
            delta_obsolete_ups (list): List of UPs to be marked as obsolete
            db_df (pd.DataFrame): DataFrame from database
            
        Returns:
            tuple: (updated_ups_df, change_log_entries)
                - updated_ups_df: DataFrame containing UPs with updated fields
                - change_log_entries: List of dictionaries with change log entries
        """
        try:
            # Get common UPs that exist in both CSV and database (not new or obsolete)
            common_ups = set(csv_df['UP']).intersection(set(db_df['UP'])) 
            
            # Filter entire df to only include common UPs
            csv_common = csv_df[csv_df['UP'].isin(common_ups)].copy()
            db_common = db_df[db_df['UP'].isin(common_ups)].copy()
            
            # Set UP as index for easier comparison
            csv_common.set_index('UP', inplace=True)
            db_common.set_index('UP', inplace=True)
            
               
            # Initialize lists for updated UPs and change log entries
            updated_ups = [] #list of UPs that have changed
            change_log_entries = [] #list of dictionaries with change log entries
            
            # Get current date for change log
            current_date = pd.Timestamp.now().strftime('%Y-%m-%d')
            
            # Check each common UP for changes in potencia and zona_regulacion
            for up in common_ups:
                #iterate through each up row of both csv and db dfs based on the common UPs
                new = csv_common.loc[up]
                old = db_common.loc[up]
        
                # Skip if new values are null, we maintain the old values, we don't even compare them
                if pd.isna(new['potencia']) or pd.isna(new['zona_regulacion']):
                    continue
                
                # Check if potencia or zona_regulacion has changed, yield true or false series with one value
                potencia_changed = new['potencia'] != old['potencia']
                zona_changed = new['zona_regulacion'] != old['zona_regulacion']
            
                
                if potencia_changed or zona_changed:  #if potencia or zona has changed or there are new ups
                    # Add to updated UPs list
                    updated_ups.append(up)
                    
                    # Create change log entry
                    if potencia_changed:
                        change_log_entries.append({
                            'UP': up,
                            'field_changed': 'potencia',
                            'old_value': old['potencia'],
                            'new_value': new['potencia'],
                            'date_updated': current_date
                        })
                    
                    if zona_changed:
                        change_log_entries.append({
                            'UP': up,
                            'field_changed': 'zona_regulacion',
                            'old_value': old['zona_regulacion'],
                            'new_value': new['zona_regulacion'],
                            'date_updated': current_date
                        })

            if not new_ups_df.empty: #if temp new ups

                #set new ups df as UP as index
                temp_new_ups_df = new_ups_df.copy()
                temp_new_ups_df.set_index('UP', inplace=True)

                for up in temp_new_ups_df.index:
                    change_log_entries.append({
                        'UP': up,
                        'field_changed': 'habilitada',
                        'old_value': False,
                        'new_value': True,
                        'date_updated': current_date
                    })

            if len(delta_obsolete_ups) > 0:
                for up in delta_obsolete_ups:
                    change_log_entries.append({
                        'UP': up,
                        'field_changed': 'obsoleta',
                        'old_value': False,
                        'new_value': True,
                        'date_updated': current_date
                    })
            
            # Create DataFrame with updated UPs
            updated_ups_df = csv_df[csv_df['UP'].isin(updated_ups)].copy()
            return updated_ups_df, change_log_entries
            
        except Exception as e:
            print(f"Error checking for UP changes: {e}")
            raise
    
    def save_change_log(self, change_log_entries: list) -> None:
        """
        Save change log entries to UP_change_log table
        
        Args:
            change_log_entries (list): List of dictionaries with change log entries
            
        Raises:
            sqlalchemy.exc.SQLAlchemyError: If there is a database error
        """
        try:
            if change_log_entries:
                #set bbdd name, which sets the bbdd_url and engine with the setter methods
                self.bbdd_name = "energy_tracker"

                # Convert list of dictionaries to DataFrame
                change_log_df = pd.DataFrame(change_log_entries)
                
                # Save to database
                change_log_df.to_sql(self.change_log_table_name, con=self.engine, if_exists='append', index=False)
                
                # Group by UP to show summary
                up_changes = {}
                for entry in change_log_entries:
                    up = entry['UP'] #get up key in change log entries
                    if up not in up_changes: #if up is not in up_changes
                        up_changes[up] = [] #add up to up_changes
                    up_changes[up].append(f"{entry['field_changed']}: {entry['old_value']} → {entry['new_value']}") #add the change to the up
                
                print("\nChanges by UP:")
                print("-" * 50)
                for up, changes in up_changes.items(): #iterate through up_changes
                    print(f"UP: {up}") #print up
                    for change in changes: #iterate through changes
                        print(f"  - {change}") #print change
                    print()
                print("-" * 50 + "\n")

                #print updated change log df
                for field in change_log_df['field_changed'].unique():
                    print(f"Changes by field: {field}")
                    print(f"Number of changes: {len(change_log_df[change_log_df['field_changed'] == field])}")
                    print()

            else:
                print("-No changes to log")
                
        except sqlalchemy.exc.SQLAlchemyError as e:
            print(f"Database error when saving change log: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error when saving change log: {e}")
            raise
    
    def update_up_changes(self, updated_ups_df: pd.DataFrame) -> None:
        """
        Update existing UPs with new values for potencia and/or zona_regulacion
        
        Args:
            updated_ups_df (pd.DataFrame): DataFrame containing UPs with updated fields
            
        Raises:
            sqlalchemy.exc.SQLAlchemyError: If there is a database error
        """
        try:
            #set bbdd name, which sets the bbdd_url and engine with the setter methods
            self.bbdd_name = "energy_tracker"

            current_date = pd.Timestamp.now().strftime('%Y-%m-%d')

            if not updated_ups_df.empty:
                # Determine which table to use
                
                # Get list of UPs to update
                ups_to_update = updated_ups_df['UP'].tolist()
                
                # Set UP as index for easier access to values
                updated_ups_df.set_index('UP', inplace=True)
                
                # Update each UP in the database
                with self.engine.connect() as connection:
                    for up in ups_to_update:
                        # Get the updated values for this UP
                        row = updated_ups_df.loc[up]
                        
                        # Create and execute UPDATE query
                        update_query = sqlalchemy.text(
                            f"""UPDATE {self.table_name} 
                               SET potencia = :potencia, 
                                   zona_regulacion = :zona_regulacion,
                                   date_updated = :date_updated
                               WHERE UP = :up"""
                        )
                        
                        connection.execute(update_query, {
                            'potencia': row['potencia'],
                            'zona_regulacion': row['zona_regulacion'],
                            'up': up,
                            'date_updated': current_date
                        })
                    
                    # Commit all updates
                    connection.commit()

                print(f"Successfully updated UPs in {self.table_name} database table")
                # print(f"\n=== UPs Updated in {table_name} ===")
                # print("\nList of updated UPs:")
                # print("-" * 20)
                # for i, up in enumerate(ups_to_update, 1):
                #     print(f"{i:2d}. {up}")
                # print("-" * 20 + "\n")
            else:
                print("-No UPs to update")
                
        except sqlalchemy.exc.SQLAlchemyError as e:
            print(f"Database error when updating UP details: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error when updating UP details: {e}")
            raise

    def process_ups(self, csv_path: str) -> None:
        """
        Main method to process and update UPs.
        
        Args:
            csv_path (str): Path to csv file containing UP data
        """
        try:
            
            # Load data from both sources
            print("\nLoading data from CSV file...")
            csv_df = self.load_csv_ups(csv_path)
            
            print("\nLoading data from database...")
            db_df = self.load_db_ups()
            
            print("\nLoading technology mappings...")
            tecnologias_df = self.load_tecnologias()
            
            # Map tecnologia_id
            print("\nMapping technology IDs...")
            csv_df = self.map_tecnologia_id(csv_df, tecnologias_df)
            
            # Compare and get changes
            print("\nIdentifying new and obsolete UPs...")
            new_ups_df, delta_obsolete_ups = self.extract_new_and_obsolete_ups(csv_df, db_df)
            
            # Check for changes in existing UPs for potencia and zona_regulacion
            print("\nChecking for changes in existing UPs...")
            updated_ups_df, change_log_entries = self.check_up_changes(csv_df, new_ups_df, delta_obsolete_ups, db_df)
            
            # Apply changes to database - handle possible errors
            print("\nAdding new UPs to database...")
            add_success = self.add_new_ups(new_ups_df)
            if add_success:
                print("✅ Success")
            else:
                print("⚠️ Skipped due to duplicate entries - continuing with next steps")

            print("\nMarking obsolete UPs in database...")
            try:
                self.mark_obsolete_ups(delta_obsolete_ups)
                print("✅ Success")
            except sqlalchemy.exc.IntegrityError as e:
                if "Duplicate entry" in str(e):
                    print(f"⚠️ Skipped due to duplicate entries - continuing with next steps")
                else:
                    raise

            print("\nUpdating UPs details in database...")
            try:
                self.update_up_changes(updated_ups_df)
                print("✅ Success")
            except sqlalchemy.exc.IntegrityError as e:
                if "Duplicate entry" in str(e):
                    print(f"⚠️ Skipped due to duplicate entries - continuing with next steps")
                else:
                    raise

            print("\nSaving change log to database...")
            self.save_change_log(change_log_entries)
            print("✅ Success")
            
            # Get updated database stats after all changes
            print("\nGetting updated database statistics...")
            updated_db_df = self.load_db_ups()
            total_ups = len(updated_db_df)
            total_obsolete = len(updated_db_df[updated_db_df['obsoleta'] == 1])
            total_active = total_ups - total_obsolete

            total_ups_before = len(db_df)
            total_obsolete_before = len(db_df[db_df['obsoleta'] == 1])
            total_active_before = total_ups_before - total_obsolete_before
            
            # Print summary
            print("\n📊 Summary of UPs in Optimize Energy database: (Before | After)")
            print(f"-Total UPs: {total_ups_before} | {total_ups}")
            print(f"-Total obsolete UPs: {total_obsolete_before} | {total_obsolete}")
            print(f"-Total active UPs: {total_active_before} | {total_active}")

            # Print summary of changes
            print("\n📊 Summary of changes: ")
            print(f"-New UPs added: {len(new_ups_df) if add_success else 0}")
            print(f"-UPs marked as obsolete: {len(delta_obsolete_ups)}")
            print(f"-UPs with changed fields: {len(updated_ups_df)}")
            print(f"-Total changes logged: {len(change_log_entries)}")
            print("\n✅ UP processing completed successfully")
            
        except Exception as e:
            print(f"\n❌ Error during UP processing: {e}")
            print("Process terminated with errors")
            raise

def main():
    """
    Main execution function
    """

    # Example usage
    tracker = UPTracker()
    #use realtive path
    csv_path = os.path.join(os.path.dirname(__file__), "..", "downloads", "tracking", "export_unidades-de-programacion_2025-03-12_16_16.csv")
    tracker.process_ups(csv_path)

if __name__ == "__main__":
    main()
