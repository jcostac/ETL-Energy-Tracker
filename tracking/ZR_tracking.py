"""
Module for tracking and updating Zonas de Regulación from ESIOS and mapping i90 IDs
"""
__all__ = ['ZRTracker']

import pandas as pd
import sqlalchemy
from pathlib import Path
import sys
import os
# Add the root directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utilidades.db_utils import DatabaseUtils, DB_URL
import pretty_errors
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from abc import ABC, abstractmethod
import re  # Import regular expressions module
import requests
import zipfile
from datetime import timedelta




class ZRTracker:
    """
    Class for tracking and updating Zonas de Regulación.
    Handles ESIOS data and maps corresponding i90 IDs from REE BSP data.
    """
    def __init__(self):
        """
        Initialize a ZRTracker instance with default database connection attributes and table names.
        
        Sets up internal variables for database name, URL, SQLAlchemy engine, and the names of the main and change log tables. Does not establish a database connection or require any parameters.
        """
        self._bbdd_name = None
        self._bbdd_url = None
        self._engine = None
        self.table_name = "zr_listado"
        self.change_log_table_name = "zr_change_log"

    @property
    def bbdd_name(self):
        """
        Gets the current database name used by the tracker.
        """
        return self._bbdd_name
    
    @bbdd_name.setter
    def bbdd_name(self, bbdd_name):
        """
        Set the database name and update the corresponding database URL and SQLAlchemy engine.
        """
        self._bbdd_name = bbdd_name
        self._bbdd_url = DB_URL(self._bbdd_name)
        self._engine = sqlalchemy.create_engine(self._bbdd_url)

    @property
    def bbdd_url(self):
        """
        Get the current database URL used for connecting to the database.
        """
        return self._bbdd_url
    
    @bbdd_url.setter
    def bbdd_url(self, bbdd_url):
        """
        Set the database URL for the tracker instance.
        """
        self._bbdd_url = bbdd_url

    @property
    def engine(self):
        """
        Returns the current SQLAlchemy engine instance used for database operations.
        """
        return self._engine
    
    @engine.setter
    def engine(self, engine):
        """
        Sets the SQLAlchemy engine for database operations and tests the connection.
        
        Raises:
            sqlalchemy.exc.SQLAlchemyError: If the database connection fails.
        """
        try:
            self._engine = engine
            with self.engine.connect() as connection:
                pass
            print(f"Successfully connected to database: {self.bbdd_name}")
        except sqlalchemy.exc.SQLAlchemyError as e:
            print(f"Database connection error: {e}")
            raise

    def get_esios_zonas(self, csv_path: str) -> pd.DataFrame:
        """
        Reads an ESIOS UP export CSV file, filters for generation units, and returns a DataFrame with total maximum power per Regulation Zone.
        
        Parameters:
        	csv_path (str): Path to the ESIOS UP export CSV file.
        
        Returns:
        	pd.DataFrame: DataFrame containing 'Zona de Regulación' and the summed 'Potencia máxima MW' for each zone.
        
        Raises:
        	Exception: If the CSV file cannot be processed or contains invalid data.
        """
        try:
            df = pd.read_csv(csv_path, sep=';')
            
            # Filter for generation type and extract required columns
            df = df[df['Tipo de UP'] == 'Generación']
            zonas_df = df[['Zona de Regulación', 'Potencia máxima MW']].copy()
            
            # Convert potencia to float - handling Spanish number format (comma as decimal point), and thousands separator eliminated
            zonas_df['Potencia máxima MW'] = zonas_df['Potencia máxima MW'].apply(
                lambda x: float(x.replace('.', '').replace(',', '.'))
            )

            #round to two decimal places
            zonas_df['Potencia máxima MW'] = zonas_df['Potencia máxima MW'].round(2)
            
            # Group by zona and sum potencia
            zonas_df = zonas_df.groupby('Zona de Regulación')['Potencia máxima MW'].sum().reset_index()

            #strip whitespace from zona de regulacion
            zonas_df['Zona de Regulación'] = zonas_df['Zona de Regulación'].str.strip()
            
            return zonas_df
            
        except Exception as e:
            print(f"Error processing ESIOS CSV file: {e}")
            raise

    def get_i90_mapping(self, bsp_path) -> Dict[str, str]:
        """
        Generate a mapping from ESIOS zone IDs to i90 IDs by reading a BSP file in CSV or Excel format.
        
        Parameters:
            bsp_path (str): Path to the BSP file (CSV or XLSX) containing ESIOS and i90 zone identifiers.
        
        Returns:
            Dict[str, str]: Dictionary mapping ESIOS zone IDs to corresponding i90 IDs.
        
        Raises:
            ValueError: If the file extension is not supported.
            Exception: If there is an error reading the file or processing the data.
        """
        try:
            if bsp_path.endswith('.xlsx'):
                df = pd.read_excel(bsp_path)
            elif bsp_path.endswith('.csv'):
                df = pd.read_csv(bsp_path)
            else:
                raise ValueError(f"Unsupported file extension: {bsp_path}")

            #delete all white spaces in both columns BSP code (i90) & BSP description (ESIOS)
            df['esios_ZR_id'] = df['Descripción corta BSP-aFRR'].str.strip()
            df['i90_ZR_id'] = df['Código BSP-aFRR'].str.strip()

            # Create mapping id esios to i90
            mapping = dict(zip(df['esios_ZR_id'], df['i90_ZR_id']))
            return mapping
            
        except Exception as e:
            print(f"Error creating i90 mapping: {e}")
            print(f"Current working directory: {os.getcwd()}")
            raise

    def identify_changes(self, esios_df: pd.DataFrame, db_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict]]:
        """
        Compares current ESIOS Regulation Zones with database records to identify new zones, obsolete zones, and changes in maximum power (potencia).
        
        Parameters:
        	esios_df (pd.DataFrame): DataFrame containing current ESIOS Regulation Zones and their maximum power.
        	db_df (pd.DataFrame): DataFrame containing existing Regulation Zones from the database.
        
        Returns:
        	new_zones_df (pd.DataFrame): DataFrame of zones present in ESIOS but not in the database.
        	obsolete_zones_df (pd.DataFrame): DataFrame of zones that are active in the database but no longer present in ESIOS.
        	change_log (List[Dict]): List of dictionaries describing changes in maximum power for zones present in both sources.
        	num_active (int): Number of active (non-obsolete) zones in the database before update.
        	num_total (int): Total number of zones in the database before update.
        """
        try:
            # Get sets of active zones
            # Strip whitespace from ESIOS zones to ensure proper comparison
            esios_zones = set(zone.strip() for zone in esios_df['Zona de Regulación'])

            # Check if db_df is empty
            if db_df.empty: #handle first run where table in db will be empty
                db_active = set()  # No active zones if the database is empty
                db_all_zones = set()  # No zones at all in the database
            else:
                # Get active zones from database where obsoleta is not 1 (not obsolete)
                db_active = set(db_df[db_df['obsoleta'] != 1]['esios_id'].dropna().unique())
                # Get all zones from database regardless of obsoleta status
                db_all_zones = set(db_df['esios_id'].dropna().unique())
            
            # Identify new and obsolete zones
            # New zones should be those that don't exist in the database at all (not just active ones)
            new_zones = esios_zones - db_all_zones
            obsolete_zones = db_active - esios_zones 
            
            # Create DataFrames
            new_zones_df = esios_df[esios_df['Zona de Regulación'].isin(new_zones)]
            obsolete_zones_df = pd.DataFrame({'esios_id': list(obsolete_zones)})
            
            # Track changes in existing zones
            change_log = []
            current_date = datetime.now().strftime('%Y-%m-%d')
            
            # Find common zones (existing in both ESIOS and database and not obsolete)
            common_active_zones = esios_zones.intersection(db_active)
            
            # Check potencia changes for existing zones
            for zone in common_active_zones:
                new_potencia = esios_df[esios_df['Zona de Regulación'] == zone]['Potencia máxima MW'].iloc[0]
                old_potencia = db_df[(db_df['esios_id'] == zone) & (db_df['obsoleta'] != 1)]['potencia'].iloc[0]
                
                # Compare rounded values to two decimals
                if round(new_potencia, 2) != round(old_potencia, 2):
                    change_log.append({
                        'ZR': zone,
                        'field_changed': 'potencia',
                        'old_value': round(old_potencia, 2),
                        'new_value': round(new_potencia, 2),
                        'date_updated': current_date
                    })

            num_potencia_changes = len(change_log)
            print(f"-Number of potencia changes detected: {num_potencia_changes}")
            if change_log:
                print("\nDetailed potencia changes by ZR:")
                for change in change_log:
                    print(f"  - ZR: {change['ZR']}, Old: {change['old_value']} MW, New: {change['new_value']} MW")
            
            return new_zones_df, obsolete_zones_df, change_log, len(db_active), len(db_all_zones)
            
        except Exception as e:
            print(f"Error identifying changes: {e}")
            raise

    def load_db_zonas(self) -> pd.DataFrame:
        """
        Load all Regulation Zones from the database table.
        
        Returns:
            pd.DataFrame: DataFrame containing all Regulation Zones records from the database.
        
        Raises:
            Exception: If an error occurs while accessing or reading from the database.
        """
        try:
            self.bbdd_name = "energy_tracker"
            engine = DatabaseUtils.create_engine(self.bbdd_name)
            db_df = DatabaseUtils.read_table(
                engine=engine,
                table_name=self.table_name
            )
            engine.dispose()
            if db_df.empty:
                print(f"No Zonas de Regulación found in database table (the table is empty)")
            else:
                print(f"Successfully loaded {len(db_df)} Zonas de Regulación from database")
                
            return db_df
            
        except Exception as e:
            print(f"Error when loading Zonas de Regulación from database: {e}")
            raise

    def update_missing_i90_mappings(self, i90_mapping: Dict[str, str]) -> List[Dict]:
        """
        Update missing i90 IDs for existing, non-obsolete zones in the database using the provided mapping.
        
        Checks for zones without an i90 ID, updates them if a mapping is available, and records each change. Handles duplicate entry errors gracefully by skipping affected updates.
        
        Parameters:
            i90_mapping (Dict[str, str]): Mapping from ESIOS zone IDs to i90 IDs.
        
        Returns:
            List[Dict]: A list of dictionaries describing each i90 ID update performed.
        """
        try:
            self.bbdd_name = "energy_tracker"
            engine = DatabaseUtils.create_engine(self.bbdd_name)
            missing_i90_df = DatabaseUtils.read_table(
                engine=engine,
                table_name=self.table_name,
                where_clause="i90_id IS NULL AND obsoleta = FALSE"
            )
            change_log = []
            current_date = datetime.now().strftime('%Y-%m-%d')
            if not missing_i90_df.empty:
                print(f"Found {len(missing_i90_df)} zone(s) with missing i90 IDs")
                for _, row in missing_i90_df.iterrows():
                    print(f"  - Zone: {row['esios_id']} missing i90 ID")
                updates = []
                for _, row in missing_i90_df.iterrows():
                    esios_id = row['esios_id']
                    if esios_id in i90_mapping:
                        updates.append({
                            'esios_id': esios_id,
                            'i90_id': i90_mapping[esios_id]
                        })
                        change_log.append({
                            'ZR': esios_id,
                            'field_changed': 'i90_id',
                            'old_value': None,
                            'new_value': i90_mapping[esios_id],
                            'date_updated': current_date
                        })
                if updates:
                    update_df = pd.DataFrame(updates)
                    try:
                        DatabaseUtils.update_table(
                            engine=engine,
                            df=update_df,
                            table_name=self.table_name,
                            key_columns=['esios_id']
                        )
                        print(f"Updated i90 IDs for {len(updates)} zones")
                    except sqlalchemy.exc.IntegrityError as e:
                        if "Duplicate entry" in str(e):
                            print(f"Skipping duplicate entry error when updating i90 IDs: {e}")
                            print("⚠️ Continuing with next steps")
                        else:
                            raise
            engine.dispose()
            if change_log:
                print("\nChanges made to i90 mappings:")
                for change in change_log:
                    old_value = change['old_value'] if change['old_value'] is not None else 'None'
                    print(f"  - Zone: {change['ZR']}, {change['field_changed']} changed from {old_value} to {change['new_value']}")
            else:
                print("No i90 mapping changes were made")
            return change_log
        except Exception as e:
            print(f"Error updating missing i90 mappings: {e}")
            raise

    def update_database(self, new_zones_df: pd.DataFrame, obsolete_zones_df: pd.DataFrame, i90_mapping: Dict[str, str], change_log: List[Dict]) -> List[Dict]:
        """
        Update the database with new zones, mark obsolete zones, update power values, and fill missing i90 mappings.
        
        This method inserts new regulation zones, marks obsolete zones as such, updates power ("potencia") values for existing zones with changes, and attempts to fill in missing i90 IDs using the provided mapping. All changes are appended to the change log. Duplicate entry errors during database operations are handled gracefully and skipped.
        
        Parameters:
            new_zones_df (pd.DataFrame): DataFrame of new zones to add.
            obsolete_zones_df (pd.DataFrame): DataFrame of zones to mark as obsolete.
            i90_mapping (Dict[str, str]): Mapping from ESIOS zone IDs to i90 IDs.
            change_log (List[Dict]): List of change log entries to append to.
        
        Returns:
            List[Dict]: The updated change log including all modifications made during the update.
        """
        try:
            self.bbdd_name = "energy_tracker"
            
            current_date = datetime.now().strftime('%Y-%m-%d')
            
            # Handle new zones
            if not new_zones_df.empty:
                # First check which zones actually don't exist in the database
                existing_zones = DatabaseUtils.read_table(
                    engine=self.engine,
                    table_name=self.table_name,
                    columns=['esios_id']
                )
                existing_zone_ids = set(existing_zones['esios_id']) if not existing_zones.empty else set()
                
                # Filter out zones that already exist
                new_zones_df = new_zones_df[~new_zones_df['Zona de Regulación'].isin(existing_zone_ids)]
                
                if not new_zones_df.empty:
                    new_zones_df = pd.DataFrame({
                        'esios_id': new_zones_df['Zona de Regulación'].str.strip(),
                        'i90_id': [i90_mapping.get(esios_id.strip()) 
                                  for esios_id in new_zones_df['Zona de Regulación']],
                        'obsoleta': False,
                        'potencia': new_zones_df['Potencia máxima MW']
                    })
                    
                    # Add change log entries for new zones
                    for _, row in new_zones_df.iterrows():
                        change_log.append({
                            'ZR': row['esios_id'],
                            'field_changed': 'habilitada',
                            'old_value': False,
                            'new_value': True,
                            'date_updated': current_date
                        })
                    
                    # Use append mode for new zones with integrity error handling
                    try:
                        DatabaseUtils.write_table(self.engine, new_zones_df, self.table_name, if_exists='append')
                    except sqlalchemy.exc.IntegrityError as e:
                        if "Duplicate entry" in str(e):
                            print(f"Skipping duplicate entry error when adding new zones: {e}")
                            print("⚠️ Continuing with next steps")
                        else:
                            raise
                
            # Handle obsolete zones using update_table
            if not obsolete_zones_df.empty:
                # Prepare update DataFrame with obsoleta column
                update_df = pd.DataFrame({
                    'esios_id': obsolete_zones_df['esios_id'],
                    'obsoleta': True
                })
                
                # Use update_table method with integrity error handling
                try:
                    DatabaseUtils.update_table(
                        engine=self.engine,
                        df=update_df,
                        table_name=self.table_name,
                        key_columns=['esios_id']
                    )
                except sqlalchemy.exc.IntegrityError as e:
                    if "Duplicate entry" in str(e):
                        print(f"Skipping duplicate entry error when marking obsolete zones: {e}")
                        print("⚠️ Continuing with next steps")
                    else:
                        raise
            
            # Update potencia for zones with changes
            if change_log:
                potencia_updates = []
                for change in change_log:
                    if change['field_changed'] == 'potencia':
                        # Always round to two decimal places before updating
                        rounded_potencia = round(change['new_value'], 2)
                        potencia_updates.append({
                            'esios_id': change['ZR'],
                            'potencia': rounded_potencia
                        })
                if potencia_updates:
                    update_df = pd.DataFrame(potencia_updates)
                    try:
                        DatabaseUtils.update_table(
                            engine=self.engine,
                            df=update_df,
                            table_name=self.table_name,
                            key_columns=['esios_id']
                        )
                        print(f"Updated potencia for {len(potencia_updates)} zones (rounded to 2 decimals)")
                    except sqlalchemy.exc.IntegrityError as e:
                        if "Duplicate entry" in str(e):
                            print(f"Skipping duplicate entry error when updating potencia: {e}")
                            print("⚠️ Continuing with next steps")
                        else:
                            raise
            
            # Check for and update any missing i90 mappings
            print("\nChecking for missing i90 mappings...")
            i90_updates = self.update_missing_i90_mappings(i90_mapping)
            if i90_updates:
                print(f"Updated {len(i90_updates)} missing i90 mappings")
                change_log.extend(i90_updates)
            
            return change_log
                
        except Exception as e:
            print(f"Error updating database: {e}")
            raise

    def save_change_log(self, change_log: List[Dict]) -> None:
        """
        Save the provided change log entries to the database table for change logs.
        
        If the change log is not empty, converts the list of change entries to a DataFrame and writes it to the database, handling duplicate entry errors by skipping them. Prints a summary of changes by zone, including the type of change and the date. Raises exceptions on other errors.
        """
        try:
            self.bbdd_name = "energy_tracker"
            
            if change_log:
                # Convert change log dict to DataFrame
                log_df = pd.DataFrame(change_log)
                
                # Save to database with integrity error handling
                try:
                    DatabaseUtils.write_table(self.engine, log_df, self.change_log_table_name, index=False)
                    
                    # Print summary of changes
                    print("\nChanges by Zone:")
                    for zr in log_df['ZR'].unique():
                        zr_changes = log_df[log_df['ZR'] == zr]
                        print(f"\nZone: {zr}")
                        
                        #for every row corresponding to the particular zone, print changes that apply and date of change
                        for _, change in zr_changes.iterrows():
                            if change['field_changed'] == 'habilitada':
                                print(f"  - New zone added")
                            elif change['field_changed'] == 'obsoleta':
                                print(f"  - Marked as obsolete")
                            else:
                                print(f"  - {change['field_changed']}: {change['old_value']} → {change['new_value']}")
                            
                            #print date of change 
                            print(f"    Date: {change['date_updated']}")
                except sqlalchemy.exc.IntegrityError as e:
                    if "Duplicate entry" in str(e):
                        print(f"Skipping duplicate entry error when saving change log: {e}")
                        print("⚠️ Continuing with next steps")
                    else:
                        raise 
            else:
                print("No changes to log")
                
        except Exception as e:
            print(f"Error saving change log: {e}")
            raise

    def process_zonas(self, esios_csv_path: str, bsp_csv_path: str) -> None:
        """
        Synchronizes Regulation Zones data from ESIOS and BSP files with the database.
        
        Loads ESIOS data and i90 mappings, compares them with the current database state to identify new, obsolete, and updated zones, applies all necessary updates, saves a detailed change log, and prints summary statistics. Raises an exception if any processing step fails.
        """
        try:
            # Load current data
            print("\n1. Loading ESIOS data...")
            esios_df = self.get_esios_zonas(esios_csv_path)
            
            print("\n2. Loading i90 mappings from BSP file...")
            i90_mapping = self.get_i90_mapping(bsp_csv_path)
            
            print("\n3. Loading current database state...")
            db_df = self.load_db_zonas()
            
            # Identify changes
            print("\n4. Identifying changes in new and obsolete zones...")
            print(f"Identifying changes in potencia of existing zones...")
            new_zones_df, obsolete_zones_df, change_log, num_active_zones_before, num_all_zones_before = self.identify_changes(esios_df, db_df)

            
            # Update database and save changes
            print("\n5. Updating database...")
            change_log_updated = self.update_database(new_zones_df, obsolete_zones_df, i90_mapping, change_log)
            
            print("\n6. Saving change log...")
            self.save_change_log(change_log_updated)
            
            # Get updated database stats after all changes
            updated_db_df = self.load_db_zonas()
            total_zones_after = len(updated_db_df)
            total_obsolete_after = len(updated_db_df[updated_db_df['obsoleta'] == 1])
            total_active_after = total_zones_after - total_obsolete_after
            
            # Print summary
            print("\n📊Summary of Zonas de Regulación in database: (Before | After)")
            print(f"- Total zones: {num_all_zones_before} | {total_zones_after}")
            print(f"- Total obsolete zones: {num_all_zones_before - num_active_zones_before} | {total_obsolete_after}")
            print(f"- Total active zones: {num_active_zones_before} | {total_active_after}")
            
            # Print summary of changes
            print("\n📊 Summary of changes:")
            print(f"- New zones added: {len(new_zones_df)}")
            print(f"- Zones marked as obsolete: {len(obsolete_zones_df)}")
            print(f"- Total changes logged: {len(change_log_updated)}")
            print("\n✅ Processing completed successfully")
            
        except Exception as e:
            print(f"\n❌ Error during processing: {e}")
            raise

