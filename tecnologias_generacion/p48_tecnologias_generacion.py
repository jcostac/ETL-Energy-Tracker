import sys
import os
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utilidades.processed_file_utils import ProcessedFileUtils
from utilidades.db_utils import DatabaseUtils

class P48TecnologiasGeneracion:
    def __init__(self):
        self.processed_file_utils = ProcessedFileUtils()
        self.mercado = "p48"
        self.dataset_type = "volumenes_i3"
        self.table_name = "tecnologias_generacion"
        self.db_name = "energy_tracker"

    def read_latest_p48_file(self) -> pd.DataFrame:
        """
        Reads the latest available P48 file from the I3 source and returns the DataFrame.
        """
        return self.processed_file_utils.read_latest_processed_file(self.mercado, self.dataset_type)

    def get_unique_p48_conceptos(self) -> set:
        """
        Reads the latest available P48 file from the I3 source and returns the unique values from the 'concepto' column.
        Returns:
            set: A set of unique 'concepto' values, or an empty set if no data is found.
        """
      
        try:
            raw_df = self.read_latest_p48_file()

            if raw_df is None or raw_df.empty:
                raise ValueError("No P48 data found.")
                
            if "Concepto" in raw_df.columns:
                unique_conceptos = set(raw_df["Concepto"].dropna().unique())
                print(f"Unique 'concepto' values: {unique_conceptos}")
                return unique_conceptos
            else:
                raise ValueError("No 'Concepto' column found in the latest P48 file.")

        except Exception as e:
            print(f"Error reading latest P48 file: {e}")
            raise e
    
    def update_tecnologias_generacion(self, p48_tecnologias: set) -> bool:
        """
        Updates the tecnologias de generacion in the database.
        Inserts new tecnologias found in the latest P48 file into energy_tracker.tecnologias_generacion
        if they do not already exist in the database.
        """

        concepto_column = "tecnologia"

        try:
            # Get unique conceptos from latest P48 file
            if not p48_tecnologias:
                raise ValueError("p48_tecnologias is empty")

            #engine var in case of error so finally block can execute gracefully 
            engine = None

            # Read existing tecnologias from the database
            try:
                 # Connect to database
                engine = DatabaseUtils.create_engine(self.db_name)
                existing_df = DatabaseUtils.read_table(engine, self.table_name, columns=[concepto_column])
                existing_tecnologias = set(existing_df[concepto_column].dropna().unique())
            
            except Exception as e:
                raise e

            # Find new tecnologias to insert
            new_tecnologias = p48_tecnologias - existing_tecnologias
            if not new_tecnologias:
                print("No new tecnologias to insert. Database is up to date.")
                return

            # Prepare DataFrame for insertion
            insert_df = pd.DataFrame({concepto_column: list(new_tecnologias)})

            # Check for duplicates in new_tecnologias
            duplicates = new_tecnologias.intersection(existing_tecnologias)
            if duplicates:
                print(f"Duplicate tecnologias found: {duplicates}")
                new_tecnologias = new_tecnologias - duplicates  # Remove duplicates from new technologies

            # Insert new tecnologias
            DatabaseUtils.write_table(engine, insert_df, self.table_name, if_exists='append', index=False)
    
            #check if insert was successful
            updated_df = DatabaseUtils.read_table(engine, self.table_name, columns=[concepto_column])
            updated_tecnologias = set(updated_df[concepto_column].dropna().unique())
            actual_length = len(updated_tecnologias)
            
            
            expected_length = len(new_tecnologias) + len(existing_tecnologias)
            
            if actual_length != expected_length:
                raise ValueError("New tecnologias were not inserted into the database.")
            else:
                print(f"Inserted {len(insert_df)} new tecnologias into {self.table_name}.")
                return True

        except Exception as e:
            print(f"Error updating tecnologias_generacion: {e}")
            raise



        finally:
            if engine:
                engine.dispose()
