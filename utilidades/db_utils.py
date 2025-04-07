"""
Utility class for database operations
"""

__all__ = ['DatabaseUtils'] #Export the class

from sqlalchemy import create_engine, text, Engine
import pandas as pd
from typing import Optional, Union, List
import sys
import os
from pathlib import Path
import pretty_errors
import sqlalchemy

# Get the absolute path to the scripts directory
SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(SCRIPTS_DIR))

from configs.config import DB_URL

class DatabaseUtils:
    """Utility class for database operations. Operations include read, write and update."""

    @staticmethod
    def create_engine(database_name: str):
        """Create SQLAlchemy engine for database operations
        
        Args:
            database_name (str): Name of the database
            
        Returns:
            Engine: SQLAlchemy engine object
        """
        try:
            engine = create_engine(DB_URL(database_name))
            # Test connection
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return engine
        except Exception as e:
            raise ConnectionError(f"Failed to connect to database {database_name}: {str(e)}")

    @staticmethod
    def read_table(engine: Engine, table_name: str, columns: Optional[List[str]] = None, where_clause: Optional[str] = None) -> pd.DataFrame:
        """Read data from a table into a DataFrame
        
        Args:
            engine: SQLAlchemy engine
            table_name (str): Name of the table to read ie 'prices'
            columns (List[str], optional): Specific columns to read ie ['column1', 'column2']
            where_clause (str, optional): SQL WHERE clause ie "column1 = 'value1'" or "column1 in ('value1', 'value2')"
            
        Returns:
            pd.DataFrame: Table data
        """
        try:
            # Construct query
            select_clause = "*" if not columns else ", ".join(columns)
            query = f"SELECT {select_clause} FROM {table_name}"
            if where_clause:
                query += f" WHERE {where_clause}"
                
            # Read data
            with engine.connect() as conn:
                df = pd.read_sql(text(query), conn)
                return df
        except Exception as e:
            raise ValueError(f"Error reading table {table_name}: {str(e)}")

    @staticmethod
    def write_table(engine: Engine, df: pd.DataFrame, table_name: str, 
                    if_exists: str = 'append', index: bool = False) -> None:
        """Write DataFrame to database table, this method is used to write data to a table. 
        Args:
            engine: SQLAlchemy engine
            df (pd.DataFrame): Data to write
            table_name (str): Target table name
            if_exists (str): How to behave if table exists ('fail', 'replace', 'append'), default is 'append'
            index (bool): Whether to write DataFrame index, default is False
        """
        try:
            df.to_sql(table_name, engine, if_exists=if_exists, index=index)
        except Exception as e:
            raise ValueError(f"Error writing to table {table_name}: {str(e)}")

    @staticmethod
    def update_table(engine, df: pd.DataFrame, table_name: str, key_columns: List[str]) -> None:
        """
        Update existing records in a table without dropping it
        
        Args:
            engine: SQLAlchemy engine
            df (pd.DataFrame): DataFrame containing update data
            table_name (str): Name of table to update
            key_columns (List[str]): Columns to match for updates
        """
        try:
            with engine.connect() as conn:
                for _, row in df.iterrows():
                    # Build WHERE clause from key columns
                    where_conditions = " AND ".join(
                        f"{col} = :{col}" for col in key_columns
                    )
                    
                    # Build SET clause from non-key columns
                    update_columns = [col for col in df.columns if col not in key_columns]
                    set_clause = ", ".join(
                        f"{col} = :{col}" for col in update_columns
                    )
                    
                    # Create update statement
                    sql = f"UPDATE {table_name} SET {set_clause} WHERE {where_conditions}"
                    
                    # Execute update with parameters from row
                    conn.execute(sqlalchemy.text(sql), row.to_dict())
                
                conn.commit()
                
        except Exception as e:
            raise ValueError(f"Error updating table {table_name}: {str(e)}")
    
def example_usage():
    """Example usage of the DatabaseUtils class"""
    engine = DatabaseUtils.create_engine("example")
    # Example: Updating daily energy prices

    # Initial data in database
    # prices table:
    # date       | hour | price | volume
    # 2024-03-19 | 1    | 45.0  | 100
    # 2024-03-19 | 2    | 46.0  | 200

    # New data to update
    new_data = pd.DataFrame({
        'date': ['2024-03-19', '2024-03-19'],
        'hour': [1, 2],
        'price': [50.0, 51.0]
    })

    # Using write_table with 'replace'
    DatabaseUtils.write_table(engine, new_data, "prices", if_exists='replace')
    # Result: Loses volume data
    # date       | hour | price
    # 2024-03-19 | 1    | 50.0
    # 2024-03-19 | 2    | 51.0

    # Using update_table
    DatabaseUtils.update_table(engine, new_data, "prices", key_columns=['date', 'hour'])
    # Result: Updates prices while preserving volume
    # date       | hour | price | volume
    # 2024-03-19 | 1    | 50.0  | 100
    # 2024-03-19 | 2    | 51.0  | 200                               

if __name__ == "__main__":
    engine = DatabaseUtils.create_engine("pruebas_BT")
    df = DatabaseUtils.read_table(engine, "Mercados")
    print(df)
