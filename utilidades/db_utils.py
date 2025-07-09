"""
Utility class for database operations
"""

__all__ = ['DatabaseUtils', 'DuckDBUtils'] #Export the class

from sqlalchemy import create_engine, text
import pandas as pd
from typing import Optional, Union, List
import sys
import os
from pathlib import Path
import pretty_errors
import sqlalchemy
import duckdb
# Get the absolute path to the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from configs.storage_config import DB_URL

class DatabaseUtils:
    """Utility class for database operations. Operations include read, write and update."""
    _engines = {} #cache for engines to avoid creating new ones

    @staticmethod
    def create_engine(database_name: str):
        """Create SQLAlchemy engine for database operations
        
        Args:
            database_name (str): Name of the database
            
        Returns:
            Engine: SQLAlchemy engine object
        """
        if database_name in DatabaseUtils._engines:
            return DatabaseUtils._engines[database_name]
        
        try:
            engine = create_engine(DB_URL(database_name))
            # Test connection
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            DatabaseUtils._engines[database_name] = engine
            return engine
        except Exception as e:
            raise ConnectionError(f"Failed to connect to database {database_name}: {str(e)}")
    
    @staticmethod
    def read_table(engine: object, table_name: str, columns: Optional[List[str]] = None, where_clause: Optional[str] = None) -> pd.DataFrame:
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
            query_str = f"SELECT {select_clause} FROM {table_name}"
            if where_clause:
                query_str += f" WHERE {where_clause}"
            
            # Create a SQLAlchemy TextClause
            query = text(query_str)
            
            # Execute the query and convert to DataFrame
            with engine.connect() as conn:
                result = conn.execute(query)
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
                return df
            
        except Exception as e:
            raise ValueError(f"Error reading table {table_name}: {str(e)}")

    @staticmethod
    def write_table(engine: object, df: pd.DataFrame, table_name: str, 
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
            df.to_sql(table_name, engine, if_exists=if_exists, index=index, method='multi')
        except Exception as e:
            raise ValueError(f"Error writing to table {table_name}: {str(e)}")

    @staticmethod
    def update_table(engine: object, df: pd.DataFrame, table_name: str, key_columns: List[str]) -> None:
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

class DuckDBUtils:
    """
    Utility class for interacting with DuckDB, especially for querying 
    partitioned Parquet datasets.
    """
    def __init__(self, db_path: Optional[str] = None):
        """
        Initializes the DuckDB connection.

        Args:
            db_path (Optional[str]): Path to a persistent DuckDB database file. 
                                     If None, an in-memory database is used.
        """
        self.db_path = db_path
        self.con = self._connect()

    def _connect(self):
        """Establishes the DuckDB connection."""
        print(f"Connecting to DuckDB {'at ' + self.db_path if self.db_path else 'in-memory'}...")
        try:
            # Connect, allowing unsigned extensions like httpfs if needed for remote data
            con = duckdb.connect(database=self.db_path if self.db_path else ':memory:', read_only=False)
            return con
        except Exception as e:
            print(f"Error connecting to DuckDB: {e}")
            raise

    def execute_query(self, query: str, params: Optional[tuple] = None) -> pd.DataFrame:
        """
        Executes a SQL query and returns the result as a Pandas DataFrame.

        Args:
            query (str): The SQL query to execute.
            params (Optional[tuple]): Parameters to bind to the query.

        Returns:
            pd.DataFrame: The result of the query.
        """
        print(f"Executing DuckDB query: {query}")
        if params:
            print(f"With parameters: {params}")
        try:
            return self.con.execute(query, params).fetchdf()
        except Exception as e:
            print(f"Error executing DuckDB query: {e}")
            # Consider more specific error handling if needed
            raise

    def query_partitioned_parquet(
        self,
        base_path: Path,
        start_date: str,
        end_date: str,
        mercados: List[str],
        mercado_ids: Optional[dict[str, List[int]]] = None,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Queries a partitioned Parquet dataset based on date, mercado, and mercado_id.

        Assumes a partition structure like:
        base_path/mercado={mercado_name}/id_mercado={id}/year={YYYY}/month={MM}/.../*.parquet

        Args:
            base_path (Path): The base directory of the partitioned dataset.
            start_date (str): Start date string (YYYY-MM-DD).
            end_date (str): End date string (YYYY-MM-DD).
            mercados (List[str]): List of market names to include.
            mercado_ids (Optional[dict[str, List[int]]]): Dictionary mapping market names 
                                                          to lists of specific IDs to include.
                                                          If None or market not in dict, 
                                                          all IDs for that market are included.
            columns (Optional[List[str]]): Specific columns to select. If None, selects all (*).

        Returns:
            pd.DataFrame: The resulting data.
        """
        select_cols = ", ".join(f'"{c}"' for c in columns) if columns else "*"
        
        # Construct the FROM clause using DuckDB's globbing and partitioning features
        # Adjust the glob pattern based on your exact partition structure (e.g., year/month/day)
        # This example assumes partitioning by year and month within id_mercado
        from_clause = f"read_parquet('{base_path / 'mercado=*' / 'id_mercado=*' / 'year=*' / 'month=*' / '*.parquet'}', hive_partitioning=1)"

        # Build the WHERE clause dynamically
        where_conditions = []

        # Date filtering (assuming a 'fecha' or similar date/timestamp column exists in the Parquet files)
        # IMPORTANT: Replace 'fecha_column_name' with the actual date/timestamp column in your Parquet files
        fecha_column = "fecha_hora" # <<<--- REPLACE THIS if needed
        where_conditions.append(f"CAST(\"{fecha_column}\" AS DATE) >= CAST(? AS DATE)")
        where_conditions.append(f"CAST(\"{fecha_column}\" AS DATE) <= CAST(? AS DATE)")
        params = [start_date, end_date]

        # Mercado filtering (uses the 'mercado' partition column)
        if mercados:
             mercado_placeholders = ', '.join(['?'] * len(mercados))
             where_conditions.append(f"mercado IN ({mercado_placeholders})")
             params.extend(mercados)

        # Mercado ID filtering (uses the 'id_mercado' partition column)
        id_conditions = []
        if mercado_ids:
            for market, ids in mercado_ids.items():
                if market in mercados and ids: # Only filter if the market is selected and IDs are provided
                     id_placeholders = ', '.join(['?'] * len(ids))
                     id_conditions.append(f"(mercado = ? AND id_mercado IN ({id_placeholders}))")
                     params.append(market)
                     params.extend(ids)
        
        # If there are markets specified without specific IDs, ensure they are included
        markets_with_all_ids = [m for m in mercados if not mercado_ids or m not in mercado_ids or not mercado_ids[m]]
        if markets_with_all_ids:
            all_ids_placeholders = ', '.join(['?'] * len(markets_with_all_ids))
            id_conditions.append(f"mercado IN ({all_ids_placeholders})")
            params.extend(markets_with_all_ids)

        if id_conditions:
             where_conditions.append(f"({' OR '.join(id_conditions)})")


        where_clause = " AND ".join(where_conditions)
        
        query = f"SELECT {select_cols} FROM {from_clause} WHERE {where_clause};"
        
        return self.execute_query(query, tuple(params))


    def close(self):
        """Closes the DuckDB connection."""
        if self.con:
            self.con.close()
            print("DuckDB connection closed.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

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
    try:
        # Create engine
        engine = DatabaseUtils.create_engine("pruebas_BT")
        print("Database engine created successfully")
        
        # Test connection
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            print("Database connection test successful")
        
        # Try reading the table
        df = DatabaseUtils.read_table(engine, "Mercados")
        print("Successfully read Mercados table")
        print(df.head())
        
    except Exception as e:
        print(f"Error: {str(e)}")
