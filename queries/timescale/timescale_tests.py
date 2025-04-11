from typing import Optional
import psycopg2
from psycopg2.extensions import connection
import os
from dotenv import load_dotenv, find_dotenv

def connect_to_timescale_db() -> Optional[connection]:
    """
    Establishes a connection to TimescaleDB using environment variables.
    
    Returns:
        Optional[psycopg2.extensions.connection]: A database connection object if successful,
                                                None if connection fails.
    
    Raises:
        psycopg2.Error: If there's an issue connecting to the database
    """
    # Load environment variables
    load_dotenv()
    
    try:
        # Use os.getenv() instead of os.environ[] for safer access
        connection_url = os.getenv("TIMESCALE_SERVICE_URL")
        if not connection_url:
            print("Error: TIMESCALE_SERVICE_URL not found in environment variables")
            return None
            
        conn = psycopg2.connect(dsn=connection_url)
        cursor = conn.cursor() # use cursor to execute queries and interact with db
        
        # Test the connection
        with cursor:
            cursor.execute("SELECT version();")
            version = cursor.fetchone()
            print(f"Successfully connected to TimescaleDB!")
            print(f"Server version: {version[0]}")
            
        return cursor
    
    except psycopg2.Error as e:
        print(f"Error connecting to TimescaleDB: {e}")
        return None
    


# Example usage
if __name__ == "__main__":
    db_connection = connect_to_timescale_db()

    if db_connection:
        # Don't forget to close the connection when done
        db_connection.close()

