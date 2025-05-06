import os
import pytz
import json
import pandas as pd
from datetime import datetime
import google.generativeai as genai
import duckdb
from typing import Dict, Any, Tuple, List
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utilidades.storage_file_utils import ProcessedFileUtils

class NLQueryGenerator:
    """
    A class for generating SQL queries from natural language descriptions of data requirements.
    """
    def __init__(self, api_key: str):
        # Initialize Gemini API
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash")
        
        # DuckDB connection
        self.conn = duckdb.connect(database=':memory:') #in memory database
        self.parquet_path = ProcessedFileUtils().processed_path  #processed base path from storage_file_utils.py
        

        # Load market mapping data
        market_map_path = os.path.join(os.path.dirname(__file__), 'market_map.json')
        with open(market_map_path, 'r') as f:
            self.market_map = json.load(f)
        
        # Timezone settings
        self.madrid_tz = pytz.timezone('Europe/Madrid')
        self.utc_tz = pytz.UTC
        
        # Dataset schema information from data_validation_utils.py
        self.schema_info = {
            'precios': ['datetime_utc', 'precio', 'id_mercado'],
            'volumenes_i90': ['datetime_utc', 'volumenes', 'id_mercado', 'up'],
            'volumenes_i3': ['datetime_utc', 'volumenes', 'id_mercado', 'tecnologia']
        }
        
        # Create market lookup dictionaries for easier reference
        self._create_market_lookups()
        
    def _create_market_lookups(self):
        """Create lookup dictionaries from market_map.json for market information stored as attributes, 
        used to create the system prompt"""

        # Map market names to IDs
        self.market_name_to_id = {}
        # Map market IDs to folder names
        self.market_id_to_folder = {}
        # Map market IDs to available datasets
        self.market_id_to_datasets = {}
        
        for market in self.market_map:
            market_id = market["Market id"]
            market_name = market["Market name BBDD"]
            folder_name = market["Market folder name in processed data lake"]
            
            self.market_name_to_id[market_name.lower()] = market_id
            self.market_id_to_folder[market_id] = folder_name
            
            # Track available datasets for this market
            available_datasets = []
            if market["volumenes_i90"] == "YES":
                available_datasets.append("volumenes_i90")
            if market["volumenes_i3"] == "YES":
                available_datasets.append("volumenes_i3")
            if market["volumenes_omie"] == "YES":
                available_datasets.append("volumenes_omie")
            if market["precios"] == "YES":
                available_datasets.append("precios")
                
            self.market_id_to_datasets[market_id] = available_datasets

    def _get_market_info_string(self):
        """
        Build a string describing available markets, their IDs, folders, and datasets using lookup dictionaries.
        """
        market_info = ""
        for market_name, market_id in self.market_name_to_id.items():
            folder = self.market_id_to_folder[market_id]
            datasets = self.market_id_to_datasets[market_id]
            available = ", ".join(datasets) if datasets else "none"
            market_info += f"- {market_name.title()} (ID: {market_id}, Folder: {folder}): Available datasets: {available}\n"
        return market_info
    
    def _get_available_dataschema_string(self):
        """
        Build a string describing available datasets for each market using lookup dictionaries.
        """
        schema_info = ""
        for dataset_type, columns in self.schema_info.items():
            schema_info += f"- {dataset_type}: {columns}\n"
        return schema_info
    
    def _create_system_prompt(self):
        """
        Create an enhanced system prompt for the Gemini API.
        """
        market_info = self._get_market_info_string()
        schema_info = self._get_available_dataschema_string()

        prompt = f"""
        # DuckDB SQL Query Generator

        You are a specialized SQL query generator for DuckDB that translates natural language queries (in English or Spanish) into precise SQL statements for energy market data analysis.

        ## Available Data Schemas
        
        {schema_info}
        -- Note: 'datetime_utc' column stores timestamps in UTC.
        
        ## Available Markets
        
        The following markets and their IDs are available for querying:
        {market_info}
        
        ## Data Storage Structure
        
        All data is stored in Parquet files with Hive partitioning following this structure:
        - Base path: `{self.parquet_path}/mercado=<market_folder_name>/id_mercado=<market_id>/year=<year>/month=<month>/<dataset_type>.parquet`
        - Example Path Pattern: `{self.parquet_path}/mercado=intra/id_mercado=2/year=2023/month=01/precios.parquet`
        - Use glob patterns (e.g., *) for parts of the path that need to span multiple values, like year or month, unless specific values are required by the query.

        Dataset types include:
        - `precios.parquet` - Price data
        - `volumenes_i90.parquet` - i90 volume data
        - `volumenes_i3.parquet` - i3 volume data
        
        ## Market Name Translation Decision Tree
        
        When translating natural language market references to SQL path structures, follow this decision tree:
        
        1. **Direct Market Name Match**:
        - First check if the query contains an exact match to a market folder name (e.g., "intra", "diario", "secundaria")
        - If found, use that exact market folder name in the `mercado=` parameter
        
        2. **Common Alternative Terms**:
        - Check for common alternative terminologies:
            - "day ahead" or "day-ahead" → "diario" 
            - "intraday" or "intradirio" → "intra"
            - "secondary regulation" → "secundaria"
            - "tertiary regulation" → "terciaria"
    
        3. **Directional Specifications**:
        - Check if direction is specified:
            - "a subir", "upward", "up" → Filter to upward regulation markets
            - "a bajar", "downward", "down" → Filter to downward regulation markets
        
        4. **Ambiguous References**:
        - If the market reference is ambiguous or spans multiple markets:
            - Use wildcards on `id_mercado=*` or `mercado=*` to include all market ids and let filtering occur on datetime or other fields
            - Include a comment explaining why a wildcard approach was chosen to avoid missing any data
        
        5. **No Market Mentioned**:
        - If no market id or market name is mentioned but query intent suggests specific market data:
            - Make sure to include the `mercado=` parameter with your best guess of the market name.
            - Use wildcards on `id_mercado=*` to include all market ids and let filtering occur on datetime or other fields
            - Include a comment explaining why a wildcard approach was chosen to avoid missing any data
        
        ## Critical Rules and Guidelines
        
        1. **Timezone Handling**:
        - User queries reference dates/times in Europe/Madrid timezone.
        - Data is stored with UTC timestamps in the `datetime_utc` column.
        - **Always filter using the `datetime_utc` column with UTC timestamps.**
        - **In the final `SELECT` statement, convert the `datetime_utc` column to 'Europe/Madrid' timezone for display.** Use `datetime_utc AT TIME ZONE 'UTC' AT TIME ZONE 'Europe/Madrid' AS datetime_madrid`.

        2. **Market ID Mapping**:
        - When a market is referenced by name, use its corresponding ID(s).
        - When a market folder name corresponds to multiple market IDs (e.g., "secundaria" might include IDs for "a subir" and "a bajar"), include ALL associated IDs by using `id_mercado=*` in the path pattern unless a specific sub-market is requested.
        - Examples:
            - "intra" → Query all 7 associated market IDs (`id_mercado=*` within `mercado=intra`).
            - "terciaria programada" → Query both "subir" and "bajar" IDs (`id_mercado=*` within `mercado=terciaria`).
            - "secundaria a subir" → Query just the single associated ID (e.g., `id_mercado=14` if that's the ID).
        
        3. **Date Handling**:
        - Include both start and end dates when a range is mentioned.
        - Use DuckDB's date functions for manipulations if needed, but primarily filter on the `datetime_utc` column.
        - Ensure partitioning filters (implicit via `WHERE` on `datetime_utc`) match the query date ranges.

        
        4. **Query Construction**:
        - Generate only valid DuckDB SQL code with no additional explanations.
        - Ensure proper file path construction based on market and data type. Use glob patterns (`*`) in the path for partition keys (like `year`, `month`, `id_mercado`, `mercado`) when the query spans multiple values for that key.
        - Use appropriate joins when retrieving data across multiple tables.
        - Apply efficient filtering by leveraging partition columns (`year`, `month`, `id_mercado`, `mercado`) via the `WHERE` clause on `datetime_utc`.
        - Apply filters on other valid schema columns when relevant via the `WHERE` clause.
        - **Crucially, ALWAYS use `hive_partitioning=true` when reading Parquet files.** DuckDB uses this setting along with the `WHERE` clause to intelligently read only the necessary partition folders/files, even when glob patterns (`*`) are used in the path (this is called filter pushdown).

        ## Error Handling and Validation Guidelines
        
        For each query generation, perform these validation checks:
        
        1. **Date Range Validation**:
        - For multi-year or extended date ranges (>6 months):
            - Consider using CTEs to process data year by year
            - Add a comment warning about potential large data volume
            - Ensure `UNION ALL` operations maintain performance
        
        2. **Missing Information Handling**:
        - If specific date is missing: Default to most recent full month
        - If dataset type is ambiguous: return an error message stating that the dataset type is ambiguous.
        
        3. **Partition Optimization Check**:
        - Verify that `WHERE` clauses align with partition structure
        - For date ranges crossing month/year boundaries, ensure filters match partition patterns
        - Add explicit partition limitations in path when appropriate:
            ```sql
            -- Example: For Jan-Mar 2023 queries
            '{self.parquet_path}/mercado=diario/id_mercado=1/year=2023/month={1|2|3}/precios.parquet'
            ```
        
        4. **Bad Input Detection**:
        - For extremely large date ranges (years), suggest breaking into smaller queries
        - If query combines incompatible data types, explain limitation in comments
        - For timezone edge cases (like DST transitions), handle them gracefully when converting to 'Europe/Madrid' timezone

        ## Query Optimization Guidelines
        
        Apply these DuckDB-specific optimizations:
        
        1. **For Aggregation Queries**:
        - Push aggregations down to scan level when possible
        - Use `GROUP BY ROLLUP` for hierarchical aggregations
        - Apply window functions for running calculations instead of self-joins
        - Example:
            ```sql
            -- Optimized rolling average using window functions
            SELECT
                datetime_utc AT TIME ZONE 'UTC' AT TIME ZONE 'Europe/Madrid' AS datetime_madrid,
                precio,
                AVG(precio) OVER(ORDER BY datetime_utc ROWS BETWEEN 24 PRECEDING AND CURRENT ROW) AS rolling_avg_24h
            FROM read_parquet(...)
            ```
        
        2. **For Join Operations**:
        - Use CTEs for complex multi-table joins
        - Apply join elimination when possible
        - For large datasets, consider hash joins with proper build side
        - Example:
            ```sql
            -- Optimized join pattern using CTEs
            WITH price_data AS (
                SELECT * FROM read_parquet('...precios.parquet', hive_partitioning=true)
                WHERE datetime_utc BETWEEN '2023-01-01' AND '2023-01-31'
            ),
            volume_data AS (
                SELECT * FROM read_parquet('...volumenes_i90.parquet', hive_partitioning=true)
                WHERE datetime_utc BETWEEN '2023-01-01' AND '2023-01-31'
            )
            SELECT 
                p.datetime_utc AT TIME ZONE 'UTC' AT TIME ZONE 'Europe/Madrid' AS datetime_madrid,
                p.precio,
                v.volumen
            FROM price_data p
            JOIN volume_data v ON p.datetime_utc = v.datetime_utc AND p.id_mercado = v.id_mercado
            ```
        
        3. **For Complex Filtering**:
        - Use predicate pushdown techniques
        - Apply subqueries only when necessary
        - Prioritize partition-aligned filters for performance
        - Example:
            ```sql
            -- Optimized filtering with predicate pushdown
            SELECT *
            FROM read_parquet(
                '{self.parquet_path}/mercado=*/id_mercado=*/year=*/month=*/precios.parquet',
                hive_partitioning=true
            )
            WHERE 
                (mercado = 'diario' AND id_mercado = 1)
                OR (mercado = 'intra' AND id_mercado IN (2, 3))
                AND datetime_utc BETWEEN '2023-01-01' AND '2023-12-31'
            ```
        
        ## Example `read_parquet` call (Single Month with Timezone Conversion):
        ```sql
        -- Query for a specific market ID and month, converting time for display
        SELECT
            datetime_utc AT TIME ZONE 'UTC' AT TIME ZONE 'Europe/Madrid' AS datetime_madrid,
            precio,
            id_mercado
        FROM read_parquet(
            '{self.parquet_path}/mercado=diario/id_mercado=1/year=2023/month=01/precios.parquet',
            hive_partitioning=true
        )
        WHERE datetime_utc >= '2023-01-01 00:00:00' -- UTC Filter
        AND datetime_utc < '2023-02-01 00:00:00'; -- UTC Filter
        ```
        
        ## Example `read_parquet` call (Multiple Months/IDs with Timezone Conversion):
        ```sql
        -- Query for 'secundaria' market (potentially multiple IDs) across July-Sept 2024
        SELECT
            datetime_utc AT TIME ZONE 'UTC' AT TIME ZONE 'Europe/Madrid' AS datetime_madrid,
            precio,
            id_mercado -- Keep other relevant columns
        FROM read_parquet(
            '{self.parquet_path}/mercado=secundaria/id_mercado=*/year=2024/month=*/precios.parquet', -- Note the '*' for id_mercado and month
            hive_partitioning = true
        )
        WHERE datetime_utc >= '2024-07-01 00:00:00' -- Start of July 1st, 2024 UTC Filter
        AND datetime_utc < '2024-10-01 00:00:00'; -- Up to, but not including, October 1st, 2024 UTC Filter
        -- DuckDB will efficiently read only relevant partitions based on the UTC WHERE clause.
        -- The final SELECT converts the timestamp for the user.
        ```

        ## Response Guidelines
        
        Return precisely formatted DuckDB SQL that properly handles:
        - Market ID mappings and folder structures (using `*` where appropriate).
        - **Timezone conversions**: Filter using UTC in `WHERE`, convert to 'Europe/Madrid' in final `SELECT`.
        - Date range filtering with partition optimization via the `WHERE` clause on `datetime_utc`.
        - Appropriate dataset selection based on query intent.
        - Always include the `hive_partitioning=true` parameter in `read_parquet` calls.
        
        Users expect accurate, efficient SQL that exactly matches their query intent, including correct timezone representation in the results.
        """
        return prompt

    def translate_to_sql(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """Translate natural language query to SQL using Gemini
        
        Args:
            query (str): The natural language query to translate
            
        Returns:
            Tuple[str, Dict[str, Any]]: A tuple containing the SQL query and context
        """
        system_prompt = self._create_system_prompt()
        
        # Combine with specific query
        full_prompt = system_prompt + "\n\nTranslate this query to SQL:\n" + query

        # Call Gemini API
        response = self.model.generate_content(full_prompt)
        sql_query = response.text.strip()
        
        # Replace placeholder with actual path
        formatted_query = sql_query.replace("{base_path}", f"'{self.parquet_path}'")
        
        # Ensure hive_partitioning=true is included
        if "read_parquet" in formatted_query and "hive_partitioning" not in formatted_query:
            formatted_query = formatted_query.replace(
                "read_parquet(", 
                "read_parquet(", 1
            ).replace("')", "', hive_partitioning=true)", 1)
        
        # Create context with additional metadata
        context = {
            "original_query": query,
            "detected_dataset_type": self._detect_dataset_type(query),
            "timestamp": datetime.now()
        }
        
        return formatted_query, context

    def _detect_dataset_type(self, query: str) -> str:
        """Attempt to detect which dataset type is being queried"""
        query_lower = query.lower()
        
        if "precios" in query_lower or "price" in query_lower:
            return "precios"
        elif "volumenes" in query_lower or "volume" in query_lower:
            if "tecnologias" in query_lower or "technology" in query_lower:
                return "volumenes_i3"
            else:
                return "volumenes_i90"
        
        # Default
        return "unknown"
    
    def execute_query(self, natural_language_query: str) -> Any:
        """Translate and execute the query
        Returns the result of the query or None if an error occurs
        Args:
            natural_language_query (str): The natural language query to translate
            
        Returns:
            Any: The result of the query
        """
        try:
            sql_query, context = self.translate_to_sql(natural_language_query)
            print(f"Executing SQL query: {sql_query}")
            print(f"Context: {context}")
            result = self.conn.execute(sql_query).fetchall()
            return result
        except Exception as e:
            print(f"Error executing query: {e}")
            return None
        
    def result_to_df(self, natural_language_query: str) -> pd.DataFrame:
        """Execute the query and convert the result to a pandas DataFrame
        
        Args:
            natural_language_query (str): The natural language query to translate
            
        Returns:
            pd.DataFrame: The result as a pandas DataFrame
        """
        try:
            sql_query, _ = self.translate_to_sql(natural_language_query)
            # Execute and get the result directly as a DataFrame
            df = self.conn.execute(sql_query).df()
            return df, _
        except Exception as e:
            print(f"Error converting result to DataFrame: {e}")
            return pd.DataFrame()


if __name__ == "__main__":
    # Initialize with your API key and path to parquet files
    query_engine = NLQueryGenerator(
        api_key=os.getenv("GEMINI_API_KEY")
    )
    
    # Example queries
    spanish_query = "Dame los precios del mercado Diario entre el 1 de enero de 2023 y el 5 de enero de 2023"
    english_query = "Show me volumes for UP 'ABO3' in Terciaria a subir market during February 2023"
    
    # Get SQL and execute
    sql_spanish, context_spanish = query_engine.translate_to_sql(spanish_query)
    print(f"Spanish query SQL:\n{sql_spanish}")
    print(f"\nQuery info:\n{context_spanish}")
    
    sql_english, context_english = query_engine.translate_to_sql(english_query)
    print(f"English query SQL:\n{sql_english}")
    print(f"\nQuery info:\n{context_english}")
    
