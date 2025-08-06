import pandas as pd
import os
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Optional
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utilidades.storage_file_utils import StorageFileUtils

class ProcessedFileUtils(StorageFileUtils):
    """
    Utility class for processing and saving processed parquet files.
    """
    def __init__(self) -> None:
        """
        Initialize the ProcessedFileUtils with the processed data path and default row group size for parquet writing.
        """
        super().__init__()
        self.processed_path = self.base_path / 'processed'
        self.row_group_size = 122880 # Example: 128k rows per group

    @staticmethod
    def drop_processed_duplicates(df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """
        Remove duplicate rows from the DataFrame according to dataset-specific columns for processed data.
        
        Duplicates are dropped based on a subset of columns determined by the dataset type. For 'volumenes_mic', duplicates are not removed. Raises a ValueError if the DataFrame is empty, and a KeyError if required columns are missing.
        
        Parameters:
            df (pd.DataFrame): The DataFrame to deduplicate.
            dataset_type (str): The type of dataset, which determines the deduplication logic.
        
        Returns:
            pd.DataFrame: The deduplicated DataFrame.
        """
        if df.empty:
            raise ValueError("Cannot process empty DataFrame")

        #print duplicate number before dropping
        print(f"Number of duplicates before dropping: {df.duplicated().sum()}")
        print(f"Columns in DataFrame: {df.columns.tolist()}")
        print(f"Dataset type: {dataset_type}")

        # Then handle subset-based duplicates
        try:
            if dataset_type == 'volumenes_i90':
                subset_cols = ['datetime_utc', 'volumenes', "up", "id_mercado"]
                if "tipo_transaccion" in df.columns: #for mercado diario 
                    subset_cols.append("tipo_transaccion")
                df = df.drop_duplicates(subset=subset_cols, keep='last')
            elif dataset_type == 'volumenes_i3':
                df = df.drop_duplicates(subset=['datetime_utc', 'volumenes', "tecnologia", "id_mercado"], keep='last')
            elif dataset_type == 'precios_i90':
                df = df.drop_duplicates(subset=['datetime_utc', 'precio', "id_mercado", "up"], keep='last')
            elif dataset_type == 'precios_esios':
                df = df.drop_duplicates(subset=['datetime_utc', 'precio', "id_mercado"], keep='last')
            elif dataset_type == 'volumenes_omie':
                df = df.drop_duplicates(subset=['datetime_utc', 'volumenes', "uof", "id_mercado"], keep='last')
            elif dataset_type == 'curtailments_i90':
                df = df.drop_duplicates(subset=['datetime_utc', 'volumenes', "up", "RTx", "tipo", "volumenes"], keep='last')
            elif dataset_type == 'curtailments_i3':
                df = df.drop_duplicates(subset=['datetime_utc', 'volumenes', "tecnologia", "RTx", "tipo", "volumenes"], keep='last')
            else:
                #we allow duplicates for mic market data, so we don't drop any duplicates
                return df
            
        except KeyError as e:
            raise KeyError(f"Missing required column for deduplication: {str(e)}")
        except Exception as e:
            raise Exception(f"Error during deduplication: {str(e)}")
        
        return df
    
    def _add_partition_cols(self, df: pd.DataFrame, mercado: str) -> pd.DataFrame:
        """
        Add 'year', 'month', and 'mercado' columns to the DataFrame for partitioning.
        
        The 'year' and 'month' columns are extracted from the 'datetime_utc' column, and 'mercado' is set to the provided value.
        
        Returns:
            pd.DataFrame: DataFrame with added partition columns.
        """
        df['year'] = df['datetime_utc'].dt.year
        df['month'] = df['datetime_utc'].dt.month
        df['mercado'] = mercado

        return df
        
    def _prepare_input_dataframe(self, df: pd.DataFrame, mercado: str) -> pd.DataFrame:
        """
        Prepare a DataFrame for partitioned storage by adding partition columns and converting the `datetime_utc` column to timezone-naive.
        
        Parameters:
            df (pd.DataFrame): Input DataFrame to be prepared.
            mercado (str): Market identifier used for partitioning.
        
        Returns:
            pd.DataFrame: DataFrame with added partition columns and a timezone-naive `datetime_utc` column.
        """
        # Add partition columns
        df = self._add_partition_cols(df, mercado)
        # Ensure datetime_utc is timezone-naive
        df = self._ensure_datetime_utc_naive(df)
        return df

    def _validate_partition_columns(self, df: pd.DataFrame, partition_cols: list) -> bool:
        """
        Check if all specified partition columns are present in the DataFrame.
        
        Returns:
            True if all required partition columns exist; False otherwise.
        """
        missing_cols = [col for col in partition_cols if col not in df.columns]
        if missing_cols:
            print(f"[ERROR] Input DataFrame is missing required partition columns: {missing_cols}. Cannot proceed.")
            return False
        return True

    def _process_single_partition(self, df: pd.DataFrame, partition, partition_cols: list, value_cols: list[str], dataset_type: str) -> None:
        """
        Processes and writes data for a single partition, combining new and existing records and ensuring deduplication.
        
        Filters the input DataFrame for the specified partition, reads any existing partition file, merges and deduplicates the data, and writes the result to the appropriate partitioned Parquet file. Raises exceptions if errors occur during processing.
        """
        # Filter new data for this partition
        new_data_partition_df = self._filter_df_for_partition(df, partition, partition_cols)
        
        if new_data_partition_df.empty:
            print(f"[DEBUG] No new data for partition: {partition.to_dict()}. Checking for existing file.")
        
        # Build file path and check for existing file
        output_file_path = self._build_partition_path(partition, partition_cols, dataset_type)
        
        # Read existing data if file exists
        existing_data_df = self._read_existing_partition_file(output_file_path, partition)
        
        # Combine, deduplicate, and write data
        self._combine_and_write_partition(
            new_data_df=new_data_partition_df,
            existing_data_df=existing_data_df,
            partition=partition,
            partition_cols=partition_cols,
            output_file_path=output_file_path,
            value_cols=value_cols,
            dataset_type=dataset_type
        )

    def _filter_df_for_partition(self, df: pd.DataFrame, partition, partition_cols: list) -> pd.DataFrame:
        """
        Return a DataFrame containing only the rows matching the specified partition values.
        
        Parameters:
            df (pd.DataFrame): The DataFrame to filter.
            partition: Partition values as a Series or dict.
            partition_cols (list): Names of columns to use for partition filtering.
        
        Returns:
            pd.DataFrame: A copy of the DataFrame filtered to rows matching all partition column values.
        """
        # Start with a True mask that will be narrowed down with each partition column
        partition_mask = True
        
        # For each partition column, add a condition to the mask
        # This builds a compound boolean mask that matches all partition criteria
        for col in partition_cols:
            partition_mask &= (df[col] == partition[col])
            
        # Return a copy of the filtered DataFrame to avoid modifying the original
        return df[partition_mask].copy()

    def _read_existing_partition_file(self, file_path: str, partition) -> pd.DataFrame:
        """
        Read an existing partition Parquet file and ensure required partition columns are present.
        
        If the file does not exist, returns an empty DataFrame. Ensures the `datetime_utc` column is timezone-naive and adds any missing partition columns with their corresponding values.
        
        Parameters:
            file_path (str): Path to the partition Parquet file.
            partition (dict): Dictionary mapping partition column names to their values.
        
        Returns:
            pd.DataFrame: DataFrame containing the data from the partition file, with all required partition columns.
        
        Raises:
            Exception: If reading the file fails for any reason.
        """
        try:
            if not os.path.exists(file_path):
                return pd.DataFrame()

            existing_data_df = pd.read_parquet(file_path)
            existing_data_df = self._ensure_datetime_utc_naive(existing_data_df)
            
            # Add partition columns back if they were dropped
            for col, value in partition.items():
                if col not in existing_data_df.columns:
                    existing_data_df[col] = value
                    
            return existing_data_df

        except Exception as e:
            raise Exception(f"Failed to read existing partition file {file_path}: {str(e)}")

    def _combine_and_write_partition(self, new_data_df: pd.DataFrame, existing_data_df: pd.DataFrame,
                                     partition_cols: list, output_file_path: str,
                                     value_cols: list[str], dataset_type: str, schema: pa.Schema = None) -> None:
        """
                                    Combine new and existing partition data, remove duplicates, and write the result as a Parquet file.
                                    
                                    Merges new and existing DataFrames for a partition, deduplicates based on dataset type, prepares the data for writing, and writes the final result to the specified Parquet file path. Raises an exception if no data is available or if any processing step fails.
                                    """
        try:
            if new_data_df.empty and existing_data_df.empty:
                raise ValueError("No data available for this partition")

            # Data combination step
            if not new_data_df.empty and not existing_data_df.empty:
                combined_df = pd.concat([existing_data_df, new_data_df], ignore_index=True)
            elif not new_data_df.empty:
                combined_df = new_data_df
            else:
                combined_df = existing_data_df

            # Deduplication step
            final_df = self.drop_processed_duplicates(combined_df, dataset_type)
            if final_df.empty:
                raise ValueError("No data remains after deduplication")

            # Prepare for writing
            final_df_sorted = self._prepare_dataframe_for_writing(final_df, partition_cols)
            if final_df_sorted is None:
                raise ValueError("Failed to prepare data for writing")

            # Write file
            row_group_size = min(self.row_group_size, len(final_df_sorted))
            if row_group_size == 0:
                raise ValueError("No rows to write")

            self._write_final_parquet(final_df_sorted, output_file_path, value_cols, row_group_size, schema)

        except Exception as e:
            raise Exception(f"Failed to combine and write partition: {str(e)}")

    def _prepare_dataframe_for_writing(self, df: pd.DataFrame, partition_cols: list) -> Optional[pd.DataFrame]:
        """
        Sorts the DataFrame by the 'datetime_utc' column and resets its index in preparation for writing.
        
        Parameters:
            df (pd.DataFrame): The DataFrame to be prepared.
            partition_cols (list): List of partition columns (not used in this method but included for interface consistency).
        
        Returns:
            pd.DataFrame: The sorted DataFrame with reset index.
        
        Raises:
            Exception: If sorting or index resetting fails.
        """
        try:
            # Sort by datetime_utc
            df_sorted = df.sort_values(by='datetime_utc').reset_index(drop=True)
                    
            return df_sorted
        except Exception as e:
            print(f"[ERROR] Failed to prepare DataFrame for writing: {e}")
            raise

    def _write_final_parquet(self, df: pd.DataFrame, output_file: str, value_cols: list[str], row_group_size: int, schema: pa.Schema = None) -> None:
        """
        Write a pandas DataFrame to a Parquet file with specified row group size, compression, statistics, and dictionary encoding.
        
        Parameters:
            df (pd.DataFrame): The DataFrame to write.
            output_file (str): Destination file path for the Parquet file.
            value_cols (list[str]): List of value columns to include in Parquet statistics.
            row_group_size (int): Number of rows per Parquet row group.
        
        Raises:
            Exception: If conversion to PyArrow Table fails or if the Parquet write operation encounters an error.
        """
        try:
            print("\n📊 WRITE OPERATION")
            print("-"*40)
            print(f"Records to write: {len(df)}")
            print(f"Row group size: {row_group_size}")
            
            # Convert to PyArrow Table
            print("\n🔄 Converting to PyArrow...")
            table = self._to_pyarrow_table(df, schema)
            if table is None:
                print("❌ Conversion failed")
                raise Exception("Conversion failed")

            # Configure write options
            print("\n⚙️ Configuring write options...")
            if schema is None:
                raise ValueError("Schema is required for writing")
            stats_cols = self._set_stats_cols(value_cols, schema)
            dict_cols = self._set_dict_cols(schema)

            # Write file
            print("\n💾 Writing to disk...")
            writer = pq.ParquetWriter(
                output_file,
                schema,
                compression='zstd',
                write_statistics=stats_cols,
                use_dictionary=dict_cols,
                data_page_size=64 * 1024,
                data_page_version='2.0'
            )
            writer.write_table(table, row_group_size)
            writer.close()

            print("\n✅ WRITE SUCCESSFUL")
            print(f"File: {output_file}")
            print("-"*40)

        except Exception as e:
            print("\n❌ WRITE FAILED")
            print(f"Error: {str(e)}")
            print("-"*40)
            raise

    def _ensure_datetime_utc_naive(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert the 'datetime_utc' column in the DataFrame to timezone-naive if it contains timezone-aware datetimes.
        
        Returns:
            DataFrame with a timezone-naive 'datetime_utc' column for Parquet compatibility.
        """
        if 'datetime_utc' in df.columns:
            # This robustly converts the column to a UTC-aware datetime series.
            # If the original data is timezone-aware, it's converted to UTC.
            # If it's naive, it's localized to UTC.
            df['datetime_utc'] = pd.to_datetime(df['datetime_utc'], utc=True)
            
            # Now, we make it timezone-naive. This simply strips the timezone information,
            # leaving the wall time as-is, which is now correctly in UTC.
            print("Converting datetime_utc to UTC naive for Parquet compatibility...")
            df['datetime_utc'] = df['datetime_utc'].dt.tz_convert(None)
        return df

    def _to_pyarrow_table(self, df: pd.DataFrame, schema: pa.Schema = None) -> Optional[pa.Table]:
        """
        Convert a pandas DataFrame to a PyArrow Table.
        
        Raises:
            Exception: If the conversion fails, an exception is raised with error details.
        
        Returns:
            pa.Table: The resulting PyArrow Table.
        """
        try:
            return pa.Table.from_pandas(df, schema=schema, preserve_index=False)
        except Exception as e:
            print(f"[ERROR] Error converting DataFrame to PyArrow Table: {e}")
            print(f"[DEBUG] DataFrame dtypes:\n{df.dtypes}")
            raise

    def _filter_partition(self, table: pa.Table, partition, partition_cols: list) -> pa.Table:
        """
        Return a PyArrow Table containing only rows that match the specified partition values.
        
        Parameters:
            table (pa.Table): The input PyArrow Table to filter.
            partition (pd.Series): Partition values to match.
            partition_cols (list): Names of columns to use for partition filtering.
        
        Returns:
            pa.Table: A new table with rows matching all partition column values.
        
        Raises:
            Exception: If filtering fails due to missing columns or computation errors.
        """
        mask = True
        try:
            for col in partition_cols:
                mask = pa.compute.and_(mask, pa.compute.equal(table[col], partition[col]))
            return table.filter(mask)
        except Exception as e:
            print(f"[ERROR] Error filtering partition: {e}")
            raise

    def _drop_partition_cols(self, table: pa.Table, partition_cols: list) -> pa.Table:
        """
        Remove specified partition columns from a PyArrow Table.
        
        Parameters:
            table (pa.Table): The input PyArrow Table.
            partition_cols (list): Names of columns to remove.
        
        Returns:
            pa.Table: A new table with the specified partition columns removed.
        """
        try:
            print(f" Dropping partition columns: {partition_cols}")
            cols_to_drop = [col for col in partition_cols if col in table.column_names]
            if not cols_to_drop:
                print(" No partition columns to drop")
                return table
            table = table.drop(cols_to_drop)
            return table
        except Exception as e:
            print(f"[ERROR] Error dropping partition columns: {e}")
            raise

    def _build_partition_path(self, partition, partition_cols: list, dataset_type: str) -> str:
        """
        Constructs the full output file path for a partitioned parquet file using Hive-style key=value directories.
        
        The resulting path is structured as `<processed_path>/<key1>=<value1>/<key2>=<value2>/.../<dataset_type>.parquet`, with the month value zero-padded.
        Creates the partition directory if it does not exist.
        
        Parameters:
            partition: Partition values, typically a pandas Series.
            partition_cols (list): Names of columns used for partitioning.
            dataset_type (str): Type of dataset, used for the output file name.
        
        Returns:
            str: Full file path for the partitioned parquet file.
        """
        # Hive-style: key=value for each partition column
        path_segments = [str(self.processed_path)]
        for col in partition_cols:
            value = partition[col] 
            # Format month as two digits, others as is
            if col == 'month':
                segment = f"{col}={value:02d}" #only month requieres 0 padding since there are single digitsfor months
            else:
                segment = f"{col}={value}"
            path_segments.append(segment)
        partition_path_str = os.path.join(*path_segments) #kwargs, join path segments ie> processed/mercado="Intra"/id_mercado="2"/year=2024/month=01
        os.makedirs(partition_path_str, exist_ok=True) #create directory if it doesn't exist

        if dataset_type == "curtailments_i90":
            print(" Naming curtailments_i90 file as volumenes_i90.parquet to homogenize processed data naming convention for curtailments parquet files")
            dataset_type = "volumenes_i90"

        if dataset_type == "curtailments_i3":
            print(" Naming curtailments_i3 file as volumenes_i3.parquet to homogenize processed data naming convention for curtailments parquet files")
            dataset_type = "volumenes_i3"
        
        return os.path.join(partition_path_str, f"{dataset_type}.parquet")

    def _set_stats_cols(self, value_cols: list[str], schema: pa.Schema) -> list[str]:
        """
        Determine which columns should be included in Parquet file statistics, including 'datetime_utc' and any value columns present in the schema.
        
        Parameters:
            value_cols (list[str]): List of value column names to include if present in the schema.
            schema (pa.Schema): PyArrow schema to check for column presence.
        
        Returns:
            list[str]: List of column names to include in Parquet statistics.
        """
        stats_cols = ['datetime_utc']
        
        # Add all valid value columns to statistics
        for value_col in value_cols:
            if value_col in schema.names:
                stats_cols.append(value_col)
            else:
                print(f"Warning: Value column '{value_col}' not found in schema")
        
        return stats_cols
    
    def _set_dict_cols(self, schema: pa.Schema) -> list[str]:
        """
        Determine which columns should use dictionary encoding (integers instead of full strings for repeated values) for Parquet writing based on the schema.
        
        Returns:
            dict_cols (list[str]): List of column names to apply dictionary encoding, such as "up", "tecnologia", or "uof", depending on the dataset type.
        """
        print("--------------------------------")
        print(f" Applying dictionary encoding...")
        dict_cols = []
        if "up" in schema.names: #for i90 market data (volumenes + precios)
            dict_cols.append("up")
        elif "tecnologia" in schema.names: #for i3 market data
            dict_cols.append("tecnologia")
        elif "uof" in schema.names: #for omie market data
            dict_cols.append("uof")
        elif "redespacho" in schema.names: #for restricciones market data
            dict_cols.append("redespacho")
        else: 
            print(f" Dictionary encoding not applied for this dataset. Only applied for volumenes data set for uof, up and tecnologia")
        
        print("--------------------------------")
        return dict_cols
    
    def write_processed_parquet(self, df: pd.DataFrame, mercado: str, value_cols: list[str], dataset_type: str, schema: pa.Schema = None) -> None:
            """
            Writes a DataFrame as partitioned Parquet files using Hive-style directory structure.
            
            The DataFrame is partitioned by 'mercado', 'id_mercado', 'year', and 'month'. For each unique partition, the method combines new and existing data, removes duplicates, and writes the result as a Parquet file with the specified value columns. Raises an exception if the input DataFrame is empty or if required partition columns are missing.
            """
            if df.empty:
                raise ValueError(f"Input DataFrame for {mercado} is empty. Cannot proceed with write operation.")
            
            try:
                # Prepare the input DataFrame
                df = self._prepare_input_dataframe(df, mercado)

                # Get unique partitions from the input DataFrame
                partition_cols = ['mercado', 'id_mercado', 'year', 'month']
                if not self._validate_partition_columns(df, partition_cols):
                    raise ValueError(f"Missing required partition columns in DataFrame")

                # Process partitions
                unique_partitions_df = df[partition_cols].drop_duplicates()
                
                for _, partition in unique_partitions_df.iterrows():
                    try:
                        output_file_path = self._build_partition_path(partition, partition_cols, dataset_type)
                        
                        # Handle existing file
                        existing_data_df = self._read_existing_partition_file(output_file_path, partition)
                        
                        # Filter new data
                        new_data_partition_df = self._filter_df_for_partition(df, partition, partition_cols)
                        
                        # Combine and write
                        self._combine_and_write_partition(
                            new_data_df=new_data_partition_df,
                            existing_data_df=existing_data_df,
                            partition_cols=partition_cols,
                            output_file_path=output_file_path,
                            value_cols=value_cols,
                            dataset_type=dataset_type,
                            schema=schema
                        )
                    except Exception as e:
                        raise Exception(f"Failed to process partition {partition.to_dict()}: {str(e)}")

            except Exception as e:
                raise Exception(f"Failed to write processed parquet: {str(e)}")

    def read_latest_processed_file(self, mercado: str, dataset_type: str) -> pd.DataFrame:
            """
            Reads the latest available processed file for a given market and dataset type.

            It searches for all matching processed files and identifies the one in the
            most recent 'year' and 'month' directory.

            Args:
                mercado (str): The market name (e.g., 'i3', 'i90').
                dataset_type (str): The type of dataset (e.g., 'p48_tecnologias_generacion').

            Returns:
                pd.DataFrame: A DataFrame with the contents of the latest processed file,
                                        or None if no file is found.
            """
            try:
                print(f"🔍 Searching for the latest processed file for market '{mercado}' and dataset '{dataset_type}'...")

        
                if dataset_type == "curtailments_i90":
                    filename = "volumenes_i90.parquet"
                elif dataset_type == "curtailments_i3":
                    filename = "volumenes_i3.parquet"
                else:
                    filename = f"{dataset_type}.parquet"
                
                # Glob pattern to find all possible files
                search_pattern = f"mercado={mercado}/**/year=*/month=*/{filename}"
                all_files = list(self.processed_path.glob(search_pattern))

                if not all_files:
                    print(f"⚠️ No processed file found for dataset '{dataset_type}' in market '{mercado}'.")
                    return None

                # Find the latest file by parsing year and month from the path
                latest_file = None
                latest_year = -1
                latest_month = -1

                for f in all_files:
                    try:
                        parts = f.parts
                        # Find year and month from path parts
                        year = int([p for p in parts if p.startswith('year=')][0].split('=')[1])
                        month = int([p for p in parts if p.startswith('month=')][0].split('=')[1])

                        if year > latest_year or (year == latest_year and month > latest_month):
                            latest_year = year
                            latest_month = month
                            latest_file = f
                    except (ValueError, IndexError):
                        # Ignore paths that don't have the expected year/month format
                        continue

                if latest_file:
                    print(f"  📄 Found latest processed file: {latest_file}")
                    return pd.read_parquet(latest_file)
                else:
                    print(f"⚠️ No valid processed file found for dataset '{dataset_type}' in market '{mercado}'.")
                    return None

            except Exception as e:
                print(f"❌ Error reading the latest processed file: {e}")
                raise
