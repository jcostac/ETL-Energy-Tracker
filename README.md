# ETL Energy Tracker Pipeline

Proof of concept for tracking of energy prices and volumes for REE.

## Project Structure

This repository is organized as a modular ETL (Extract, Transform, Load) pipeline for energy data. Below is an overview of the main folders and their contents:

### `extract/`
Contains scripts for extracting raw data from external sources:
- `i90_extractor.py`, `esios_extractor.py`: Main extractors for I90 and ESIOS data.
- `_descargador_i90.py`, `_descargador_esios.py`: Helper modules fr downloading data sources.
- `README.md`: Additional documentation for extraction logic.

### `transform/`
Handles data cleaning, date transformation, and  parquet partitioning:
- `i90_transform.py`, `esios_transform.py`: Transform raw extracted data into a standardized format.
- `_procesador_i90.py`, `_procesador_esios.py`: Helper modules for processing specific data types.
- `carga_i90.py`: Likely handles loading or further transformation for I90 data.

### `load/`
Scripts for loading processed data into storage solutions (local or cloud):
- `s3_data_lake_loader.py`, `local_data_lake_loader.py`, `data_lake_loader.py`: Loaders for different storage solutions (S3 or local storage).
- `__init__.py`: Marks the folder as a Python package.

### `read/`
Utilities for reading and querying stored data:
- `read_data.py`, `read_ESIOS_data.py`, `read_i90_data.py`: Read and process data from storage.



### `utilidades/`
General utilities and helper functions:
- `env_utils.py`: Environment variable helpers.
- `storage_file_utils.py`, `etl_date_utils.py`, `data_validation_utils.py`, `db_utils.py`: Utilities for file handling, date processing, validation, and database operations.

### `data/`
Data storage directory:
- `raw/`: Stores raw extracted data.
- `processed/`: Stores processed/cleaned data.
- `temporary/`: Temporary files used during processing.

### `configs/`
Configuration files for the pipeline:
- `storage_config.py`, `i90_config.py`, `esios_config.py`: Configuration for storage and data sources.

### `queries/`
SQL or script-based queries for data analysis or database operations:
- `timescale/`: Contains `timescale_tests.py` for testing TimescaleDB queries.

### `dags/`
Airflow DAGs or similar orchestration scripts:
- `i90_precios_etl_dag.py`, `esios_precios_etl_dag.py`: Define ETL workflows for I90 and ESIOS data.

### `tests/`
Unit and integration tests for the pipeline:
- `extractor_tests.py`: Tests for extraction logic.

---

## Getting Started

1. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

2. **Configure your environment:**
   - Edit files in `configs/` as needed for your data sources and storage.

3. **Run the ETL pipeline:**
   - Use scripts in `extract/`, `transform/`, and `load/` as needed, or orchestrate with DAGs in `dags/`.

## Notes

- This is a proof of concept and may require adaptation for production use.
- See individual folder `README.md` files (where present) for more details.
