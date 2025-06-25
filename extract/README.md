# Extract Module Documentation

This directory contains scripts responsible for extracting data from the ESIOS API and I90 files provided by REE (Red Eléctrica de España), saving the raw data for further processing.

## File Descriptions

### `descargador_esios.py`

*   **Purpose:** Downloads **price data** directly from the ESIOS API for various electricity markets (Diario, Intradia, Secundaria, Terciaria, RR).
*   **Functionality:** Base class `DescargadorESIOS` manages API requests, authentication, and data parsing; subclasses handle market-specific logic, indicator IDs, and regulatory changes.
*   **Inputs:** API token, indicator IDs, date ranges, market-specific parameters.
*   **Outputs:** Pandas DataFrames with price data, `granularidad`, and `indicador_id`.
*   **Dependencies:** `requests`, `pandas`, `pytz`, `dotenv`, `datetime`, `utilidades.db_utils`, `configs.esios_config`.

### `descargador_i90.py`

*   **Purpose:** Downloads and processes **I90 files** (primarily **volume data**) from the ESIOS archives API.
*   **Functionality:** Base class `I90Downloader` handles downloading, extracting `.xls` files, managing configured errors, filtering sheets, and parsing data; subclasses specify relevant sheets for different markets/data types.
*   **Inputs:** API key, date, sheet configurations, error list, temporary download path.
*   **Outputs:** Pandas DataFrames with volume or price data from specific I90 sheets.
*   **Dependencies:** `requests`, `pandas`, `zipfile`, `openpyxl` (implicitly via pandas), `utilidades.db_utils`, `configs.i90_config`.

### `esios_extractor.py`

*   **Purpose:** High-level wrapper for `descargador_esios.py`, orchestrating the download and saving of **price data**.
*   **Functionality:** Initializes market-specific downloaders, handles date validation, iterates daily requests, and uses `RawFileUtils` to save results.
*   **Inputs:** Start/end dates, market-specific parameters (e.g., `intra_lst`), `dev` flag.
*   **Outputs:** Raw data files (CSV/Parquet) in the `raw_data` directory structure.
*   **Dependencies:** `pandas`, `datetime`, `extract.descargador_esios`, `utilidades.storage_file_utils`.

### `i90_extractor.py`

*   **Purpose:** High-level wrapper for `descargador_i90.py`, orchestrating the download, extraction, and saving of **volume and price data** from I90 files.
*   **Functionality:** Initializes downloaders, handles date validation, manages daily file downloads/cleanup via `I90Downloader`, delegates extraction to subclasses (`I90VolumenesExtractor`, `I90PreciosExtractor`), and uses `RawFileUtils` to save results.
*   **Inputs:** Start/end dates, `dev` flag.
*   **Outputs:** Raw data files (CSV/Parquet) in the `raw_data` directory structure.
*   **Dependencies:** `pandas`, `datetime`, `extract.descargador_i90`, `utilidades.storage_file_utils`, `utilidades.db_utils`.

## Configuration

*   **API Credentials:** Create a `.env` file in the project root directory containing:
    ```dotenv
    ESIOS_TOKEN=your_esios_api_token_here
    ```
*   **Market/File Settings:** Specific configurations (indicator IDs, sheet numbers, regulatory dates, error lists, paths) are managed within:
    *   `configs/esios_config.py`
    *   `configs/i90_config.py`
*   **Output Storage:** The location and format of saved raw data are determined by `utilidades/storage_file_utils.py`.

## Usage Examples

These examples demonstrate how to use the extractor classes programmatically.

```python
from extract.esios_extractor import ESIOSPreciosExtractor
from extract.i90_extractor import I90VolumenesExtractor, I90PreciosExtractor
from datetime import datetime, timedelta

# --- ESIOS Price Extraction ---

# Define date range (e.g., yesterday)
end_date = datetime.now() - timedelta(days=1)
start_date = end_date - timedelta(days=2) # Example: 3 days total
start_str = start_date.strftime('%Y-%m-%d')
end_str = end_date.strftime('%Y-%m-%d')

# Initialize ESIOS price extractor
esios_extractor = ESIOSPreciosExtractor()

# Extract daily market prices (saves to CSV if dev=True, else Parquet)
print(f"Extracting ESIOS Diario prices from {start_str} to {end_str}...")
esios_extractor.extract_diario(
    fecha_inicio_carga=start_str,
    fecha_fin_carga=end_str,
    dev=False # Set dev=True for CSV output, False for Parquet
)

# Extract specific intraday markets (e.g., 1 and 2)
print(f"Extracting ESIOS Intra prices (1, 2) from {start_str} to {end_str}...")
esios_extractor.extract_intra(
    fecha_inicio_carga=start_str,
    fecha_fin_carga=end_str,
    intra_lst=[1, 2],
    dev=False
)

# --- I90 Volume & Price Extraction ---

# Define date range for I90 (Note: I90 data has a delay, adjust dates accordingly)
# Example: Extract data published ~90 days ago
i90_end_date = datetime.now() - timedelta(days=90)
i90_start_date = i90_end_date - timedelta(days=1) # Example: 2 days total
i90_start_str = i90_start_date.strftime('%Y-%m-%d')
i90_end_str = i90_end_date.strftime('%Y-%m-%d')

# Initialize I90 volume extractor
i90_vol_extractor = I90VolumenesExtractor()

# Extract all available I90 volumes for the date range
print(f"Extracting I90 Volumes from {i90_start_str} to {i90_end_str}...")
i90_vol_extractor.extract_data(
    fecha_inicio_carga=i90_start_str,
    fecha_fin_carga=i90_end_str,
    dev=True # Set dev=True for CSV output, False for Parquet
)

# Initialize I90 price extractor (primarily for 'restricciones')
i90_pre_extractor = I90PreciosExtractor()

# Extract I90 prices (e.g., restrictions) for the date range
print(f"Extracting I90 Precios (Restricciones) from {i90_start_str} to {i90_end_str}...")
i90_pre_extractor.extract_data(
    fecha_inicio_carga=i90_start_str,
    fecha_fin_carga=i90_end_str,
    dev=True
)

print("Extraction complete.")
```

## Interdependencies & Workflow

1.  **Configuration:** Load API keys from `.env` and market/file settings from `configs/*.py`. Set datalake path in the `.env` as well
2.  **Execution Start:** Instantiate an extractor class (`ESIOSPreciosExtractor`, `I90VolumenesExtractor`, etc.).
3.  **Extractor Layer:** The extractor validates input (dates) and iterates through the requested period.
4.  **Downloader Layer:** For each day/request, the extractor calls the relevant downloader class (`descargador_esios.py` or `descargador_i90.py`).
    *   `descargador_esios.py` queries the ESIOS API.
    *   `descargador_i90.py` downloads/extracts the daily I90 file and parses configured sheets.
5.  **Data Processing:** Downloaders process raw responses (JSON/Excel) into Pandas DataFrames.
6.  **Output/Storage:** The extractor passes the DataFrame to `utilidades.storage_file_utils.RawFileUtils` for saving (Parquet/CSV) in the `raw_data` directory. 