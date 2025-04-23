import pandas as pd
from typing import Dict, List, Optional, Any
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Import actual config classes from configs.i90_config
from configs.i90_config import (
        I90Config, # Base might be useful too
        DiarioConfig, SecundariaConfig, TerciariaConfig, RRConfig,
        CurtailmentConfig, P48Config, IndisponibilidadesConfig, RestriccionesConfig
    )
from utilidades.etl_date_utils import DateUtilsETL
from utilidades.data_validation_utils import DataValidationUtils


class I90Processor:
    """
    Processor class for I90 data (Volumenes and Precios).
    Handles data cleaning, validation, filtering, and transformation.
    """
    def __init__(self):
        """
        Initialize the processor.
        """
        self.date_utils = DateUtilsETL()
        # Potential future enhancements: Load all configs at init if needed

    # === MAIN PIPELINE ===
    def transform_data(self, df: pd.DataFrame, market_config: I90Config, dataset_type: str) -> pd.DataFrame:
        """
        Main transformation pipeline for I90 data.
        """
        if df.empty:
            print("Input DataFrame is empty. Skipping transformation.")
            return self._empty_output_df(dataset_type)

        try:
            df = self._apply_market_filters_and_id(df, market_config)
            if df.empty: raise ValueError("DataFrame empty after applying market filters.")

            df = self._standardize_datetime(df)
            if df.empty: raise ValueError("DataFrame empty after standardizing datetime.")

            df = self._select_and_finalize_columns(df, dataset_type)
            if df.empty: raise ValueError("DataFrame empty after selecting final columns.")

            df = self._validate_final_data(df, dataset_type)
            print("I90 transformation pipeline completed successfully.")
            return df

        except Exception as e:
            print(f"Error during I90 transformation pipeline: {e}")
            import traceback
            print(traceback.format_exc())
            return self._empty_output_df(dataset_type)

    # === FILTERING ===
    def _apply_market_filters_and_id(self, df: pd.DataFrame, market_config: I90Config) -> pd.DataFrame:
        """
        Filters the DataFrame based on market config and adds id_mercado.
        
        Args:
            df (pd.DataFrame): Input DataFrame (potentially pre-processed by extractor).
                               Expected columns might include 'Sentido', 'Redespacho', 'UP', 'hora', 'volumen'/'precio', 'fecha'/'datetime_utc'.
            market_config (I90Config): The configuration object instance for the specific market
                                     (e.g., SecundariaConfig(), RestriccionesConfig()).
            
        Returns:
            pd.DataFrame: DataFrame filtered and augmented with 'id_mercado',
                          or an empty DataFrame if no data matches.
        """
        if df.empty:
            return pd.DataFrame() # Return empty if input is empty

        all_market_dfs = []

        required_cols = ['volumen'] # Base columns often present
        if 'Sentido' in df.columns: required_cols.append('Sentido')
        if 'Redespacho' in df.columns: required_cols.append('Redespacho')

        # Ensure required columns exist
        if not all(col in df.columns for col in required_cols if col in ['Sentido', 'Redespacho']):
             print(f"Warning: Input DataFrame for market transformation might be missing 'Sentido' or 'Redespacho' columns if required by config.")
             # Decide if this is fatal or if filtering can proceed partially

        # Ensure market_config has the necessary attributes
        if not hasattr(market_config, 'market_ids') or not hasattr(market_config, 'sentido_map') or not hasattr(market_config, 'get_redespacho_filter'):
             print(f"Error: Provided market_config object ({type(market_config).__name__}) is missing required attributes/methods (market_ids, sentido_map, get_redespacho_filter).")
             return pd.DataFrame()

        for market_id in market_config.market_ids:
            # market_id from config is already a string
            sentido = market_config.sentido_map.get(market_id) # Use market_id directly
            redespacho_filter = market_config.get_redespacho_filter(market_id) # Use market_id directly

            filtered_df = df.copy() # Start with the full data for this specific market_id iteration

            # Apply sentido filter
            if sentido and 'Sentido' in filtered_df.columns:
                 # Ensure consistent comparison (e.g., handle case sensitivity if needed)
                 # Assuming Sentido in DataFrame and config map are comparable (e.g., 'Subir', 'Bajar', None)
                filtered_df = filtered_df[filtered_df['Sentido'] == sentido]
            elif sentido and 'Sentido' not in filtered_df.columns:
                 print(f"Warning: Config requires filtering by Sentido='{sentido}' for market_id {market_id}, but 'Sentido' column is missing.")
                 # If sentido filter is required but column missing, no rows will match
                 filtered_df = pd.DataFrame() # Clear DF

            # Apply redespacho filter
            if redespacho_filter and 'Redespacho' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['Redespacho'].isin(redespacho_filter)]
            elif redespacho_filter and 'Redespacho' not in filtered_df.columns:
                 print(f"Warning: Config requires filtering by Redespacho='{redespacho_filter}' for market_id {market_id}, but 'Redespacho' column is missing.")
                 # If redespacho filter is required but column missing, no rows will match
                 filtered_df = pd.DataFrame() # Clear DF


            if not filtered_df.empty:
                # Add the market_id (which is already a string from the config)
                filtered_df['id_mercado'] = market_id
                all_market_dfs.append(filtered_df)
                print(f"Processed Market ID: {market_id}, Filtered rows: {len(filtered_df)}")
            else:
                 print(f"No data matched filters for Market ID: {market_id}")


        if not all_market_dfs:
            return pd.DataFrame() # Return empty if no market_id yielded data

        # Combine results for all market_ids handled by this config
        final_df = pd.concat(all_market_dfs, ignore_index=True)
        # Ensure id_mercado is string type after concat (should be if added as string)
        if 'id_mercado' in final_df.columns:
             final_df['id_mercado'] = final_df['id_mercado'].astype(str)
        return final_df

    # === DATETIME HANDLING ===
    def _standardize_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensures a standard UTC datetime column."""
        if df.empty: return df

        if 'hora' in df.columns and 'fecha' in df.columns:
             # Combine fecha and hora, assuming 'hora' might need adjustment (e.g., from 1-24 to 0-23)
             # The extractor might already handle this; this is a fallback/standardization step.
             # This logic needs refinement based on the exact output of the extractor.
             # For now, assume 'hora' is datetime.time or similar and 'fecha' is datetime.date/datetime
             # Handle potential errors during combination
             try:
                df['datetime_local'] = df.apply(lambda row: pd.Timestamp.combine(row['fecha'].date(), row['hora']), axis=1)
                df['datetime_utc'] = self.date_utils.convert_local_to_utc(df['datetime_local']) # Assumes local is Europe/Madrid
                df = df.drop(columns=['datetime_local', 'fecha', 'hora'], errors='ignore')
             except Exception as e:
                 print(f"Error combining 'fecha' and 'hora': {e}. Cannot create 'datetime_utc'.")
                 # Decide handling: return empty or raise? Returning df without datetime_utc for now.
                 if 'datetime_utc' not in df.columns: # Ensure it doesn't exist if creation failed
                     pass # Or add an empty column: df['datetime_utc'] = pd.NaT
                 return df.drop(columns=['fecha', 'hora'], errors='ignore') # Drop original cols

        elif 'datetime_utc' in df.columns:
             try:
                df['datetime_utc'] = pd.to_datetime(df['datetime_utc'], utc=True)
             except Exception as e:
                print(f"Error converting existing 'datetime_utc' column: {e}. Setting to NaT.")
                df['datetime_utc'] = pd.NaT
        elif 'datetime' in df.columns: # If extractor provides a generic datetime
             try:
                df['datetime_utc'] = pd.to_datetime(df['datetime'], utc=True)
                if 'datetime' != 'datetime_utc':
                    df = df.drop(columns=['datetime'], errors='ignore')
             except Exception as e:
                 print(f"Error converting existing 'datetime' column: {e}. Setting to NaT.")
                 df['datetime_utc'] = pd.NaT
                 if 'datetime' != 'datetime_utc': df = df.drop(columns=['datetime'], errors='ignore')
                    
             print("Warning: Could not find suitable columns ('fecha'/'hora', 'datetime_utc', 'datetime') to create standardized 'datetime_utc'.")
             # Decide handling: error or return df as is? Returning as is for now.
             return df

        # Drop rows where datetime conversion might have failed
        df = df.dropna(subset=['datetime_utc'])

        return df

    # === COLUMN FINALIZATION ===
    def _select_and_finalize_columns(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """Selects, orders, and standardizes final columns, outputting 'up' for unit identifier."""
       
        pass

    def _get_value_col(self, dataset_type: str) -> Optional[str]:
        if dataset_type == 'volumenes_i90':
            return 'volumen'
        elif dataset_type == 'precios_i90':
            return 'precio'
        return None

    # === VALIDATION ===
    def _validate_final_data(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """Validate final data structure."""
        if not df.empty:
            validation_schema_type = "volumenes_i90" if dataset_type == 'volumenes_i90' else "precios_i90" # Map to validation schema names more specifically
            try:
                 # Assuming DataValidationUtils.validate_data expects specific schema names
                 DataValidationUtils.validate_data(df, validation_schema_type)
                 print("Final data validation successful.")
            except KeyError as e:
                 print(f"Validation Error: Schema '{validation_schema_type}' not found in DataValidationUtils. Skipping validation. Error: {e}")
            except Exception as e:
                 print(f"Error during final data validation: {e}")
                 # Decide if this should return empty df or raise
                 # Raising allows the error to propagate up
                 raise # Reraise validation error
            
            print("Skipping validation for empty DataFrame.")
        return df

    # === UTILITY ===
    def _empty_output_df(self, dataset_type: str) -> pd.DataFrame:
        """
        Returns an empty DataFrame with the expected columns for the dataset_type.
        """
        value_col = self._get_value_col(dataset_type)
        cols = ['id_mercado', 'datetime_utc', value_col]
        if dataset_type == 'volumenes_i90':
            cols.append('up')
        df = pd.DataFrame(columns=cols)
        df.index.name = 'id'
        return df

 
 