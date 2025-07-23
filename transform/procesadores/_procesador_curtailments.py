import pandas as pd
from typing import Optional
import traceback

from configs.curtailment_config import CurtailmentConfig
from utilidades.etl_date_utils import DateUtilsETL
from utilidades.data_validation_utils import DataValidationUtils
from utilidades.raw_file_utils import RawFileUtils
from utilidades.progress_utils import with_progress

class CurtailmentProcessor:
    """
    Processor class for Curtailment data (Volumenes).
    Handles data cleaning, validation, filtering, transformation, and adds curtailment-specific columns.
    Similar to I90Processor but tailored for curtailment with fixed market ID 13.
    """

    def __init__(self):
        """
        Initializes the CurtailmentProcessor with utilities for date handling, data validation, and file operations.
        """
        self.date_utils = DateUtilsETL()
        self.data_validation_utils = DataValidationUtils()
        self.raw_file_utils = RawFileUtils()
        self.curtailment_config = CurtailmentConfig()

    # === FILTERING ===
    def _apply_market_filters_and_id(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filters the DataFrame and assigns market ID 13 for curtailment data.
        Applies redespacho filters based on sheet if available.
        """
        if df.empty:
            return pd.DataFrame()

        print(f"Applying curtailment filters and setting id_mercado to 13. Shape: {df.shape}")

        #filter df by sentido Bajar (curtailments are only in Bajar)
        df = df[df['Sentido'] == 'Bajar']

        #apply filters from config  
        if "Redespacho" in df.columns:
            def assign_rtx(redespacho):
                if redespacho in self.curtailment_config.rt1_redespacho_filter:
                    return "R1"
                if redespacho in self.curtailment_config.rt5_redespacho_filter:
                    return "R5"
                return None
            df['RTx'] = df['Redespacho'].apply(assign_rtx)

            #filter df by redespacho
            df = df[df['Redespacho'].isin(self.curtailment_config.rt1_redespacho_filter + self.curtailment_config.rt5_redespacho_filter)]
            
        else:
            raise ValueError("Redespacho column not found in dataframe")
        
        #add id_mercado column
        df['id_mercado'] = 13
        return df

    # === DATETIME HANDLING ===
    def _standardize_datetime(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        return self.date_utils.standardize_datetime(df, dataset_type)

    @with_progress(message="Processing hourly data...", interval=2)
    def _process_hourly_data(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        return self.date_utils.process_hourly_data(df, dataset_type)

    @with_progress(message="Processing 15-minute data...", interval=2)
    def _process_15min_data(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.date_utils.process_15min_data(df)

    @with_progress(message="Processing hourly data (vectorized)...", interval=2)
    def _process_hourly_data_vectorized(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        return self.date_utils.process_hourly_data_vectorized(df, dataset_type)

    @with_progress(message="Processing 15-minute data (vectorized)...", interval=2)
    def _process_15min_data_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.date_utils.process_15min_data_vectorized(df)

    # === COLUMN FINALIZATION ===
    def _select_and_finalize_columns(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """Selects and renames columns to the final required format for curtailment."""
        
        if "Unidad de Programación" in df.columns:
            df = df.rename(columns={"Unidad de Programación": "up"})
        
        if "Tipo Restricción" in df.columns:
            df = df.rename(columns={"Tipo Restricción": "tipo"})

        if "Concepto" in df.columns:
            df = df.rename(columns={"Concepto": "tecnologia"})

        if dataset_type == "curtailments_i90":
            required_cols = self.data_validation_utils.processed_curtailments_i90_required_cols
        elif dataset_type == "curtailments_i3":
            required_cols = self.data_validation_utils.processed_curtailments_i3_required_cols
        else:
            raise ValueError(f"Invalid dataset type: {dataset_type}")

        return df[[col for col in required_cols if col in df.columns]]
    
    def _validate_final_data(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """Validates the final processed data using curtailment schema."""
        if not df.empty:

            if dataset_type == "curtailments_i90":
                try:
                    self.data_validation_utils.validate_processed_data(df, "curtailments_i90")
                    print("Final data validation successful.")
                except Exception as e:
                    print(f"Error during final data validation: {e}")
                    raise
            elif dataset_type == "curtailments_i3":
                try:
                    self.data_validation_utils.validate_processed_data(df, "curtailments_i3")
                    print("Final data validation successful.")
                except Exception as e:
                    print(f"Error during final data validation: {e}")
                    raise
        return df

    # === UTILITY ===
    def _empty_output_df(self, dataset_type: str) -> pd.DataFrame:
        if dataset_type == "curtailments_i90":
            cols = self.data_validation_utils.processed_curtailments_i90_required_cols
        elif dataset_type == "curtailments_i3":
            cols = self.data_validation_utils.processed_curtailments_i3_required_cols
        else:
            raise ValueError(f"Invalid dataset type: {dataset_type}")

        df = pd.DataFrame(columns=cols)
        df.index.name = 'id'
        return df

    # === MAIN PIPELINE ===
    def transform_raw_curtailment_data(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        print("\n" + "="*80)
        print(f"🔄 STARTING CURTAILMENT TRANSFORMATION")
        print("="*80)

        if df.empty:
            print("\n❌ EMPTY INPUT")
            return self._empty_output_df(dataset_type)

        pipeline = [
            (self._apply_market_filters_and_id, {}),
            (self._standardize_datetime, {"dataset_type": dataset_type}),
            (self._select_and_finalize_columns, {"dataset_type": dataset_type}),
            (self._validate_final_data, {"dataset_type": dataset_type}),
        ]

        try:
            df_processed = df.copy()
            total_steps = len(pipeline)

            for i, (step_func, step_kwargs) in enumerate(pipeline, 1):
                print("\n" + "-"*50)
                print(f"📍 STEP {i}/{total_steps}: {step_func.__name__.replace('_', ' ').title()}")
                print("-"*50)

                df_processed = step_func(df_processed, **step_kwargs or {})

                print(f"\n📊 Data Status:")
                print(f"   Rows: {df_processed.shape[0]}")
                print(f"   Columns: {df_processed.shape[1]}")

                if df_processed.empty and step_func != self._validate_final_data:
                    raise ValueError(f"DataFrame empty after: {step_func.__name__}")

            print("\n✅ TRANSFORMATION COMPLETE")
            print(f"Final shape: {df_processed.shape}")
            print("="*80 + "\n")
            return df_processed

        except ValueError as e:
            print(f"\n❌ PROCESSING ERROR: {str(e)}")
            print("="*80 + "\n")
            raise

        except Exception as e:
            print(f"\n❌ UNEXPECTED ERROR: {str(e)}")
            print(traceback.format_exc())
            print("="*80 + "\n")
            raise