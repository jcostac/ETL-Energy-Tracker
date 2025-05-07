from __future__ import annotations

import pendulum

from airflow.models.dag import DAG
from airflow.providers.standard.operators.python import PythonOperator
# Assuming your scripts are importable
from extract.esios_extractor import ESIOSPreciosExtractor
from transform.esios_transform import TransformadorESIOS
from load.local_data_lake_loader import LocalDataLakeLoader
from configs.esios_config import ESIOSConfig # To get market list and value columns

# --- Define Default Args ---
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': pendulum.duration(minutes=5),
}

# --- Helper Functions to be called by PythonOperator ---

def extract_daily_data(ds, **kwargs):
    """Extracts data for a specific day.
    
    Args:
        ds: The date to extract data for.
        kwargs: Additional keyword arguments.

    Returns:
       Nothing. The extractor saves files, so no explicit return needed since transform task reads from those files. 
    """
    print(f"Extracting data for date: {ds}")
    extractor = ESIOSPreciosExtractor()
    # For a daily backfill, start and end date are the same
    # The extractor saves files, so no explicit return needed if transform reads from those files
    extractor.extract_data_for_all_markets(fecha_inicio_carga=ds, fecha_fin_carga=ds)
    print(f"Extraction complete for {ds}")

def transform_daily_data(ds, **kwargs):
    """Transforms data for a specific day.
    
    Args:
        ds: The date to transform data for.
        kwargs: Additional keyword arguments.

    Returns:
        A dictionary of processed data for each market.
    """
    print(f"Transforming data for date: {ds}")
    transformer = TransformadorESIOS()
    # Mode 'single' uses the start_date for the specific day
    # mercados can be fetched from config or defined here
    mercados_to_process = ESIOSConfig().esios_precios_markets
    processed_data_map = transformer.transform_data_for_all_markets(
        start_date=ds,
        mode='single',
        mercados=mercados_to_process
    )
    print(f"Transformation complete for {ds}")
    # Filter out None or empty DataFrames before returning
    return {
        market: df 
        for market, df in processed_data_map.items() 
        if df is not None and not df.empty
    }


def load_daily_data_for_market(market_name, processed_df_for_market, **kwargs):
    """Loads transformed data for a specific market."""
    print(f"Loading data for market: {market_name}")
    loader = LocalDataLakeLoader()
    
    # You'll need a way to determine the value_col, perhaps from a config or a mapping
    # For example:
    value_col_mapping = {
        'diario': 'precio',
        'intra': 'precio_intra', # Fictional, adjust to your actual column names
        'secundaria': 'precio_secundaria',
        'terciaria': 'precio_terciaria',
        'rr': 'precio_rr'
        # Add other markets and their respective value columns
    }
    value_col = value_col_mapping.get(market_name, 'valor') # Default if not found

    if processed_df_for_market is not None and not processed_df_for_market.empty:
        loader.save_processed_data(
            processed_df=processed_df_for_market,
            mercado=market_name,
            value_col=value_col, # Make sure this is correct for each market
            dataset_type='precios'
        )
        print(f"Loading complete for market: {market_name}")
    else:
        print(f"No data to load for market: {market_name}")

# --- DAG Definition ---
with DAG(
    dag_id='esios_daily_backfill_etl',
    default_args=default_args,
    description='Daily ETL pipeline for ESIOS data backfill from 2022 onwards.',
    schedule_interval='@daily', # Runs daily
    start_date=pendulum.datetime(2022, 1, 1, tz="UTC"), # Backfill start
    catchup=True, # IMPORTANT for backfilling
    tags=['esios', 'backfill'],
) as dag:

    extract_task = PythonOperator(
        task_id='extract_esios_data',
        python_callable=extract_daily_data,
        # op_kwargs={'ds': '{{ ds }}'} is implicitly passed if provide_context=True (default)
    )

    transform_task = PythonOperator(
        task_id='transform_esios_data',
        python_callable=transform_daily_data,
        # op_kwargs={'ds': '{{ ds }}'}
    )

    # Dynamically create load tasks for each market
    # This assumes ESIOSConfig().esios_precios_markets is available and correct
    # and that transform_task returns a dictionary as described.
    
    # Get the dictionary of processed data from transform_task
    # This uses XComs implicitly
    processed_market_data = transform_task.output 

    for market in ESIOSConfig().esios_precios_markets:
        # Using a PythonOperator that pulls the specific market's DataFrame
        def _callable_load_market_specific(market_name_param, processed_data_dict_param, **context):
            # The processed_data_dict_param will be the entire dictionary from transform_task.output
            # We need to extract the DataFrame for the current market_name_param
            df_for_this_market = processed_data_dict_param.get(market_name_param)
            if df_for_this_market is not None:
                load_daily_data_for_market(market_name=market_name_param, processed_df_for_market=df_for_this_market)
            else:
                print(f"No processed data found for market {market_name_param} in the XCom result.")

        load_market_task = PythonOperator(
            task_id=f'load_{market}_data',
            python_callable=_callable_load_market_specific,
            op_kwargs={
                'market_name_param': market,
                'processed_data_dict_param': processed_market_data # Pass the XComArg here
            },
        )
        
        # Define dependencies: extract -> transform -> load_market_task
        extract_task >> transform_task >> load_market_task
