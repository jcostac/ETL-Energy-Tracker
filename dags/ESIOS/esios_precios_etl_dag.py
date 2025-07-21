from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta, timezone
import pendulum
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from extract.esios_extractor import ESIOSPreciosExtractor
from transform.esios_transform import TransformadorESIOS
from load.data_lake_loader import DataLakeLoader
from helpers.email_triggers import dag_failure_email, dag_success_email, task_failure_email, task_success_email
from helpers.pipeline_status_helpers import update_pipeline_stage_status


default_args = {
    #DAG has no retry logic for failures. 
    'owner': 'jcosta',
    'depends_on_past': False,
    'email_on_failure': False, #set to false bc we have custom callbacks (email_triggers.py)
    'email_on_retry': False, #set to false bc we have custom callbacks (email_triggers.py)
    'email_on_success': False, #set to false bc we have custom callbacks (email_triggers.py)
}

# -- DAG Definition --
dag_esios_precios = DAG(
    'esios_precios_etl', #unique identifier for the DAG
    default_args=default_args,
    description='ETL pipeline for downloading and processing ESIOS electricity price data',
    schedule='0 23 * * *',  # Daily at 22:00 UTC
    start_date=pendulum.datetime(2025,  1, 1), # May 1st 2025
    catchup=True, # This will backfill data for all days since the start date
    tags=['esios', 'electricidad', 'precios', 'etl'],
    #dag_run_timeout=timedelta(hours=1), # This is the maximum time the DAG can run before being killed
    
    #custom callbacks for fails (email_triggers.py), if any of the tasks fail, the email_triggers.py will be called for failure.
    on_failure_callback=dag_failure_email
)

# Task functions to defer instantiation on load
def extract_esios_prices_func(fecha_inicio, fecha_fin):
    """
    Extracts electricity price data for all markets from ESIOS between the specified start and end dates.
    
    Parameters:
        fecha_inicio: Start date for data extraction.
        fecha_fin: End date for data extraction.
    
    Returns:
        dict: Extracted data for all markets within the given date range.
    """
    extractor = ESIOSPreciosExtractor()
    return extractor.extract_data_for_all_markets(
        fecha_inicio=fecha_inicio,
        fecha_fin=fecha_fin
    )

def transform_esios_prices_func(fecha_inicio, fecha_fin, mode):
    """
    Transforms ESIOS electricity price data for all markets within the specified date range and mode.
    
    Parameters:
        fecha_inicio (str): Start date for the transformation period.
        fecha_fin (str): End date for the transformation period.
        mode (str): Transformation mode to apply.
    
    Returns:
        dict: Transformed data for all markets within the given date range.
    """
    transformer = TransformadorESIOS()
    return transformer.transform_data_for_all_markets(
        fecha_inicio=fecha_inicio,
        fecha_fin=fecha_fin,
        mode=mode
    )

def load_esios_prices_func(transformed_data_dict):
    """
    Load transformed ESIOS electricity price data into the local data lake.
    
    Parameters:
        transformed_data_dict (dict): Dictionary containing transformed ESIOS price data to be loaded.
    
    Returns:
        dict: Result of the load operation, typically including status and metadata.
    """
    loader = DataLakeLoader()
    return loader.load_transformed_data_esios(
        transformed_data_dict=transformed_data_dict
    )

# -- Task 1: Extract ESIOS price data --
extract_esios_prices = PythonOperator(
    task_id='extract_esios_prices',
    python_callable=extract_esios_prices_func,
    op_kwargs={'fecha_inicio': '{{ ds }}', 'fecha_fin': '{{ ds }}'},
    dag=dag_esios_precios
)

# -- Task 2: Process extraction output and update status
check_extraction = PythonOperator(
    task_id='check_extraction',
    python_callable=update_pipeline_stage_status,
    op_kwargs={
        'stage_name': 'extraction',
        'current_stage_task_id': 'extract_esios_prices'
    },
    provide_context=True,
    dag=dag_esios_precios,
    on_failure_callback=task_failure_email
)

# -- Task 3: Transform ESIOS price data --
transform_esios_prices = PythonOperator(
    task_id='transform_esios_prices',
    python_callable=transform_esios_prices_func,
    op_kwargs={
        'fecha_inicio': '{{ ds }}', 
        'fecha_fin': '{{ ds }}', 
        'mode': 'single'
    },
    dag=dag_esios_precios
)

# -- Task 4: Process transform output and update status
check_transform = PythonOperator(
    task_id='check_transform',
    python_callable=update_pipeline_stage_status,
    op_kwargs={
        'stage_name': 'transformation',
        'current_stage_task_id': 'transform_esios_prices'
    },
    provide_context=True,
    dag=dag_esios_precios,
    on_failure_callback=task_failure_email
)

# -- Task 5: Load ESIOS price data --
load_esios_prices_to_datalake = PythonOperator(
    task_id='load_esios_prices_to_datalake',
    python_callable=load_esios_prices_func,
    op_kwargs={
        'transformed_data_dict': "{{ ti.xcom_pull(task_ids='check_transform') }}"
    },
    dag=dag_esios_precios,
)

# -- Task 6: Process load output and finalize pipeline status
finalize_pipeline = PythonOperator(
    task_id='finalize_pipeline',
    python_callable=update_pipeline_stage_status,
    op_kwargs={
        'stage_name': 'load',
        'current_stage_task_id': 'load_esios_prices_to_datalake'
    },
    provide_context=True,
    dag=dag_esios_precios,
    on_failure_callback=task_failure_email
)

# -- Finally: Define task dependencies --
extract_esios_prices >> check_extraction >> transform_esios_prices >> check_transform >> load_esios_prices_to_datalake >> finalize_pipeline
