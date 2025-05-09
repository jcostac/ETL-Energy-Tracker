from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os
from pathlib import Path
import pendulum

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Import necessary modules
from extract.i90_extractor import I90PreciosExtractor
from transform.i90_transform import TransformadorI90
from load.local_data_lake_loader import LocalDataLakeLoader
from helpers.email_triggers import dag_failure_email, task_failure_email
from helpers.pipeline_status_helpers import update_pipeline_stage_status

default_args = {
    'owner': 'jcosta',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# DAG Definition
dag = DAG(
    'i90_precios_etl',
    default_args=default_args,
    description='ETL pipeline for I90 market price data',
    schedule_interval='0 3 * * *',  # Daily at 03:00 UTC
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    catchup=False,
    tags=['i90', 'market', 'prices', 'etl'],
    on_failure_callback=dag_failure_email
)

# Task functions to defer instantiation on load
def extract_i90_prices_func(fecha_inicio_carga, fecha_fin_carga):
    extractor = I90PreciosExtractor()
    return extractor.extract_data_for_all_markets(
        fecha_inicio_carga=fecha_inicio_carga,
        fecha_fin_carga=fecha_fin_carga
    )

def transform_i90_prices_func(start_date, end_date, dataset_types, transform_type):
    transformer = TransformadorI90()
    return transformer.transform_data_for_all_markets(
        start_date=start_date,
        end_date=end_date,
        dataset_type=dataset_types,
        transform_type=transform_type
    )

def load_i90_prices_func(transformed_data_dict):
    loader = LocalDataLakeLoader()
    return loader.load_transformed_data_precios_i90(
        transformed_data_dict=transformed_data_dict
    )

# Task 1: Extract I90 price data
extract_i90_prices = PythonOperator(
    task_id='extract_i90_prices',
    python_callable=extract_i90_prices_func,
    op_kwargs={'fecha_inicio_carga': '{{ ds }}', 'fecha_fin_carga': '{{ ds }}'},
    dag=dag,
)

# Task 2: Check extraction status
check_extraction_i90_prices = PythonOperator(
    task_id='check_extraction_i90_prices',
    python_callable=update_pipeline_stage_status,
    op_kwargs={
        'stage_name': 'extraction',
        'current_stage_task_id': 'extract_i90_prices'
    },
    provide_context=True,
    dag=dag,
    on_failure_callback=task_failure_email
)

# Task 3: Transform I90 price data
transform_i90_prices = PythonOperator(
    task_id='transform_i90_prices',
    python_callable=transform_i90_prices_func,
    op_kwargs={
        'start_date': '{{ ds }}',
        'end_date': '{{ ds }}',
        'dataset_types':'precios_i90',
        'transform_type': 'single'
    },
    dag=dag,
)

# Task 4: Check transformation status
check_transform_i90_prices = PythonOperator(
    task_id='check_transform_i90_prices',
    python_callable=update_pipeline_stage_status,
    op_kwargs={
        'stage_name': 'transformation',
        'current_stage_task_id': 'transform_i90_prices'
    },
    provide_context=True,
    dag=dag,
    on_failure_callback=task_failure_email
)

# Task 5: Load I90 price data to data lake
load_i90_prices_to_datalake = PythonOperator(
    task_id='load_i90_prices_to_datalake',
    python_callable=load_i90_prices_func,
    op_kwargs={
        'transformed_data_dict': "{{ ti.xcom_pull(task_ids='check_transform_i90_prices') }}" },
    dag=dag,
)

# Task 6: Finalize pipeline status
finalize_pipeline_i90_prices = PythonOperator(
    task_id='finalize_pipeline_i90_prices',
    python_callable=update_pipeline_stage_status,
    op_kwargs={
        'stage_name': 'load',
        'current_stage_task_id': 'load_i90_prices_to_datalake'
    },
    provide_context=True,
    dag=dag,
    on_failure_callback=task_failure_email
)

# Define task dependencies
extract_i90_prices >> check_extraction_i90_prices >> \
transform_i90_prices >> check_transform_i90_prices >> \
load_i90_prices_to_datalake >> finalize_pipeline_i90_prices
