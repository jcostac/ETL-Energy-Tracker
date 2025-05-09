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
from extract.i90_extractor import I90VolumenesExtractor
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
    'i90_volumenes_etl',
    default_args=default_args,
    description='ETL pipeline for downloading and processing I90 market volume data',
    schedule_interval='30 3 * * *',  # Daily at 03:30 UTC
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    catchup=False,
    tags=['i90', 'market', 'volumes', 'etl'],
    on_failure_callback=dag_failure_email
)

# Task functions to defer instantiation on load (which can cause issues with the DAG like SQL too many connections error)
def extract_i90_volumes_func(fecha_inicio_carga, fecha_fin_carga):
    extractor = I90VolumenesExtractor()
    return extractor.extract_data_for_all_markets(
        fecha_inicio_carga=fecha_inicio_carga, 
        fecha_fin_carga=fecha_fin_carga
    )

def transform_i90_volumes_func(start_date, end_date, dataset_types, transform_type):
    transformer = TransformadorI90()
    return transformer.transform_data_for_all_markets(
        start_date=start_date,
        end_date=end_date,
        dataset_type=dataset_types,
        transform_type=transform_type
    )

def load_i90_volumes_func(transformed_data_dict):
    loader = LocalDataLakeLoader()
    return loader.load_transformed_data_volumenes_i90(
        transformed_data_dict=transformed_data_dict
    )

# Task 1: Extract I90 volume data
extract_i90_volumes = PythonOperator(
    task_id='extract_i90_volumes',
    python_callable=extract_i90_volumes_func,
    op_kwargs={'fecha_inicio_carga': '{{ ds }}', 'fecha_fin_carga': '{{ ds }}'},
    dag=dag,
)

# Task 2: Check extraction status
check_extraction_i90_volumes = PythonOperator(
    task_id='check_extraction_i90_volumes',
    python_callable=update_pipeline_stage_status,
    op_kwargs={
        'stage_name': 'extraction',
        'current_stage_task_id': 'extract_i90_volumes'
    },
    provide_context=True,
    dag=dag,
    on_failure_callback=task_failure_email
)

# Task 3: Transform I90 volume data
transform_i90_volumes = PythonOperator(
    task_id='transform_i90_volumes',
    python_callable=transform_i90_volumes_func,
    op_kwargs={
        'start_date': '{{ ds }}',
        'end_date': '{{ ds }}',
        'dataset_types': 'volumenes_i90',
        'transform_type': 'single'
    },
    dag=dag,
)

# Task 4: Check transformation status
check_transform_i90_volumes = PythonOperator(
    task_id='check_transform_i90_volumes',
    python_callable=update_pipeline_stage_status,
    op_kwargs={
        'stage_name': 'transformation',
        'current_stage_task_id': 'transform_i90_volumes'
    },
    provide_context=True,
    dag=dag,
    on_failure_callback=task_failure_email
)

# Task 5: Load I90 volume data to data lake
load_i90_volumes_to_datalake = PythonOperator(
    task_id='load_i90_volumes_to_datalake',
    python_callable=load_i90_volumes_func,
    op_kwargs={
        'transformed_data_dict': "{{ ti.xcom_pull(task_ids='check_transform_i90_volumes') }}" },
    dag=dag,
)

# Task 6: Finalize pipeline status
finalize_pipeline_i90_volumes = PythonOperator(
    task_id='finalize_pipeline_i90_volumes',
    python_callable=update_pipeline_stage_status,
    op_kwargs={
        'stage_name': 'load',
        'current_stage_task_id': 'load_i90_volumes_to_datalake'
    },
    provide_context=True,
    dag=dag,
    on_failure_callback=task_failure_email
)

# Define task dependencies
extract_i90_volumes >> check_extraction_i90_volumes >> \
transform_i90_volumes >> check_transform_i90_volumes >> \
load_i90_volumes_to_datalake >> finalize_pipeline_i90_volumes 