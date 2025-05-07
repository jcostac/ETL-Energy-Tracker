from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

# Import necessary modules
from extract.i90_extractor import extract_i90_data
from transform.i90_transform import transform_i90_data
from transform.carga_i90 import process_i90_data
from load.local_data_lake_loader import LocalDataLakeLoader

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': True,
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
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['i90', 'market', 'volumes', 'etl'],
)

# Task 1: Extract I90 volume data
extract_i90_volumes = PythonOperator(
    task_id='extract_i90_volumes',
    python_callable=extract_i90_data,
    op_kwargs={'data_type': 'volumes'},
    dag=dag,
)

# Task 2: Transform I90 volume data
transform_i90_volumes = PythonOperator(
    task_id='transform_i90_volumes',
    python_callable=transform_i90_data,
    op_kwargs={'data_type': 'volumes'},
    dag=dag,
)

# Task 3: Process I90 volume data with specialized logic
load_i90_volumes_specific = PythonOperator(
    task_id='load_i90_volumes_specific',
    python_callable=process_i90_data,
    op_kwargs={'data_type': 'volumes'},
    dag=dag,
)

# Task 4: Load I90 volume data to data lake
load_i90_volumes_to_datalake = PythonOperator(
    task_id='load_i90_volumes_to_datalake',
    python_callable=load_data_to_datalake,
    op_kwargs={'source': 'i90', 'data_type': 'volumes'},
    dag=dag,
)

# Define task dependencies
extract_i90_volumes >> transform_i90_volumes >> load_i90_volumes_specific >> load_i90_volumes_to_datalake 