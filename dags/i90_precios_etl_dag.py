from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

# Import necessary modules
from extract.i90_extractor import extract_i90_data
from transform.i90_transform import transform_i90_data
from transform.carga_i90 import process_i90_data
from load.data_lake_loader import load_data_to_datalake

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
    'i90_precios_etl',
    default_args=default_args,
    description='ETL pipeline for I90 market price data',
    schedule_interval='0 3 * * *',  # Daily at 03:00 UTC
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['i90', 'market', 'prices', 'etl'],
)

# Task 1: Extract I90 price data
extract_i90_prices = PythonOperator(
    task_id='extract_i90_prices',
    python_callable=extract_i90_data,
    op_kwargs={'data_type': 'prices'},
    dag=dag,
)

# Task 2: Transform I90 price data
transform_i90_prices = PythonOperator(
    task_id='transform_i90_prices',
    python_callable=transform_i90_data,
    op_kwargs={'data_type': 'prices'},
    dag=dag,
)

# Task 3: Process I90 price data with specialized logic
load_i90_prices_specific = PythonOperator(
    task_id='load_i90_prices_specific',
    python_callable=process_i90_data,
    op_kwargs={'data_type': 'prices'},
    dag=dag,
)

# Task 4: Load I90 price data to data lake
load_i90_prices_to_datalake = PythonOperator(
    task_id='load_i90_prices_to_datalake',
    python_callable=load_data_to_datalake,
    op_kwargs={'source': 'i90', 'data_type': 'prices'},
    dag=dag,
)

# Define task dependencies
extract_i90_prices >> transform_i90_prices >> load_i90_prices_specific >> load_i90_prices_to_datalake
