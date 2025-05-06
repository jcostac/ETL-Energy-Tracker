from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

# Import necessary modules
from extract.esios_extractor import extract_esios_data
from transform.esios_transform import transform_esios_data
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
    'esios_precios_etl',
    default_args=default_args,
    description='ETL pipeline for ESIOS electricity price data',
    schedule_interval='0 2 * * *',  # Daily at 02:00 UTC
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['esios', 'electricity', 'prices', 'etl'],
)

# Task 1: Extract ESIOS price data
extract_esios_prices = PythonOperator(
    task_id='extract_esios_prices',
    python_callable=extract_esios_data,
    op_kwargs={'data_type': 'prices'},
    dag=dag,
)

# Task 2: Transform ESIOS price data
transform_esios_prices = PythonOperator(
    task_id='transform_esios_prices',
    python_callable=transform_esios_data,
    op_kwargs={'data_type': 'prices'},
    dag=dag,
)

# Task 3: Load ESIOS price data to data lake
load_esios_prices_to_datalake = PythonOperator(
    task_id='load_esios_prices_to_datalake',
    python_callable=load_data_to_datalake,
    op_kwargs={'source': 'esios', 'data_type': 'prices'},
    dag=dag,
)

# Define task dependencies
extract_esios_prices >> transform_esios_prices >> load_esios_prices_to_datalake
