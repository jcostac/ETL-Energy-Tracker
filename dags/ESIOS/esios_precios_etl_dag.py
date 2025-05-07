from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime, timedelta, timezone
import pendulum
# Import necessary modules
from extract.esios_extractor import ESIOSPreciosExtractor
from transform.esios_transform import TransformadorESIOS
from load.local_data_lake_loader import LocalDataLakeLoader
from email_triggers import dag_failure_email, dag_success_email, task_failure_email, task_success_email


default_args = {
    'owner': 'jcosta',
    'depends_on_past': False,
    'email_on_failure': False, #set to false bc we have custom callbacks (email_triggers.py)
    'email_on_retry': False, #set to false bc we have custom callbacks (email_triggers.py)
    'email_on_success': False, #set to false bc we have custom callbacks (email_triggers.py)
    'retries': 4, #4 retries
    'retry_delay': timedelta(minutes=10), #10 minutes delay between retries
}

# DAG Definition
dag_esios_precios = DAG(
    'esios_precios_etl', #unique identifier for the DAG
    default_args=default_args,
    description='ETL pipeline for downloading and processing ESIOS electricity price data',
    schedule_interval='0 22 * * *',  # Daily at 22:00 UTC
    start_date=pendulum.datetime(2025,  1, 1, tz="UTC"), # May 1st 2025
    catchup=False, # This will backfill data for all days since the start date
    tags=['esios', 'electricidad', 'precios', 'etl'],
    dag_run_timeout=timedelta(hours=1), # This is the maximum time the DAG can run before being killed
    
    #custom callbacks for fails and successes (email_triggers.py)
    on_failure_callback=dag_failure_email,
    on_success_callback=dag_success_email,
)

# Task 1: Extract ESIOS price data
extract_esios_prices = PythonOperator(
    task_id='extract_esios_prices',
    python_callable=ESIOSPreciosExtractor().extract_data_for_all_markets,
    op_kwargs={'data_type': 'prices'},
    dag=dag_esios_precios,

    #custom callbacks for fails and successes (email_triggers.py)
    on_failure_callback=task_failure_email
)

# Task 2: Transform ESIOS price data
transform_esios_prices = PythonOperator(
    task_id='transform_esios_prices',
    python_callable=TransformadorESIOS().transform_data_for_all_markets,
    op_kwargs={'data_type': 'prices'},
    dag=dag_esios_precios,

    #custom callbacks for fails and successes (email_triggers.py)
    on_failure_callback=task_failure_email
)

# Task 3: Load ESIOS price data to data lake
load_esios_prices_to_datalake = PythonOperator(
    task_id='load_esios_prices_to_datalake',
    python_callable=LocalDataLakeLoader().save_processed_data,
    op_kwargs={'source': 'esios', 'data_type': 'prices'},
    dag=dag_esios_precios,

    #custom callbacks for fails and successes (email_triggers.py)
    on_failure_callback=task_failure_email
)

# Define task dependencies
extract_esios_prices >> transform_esios_prices >> load_esios_prices_to_datalake
