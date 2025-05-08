from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime, timedelta, timezone
import pendulum
from extract.esios_extractor import ESIOSPreciosExtractor
from transform.esios_transform import TransformadorESIOS
from load.local_data_lake_loader import LocalDataLakeLoader
from helpers.email_triggers import dag_failure_email, dag_success_email, task_failure_email, task_success_email
from helpers.pipeline_status_helpers import process_extraction_output, process_transform_output, finalize_pipeline_status, process_load_output


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
    schedule_interval='0 23 * * *',  # Daily at 22:00 UTC
    start_date=pendulum.datetime(2025,  1, 1), # May 1st 2025
    catchup=True, # This will backfill data for all days since the start date
    tags=['esios', 'electricidad', 'precios', 'etl'],
    dag_run_timeout=timedelta(hours=1), # This is the maximum time the DAG can run before being killed
    
    #custom callbacks for fails (email_triggers.py), if any of the tasks fail, the email_triggers.py will be called for failure.
    on_failure_callback=dag_failure_email
)


# -- Task 1: Extract ESIOS price data --
extract_esios_prices = PythonOperator(
    task_id='extract_esios_prices',
    python_callable=ESIOSPreciosExtractor().extract_data_for_all_markets,
    op_kwargs={'fecha_inicio_carga': '{{ ds }}', 'fecha_fin_carga': '{{ ds }}'},
    dag=dag_esios_precios
)

# -- Task 2: Process extraction output -> sends email if fails 
check_extraction_output = PythonOperator(
    task_id='check_extraction_output',
    python_callable=process_extraction_output,
    provide_context=True,
    dag=dag_esios_precios,
    on_failure_callback=task_failure_email
)

# -- Task 3: Transform ESIOS price data --
transform_esios_prices = PythonOperator(
    task_id='transform_esios_prices',
    python_callable=TransformadorESIOS().transform_data_for_all_markets,
    op_kwargs={'fecha_inicio': '{{ ds }}', 'fecha_fin': '{{ ds }}', 'mode': 'single'},
    dag=dag_esios_precios
)

# -- Task 4: Process transform output -> sends email if fails 
check_transform_output = PythonOperator(
    task_id='check_transform_output',
    python_callable=process_transform_output,
    provide_context=True,
    dag=dag_esios_precios,
    on_failure_callback=task_failure_email
)

# -- Task 5: Load ESIOS price data --
load_esios_prices_to_datalake = PythonOperator(
    task_id='load_esios_prices_to_datalake',
    python_callable=LocalDataLakeLoader().load_transformed_data_esios,
    op_kwargs={'transformed_data_dict': "{{ ti.xcom_pull(task_ids='process_transform') }}"},
    dag=dag_esios_precios,
)

# -- Task 6: Process load output -> sends email if fails 
process_load_output = PythonOperator(
    task_id='process_load_output',
    python_callable=process_load_output,
    provide_context=True,
    dag=dag_esios_precios,
    on_failure_callback=task_failure_email
)
# -- Task 7: Finalize pipeline status -> sends email if fails 
finalize_pipeline = PythonOperator(
    task_id='finalize_pipeline',
    python_callable=finalize_pipeline_status,
    provide_context=True,
    dag=dag_esios_precios,
    on_failure_callback=task_failure_email
)

# -- Finally: Define task dependencies --
extract_esios_prices >> check_extraction_output >> transform_esios_prices >> check_transform_output >> load_esios_prices_to_datalake >> finalize_pipeline
