"""
DAG for UP (Unidades de Programación) tracking
"""
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
from tracking.UP_tracking import UPTracker
from tracking.descarga_up_list import descargador_UP_list
from dags.tracking.tracking_helpers import check_required_files, get_latest_file_by_pattern, setup_tracking_directories
from dags.helpers.email_triggers import dag_failure_email, task_failure_email
from dags.helpers.pipeline_status_helpers import update_pipeline_stage_status

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
    'up_tracking_etl',
    default_args=default_args,
    description='UP (Unidades de Programación) tracking pipeline',
    schedule_interval='0 4 * * *',  # Daily at 04:00 UTC
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    catchup=False,
    tags=['tracking', 'up', 'esios', 'unidades_programacion'],
    on_failure_callback=dag_failure_email
)

# Task functions
def check_up_files_func(**context):
    """
    Checks whether recent UP export CSV files exist in the designated download directory.
    
    Returns:
        dict: A dictionary indicating success status, the download directory path, and details about found or missing files.
    """
    download_dir = setup_tracking_directories()
    
    # Check for existing UP export files
    file_patterns = ['export_unidades-de-programacion*.csv']
    result = check_required_files(download_dir, file_patterns, max_age_hours=24)
    
    if result['success']:
        print("✅ Required UP files found and are recent enough")
        return {
            'success': True,
            'download_dir': download_dir,
            'files_found': result['files_found'],
            'details': result['details']
        }
    else:
        print("⚠️ Required UP files not found or too old, need to download")
        return {
            'success': False,
            'download_dir': download_dir,
            'missing_patterns': result['missing_patterns'],
            'details': result['details']
        }

async def download_up_files_func(**context):
    """
    Download UP export files from ESIOS if they are missing or outdated.
    
    Checks if the required UP files are already present and recent; if so, skips downloading. Otherwise, downloads the files asynchronously to the specified directory. Returns a dictionary indicating success or failure, along with relevant details.
    """
    ti = context['ti']
    check_result = ti.xcom_pull(task_ids='check_up_files')
    
    if check_result['success']:
        print("Files already exist, skipping download")
        return {
            'success': True,
            'download_dir': check_result['download_dir'],
            'message': 'Files already exist',
            'details': check_result['details']
        }
    
    try:
        download_dir = check_result['download_dir']
        print(f"Downloading UP files to {download_dir}")
        
        # Download UP list
        await descargador_UP_list(download_dir)
        
        return {
            'success': True,
            'download_dir': download_dir,
            'message': 'Files downloaded successfully',
            'details': {}
        }
    except Exception as e:
        print(f"Error downloading UP files: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'details': {}
        }

def process_up_tracking_func(**context):
    """
    Processes the UP tracking data by locating and handling the latest UP export file.
    
    This function retrieves the result of the UP file download task, raises an error if the download failed, and processes the most recent UP export CSV file using the UPTracker. Returns a dictionary indicating success or failure, along with relevant details.
    
    Returns:
        dict: A dictionary containing the success status, processed file path (if successful), message, and error details if any.
    """
    ti = context['ti']
    download_result = ti.xcom_pull(task_ids='download_up_files')
    
    if not download_result['success']:
        raise ValueError(f"Cannot process UP tracking: {download_result.get('error', 'Download failed')}")
    
    try:
        download_dir = download_result['download_dir']
        
        # Find the most recent UP export file
        latest_file = get_latest_file_by_pattern(download_dir, 'export_unidades-de-programacion*.csv')
        
        if not latest_file:
            raise FileNotFoundError("No UP export files found after download")
        
        print(f"Processing UP tracking with file: {latest_file}")
        
        # Process UPs
        up_tracker = UPTracker()
        up_tracker.process_ups(latest_file)
        
        return {
            'success': True,
            'file_processed': latest_file,
            'message': 'UP tracking completed successfully',
            'details': {}
        }
        
    except Exception as e:
        print(f"Error in UP tracking: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'details': {}
        }

def cleanup_up_files_func(**context):
    """
    Deletes the most recent UP export CSV file from the download directory if it exists.
    
    Removes the latest file matching the UP export pattern after pipeline execution, logging any errors encountered during deletion.
    """
    ti = context['ti']
    download_result = ti.xcom_pull(task_ids='download_up_files')
    
    if download_result and download_result.get('download_dir'):
        download_dir = download_result['download_dir']
        latest_file = get_latest_file_by_pattern(download_dir, 'export_unidades-de-programacion*.csv')
        
        if latest_file and os.path.exists(latest_file):
            try:
                os.remove(latest_file)
                print(f"🗑️ Cleaned up file: {latest_file}")
            except Exception as e:
                print(f"⚠️ Error cleaning up file {latest_file}: {e}")

# Task 1: Check for existing UP files
check_up_files = PythonOperator(
    task_id='check_up_files',
    python_callable=check_up_files_func,
    dag=dag,
)

# Task 2: Download UP files if needed
download_up_files = PythonOperator(
    task_id='download_up_files',
    python_callable=download_up_files_func,
    dag=dag,
    on_failure_callback=task_failure_email
)

# Task 3: Check download status
check_download_status = PythonOperator(
    task_id='check_download_status',
    python_callable=update_pipeline_stage_status,
    op_kwargs={
        'stage_name': 'extraction',
        'current_stage_task_id': 'download_up_files'
    },
    provide_context=True,
    dag=dag,
    on_failure_callback=task_failure_email
)

# Task 4: Process UP tracking
process_up_tracking = PythonOperator(
    task_id='process_up_tracking',
    python_callable=process_up_tracking_func,
    dag=dag,
    on_failure_callback=task_failure_email
)

# Task 5: Check processing status
check_processing_status = PythonOperator(
    task_id='check_processing_status',
    python_callable=update_pipeline_stage_status,
    op_kwargs={
        'stage_name': 'transformation',
        'current_stage_task_id': 'process_up_tracking'
    },
    provide_context=True,
    dag=dag,
    on_failure_callback=task_failure_email
)

# Task 6: Cleanup files
cleanup_files = PythonOperator(
    task_id='cleanup_files',
    python_callable=cleanup_up_files_func,
    dag=dag,
    trigger_rule='all_done'  # Run regardless of upstream success/failure
)

# Task 7: Finalize pipeline
finalize_pipeline = PythonOperator(
    task_id='finalize_pipeline',
    python_callable=update_pipeline_stage_status,
    op_kwargs={
        'stage_name': 'load',
        'current_stage_task_id': 'process_up_tracking'
    },
    provide_context=True,
    dag=dag,
    on_failure_callback=task_failure_email
)

# Define task dependencies
check_up_files >> download_up_files >> check_download_status >> \
process_up_tracking >> check_processing_status >> finalize_pipeline >> cleanup_files 