"""
DAG for UOF (Unidades de Oferta) tracking
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
from tracking.UOF_tracking import UOFTracker
from tracking.descarga_uofs_omie import download_uofs_from_omie
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
    'uof_tracking_etl',
    default_args=default_args,
    description='UOF (Unidades de Oferta) tracking pipeline',
    schedule_interval='15 4 * * *',  # Daily at 04:15 UTC
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    catchup=False,
    tags=['tracking', 'uof', 'omie', 'unidades_oferta'],
    on_failure_callback=dag_failure_email
)

# Task functions
def check_uof_files_func(**context):
    """
    Checks for the presence and recency of required UOF Excel files in the download directory.
    
    Returns:
        dict: A dictionary indicating whether the required files are present and recent (`success`), the download directory path, details about found files or missing patterns, and additional information.
    """
    download_dir = setup_tracking_directories()
    
    # Check for existing UOF files
    file_patterns = ['listado_unidades*.xlsx']
    result = check_required_files(download_dir, file_patterns, max_age_hours=24)
    
    if result['success']:
        print("✅ Required UOF files found and are recent enough")
        return {
            'success': True,
            'download_dir': download_dir,
            'files_found': result['files_found'],
            'details': result['details']
        }
    else:
        print("⚠️ Required UOF files not found or too old, need to download")
        return {
            'success': False,
            'download_dir': download_dir,
            'missing_patterns': result['missing_patterns'],
            'details': result['details']
        }

async def download_uof_files_func(**context):
    """
    Asynchronously downloads UOF files from OMIE if they are missing or outdated.
    
    Checks for the presence and recency of required UOF files using the result from the previous task. If files are already present and up-to-date, the download is skipped. Otherwise, attempts to download the files and returns a structured result indicating success or failure.
    
    Returns:
        dict: A dictionary with keys indicating success status, download directory, messages, and error details if applicable.
    """
    ti = context['ti']
    check_result = ti.xcom_pull(task_ids='check_uof_files')
    
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
        print(f"Downloading UOF files to {download_dir}")
        
        # Download UOF list
        await download_uofs_from_omie(download_dir)
        
        return {
            'success': True,
            'download_dir': download_dir,
            'message': 'Files downloaded successfully',
            'details': {}
        }
    except Exception as e:
        print(f"Error downloading UOF files: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'details': {}
        }

def process_uof_tracking_func(**context):
    """
    Processes UOF tracking data by locating the latest downloaded UOF Excel file and invoking the UOFTracker to process its contents.
    
    Raises a ValueError if the download task failed or a FileNotFoundError if no UOF file is found. Returns a dictionary indicating success or failure, including the processed file path or error details.
    """
    ti = context['ti']
    download_result = ti.xcom_pull(task_ids='download_uof_files')
    
    if not download_result['success']:
        raise ValueError(f"Cannot process UOF tracking: {download_result.get('error', 'Download failed')}")
    
    try:
        download_dir = download_result['download_dir']
        
        # Find the most recent UOF file
        latest_file = get_latest_file_by_pattern(download_dir, 'listado_unidades*.xlsx')
        
        if not latest_file:
            raise FileNotFoundError("No UOF files found after download")
        
        print(f"Processing UOF tracking with file: {latest_file}")
        
        # Process UOFs
        uof_tracker = UOFTracker()
        uof_tracker.process_uofs(latest_file)
        
        return {
            'success': True,
            'file_processed': latest_file,
            'message': 'UOF tracking completed successfully',
            'details': {}
        }
        
    except Exception as e:
        print(f"Error in UOF tracking: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'details': {}
        }

def cleanup_uof_files_func(**context):
    """
    Removes the most recent downloaded UOF Excel file from the download directory if it exists.
    
    This function is intended to clean up temporary files after processing, ensuring that the latest UOF file is deleted regardless of upstream task outcomes.
    """
    ti = context['ti']
    download_result = ti.xcom_pull(task_ids='download_uof_files')
    
    if download_result and download_result.get('download_dir'):
        download_dir = download_result['download_dir']
        latest_file = get_latest_file_by_pattern(download_dir, 'listado_unidades*.xlsx')
        
        if latest_file and os.path.exists(latest_file):
            try:
                os.remove(latest_file)
                print(f"🗑️ Cleaned up file: {latest_file}")
            except Exception as e:
                print(f"⚠️ Error cleaning up file {latest_file}: {e}")

# Task 1: Check for existing UOF files
check_uof_files = PythonOperator(
    task_id='check_uof_files',
    python_callable=check_uof_files_func,
    dag=dag,
)

# Task 2: Download UOF files if needed
download_uof_files = PythonOperator(
    task_id='download_uof_files',
    python_callable=download_uof_files_func,
    dag=dag,
    on_failure_callback=task_failure_email
)

# Task 3: Check download status
check_download_status = PythonOperator(
    task_id='check_download_status',
    python_callable=update_pipeline_stage_status,
    op_kwargs={
        'stage_name': 'extraction',
        'current_stage_task_id': 'download_uof_files'
    },
    provide_context=True,
    dag=dag,
    on_failure_callback=task_failure_email
)

# Task 4: Process UOF tracking
process_uof_tracking = PythonOperator(
    task_id='process_uof_tracking',
    python_callable=process_uof_tracking_func,
    dag=dag,
    on_failure_callback=task_failure_email
)

# Task 5: Check processing status
check_processing_status = PythonOperator(
    task_id='check_processing_status',
    python_callable=update_pipeline_stage_status,
    op_kwargs={
        'stage_name': 'transformation',
        'current_stage_task_id': 'process_uof_tracking'
    },
    provide_context=True,
    dag=dag,
    on_failure_callback=task_failure_email
)

# Task 6: Cleanup files
cleanup_files = PythonOperator(
    task_id='cleanup_files',
    python_callable=cleanup_uof_files_func,
    dag=dag,
    trigger_rule='all_done'  # Run regardless of upstream success/failure
)

# Task 7: Finalize pipeline
finalize_pipeline = PythonOperator(
    task_id='finalize_pipeline',
    python_callable=update_pipeline_stage_status,
    op_kwargs={
        'stage_name': 'load',
        'current_stage_task_id': 'process_uof_tracking'
    },
    provide_context=True,
    dag=dag,
    on_failure_callback=task_failure_email
)

# Define task dependencies
check_uof_files >> download_uof_files >> check_download_status >> \
process_uof_tracking >> check_processing_status >> finalize_pipeline >> cleanup_files 