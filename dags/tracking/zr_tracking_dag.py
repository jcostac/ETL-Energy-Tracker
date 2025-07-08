"""
DAG for ZR (Zonas de Regulación) tracking
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
from tracking.ZR_tracking import ZRTracker
from tracking.descarga_up_list import descargador_UP_list
from tracking.descarga_bsp_esios import download_bsp_list  # You'll need to implement this
from tracking.tracking_helpers import check_required_files, get_latest_file_by_pattern, setup_tracking_directories
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
    'zr_tracking_etl',
    default_args=default_args,
    description='ZR (Zonas de Regulación) tracking pipeline',
    schedule_interval='30 4 * * *',  # Daily at 04:30 UTC
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    catchup=False,
    tags=['tracking', 'zr', 'zonas_regulacion', 'esios'],
    on_failure_callback=dag_failure_email
)

# Task functions
def check_zr_files_func(**context):
    """Check if ZR files exist"""
    download_dir = setup_tracking_directories()
    
    # Check for existing UP export files and BSP files
    file_patterns = [
        'export_unidades-de-programacion*.csv',
        'BSP-aFRR*.xlsx',
        'BSP-aFRR*.csv'
    ]
    result = check_required_files(download_dir, file_patterns, max_age_hours=24)
    
    # For ZR, we need at least the UP export file and one BSP file
    up_files = [f for pattern, f in result['files_found'].items() 
                if 'export_unidades-de-programacion' in pattern]
    bsp_files = [f for pattern, f in result['files_found'].items() 
                 if 'BSP-aFRR' in pattern]
    
    if up_files and bsp_files:
        print("✅ Required ZR files found and are recent enough")
        return {
            'success': True,
            'download_dir': download_dir,
            'files_found': result['files_found'],
            'details': result['details']
        }
    else:
        missing = []
        if not up_files:
            missing.append('UP export file')
        if not bsp_files:
            missing.append('BSP file')
        
        print(f"⚠️ Required ZR files not found: {', '.join(missing)}")
        return {
            'success': False,
            'download_dir': download_dir,
            'missing_files': missing,
            'details': result['details']
        }

async def download_zr_files_func(**context):
    """Download ZR files (UP export and BSP)"""
    ti = context['ti']
    check_result = ti.xcom_pull(task_ids='check_zr_files')
    
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
        print(f"Downloading ZR files to {download_dir}")
        
        missing_files = check_result.get('missing_files', [])
        
        # Download UP export if missing
        if 'UP export file' in missing_files:
            await descargador_UP_list(download_dir)
            
        # Note: BSP file download is currently manual in your daemon
        # You'll need to implement download_bsp_list or handle this differently
        if 'BSP file' in missing_files:
            print("⚠️ BSP file download not implemented - this needs to be uploaded manually")
            # Uncomment when BSP download is implemented:
            # download_bsp_list(download_dir)
        
        return {
            'success': True,
            'download_dir': download_dir,
            'message': 'Available files downloaded successfully',
            'details': {'note': 'BSP list file needs manual upload'}
        }
    except Exception as e:
        print(f"Error downloading ZR files: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'details': {}
        }

def process_zr_tracking_func(**context):
    """Process ZR tracking"""
    ti = context['ti']
    download_result = ti.xcom_pull(task_ids='download_zr_files')
    
    if not download_result['success']:
        raise ValueError(f"Cannot process ZR tracking: {download_result.get('error', 'Download failed')}")
    
    try:
        download_dir = download_result['download_dir']
        
        # Find the most recent UP export file
        up_file = get_latest_file_by_pattern(download_dir, 'export_unidades-de-programacion*.csv')
        
        # Find the most recent BSP file (try both xlsx and csv)
        bsp_file = (get_latest_file_by_pattern(download_dir, 'BSP-aFRR*.xlsx') or 
                   get_latest_file_by_pattern(download_dir, 'BSP-aFRR*.csv'))
        
        if not up_file:
            raise FileNotFoundError("No UP export files found after download")
        if not bsp_file:
            raise FileNotFoundError("No BSP files found - may need manual upload")
        
        print(f"Processing ZR tracking with files:")
        print(f"  UP file: {up_file}")
        print(f"  BSP file: {bsp_file}")
        
        # Process ZRs
        zr_tracker = ZRTracker()
        zr_tracker.process_zonas(up_file, bsp_file)
        
        return {
            'success': True,
            'files_processed': {'up_file': up_file, 'bsp_file': bsp_file},
            'message': 'ZR tracking completed successfully',
            'details': {}
        }
        
    except Exception as e:
        print(f"Error in ZR tracking: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'details': {}
        }

def cleanup_zr_files_func(**context):
    """Cleanup downloaded files"""
    ti = context['ti']
    download_result = ti.xcom_pull(task_ids='download_zr_files')
    
    if download_result and download_result.get('download_dir'):
        download_dir = download_result['download_dir']
        
        # Clean up UP file
        up_file = get_latest_file_by_pattern(download_dir, 'export_unidades-de-programacion*.csv')
        if up_file and os.path.exists(up_file):
            try:
                os.remove(up_file)
                print(f"🗑️ Cleaned up UP file: {up_file}")
            except Exception as e:
                print(f"⚠️ Error cleaning up UP file {up_file}: {e}")
        
        # Note: We typically don't clean up BSP files as they're manually uploaded
        # and may be used by other processes

# Task 1: Check for existing ZR files
check_zr_files = PythonOperator(
    task_id='check_zr_files',
    python_callable=check_zr_files_func,
    dag=dag,
)

# Task 2: Download ZR files if needed
download_zr_files = PythonOperator(
    task_id='download_zr_files',
    python_callable=download_zr_files_func,
    dag=dag,
    on_failure_callback=task_failure_email
)

# Task 3: Check download status
check_download_status = PythonOperator(
    task_id='check_download_status',
    python_callable=update_pipeline_stage_status,
    op_kwargs={
        'stage_name': 'extraction',
        'current_stage_task_id': 'download_zr_files'
    },
    provide_context=True,
    dag=dag,
    on_failure_callback=task_failure_email
)

# Task 4: Process ZR tracking
process_zr_tracking = PythonOperator(
    task_id='process_zr_tracking',
    python_callable=process_zr_tracking_func,
    dag=dag,
    on_failure_callback=task_failure_email
)

# Task 5: Check processing status
check_processing_status = PythonOperator(
    task_id='check_processing_status',
    python_callable=update_pipeline_stage_status,
    op_kwargs={
        'stage_name': 'transformation',
        'current_stage_task_id': 'process_zr_tracking'
    },
    provide_context=True,
    dag=dag,
    on_failure_callback=task_failure_email
)

# Task 6: Cleanup files
cleanup_files = PythonOperator(
    task_id='cleanup_files',
    python_callable=cleanup_zr_files_func,
    dag=dag,
    trigger_rule='all_done'  # Run regardless of upstream success/failure
)

# Task 7: Finalize pipeline
finalize_pipeline = PythonOperator(
    task_id='finalize_pipeline',
    python_callable=update_pipeline_stage_status,
    op_kwargs={
        'stage_name': 'load',
        'current_stage_task_id': 'process_zr_tracking'
    },
    provide_context=True,
    dag=dag,
    on_failure_callback=task_failure_email
)

# Define task dependencies
check_zr_files >> download_zr_files >> check_download_status >> \
process_zr_tracking >> check_processing_status >> finalize_pipeline ## >> cleanup_files 