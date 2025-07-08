import pretty_errors
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.email import EmailOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.email import send_email

def dag_success_callback(context):
    """Callback function to send email on DAG success"""
    return EmailOperator(
        task_id='dag_success_email',
        to='jj_costa@outlook.es',
        subject='DAG Completed Successfully: {{ dag.dag_id }}',
        html_content='The DAG has completed successfully at {{ ts }}.',
        dag=context['dag']
    ).execute(context=context)

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email': ['jj_costa@outlook.es'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'on_success_callback': dag_success_callback
}

# Define your DAG
with DAG(
    'example_dag',
    default_args=default_args,
    description='A simple example DAG',
    schedule='@daily',
    start_date=datetime(2025, 5, 1),
    catchup=False,
) as dag:
    
    start = EmptyOperator(
        task_id='start',
    )
    
    end = EmptyOperator(
        task_id='end',
    )
    
    start >> end