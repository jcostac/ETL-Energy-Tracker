from airflow.utils.email import send_email
from typing import Dict, Any
import os

# Email recipients can be defined as constants or loaded from environment variables
RECIPIENTS = os.environ.get('AIRFLOW_ALERT_EMAILS', 'jcosta@optimizeenergy.es,psanchez@optimizeenergy.es').split(',')

def dag_success_email_func(**kwargs: Dict[str, Any]) -> None:
    """Send email notification when a DAG completes successfully.
    
    Args:
        **kwargs: Keyword arguments from Airflow context
    """
    dag_id = kwargs['dag'].dag_id
    execution_date = kwargs['execution_date'].strftime('%Y-%m-%d %H:%M:%S')
    
    send_email(
        to=RECIPIENTS,
        subject=f"Airflow Success: {dag_id}",
        html_content=f"""
        <p>Your DAG <b>{dag_id}</b> completed successfully!</p>
        <p>Execution date: {execution_date}</p>
        """,
    )

def dag_failure_email_func(**kwargs: Dict[str, Any]) -> None:
    """Send email notification when a DAG fails.
    
    Args:
        **kwargs: Keyword arguments from Airflow context
    """
    dag_id = kwargs['dag'].dag_id
    execution_date = kwargs['execution_date'].strftime('%Y-%m-%d %H:%M:%S')
    exception = kwargs.get('exception', 'No exception information available')
    
    send_email(
        to=RECIPIENTS,
        subject=f"❌ Airflow Failure: {dag_id}",
        html_content=f"""
        <p>Your DAG <b>{dag_id}</b> failed!</p>
        <p>Execution date: {execution_date}</p>
        <p>Error details:</p>
        <pre>{exception}</pre>
        """,
    )

def task_success_email(**kwargs: Dict[str, Any]) -> None:
    """Send email notification when a task completes successfully.
    
    Args:
        **kwargs: Keyword arguments from Airflow context
    """
    task_id = kwargs['task_instance'].task_id
    dag_id = kwargs['task_instance'].dag_id
    execution_date = kwargs['execution_date'].strftime('%Y-%m-%d %H:%M:%S')
    
    send_email(
        to=RECIPIENTS,
        subject=f"Airflow Task Success: {task_id} in {dag_id}",
        html_content=f"""
        <p>Task <b>{task_id}</b> in DAG <b>{dag_id}</b> completed successfully!</p>
        <p>Execution date: {execution_date}</p>
        """,
    )

def task_failure_email(**kwargs: Dict[str, Any]) -> None:
    """Send email notification when a task fails.
    
    Args:
        **kwargs: Keyword arguments from Airflow context
    """
    task_id = kwargs['task_instance'].task_id
    dag_id = kwargs['task_instance'].dag_id
    execution_date = kwargs['execution_date'].strftime('%Y-%m-%d %H:%M:%S')
    exception = kwargs.get('exception', 'No exception information available')
    
    send_email(
        to=RECIPIENTS,
        subject=f"❌ Airflow Task Failure: {task_id} in {dag_id}",
        html_content=f"""
        <p>Task <b>{task_id}</b> in DAG <b>{dag_id}</b> failed!</p>
        <p>Execution date: {execution_date}</p>
        <p>Error details:</p>
        <pre>{exception}</pre>
        """,
    )