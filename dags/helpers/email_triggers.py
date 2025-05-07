from airflow.utils.email import send_email
from typing import Dict, Any
import os
from datetime import datetime
import traceback

# Email recipients can be defined as constants or loaded from environment variables
RECIPIENTS = os.environ.get('AIRFLOW_ALERT_EMAILS', 'jcosta@optimizeenergy.es,psanchez@optimizeenergy.es').split(',')

def _get_email_style() -> str:
    """Returns common CSS styles for email templates."""
    return """
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
        .container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .header { padding: 20px; text-align: center; border-radius: 5px; }
        .success { background-color: #d4edda; color: #155724; }
        .failure { background-color: #f8d7da; color: #721c24; }
        .content { padding: 20px; background-color: #f8f9fa; border-radius: 5px; }
        .details { margin-top: 20px; padding: 15px; background-color: #fff; border-radius: 5px; }
        .timestamp { color: #6c757d; font-size: 0.9em; }
        .error-details { background-color: #fff; padding: 15px; border-left: 4px solid #dc3545; }
        pre { white-space: pre-wrap; word-wrap: break-word; }
    </style>
    """

def _format_timestamp(dt: datetime) -> str:
    """Formats datetime for display in emails."""
    return dt.strftime('%Y-%m-%d %H:%M:%S UTC')

def dag_success_email(**kwargs: Dict[str, Any]) -> None:
    """Send enhanced email notification when a DAG completes successfully."""
    dag_id = kwargs['dag'].dag_id
    execution_date = kwargs['execution_date']
    duration = kwargs.get('duration', 'N/A')
    
    html_content = f"""
    {_get_email_style()}
    <div class="container">
        <div class="header success">
            <h2>✅ DAG Execution Successful</h2>
        </div>
        <div class="content">
            <h3>DAG Details</h3>
            <p><strong>DAG ID:</strong> {dag_id}</p>
            <p><strong>Execution Date:</strong> {_format_timestamp(execution_date)}</p>
            <p><strong>Duration:</strong> {duration}</p>
            
            <div class="details">
                <h4>Additional Information</h4>
                <p>All tasks in the DAG completed successfully.</p>
                <p class="timestamp">Email sent at: {_format_timestamp(datetime.now())}</p>
            </div>
        </div>
    </div>
    """
    
    send_email(
        to=RECIPIENTS,
        subject=f"✅ Success: {dag_id} - {_format_timestamp(execution_date)}",
        html_content=html_content,
    )

def dag_failure_email(**kwargs: Dict[str, Any]) -> None:
    """Send enhanced email notification when a DAG fails."""
    dag_id = kwargs['dag'].dag_id
    execution_date = kwargs['execution_date']
    exception = kwargs.get('exception', 'No exception information available')
    duration = kwargs.get('duration', 'N/A')
    
    # Get the full traceback if available
    try:
        full_traceback = traceback.format_exc()
    except:
        full_traceback = str(exception)
    
    html_content = f"""
    {_get_email_style()}
    <div class="container">
        <div class="header failure">
            <h2>❌ DAG Execution Failed</h2>
        </div>
        <div class="content">
            <h3>DAG Details</h3>
            <p><strong>DAG ID:</strong> {dag_id}</p>
            <p><strong>Execution Date:</strong> {_format_timestamp(execution_date)}</p>
            <p><strong>Duration:</strong> {duration}</p>
            
            <div class="details">
                <h4>Error Information</h4>
                <div class="error-details">
                    <pre>{full_traceback}</pre>
                </div>
                <p class="timestamp">Email sent at: {_format_timestamp(datetime.now())}</p>
            </div>
        </div>
    </div>
    """
    
    send_email(
        to=RECIPIENTS,
        subject=f"❌ Failure: {dag_id} - {_format_timestamp(execution_date)}",
        html_content=html_content,
    )

def task_success_email(**kwargs: Dict[str, Any]) -> None:
    """Send enhanced email notification when a task completes successfully."""
    task_id = kwargs['task_instance'].task_id
    dag_id = kwargs['task_instance'].dag_id
    execution_date = kwargs['execution_date']
    duration = kwargs['task_instance'].duration
    
    html_content = f"""
    {_get_email_style()}
    <div class="container">
        <div class="header success">
            <h2>✅ Task Execution Successful</h2>
        </div>
        <div class="content">
            <h3>Task Details</h3>
            <p><strong>Task ID:</strong> {task_id}</p>
            <p><strong>DAG ID:</strong> {dag_id}</p>
            <p><strong>Execution Date:</strong> {_format_timestamp(execution_date)}</p>
            <p><strong>Duration:</strong> {duration:.2f} seconds</p>
            
            <div class="details">
                <h4>Additional Information</h4>
                <p>Task completed successfully without any errors.</p>
                <p class="timestamp">Email sent at: {_format_timestamp(datetime.now())}</p>
            </div>
        </div>
    </div>
    """
    
    send_email(
        to=RECIPIENTS,
        subject=f"✅ Success: {task_id} in {dag_id} - {_format_timestamp(execution_date)}",
        html_content=html_content,
    )

def task_failure_email(**kwargs: Dict[str, Any]) -> None:
    """Send enhanced email notification when a task fails."""
    task_id = kwargs['task_instance'].task_id
    dag_id = kwargs['task_instance'].dag_id
    execution_date = kwargs['execution_date']
    duration = kwargs['task_instance'].duration
    exception = kwargs.get('exception', 'No exception information available')
    
    # Get the full traceback if available
    try:
        full_traceback = traceback.format_exc()
    except:
        full_traceback = str(exception)
    
    html_content = f"""
    {_get_email_style()}
    <div class="container">
        <div class="header failure">
            <h2>❌ Task Execution Failed</h2>
        </div>
        <div class="content">
            <h3>Task Details</h3>
            <p><strong>Task ID:</strong> {task_id}</p>
            <p><strong>DAG ID:</strong> {dag_id}</p>
            <p><strong>Execution Date:</strong> {_format_timestamp(execution_date)}</p>
            <p><strong>Duration:</strong> {duration:.2f} seconds</p>
            
            <div class="details">
                <h4>Error Information</h4>
                <div class="error-details">
                    <pre>{full_traceback}</pre>
                </div>
                <p class="timestamp">Email sent at: {_format_timestamp(datetime.now())}</p>
            </div>
        </div>
    </div>
    """
    
    send_email(
        to=RECIPIENTS,
        subject=f"❌ Failure: {task_id} in {dag_id} - {_format_timestamp(execution_date)}",
        html_content=html_content,
    )