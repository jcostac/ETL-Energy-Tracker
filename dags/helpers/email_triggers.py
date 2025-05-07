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
        /* Base Styles */
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            margin: 0;
            padding: 0;
            background-color: #f5f7fa;
        }
        
        /* Container */
        .container {
            max-width: 650px;
            margin: 0 auto;
            padding: 0;
            background-color: #ffffff;
            border-radius: 8px;
            box-shadow: 0 3px 10px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }
        
        /* Headers */
        .header {
            padding: 24px;
            text-align: center;
            border-bottom: 1px solid rgba(0, 0, 0, 0.08);
        }
        
        .header h2 {
            margin: 0;
            font-size: 22px;
            font-weight: 600;
        }
        
        .success-header {
            background-color: #edf7ed;
            color: #1e4620;
            border-top: 6px solid #4caf50;
        }
        
        .failure-header {
            background-color: #fdeded;
            color: #5f2120;
            border-top: 6px solid #ef5350;
        }
        
        /* Content Sections */
        .content {
            padding: 24px;
            background-color: #ffffff;
        }
        
        /* Section Headings */
        .section-title {
            font-size: 18px;
            font-weight: 600;
            margin: 0 0 16px 0;
            padding-bottom: 8px;
            border-bottom: 1px solid #eaeaea;
            color: #424242;
        }
        
        /* Data Points */
        .data-point {
            margin-bottom: 12px;
            display: flex;
            align-items: flex-start;
        }
        
        .label {
            font-weight: 600;
            color: #616161;
            width: 140px;
            text-align: right;
            padding-right: 16px;
        }
        
        .value {
            flex: 1;
            color: #212121;
        }
        
        /* Status Indicators */
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 10px;
        }
        
        .success-indicator {
            background-color: #4caf50;
        }
        
        .failure-indicator {
            background-color: #ef5350;
        }
        
        /* Details Card */
        .details-card {
            margin-top: 24px;
            padding: 20px;
            background-color: #f9f9f9;
            border-radius: 6px;
            border: 1px solid #eaeaea;
        }
        
        /* Error Display */
        .error-details {
            background-color: #fff;
            padding: 15px;
            border-left: 4px solid #ef5350;
            font-family: 'Courier New', monospace;
            font-size: 13px;
            overflow-x: auto;
            margin-top: 12px;
            border-radius: 4px;
        }
        
        /* Code Block */
        pre {
            white-space: pre-wrap;
            word-wrap: break-word;
            margin: 0;
            font-size: 13px;
            background-color: #f8f9fb;
            padding: 12px;
            border-radius: 4px;
            border: 1px solid #e0e0e0;
        }
        
        /* Footer */
        .footer {
            padding: 16px 24px;
            color: #757575;
            font-size: 12px;
            text-align: center;
            background-color: #f5f7fa;
            border-top: 1px solid #eaeaea;
        }
        
        /* Timestamp */
        .timestamp {
            color: #9e9e9e;
            font-size: 13px;
            margin-top: 20px;
            text-align: right;
            font-style: italic;
        }
        
        /* Responsive adjustments */
        @media only screen and (max-width: 600px) {
            .data-point {
                flex-direction: column;
            }
            
            .label {
                width: 100%;
                text-align: left;
                padding-right: 0;
                margin-bottom: 4px;
            }
        }
    </style>
    """

def _format_timestamp(dt: datetime) -> str:
    """Formats datetime for display in emails."""
    return dt.strftime('%Y-%m-%d %H:%M:%S UTC')

def _format_duration(seconds: float) -> str:
    """Formats duration in a human-readable way."""
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        return f"{int(minutes)} min {int(remaining_seconds)} sec"
    else:
        hours = seconds // 3600
        remaining = seconds % 3600
        minutes = remaining // 60
        seconds = remaining % 60
        return f"{int(hours)} hr {int(minutes)} min {int(seconds)} sec"

def dag_success_email(**kwargs: Dict[str, Any]) -> None:
    """Send enhanced email notification when a DAG completes successfully."""
    dag_id = kwargs['dag'].dag_id
    execution_date = kwargs['execution_date']
    duration_raw = kwargs.get('duration', None)
    
    # Format duration if available
    if duration_raw and isinstance(duration_raw, (int, float)):
        duration = _format_duration(float(duration_raw))
    else:
        duration = 'N/A'
    
    html_content = f"""
    {_get_email_style()}
    <div class="container">
        <div class="header success-header">
            <h2>✅ DAG Execution Successful</h2>
        </div>
        
        <div class="content">
            <h3 class="section-title">DAG Details</h3>
            
            <div class="data-point">
                <div class="label">DAG ID:</div>
                <div class="value"><strong>{dag_id}</strong></div>
            </div>
            
            <div class="data-point">
                <div class="label">Execution Date:</div>
                <div class="value">{_format_timestamp(execution_date)}</div>
            </div>
            
            <div class="data-point">
                <div class="label">Duration:</div>
                <div class="value">{duration}</div>
            </div>
            
            <div class="details-card">
                <h4 class="section-title">Execution Summary</h4>
                <p>All tasks in the DAG completed successfully without any errors.</p>
                
                <p class="timestamp">Email sent at: {_format_timestamp(datetime.now())}</p>
            </div>
        </div>
        
        <div class="footer">
            This is an automated notification from Airflow. Please do not reply to this email.
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
    duration_raw = kwargs.get('duration', None)
    
    # Format duration if available
    if duration_raw and isinstance(duration_raw, (int, float)):
        duration = _format_duration(float(duration_raw))
    else:
        duration = 'N/A'
    
    # Get the full traceback if available
    try:
        full_traceback = traceback.format_exc()
        if full_traceback == 'NoneType: None\n':  # Common case when exception is passed directly
            full_traceback = str(exception)
    except:
        full_traceback = str(exception)
    
    html_content = f"""
    {_get_email_style()}
    <div class="container">
        <div class="header failure-header">
            <h2>❌ DAG Execution Failed</h2>
        </div>
        
        <div class="content">
            <h3 class="section-title">DAG Details</h3>
            
            <div class="data-point">
                <div class="label">DAG ID:</div>
                <div class="value"><strong>{dag_id}</strong></div>
            </div>
            
            <div class="data-point">
                <div class="label">Execution Date:</div>
                <div class="value">{_format_timestamp(execution_date)}</div>
            </div>
            
            <div class="data-point">
                <div class="label">Duration:</div>
                <div class="value">{duration}</div>
            </div>
            
            <div class="details-card">
                <h4 class="section-title">Error Information</h4>
                <div class="data-point">
                    <div class="label">Status:</div>
                    <div class="value">
                        <span class="status-indicator failure-indicator"></span>
                        <strong>Failed</strong>
                    </div>
                </div>
                
                <div class="error-details">
                    <pre>{full_traceback}</pre>
                </div>
                
                <p class="timestamp">Email sent at: {_format_timestamp(datetime.now())}</p>
            </div>
        </div>
        
        <div class="footer">
            This is an automated notification from Airflow. Please do not reply to this email.
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
    
    # Format duration
    duration_raw = kwargs['task_instance'].duration
    duration = _format_duration(duration_raw) if duration_raw is not None else 'N/A'
    
    html_content = f"""
    {_get_email_style()}
    <div class="container">
        <div class="header success-header">
            <h2>✅ Task Execution Successful</h2>
        </div>
        
        <div class="content">
            <h3 class="section-title">Task Details</h3>
            
            <div class="data-point">
                <div class="label">Task ID:</div>
                <div class="value"><strong>{task_id}</strong></div>
            </div>
            
            <div class="data-point">
                <div class="label">DAG ID:</div>
                <div class="value">{dag_id}</div>
            </div>
            
            <div class="data-point">
                <div class="label">Execution Date:</div>
                <div class="value">{_format_timestamp(execution_date)}</div>
            </div>
            
            <div class="data-point">
                <div class="label">Duration:</div>
                <div class="value">{duration}</div>
            </div>
            
            <div class="data-point">
                <div class="label">Status:</div>
                <div class="value">
                    <span class="status-indicator success-indicator"></span>
                    <strong>Success</strong>
                </div>
            </div>
            
            <div class="details-card">
                <h4 class="section-title">Execution Summary</h4>
                <p>Task completed successfully without any errors.</p>
                
                <p class="timestamp">Email sent at: {_format_timestamp(datetime.now())}</p>
            </div>
        </div>
        
        <div class="footer">
            This is an automated notification from Airflow. Please do not reply to this email.
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
    exception = kwargs.get('exception', 'No exception information available')
    
    # Format duration
    duration_raw = kwargs['task_instance'].duration
    duration = _format_duration(duration_raw) if duration_raw is not None else 'N/A'
    
    # Get the full traceback if available
    try:
        full_traceback = traceback.format_exc()
        if full_traceback == 'NoneType: None\n':  # Common case when exception is passed directly
            full_traceback = str(exception)
    except:
        full_traceback = str(exception)
    
    html_content = f"""
    {_get_email_style()}
    <div class="container">
        <div class="header failure-header">
            <h2>❌ Task Execution Failed</h2>
        </div>
        
        <div class="content">
            <h3 class="section-title">Task Details</h3>
            
            <div class="data-point">
                <div class="label">Task ID:</div>
                <div class="value"><strong>{task_id}</strong></div>
            </div>
            
            <div class="data-point">
                <div class="label">DAG ID:</div>
                <div class="value">{dag_id}</div>
            </div>
            
            <div class="data-point">
                <div class="label">Execution Date:</div>
                <div class="value">{_format_timestamp(execution_date)}</div>
            </div>
            
            <div class="data-point">
                <div class="label">Duration:</div>
                <div class="value">{duration}</div>
            </div>
            
            <div class="data-point">
                <div class="label">Status:</div>
                <div class="value">
                    <span class="status-indicator failure-indicator"></span>
                    <strong>Failed</strong>
                </div>
            </div>
            
            <div class="details-card">
                <h4 class="section-title">Error Information</h4>
                <div class="error-details">
                    <pre>{full_traceback}</pre>
                </div>
                
                <p class="timestamp">Email sent at: {_format_timestamp(datetime.now())}</p>
            </div>
        </div>
        
        <div class="footer">
            This is an automated notification from Airflow. Please do not reply to this email.
        </div>
    </div>
    """
    
    send_email(
        to=RECIPIENTS,
        subject=f"❌ Failure: {task_id} in {dag_id} - {_format_timestamp(execution_date)}",
        html_content=html_content,
    )