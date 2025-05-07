from dags.helpers.pipeline_status import ETLPipelineStatus

def process_extraction_output(**context):
    """Process extraction output and update pipeline status"""
    # Get extraction output
    extraction_output = context['ti'].xcom_pull(task_ids='extract_esios_prices')
    
    # Initialize pipeline status
    pipeline_status = ETLPipelineStatus()
    pipeline_status.set_extraction_status(
        extraction_output.get('success', False), 
        extraction_output.get('details', {})
    )
    
    # Push pipeline status to XCom
    context['ti'].xcom_push(key='pipeline_status', value=pipeline_status.get_summary())
    
    # If extraction failed, raise exception to fail the task
    if not extraction_output.get('success', False):
        raise ValueError(f"Extraction failed: {extraction_output.get('details', {})}")
    
    # Return the raw extraction data for transform task
    return extraction_output

def process_transform_output(**context):
    """Process transform output and update pipeline status"""
    # Get transform output
    transform_output = context['ti'].xcom_pull(task_ids='transform_esios_prices')
    
    # Get pipeline status from XCom
    pipeline_status_dict = context['ti'].xcom_pull(key='pipeline_status')
    pipeline_status = ETLPipelineStatus()
    pipeline_status.set_extraction_status(
        pipeline_status_dict['extraction']['success'],
        pipeline_status_dict['extraction']['details']
    )
    
    # Set transform status
    transform_status = transform_output.get('status', {'success': False})
    pipeline_status.set_transform_status(
        transform_status.get('success', False),
        transform_status.get('details', {})
    )
    
    # Push updated pipeline status to XCom
    context['ti'].xcom_push(key='pipeline_status', value=pipeline_status.get_summary())
    
    # If transform failed, raise exception to fail the task
    if not transform_status.get('success', False):
        raise ValueError(f"Transformation failed: {transform_status.get('details', {})}")
    
    # Return the transformed data for load task
    return transform_output.get('data', {})

def finalize_pipeline_status(**context):
    """Finalize pipeline status after load task"""
    # Get load output
    load_output = context['ti'].xcom_pull(task_ids='load_esios_prices_to_datalake')
    
    # Get pipeline status from XCom
    pipeline_status_dict = context['ti'].xcom_pull(key='pipeline_status')
    pipeline_status = ETLPipelineStatus()
    pipeline_status.set_extraction_status(
        pipeline_status_dict['extraction']['success'],
        pipeline_status_dict['extraction']['details']
    )
    pipeline_status.set_transform_status(
        pipeline_status_dict['transformation']['success'],
        pipeline_status_dict['transformation']['details']
    )
    
    # Set load status
    pipeline_status.set_load_status(
        load_output.get('success', False),
        load_output.get('messages', [])
    )
    
    # Push final pipeline status to XCom
    context['ti'].xcom_push(key='pipeline_status', value=pipeline_status.get_summary())
    
    # If load failed, raise exception to fail the task
    if not load_output.get('success', False):
        raise ValueError(f"Load failed: {load_output.get('messages', [])}")
    
    # Log success message
    print(f"Pipeline completed successfully: {pipeline_status.get_summary()}")
