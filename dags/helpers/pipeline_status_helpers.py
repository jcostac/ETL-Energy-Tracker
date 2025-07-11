from helpers.pipeline_status import ETLPipelineStatus


def update_pipeline_stage_status(stage_name: str, current_stage_task_id: str, **context) -> any:
    """
    Update the ETL pipeline status for a given stage, propagate the status via Airflow XCom, and raise an error if the stage failed.
    
    Processes the output of the specified pipeline stage, updates the cumulative pipeline status using an `ETLPipelineStatus` object, and pushes the updated status summary to XCom for downstream tasks. If the current stage fails, raises a `ValueError` with a detailed failure message. Returns the relevant data from the current stage's output for use by the next pipeline task.
    
    Parameters:
        stage_name (str): The name of the current pipeline stage ("extraction", "transformation", or "load").
        current_stage_task_id (str): The Airflow task ID that produced the output for this stage.
    
    Returns:
        The relevant data output from the processed stage, suitable for downstream pipeline tasks.
    
    Raises:
        ValueError: If the stage output is missing from XCom, the stage name is unrecognized, or the current stage reports failure.
    """
    ti = context['ti']
    stage_output = ti.xcom_pull(task_ids=current_stage_task_id)

    if stage_output is None:
        raise ValueError(f"No output found in XCom for task_id: {current_stage_task_id} in stage: {stage_name}")

    pipeline_status = _get_previous_pipeline_status(ti)
    
    current_stage_success = False
    current_stage_details_or_messages = {} 
    data_to_return = None
    failure_error_message = None

    if stage_name == "extraction":
        current_stage_success, current_stage_details_or_messages, data_to_return = _handle_extraction_stage(stage_output, pipeline_status)
    
    elif stage_name == "transformation":
        # Maintain previous extraction status
        summary = pipeline_status.get_summary()
        pipeline_status.set_extraction_status(
            summary['extraction']['success'],
            summary['extraction']['details']
        )
        current_stage_success, current_stage_details_or_messages, data_to_return = _handle_transformation_stage(stage_output, pipeline_status)
        
    elif stage_name == "load":
        # Maintain previous extraction and transformation statuses
        summary = pipeline_status.get_summary()
        pipeline_status.set_extraction_status(
            summary['extraction']['success'],
            summary['extraction']['details']
        )
        pipeline_status.set_transform_status(
            summary['transformation']['success'],
            summary['transformation']['details']
        )
        current_stage_success, current_stage_details_or_messages, data_to_return, failure_error_message = _handle_load_stage(stage_output, pipeline_status)
    
    else:
        raise ValueError(f"Unknown stage_name: {stage_name}")

    ti.xcom_push(key='pipeline_status', value=pipeline_status.get_summary())

    if not current_stage_success:
        error_message_to_raise = failure_error_message if stage_name == "load" and failure_error_message else f"{stage_name.capitalize()} stage failed: {current_stage_details_or_messages}"
        raise ValueError(error_message_to_raise)

    return data_to_return


def _get_previous_pipeline_status(ti) -> ETLPipelineStatus:
    """
    Retrieve and initialize the ETL pipeline status from Airflow XCom.
    
    If a previous pipeline status summary exists in XCom, sets the extraction and transformation statuses on a new ETLPipelineStatus object accordingly.
    
    Returns:
        ETLPipelineStatus: The initialized pipeline status object reflecting any prior extraction and transformation results.
    """
    previous_pipeline_status_summary = ti.xcom_pull(key='pipeline_status')
    pipeline_status = ETLPipelineStatus()

    if previous_pipeline_status_summary:
        pipeline_status.set_extraction_status(
            previous_pipeline_status_summary['extraction']['success'],
            previous_pipeline_status_summary['extraction']['details']
        )
        pipeline_status.set_transform_status(
            previous_pipeline_status_summary['transformation']['success'],
            previous_pipeline_status_summary['transformation']['details']
        )
    return pipeline_status


def _handle_extraction_stage(stage_output: dict, pipeline_status: ETLPipelineStatus) :
    """
    Update the pipeline status with the results of the extraction stage and return extraction outcome details.
    
    Parameters:
    	stage_output (dict): Output dictionary from the extraction stage, expected to contain 'success' and 'details' keys.
    
    Returns:
    	tuple: (success, details, stage_output), where success is a boolean indicating extraction success, details is a dictionary with extraction metadata, and stage_output is the original extraction output.
    """
    success = stage_output.get('success', False)
    details = stage_output.get('details', {})
    pipeline_status.set_extraction_status(success, details)
    return success, details, stage_output


def _handle_transformation_stage(stage_output: dict, pipeline_status: ETLPipelineStatus) :
    """
    Update the pipeline status with the results of the transformation stage and extract transformation data.
    
    Parameters:
        stage_output (dict): Output dictionary from the transformation stage, expected to contain a 'status' dict and optional 'data'.
        pipeline_status (ETLPipelineStatus): Pipeline status object to be updated with transformation results.
    
    Returns:
        tuple: (success (bool), details (dict), data_to_return (any)) where 'success' indicates if the transformation succeeded, 'details' provides additional information, and 'data_to_return' contains the transformation output data.
    """
    transform_status_dict = stage_output.get('status', {'success': False, 'details': {}})
    success = transform_status_dict.get('success', False)
    details = transform_status_dict.get('details', {})
    pipeline_status.set_transform_status(success, details)
    data_to_return = stage_output.get('data', {})
    return success, details, data_to_return


def _handle_load_stage(stage_output: dict, pipeline_status: ETLPipelineStatus) :
    """
    Process the output of the load stage, update the pipeline status, and generate failure details if applicable.
    
    Parameters:
        stage_output (dict): Output dictionary from the load stage, expected to include 'success', 'messages', and optionally 'market_status'.
        pipeline_status (ETLPipelineStatus): The pipeline status object to update with load results.
    
    Returns:
        tuple: (success, messages, stage_output, failure_error_message), where `failure_error_message` is a string describing failed markets or general failure, or None if the load succeeded.
    """
    success = stage_output.get('success', False)
    messages = stage_output.get('messages', [])
    pipeline_status.set_load_status(success, messages)
    
    failure_error_message = None
    if not success:
        failed_markets = [
            market for market, market_success in stage_output.get('market_status', {}).items() if not market_success
        ]
        if failed_markets:
            failure_error_message = f"Load failed for markets: {', '.join(failed_markets)}"
        else:
            failure_error_message = f"Load failed: {messages}"
    else:
        for message in messages: # Log success messages from load
            print(message)
        print(f"Pipeline completed successfully: {pipeline_status.get_summary()}")
            
    return success, messages, stage_output, failure_error_message



