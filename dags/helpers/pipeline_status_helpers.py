from helpers.pipeline_status import ETLPipelineStatus


def update_pipeline_stage_status(stage_name: str, current_stage_task_id: str, **context) -> any:
    """
    Processes the output of a pipeline stage, updates the overall pipeline status,
    pushes it to XCom, and raises an error if the current stage failed.
    Returns the relevant data from the current stage's output for the next task.

    Args:
        stage_name (str): The name of the current stage (e.g., 'extraction', 'transformation', 'load').
        current_stage_task_id (str): The task_id of the Airflow operator that produced the output for this stage.
        **context: Airflow context, providing access to XComs and task instance information.

    Returns:
        any: The data output from the processed stage, intended for use by the next task in the pipeline.

    Raises:
        ValueError: If the stage output is not found in XCom, if the stage_name is unknown,
                    or if the processed stage itself reported a failure.
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
    Retrieves the previous pipeline status from XCom and initializes an ETLPipelineStatus object.

    Args:
        ti: Airflow task instance.

    Returns:
        ETLPipelineStatus: An initialized ETLPipelineStatus object.
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
    Handles the logic for the extraction stage.

    Args:
        stage_output (dict): The output from the extraction task.
        pipeline_status (ETLPipelineStatus): The current pipeline status object.

    Returns:
        tuple[bool, dict, any]: A tuple containing the success status, details, and data to return.
    """
    success = stage_output.get('success', False)
    details = stage_output.get('details', {})
    pipeline_status.set_extraction_status(success, details)
    return success, details, stage_output


def _handle_transformation_stage(stage_output: dict, pipeline_status: ETLPipelineStatus) :
    """
    Handles the logic for the transformation stage.

    Args:
        stage_output (dict): The output from the transformation task.
        pipeline_status (ETLPipelineStatus): The current pipeline status object.

    Returns:
        tuple[bool, dict, any]: A tuple containing the success status, details, and data to return.
    """
    transform_status_dict = stage_output.get('status', {'success': False, 'details': {}})
    success = transform_status_dict.get('success', False)
    details = transform_status_dict.get('details', {})
    pipeline_status.set_transform_status(success, details)
    data_to_return = stage_output.get('data', {})
    return success, details, data_to_return


def _handle_load_stage(stage_output: dict, pipeline_status: ETLPipelineStatus) :
    """
    Handles the logic for the load stage.

    Args:
        stage_output (dict): The output from the load task.
        pipeline_status (ETLPipelineStatus): The current pipeline status object.

    Returns:
        tuple[bool, list, any, str | None]: A tuple containing the success status, messages, data to return, 
                                         and a specific failure error message if applicable.
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



