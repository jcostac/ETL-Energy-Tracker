class ETLPipelineStatus:
    """
    Class to track the status of the ETL pipeline.
    """
    def __init__(self):
        """
        Initialize the ETLPipelineStatus with default failure states and empty details for extraction, transformation, and loading stages, and set the overall success flag to False.
        """
        self.extraction_status = {"success": False, "details": {}}
        self.transform_status = {"success": False, "details": {}}
        self.load_status = {"success": False, "details": {}}
        self.overall_success = False
        
    def set_extraction_status(self, success, details):
        """
        Update the extraction stage status with the given success flag and details.
        
        Parameters:
            success (bool): Indicates whether the extraction stage succeeded.
            details (dict): Additional information about the extraction stage outcome.
        """
        self.extraction_status = {"success": success, "details": details}
        self._update_overall_status()

    def set_transform_status(self, success, details):
        """
        Update the transformation stage status with the provided success flag and details.
        
        Parameters:
        	success (bool): Indicates whether the transformation stage succeeded.
        	details (dict): Additional information about the transformation stage execution.
        """
        self.transform_status = {"success": success, "details": details}
        self._update_overall_status()

    def set_load_status(self, success, details):
        """
        Update the status and details of the load stage in the ETL pipeline.
        
        Parameters:
        	success (bool): Indicates whether the load stage was successful.
        	details (dict): Additional information about the load stage execution.
        """
        self.load_status = {"success": success, "details": details}
        self._update_overall_status()
    
    def _update_overall_status(self):
        """
        Update the overall pipeline success flag based on the success of extraction, transformation, and loading stages.
        
        The overall success is set to True only if all individual stage success flags are True; otherwise, it is set to False.
        """
        # Pipeline is successful only if all stages are successful
        self.overall_success = (
            self.extraction_status["success"] and
            self.transform_status["success"] and
            self.load_status["success"]
        )
    
    def get_summary(self):
        """
        Return a dictionary summarizing the ETL pipeline execution, including overall success and detailed status for each stage.
        
        Returns:
            dict: A summary containing the overall success flag and the status dictionaries for extraction, transformation, and loading stages.
        """
        return {
            "overall_success": self.overall_success,
            "extraction": self.extraction_status,
            "transformation": self.transform_status,
            "loading": self.load_status
        }