class ETLPipelineStatus:
    """
    Class to track the status of the ETL pipeline.
    """
    def __init__(self):
        self.extraction_status = {"success": False, "details": {}}
        self.transform_status = {"success": False, "details": {}}
        self.load_status = {"success": False, "details": {}}
        self.overall_success = False
        
    def set_extraction_status(self, success, details):
        """
        Set the status of the extraction stage.
        """
        self.extraction_status = {"success": success, "details": details}
        self._update_overall_status()

    def set_transform_status(self, success, details):
        """
        Set the status of the transformation stage.
        """
        self.transform_status = {"success": success, "details": details}
        self._update_overall_status()

    def set_load_status(self, success, details):
        """
        Set the status of the load stage.
        """
        self.load_status = {"success": success, "details": details}
        self._update_overall_status()
    
    def _update_overall_status(self):
        """
        Update the overall success status of the pipeline.
        """
        # Pipeline is successful only if all stages are successful
        self.overall_success = (
            self.extraction_status["success"] and
            self.transform_status["success"] and
            self.load_status["success"]
        )
    
    def get_summary(self):
        """Returns a complete summary of the pipeline execution"""
        return {
            "overall_success": self.overall_success,
            "extraction": self.extraction_status,
            "transformation": self.transform_status,
            "loading": self.load_status
        }
        
