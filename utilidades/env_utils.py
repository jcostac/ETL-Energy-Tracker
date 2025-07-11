import os
import pretty_errors
from dotenv import load_dotenv

class EnvUtils:
    _env_checked = False

    def __init__(self):
        """
        Initialize the EnvUtils instance by loading environment variables from a .env file and verifying that all required variables are set.
        """
        load_dotenv() #load .env file
        self.check_env() #check that all variables needed are present in the environment on init

    def check_env(self):
        """
        Verify that all required environment variables are set, raising an error if any are missing.
        
        Raises:
            ValueError: If any of the required environment variables are not set.
        """
        if self._env_checked:
            return
        
        #check that all variables needed are present in the environment
        if os.getenv('DB_TYPE') is None:
            raise ValueError("DB_TYPE is not set in the environment.")
        if os.getenv('DB_USER') is None:
            raise ValueError("DB_USER is not set in the environment.")
        if os.getenv('DB_PASSWORD') is None:
            raise ValueError("DB_PASSWORD is not set in the environment.")
        if os.getenv('DB_HOST') is None:
            raise ValueError("DB_HOST is not set in the environment.")
        if os.getenv('DB_PORT') is None:
            raise ValueError("DB_PORT is not set in the environment.")
        if os.getenv('DATA_LAKE_PATH') is None:
            raise ValueError("DATA_LAKE_PATH is not set in the environment.")
        if os.getenv('GEMINI_API_KEY') is None:
            raise ValueError("GEMINI_API_KEY is not set in the environment.")
        if os.getenv('ESIOS_TOKEN') is None:
            raise ValueError("ESIOS_TOKEN is not set in the environment.")
        
        print("Environment variables checked successfully.")
        EnvUtils._env_checked = True

if __name__ == "__main__":
    env_utils = EnvUtils()
    