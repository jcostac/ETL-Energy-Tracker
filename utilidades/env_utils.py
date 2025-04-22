import os
import pretty_errors
from dotenv import load_dotenv

class EnvUtils:
    def __init__(self):
        load_dotenv() #load .env file

    def check_dev_env(self):
        """
        Check if the environment is development or production.
        """
        dev = os.getenv('DEV') == 'True'
        prod = os.getenv('PROD') == 'True'
        
        if dev:
            print("Working in development environment.")
            return dev, prod
        elif prod:
            print("Working in production environment.")
            return dev, prod
        else:
            raise ValueError("Environment not set. Please check your .env file for DEV and PROD variables.")
            

if __name__ == "__main__":
    env_utils = EnvUtils()
    print(env_utils.check_dev_env())
    