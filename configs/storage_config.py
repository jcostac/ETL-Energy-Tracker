from dotenv import load_dotenv
import os

# Add debug print to show where .env is being searched
#print(f"Current working directory: {os.getcwd()}")

# Load .env file and get the result
result = load_dotenv()
#print(f"Loaded .env file: {result}")  # Should print True if .env was found and loaded

#Data lake base path
DATA_LAKE_BASE_PATH_DEV= os.getenv('DATA_LAKE_BASE_PATH_DEV')
DATA_LAKE_BASE_PATH_PROD = os.getenv('DATA_LAKE_BASE_PATH_PROD')

# Add a default value in case ENV is None
DATA_LAKE_BASE_PATH = DATA_LAKE_BASE_PATH_DEV if os.getenv('DEV') == 'True' else DATA_LAKE_BASE_PATH_PROD

#mySQL DB connection url function using lambda
DB_URL = lambda database: f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{database}"

#valid dataset types for raw and processed files
VALID_DATASET_TYPES = ['precios', 'volumenes_i90', 'volumenes_i3', 'ingresos', "precios_i90", "volumenes_omie"]
