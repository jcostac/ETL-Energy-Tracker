from dotenv import load_dotenv
import os

# Add debug print to show where .env is being searched
#print(f"Current working directory: {os.getcwd()}")

# Load .env file and get the result
result = load_dotenv()
#print(f"Loaded .env file: {result}")  # Should print True if .env was found and loaded

#mySQL DB connection url function using lambda
DB_URL = lambda database: f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{database}"

#valid dataset types for raw and processed files
VALID_DATASET_TYPES = ['precios', 'volumenes_i90', 'volumenes_i3', 'ingresos', "precios_i90", "volumenes_omie"]
