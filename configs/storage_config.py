from dotenv import load_dotenv
import os
load_dotenv()

#Data lake base path
DATA_LAKE_BASE_PATH = "/Users/jjcosta/Desktop/git repo/timescale_v_duckdb_testing/data"

#mySQL DB connection url function using lambda
DB_URL = lambda database: f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{database}"

#valid dataset types for raw and processed files
VALID_DATASET_TYPES = ['precios', 'volumenes_i90', 'volumenes_i3', 'ingresos']